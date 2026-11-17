# -*- coding: utf-8 -*-
"""
copilot.py — POS売上分析Copilot（Tool Calling ＋ 構造化出力 ＋ 回答検証）

構成
  1. データ概要・利用可能期間・基準日・使えるツールを質問とともに生成AIへ渡す
  2. 生成AIからツール名と引数を受け取る（条件が足りなければ確認質問を返す）
  3. Python側でツール名・型・期間・対象値・呼出し回数上限を検証する（ToolGuard）
  4. 許可した関数を実行し、結果ID・集計条件・表・警告を保存する（ResultStore）
  5. 生成AIに集計結果を渡し、事実・仮説・追加確認を指定のJSON形式で返させる（構造化出力）
  6. 結果ID・指標キー・数値を検証し、確定数値はPython側の結果から表示する（verify_answer）
  7. 表・グラフ・CSVにも同じ結果を使う

生成AIの提供元は OpenAI をリファレンス実装とし、モデル名は環境変数 POS_COPILOT_MODEL で切り替える。
Claude（Anthropic）・Gemini（Google）は llm_providers.py の接続層で差し替えられる（provider="anthropic" / "gemini"）。
APIキーは環境変数 OPENAI_API_KEY から読む（Notebookや配布コードに直接書かない）。
APIに接続できないときは「模擬モード」（MockLLM：ルールベース）で同じ処理の流れを体験できる。
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from typing import Optional

import pandas as pd

import analysis_tools as at

DEFAULT_MODEL = "gpt-6.1-terra" # "gpt-5.6-luna"

# 推論の強さ。gpt-5 系は Chat Completions で関数ツール（Tool Calling）を使うとき
# reasoning_effort="none" が必要（そうしないと 400 invalid_request_error になる）。
# reasoning_effort に対応しないモデル（gpt-4o など）では、下の _create() が自動で外して呼び直す。
DEFAULT_REASONING_EFFORT = os.environ.get("POS_COPILOT_REASONING_EFFORT", "none")
MAX_TOOL_CALLS = 4             # 1回の質問で許可するツール呼出しの上限
MAX_TABLE_ROWS_TO_AI = 25      # 生成AIに渡す表の最大行数（数値の正本はPython側に残る）

# ---------------------------------------------------------------------------
# システム指示
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """あなたはD店の売上分析を支援します。
数値は必ず提供された分析ツールの結果に基づいてください。
利用可能な期間と基準日を確認し、曖昧な条件は質問してください。
事実には対応する結果IDと指標キーを付けてください。
売上の増減の内訳と、因果的な理由を区別してください。
在庫、天候、競合、販促等は、根拠データがなければ仮説としてください。
取引IDがないデータから客数や客単価を作らないでください。
集計できない場合は理由と、次に指定できる条件を返してください。

補足ルール:
- 「先週」「前週」などの相対的な期間は、必ずコンテキストに示した基準日から解決した日付を使い、回答にもその日付を含めてください。
- 期間・カテゴリ・商品がデータに存在しない場合、似た名前に勝手に置き換えず、ツールが返す候補を示して再指定を求めてください。
- ツールが error を返したら、その理由を利用者に伝え、数値を作らないでください。
- 重点商品（ABC）は売上構成上の重要度であり、利益や施策効果の大きさではありません。
"""

# ---------------------------------------------------------------------------
# AIに公開するツール定義（引数は文字列の日付・許可済みカテゴリ・列挙型・上限付き整数に制限）
# ---------------------------------------------------------------------------
DATE_DESC = "YYYY-MM-DD 形式。利用可能期間の中で指定する"


def build_tool_specs(categories: list[str]) -> list[dict]:
    cat_schema = {"anyOf": [{"type": "string", "enum": categories}, {"type": "null"}],
                  "description": "カテゴリで絞る場合に指定。全カテゴリなら null"}
    prod_schema = {"anyOf": [{"type": "string"}, {"type": "null"}],
                   "description": "特定の商品名で絞る場合に指定（正確な商品名）。指定しないなら null"}
    return [
        {"type": "function", "function": {
            "name": "summarize_sales",
            "description": "指定期間・条件の売上を集計する。unit=day で日別推移、product で商品別ランキング、category でカテゴリ別。",
            "strict": True,
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {"start": {"type": "string", "description": "開始日。" + DATE_DESC},
                                          "end": {"type": "string", "description": "終了日。" + DATE_DESC},
                                          "category": cat_schema,
                                          "unit": {"type": "string", "enum": list(at.UNITS), "description": "集計単位"},
                                          "product": prod_schema},
                           "required": ["start", "end", "category", "unit", "product"]}}},
        {"type": "function", "function": {
            "name": "weekday_top_products",
            "description": "曜日別の売上合計・収録日数・1日平均と、曜日ごとの売れ筋トップN商品（売上金額基準）。",
            "strict": True,
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {"start": {"type": "string", "description": "開始日。" + DATE_DESC},
                                          "end": {"type": "string", "description": "終了日。" + DATE_DESC},
                                          "category": cat_schema,
                                          "top_n": {"type": "integer", "description": "曜日ごとの上位件数（1〜10）"}},
                           "required": ["start", "end", "category", "top_n"]}}},
        {"type": "function", "function": {
            "name": "abc_analysis",
            "description": "商品別売上の累積構成比で商品をA・B・Cに分類する（重点商品の把握）。売上構成上の重要度であり利益ではない。",
            "strict": True,
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {"start": {"type": "string", "description": "開始日。" + DATE_DESC},
                                          "end": {"type": "string", "description": "終了日。" + DATE_DESC},
                                          "category": cat_schema},
                           "required": ["start", "end", "category"]}}},
        {"type": "function", "function": {
            "name": "compare_periods",
            "description": "当期と前期の売上を比較し、全体・商品別・カテゴリ別の増減額・増減率を返す。「先週から落ちた商品」などに使う。",
            "strict": True,
            "parameters": {"type": "object", "additionalProperties": False,
                           "properties": {"cur_start": {"type": "string", "description": "当期の開始日。" + DATE_DESC},
                                          "cur_end": {"type": "string", "description": "当期の終了日。" + DATE_DESC},
                                          "prev_start": {"type": "string", "description": "前期の開始日。" + DATE_DESC},
                                          "prev_end": {"type": "string", "description": "前期の終了日。" + DATE_DESC},
                                          "category": cat_schema,
                                          "product": prod_schema},
                           "required": ["cur_start", "cur_end", "prev_start", "prev_end", "category", "product"]}}},
    ]


# 構造化出力のスキーマ
ANSWER_SCHEMA = {
    "name": "sales_answer",
    "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "summary": {"type": "string", "description": "店長向けの結論（1〜3文）。数値は必ずツール結果のものだけ"},
            "facts": {"type": "array", "description": "データで確認できた事実。各項目に結果IDと指標キーを付ける",
                      "items": {"type": "object", "additionalProperties": False,
                                "properties": {"text": {"type": "string"},
                                               "result_id": {"type": "string", "description": "根拠となる結果ID（例 r001）"},
                                               "metric_key": {"type": "string", "description": "metrics のキー名。表の値を根拠にする場合は table"}},
                                "required": ["text", "result_id", "metric_key"]}},
            "hypotheses": {"type": "array", "items": {"type": "string"}, "description": "データでは確認できない原因候補（仮説）"},
            "next_checks": {"type": "array", "items": {"type": "string"}, "description": "次に確認すべきこと・必要なデータ"},
            "limitations": {"type": "array", "items": {"type": "string"}, "description": "このデータ・集計で言えないこと"},
            "needs_clarification": {"type": "boolean", "description": "条件が足りず利用者に確認が必要なら true"},
            "clarification_question": {"type": "string", "description": "確認が必要な場合の質問文。不要なら空文字"},
        },
        "required": ["summary", "facts", "hypotheses", "next_checks", "limitations",
                     "needs_clarification", "clarification_question"],
    },
}


# ---------------------------------------------------------------------------
# 結果の保管（数値と条件の正本）
# ---------------------------------------------------------------------------
class ResultStore:
    def __init__(self, offset: int = 0):
        self._items: dict[str, dict] = {}
        self._n = 0
        self._offset = offset     # セッション内で結果IDが重複しないようにする

    def add(self, result: dict) -> str:
        self._n += 1
        rid = f"r{self._offset + self._n:03d}"
        result = dict(result)
        result["result_id"] = rid
        self._items[rid] = result
        return rid

    def get(self, rid: str) -> Optional[dict]:
        return self._items.get(rid)

    def ids(self) -> list[str]:
        return list(self._items)

    def all(self) -> dict[str, dict]:
        return dict(self._items)

    def all_numbers(self) -> set:
        """保存済み結果に含まれるすべての数値（自由文の数字の照合用）"""
        nums = set()

        def walk(x):
            if isinstance(x, bool):
                return
            if isinstance(x, (int, float)):
                nums.add(round(abs(float(x)), 1))          # 自由文では「-12,930円」のように符号と数字が分かれるため絶対値で照合
                nums.add(float(int(round(abs(x)))))
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
        for r in self._items.values():
            walk(r.get("metrics", {}))
            walk(r.get("table", []))
            walk(r.get("extra_tables", {}))
            walk(r.get("filters", {}))
        return nums


# ---------------------------------------------------------------------------
# 引数検証と実行（AIが何を要求しても、ここを通らない処理は動かない）
# ---------------------------------------------------------------------------
class ToolGuard:
    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    def __init__(self, df: pd.DataFrame, store: ResultStore, max_calls: int = MAX_TOOL_CALLS):
        self.df = df
        self.store = store
        self.max_calls = max_calls
        self.calls = 0
        self.categories = sorted(df["カテゴリ"].unique().tolist())
        self.products = sorted(df["商品名"].unique().tolist())
        self.funcs = {"summarize_sales": at.summarize_sales, "weekday_top_products": at.weekday_top_products,
                      "abc_analysis": at.abc_analysis, "compare_periods": at.compare_periods}
        self.allowed_args = {"summarize_sales": {"start", "end", "category", "unit", "product"},
                             "weekday_top_products": {"start", "end", "category", "top_n"},
                             "abc_analysis": {"start", "end", "category"},
                             "compare_periods": {"cur_start", "cur_end", "prev_start", "prev_end", "category", "product"}}

    def validate(self, name: str, args: dict) -> Optional[dict]:
        """問題があればエラー結果辞書を返す。なければ None"""
        if name not in self.funcs:
            return at._err(name, "UNKNOWN_TOOL", f"ツール「{name}」は許可されていません", {"許可ツール": list(self.funcs)})
        if self.calls >= self.max_calls:
            return at._err(name, "CALL_LIMIT", f"ツール呼出しの上限（{self.max_calls}回）に達しました")
        extra = set(args) - self.allowed_args[name]
        if extra:
            return at._err(name, "UNKNOWN_ARGUMENT", f"不明な引数 {sorted(extra)}", {"許可引数": sorted(self.allowed_args[name])})
        for k, v in args.items():
            if k in ("start", "end", "cur_start", "cur_end", "prev_start", "prev_end"):
                if not isinstance(v, str) or not self.DATE_RE.match(v):
                    return at._err(name, "INVALID_DATE", f"{k}=「{v}」は YYYY-MM-DD 形式で指定してください")
            elif k == "category":
                if v is not None and v not in self.categories:
                    return at._err(name, "UNKNOWN_CATEGORY", f"カテゴリ「{v}」はデータにありません", {"カテゴリ候補": self.categories})
            elif k == "unit":
                if v not in at.UNITS:
                    return at._err(name, "INVALID_UNIT", f"unit=「{v}」は使えません", {"集計単位": list(at.UNITS)})
            elif k == "top_n":
                if not isinstance(v, int) or isinstance(v, bool) or not (1 <= v <= 10):
                    return at._err(name, "INVALID_TOP_N", "top_n は1〜10の整数で指定してください")
            elif k == "product":
                if v is not None and not isinstance(v, str):
                    return at._err(name, "INVALID_PRODUCT", "product は文字列で指定してください")
        return None

    def execute(self, name: str, args: dict) -> dict:
        err = self.validate(name, args)
        if err:
            err["result_id"] = self.store.add(err)
            return err
        self.calls += 1
        kwargs = {k: v for k, v in args.items() if v is not None}
        result = self.funcs[name](self.df, **kwargs)
        result["result_id"] = self.store.add(result)
        return result


def result_for_ai(result: dict) -> str:
    """生成AIに渡す結果（表は先頭N行に制限。数値の正本はResultStoreに残る）"""
    if result.get("status") != "ok":
        return json.dumps({k: result.get(k) for k in ("result_id", "status", "tool", "error_code", "message", "available_options")},
                          ensure_ascii=False)
    out = {"result_id": result["result_id"], "status": "ok", "tool": result["tool"], "filters": result["filters"],
           "metrics": result["metrics"], "warnings": result["warnings"],
           "table_rows": len(result["table"]), "table_head": result["table"][:MAX_TABLE_ROWS_TO_AI]}
    for k, v in result.get("extra_tables", {}).items():
        out[f"extra_{k}_head"] = v[:MAX_TABLE_ROWS_TO_AI]
    return json.dumps(out, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# 回答の検証（JSONとして妥当でも内容が正しいとは限らない）
# ---------------------------------------------------------------------------
NUM_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*[％%])?")


def _numbers_in_text(text: str) -> list[float]:
    """照合対象の数値を取り出す。100以上の数、小数、％付きの数を対象にし、「3つ」「5日」のような小さな整数は除外"""
    out = []
    for m in NUM_RE.finditer(text):
        s = m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else "")
        v = float(s)
        if v >= 100 or m.group(2) is not None or m.group(3):
            out.append(v)
    return out


def verify_answer(answer: dict, store: ResultStore) -> dict:
    """結果IDと指標キーの存在、自由文の数値の照合を行う。戻り値 {"ok": bool, "problems": [...]}"""
    problems = []
    for i, f in enumerate(answer.get("facts", [])):
        r = store.get(f.get("result_id", ""))
        if r is None:
            problems.append(f"facts[{i}] の結果ID「{f.get('result_id')}」は存在しません")
            continue
        if r.get("status") != "ok":
            problems.append(f"facts[{i}] はエラー結果 {f.get('result_id')} を根拠にしています")
            continue
        mk = f.get("metric_key", "")
        if mk not in ("table", "") and mk not in r.get("metrics", {}):
            problems.append(f"facts[{i}] の指標キー「{mk}」は {f.get('result_id')} の metrics にありません")
            continue
        mv = r.get("metrics", {}).get(mk)
        if isinstance(mv, (int, float)) and not isinstance(mv, bool):
            # 指標の値が事実の文中に出てくるか（カンマ有無・小数1桁で照合）
            nums = {round(abs(v), 1) for v in _numbers_in_text(f.get("text", ""))}
            if nums and round(abs(float(mv)), 1) not in nums and float(int(round(abs(mv)))) not in nums:
                problems.append(f"facts[{i}] の文中の数値が {f.get('result_id')}.{mk}={mv} と一致しません")
    # 自由文に出てくる数値（年月日・ID・100未満の小さな数は除外）が結果のどこかに存在するか
    known = store.all_numbers()
    texts = [answer.get("summary", "")] + [f.get("text", "") for f in answer.get("facts", [])]
    for t in texts:
        t2 = re.sub(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?", " ", t)   # 日付を除去
        t2 = re.sub(r"\d{1,2}月\d{1,2}日|\d{1,2}/\d{1,2}|\d{4}年|\d{1,2}月|r\d{3}", " ", t2)
        for v in _numbers_in_text(t2):
            if round(v, 1) not in known and float(int(round(v))) not in known:
                problems.append(f"文中の数値 {v:g} はツール結果に存在しません:「{t[:40]}…」")
    return {"ok": not problems, "problems": problems}


def fallback_answer(question: str, store: ResultStore, reason: str) -> dict:
    """検証に失敗したときの定型回答（数値は結果から機械的に組み立てる）"""
    facts = []
    for rid, r in store.all().items():
        if r.get("status") != "ok":
            continue
        m = r["metrics"]
        if r["tool"] == "compare_periods":
            facts.append({"text": f"当期売上 {m['当期売上']:,}円、前期売上 {m['前期売上']:,}円、増減額 {m['増減額']:+,}円", "result_id": rid, "metric_key": "増減額"})
            for name, v in m.get("減少額上位", []):
                facts.append({"text": f"{name} は {v:+,}円", "result_id": rid, "metric_key": "table"})
        elif r["tool"] == "abc_analysis":
            facts.append({"text": f"A商品は {len(m['A商品'])}品（{'、'.join(m['A商品'])}）で売上の {m['A売上構成比']}％", "result_id": rid, "metric_key": "A売上構成比"})
        elif r["tool"] == "summarize_sales":
            facts.append({"text": f"売上合計 {m['売上合計']:,}円（{m['日数']}日）", "result_id": rid, "metric_key": "売上合計"})
        elif r["tool"] == "weekday_top_products":
            facts.append({"text": f"売上合計が最も高い曜日は {m['最高売上の曜日']}、1日平均が最も高い曜日は {m['1日平均が最高の曜日']}", "result_id": rid, "metric_key": "最高売上の曜日"})
    return {"summary": f"生成AIの回答が検証を通らなかったため（{reason}）、集計結果を定型で表示します。数値は下の根拠表で確認してください。",
            "facts": facts, "hypotheses": [], "next_checks": ["根拠表を確認し、必要なら質問を具体的に言い換えてください"],
            "limitations": ["この回答は生成AIの説明を含みません"], "needs_clarification": False, "clarification_question": "",
            "_fallback": True}


# ---------------------------------------------------------------------------
# LLM：OpenAI 実接続
# ---------------------------------------------------------------------------
class OpenAILLM:
    """Chat Completions API を使う最小の接続層。plan() でツール要求、answer() で構造化回答"""

    def __init__(self, model: Optional[str] = None, client=None, reasoning_effort: Optional[str] = None):
        from openai import OpenAI   # 遅延import（模擬モードでは不要）
        self.model = model or os.environ.get("POS_COPILOT_MODEL", DEFAULT_MODEL)
        self.client = client or OpenAI()      # OPENAI_API_KEY を環境変数から読む
        self.messages: list[dict] = []
        self.usage = []
        self.reasoning_effort = DEFAULT_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        self._send_effort = bool(self.reasoning_effort)   # 非対応モデルと分かったら False にする

    def _create(self, **kwargs):
        """chat.completions.create の薄いラッパ。

        gpt-5 系は「関数ツール＋推論あり」を Chat Completions で受け付けないため
        reasoning_effort="none" を付けて呼ぶ。この引数に対応しないモデルでは
        400 が返るので、その場合だけ引数を外して1回だけ呼び直す（以降は付けない）。
        """
        if self._send_effort:
            try:
                return self.client.chat.completions.create(reasoning_effort=self.reasoning_effort, **kwargs)
            except Exception as e:
                if "reasoning_effort" not in str(e):
                    raise
                self._send_effort = False
                print(f"（注意）このモデルは reasoning_effort を受け付けないため、指定せずに呼び直します: {self.model}")
        return self.client.chat.completions.create(**kwargs)

    def start(self, system_prompt: str, context: str, question: str):
        self.messages = [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": f"# コンテキスト\n{context}\n\n# 質問\n{question}"}]

    def plan(self, tools: list[dict]) -> dict:
        """戻り値 {"tool_calls": [{"id","name","args"}], "text": str}"""
        resp = self._create(model=self.model, messages=self.messages, tools=tools, tool_choice="auto")
        self.usage.append(getattr(resp, "usage", None))
        msg = resp.choices[0].message
        calls = []
        raw_calls = []
        for tc in (msg.tool_calls or []):
            if getattr(tc, "type", "function") != "function":
                continue
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_invalid_json": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
            raw_calls.append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
        # 会話履歴にはSDKのオブジェクトではなく辞書で追加する（後続の tool メッセージと対応づけるため）
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if raw_calls:
            assistant_msg["tool_calls"] = raw_calls
        self.messages.append(assistant_msg)
        return {"tool_calls": calls, "text": msg.content or ""}

    def tool_result(self, call_id: str, content: str):
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": content})

    def answer(self, extra_instruction: str = "") -> dict:
        self.messages.append({"role": "user", "content": "以上のツール結果だけを根拠に、指定のJSON形式で店長向けの回答を作ってください。" + extra_instruction})
        resp = self._create(model=self.model, messages=self.messages,
                            response_format={"type": "json_schema", "json_schema": ANSWER_SCHEMA})
        self.usage.append(getattr(resp, "usage", None))
        msg = resp.choices[0].message
        self.messages.append({"role": "assistant", "content": msg.content})
        return json.loads(msg.content)


# ---------------------------------------------------------------------------
# LLM：模擬モード（ルールベース）。API接続なしで同じ流れを体験する
# ---------------------------------------------------------------------------
class MockLLM:
    """質問文をルールで解釈してツール要求と回答を作る。実接続とは区別して表示すること"""

    def __init__(self, base_date: str, categories: list[str], products: list[str], period: tuple[str, str]):
        self.base_date = base_date
        self.categories = categories
        self.products = products
        self.period = period
        self.question = ""
        self.results: list[dict] = []
        self.planned = False
        self.note = ""

    def start(self, system_prompt: str, context: str, question: str):
        self.question = question
        self.results, self.planned, self.note = [], False, ""

    # --- 質問の解釈 -----------------------------------------------------------
    def _periods(self) -> list[dict]:
        q = self.question
        found = []
        for m in re.finditer(r"(\d{4})年(\d{1,2})月(\d{1,2})日?[〜~～\-–](?:(\d{4})年)?(?:(\d{1,2})月)?(\d{1,2})日?", q):
            y1, m1, d1, y2, m2, d2 = m.groups()
            y2 = y2 or y1
            m2 = m2 or m1
            found.append({"開始日": f"{int(y1):04d}-{int(m1):02d}-{int(d1):02d}", "終了日": f"{int(y2):04d}-{int(m2):02d}-{int(d2):02d}"})
        q = re.sub(r"(\d{4})年(\d{1,2})月(\d{1,2})日?[〜~～\-–](?:(\d{4})年)?(?:(\d{1,2})月)?(\d{1,2})日?", " ", q)   # 年付きの期間を除いた残りから月日形式を探す
        for m in re.finditer(r"(\d{1,2})月(\d{1,2})日?[〜~～\-–](?:(\d{1,2})月)?(\d{1,2})日?", q):
            m1, d1, m2, d2 = m.groups()
            m2 = m2 or m1
            found.append({"開始日": f"2026-{int(m1):02d}-{int(d1):02d}", "終了日": f"2026-{int(m2):02d}-{int(d2):02d}"})
        if found:
            return found
        m = re.search(r"(?:(\d{4})年)?(\d{1,2})月(?!\d)", q)
        if m and not re.search(r"\d{1,2}月\d{1,2}日", q):
            y, mo = int(m.group(1) or 2026), int(m.group(2))
            last = (date(y, mo % 12 + 1, 1) - timedelta(days=1)) if mo < 12 else date(y, 12, 31)
            return [{"開始日": f"{y:04d}-{mo:02d}-01", "終了日": last.isoformat()}]
        for kw in ("先週", "前週", "今月", "先月"):
            if kw in q:
                return [at.resolve_relative_period(self.base_date, kw)]
        if "全期間" in q or "全体" in q:
            return [{"開始日": self.period[0], "終了日": self.period[1]}]
        return []

    def _category(self):
        for c in self.categories:
            if c in self.question:
                return c
        return None

    def _product(self):
        for p in sorted(self.products, key=len, reverse=True):
            if p in self.question:
                return p
        m = re.search(r"([^\s、。「」]+?)の(?:売上|販売)", self.question)
        if m and m.group(1) not in self.categories and not re.search(r"\d", m.group(1)) and m.group(1) not in ("全体", "先週", "前週", "曜日", "商品", "全期間", "店"):
            return m.group(1)   # データにない商品名 → ツール側で候補を返す
        return None

    def plan(self, tools: list[dict]) -> dict:
        if self.planned:
            return {"tool_calls": [], "text": ""}
        self.planned = True
        q = self.question
        cat, prod = self._category(), self._product()
        periods = self._periods()
        if re.search(r"客単価|客数|来店|利益|粗利|原価|在庫数", q) and not re.search(r"落ち|減|増|比較", q):
            self.note = "no_tool_metric"
            return {"tool_calls": [], "text": ""}
        compare_kw = re.search(r"落ち|減っ|減少|増え|増加|伸び|比べ|比較|前週|先週から|変わ|変化|なぜ|原因|理由", q)
        if compare_kw:
            if "先週" in q or "前週" in q or not periods:
                if not periods and "先週" not in q and "前週" not in q:
                    self.note = "clarify_period"
                    return {"tool_calls": [], "text": "比較する期間を指定してください（例：先週と前週、2026年8月24〜30日と8月17〜23日）"}
                cur = at.resolve_relative_period(self.base_date, "先週")
                prev = at.resolve_relative_period(self.base_date, "前週")
            elif len(periods) >= 2:
                cur, prev = periods[0], periods[1]
            else:
                s = date.fromisoformat(periods[0]["開始日"]); e = date.fromisoformat(periods[0]["終了日"])
                n = (e - s).days + 1
                cur, prev = periods[0], {"開始日": (s - timedelta(days=n)).isoformat(), "終了日": (s - timedelta(days=1)).isoformat()}
            args = {"cur_start": cur["開始日"], "cur_end": cur["終了日"], "prev_start": prev["開始日"], "prev_end": prev["終了日"],
                    "category": cat, "product": prod}
            calls = [{"id": "mock_1", "name": "compare_periods", "args": args}]
            if re.search(r"重点|ABC", q):
                calls.append({"id": "mock_2", "name": "abc_analysis", "args": {"start": cur["開始日"], "end": cur["終了日"], "category": cat}})
            return {"tool_calls": calls, "text": ""}
        if not periods and prod:
            # 商品名だけが指定された質問は全期間を対象にする（回答に期間を明示する）
            periods = [{"開始日": self.period[0], "終了日": self.period[1]}]
        if not periods:
            self.note = "clarify_period"
            return {"tool_calls": [], "text": f"対象期間を指定してください（利用可能期間 {self.period[0]}〜{self.period[1]}、基準日 {self.base_date}）"}
        p = periods[0]
        if re.search(r"ABC|重点|重要商品|売上構成", q):
            return {"tool_calls": [{"id": "mock_1", "name": "abc_analysis", "args": {"start": p["開始日"], "end": p["終了日"], "category": cat}}], "text": ""}
        if "曜日" in q:
            m = re.search(r"(\d+)\s*(?:つ|件|商品|位)", q)
            n = int(m.group(1)) if m else 3
            return {"tool_calls": [{"id": "mock_1", "name": "weekday_top_products", "args": {"start": p["開始日"], "end": p["終了日"], "category": cat, "top_n": n}}], "text": ""}
        unit = "day"
        if re.search(r"商品別|ランキング|売れ筋|上位", q):
            unit = "product"
        elif re.search(r"カテゴリ別|カテゴリごと", q):
            unit = "category"
        return {"tool_calls": [{"id": "mock_1", "name": "summarize_sales",
                                "args": {"start": p["開始日"], "end": p["終了日"], "category": cat, "unit": unit, "product": prod}}], "text": ""}

    def tool_result(self, call_id: str, content: str):
        self.results.append(json.loads(content))

    # --- 回答の生成（結果から機械的に文章化） ----------------------------------------
    def answer(self, extra_instruction: str = "") -> dict:
        q = self.question
        ans = {"summary": "", "facts": [], "hypotheses": [], "next_checks": [], "limitations": [],
               "needs_clarification": False, "clarification_question": ""}
        if self.note == "no_tool_metric":
            ans["summary"] = "このデータは日別・商品別の販売集計で取引IDがないため、客数・客単価・利益は算出できません。"
            ans["limitations"] = ["客数・客単価にはレシート（取引）単位のデータが必要です", "利益には原価データが必要です"]
            ans["next_checks"] = ["POSの取引明細（レシートID付き）や商品マスタの原価を入手できるか確認する"]
            return ans
        if self.note == "clarify_period" or not self.results:
            ans["needs_clarification"] = True
            ans["clarification_question"] = f"対象期間を指定してください（利用可能期間 {self.period[0]}〜{self.period[1]}、基準日 {self.base_date}。例：「先週」「2026年8月」）"
            ans["summary"] = "条件が足りないため集計していません。"
            if re.search(r"落ち|減|なぜ|原因|理由", q):
                ans["summary"] += " 期間を指定いただければ、当期と前期の売上を比較して増減の内訳（商品別・カテゴリ別）を示します。"
                ans["next_checks"].append("例：「先週から売上が落ちた商品は？」（基準日から先週・前週を解決して比較します）")
            if re.search(r"雨|天候|天気", q):
                ans["limitations"].append("このデータに天候の情報はないため、天候が原因かどうかは確認できません（仮説として扱います）")
                ans["next_checks"].append("天候の影響を調べるには、日別の気象データを別途入手して照合する必要があります")
            if re.search(r"在庫|欠品", q):
                ans["limitations"].append("このデータに在庫の情報はないため、欠品が原因かどうかは確認できません（販売0の日があっても断定できません）")
                ans["next_checks"].append("在庫履歴・欠品記録を確認する")
            return ans
        errs = [r for r in self.results if r.get("status") != "ok"]
        if errs:
            e = errs[0]
            ans["summary"] = f"集計できませんでした：{e.get('message')}"
            opts = e.get("available_options") or {}
            if opts:
                ans["next_checks"] = [f"{k}: {v}" for k, v in opts.items()][:3]
            ans["limitations"] = ["指定条件を修正して再度質問してください"]
            return ans
        causal = re.search(r"なぜ|原因|理由|せい|ですよね", q)
        for r in self.results:
            rid, m, t = r["result_id"], r["metrics"], r["tool"]
            if t == "compare_periods":
                f = r["filters"]
                ans["summary"] += (f"当期（{f['当期'][0]}〜{f['当期'][1]}）の売上は {m['当期売上']:,}円で、前期（{f['前期'][0]}〜{f['前期'][1]}）の {m['前期売上']:,}円から {m['増減額']:+,}円"
                                   + (f"（{m['増減率']:+}％）" if m["増減率"] is not None else "（前期売上0のため増減率は算出不能）") + "でした。")
                ans["facts"].append({"text": f"全体の増減額は {m['増減額']:+,}円", "result_id": rid, "metric_key": "増減額"})
                for name, v in m.get("減少額上位", []):
                    ans["facts"].append({"text": f"減少額が大きい商品：{name}（{v:+,}円）", "result_id": rid, "metric_key": "table"})
                for name, v in m.get("増加額上位", [])[:2]:
                    ans["facts"].append({"text": f"増加した商品：{name}（{v:+,}円）", "result_id": rid, "metric_key": "table"})
                ans["hypotheses"] += ["減少した商品は欠品・売場変更・販促終了などの可能性がありますが、POSデータだけでは確認できません"]
                ans["next_checks"] += ["減少商品の在庫履歴・欠品記録を確認する", "販促・売場変更の記録と日付を照合する"]
                ans["limitations"] += ["増減の内訳は分かりますが、原因はこのデータからは断定できません"]
            elif t == "abc_analysis":
                f = r["filters"]
                ans["summary"] += f" 期間 {f['開始日']}〜{f['終了日']} のA商品は {len(m['A商品'])}品で売上の {m['A売上構成比']}％を占めます。"
                ans["facts"].append({"text": f"A商品：{'、'.join(m['A商品'])}（売上構成比 {m['A売上構成比']}％）", "result_id": rid, "metric_key": "A売上構成比"})
                ans["limitations"] += ["ABCは売上構成上の重要度であり、利益や施策効果の大きさではありません"]
            elif t == "summarize_sales":
                f = r["filters"]
                ans["summary"] += f" 期間 {f['開始日']}〜{f['終了日']}" + (f"（{f['カテゴリ']}）" if f.get("カテゴリ") else "") + f" の売上合計は {m['売上合計']:,}円（{m['日数']}日、1日平均 {m['1日平均売上']:,}円）です。"
                ans["facts"].append({"text": f"売上合計 {m['売上合計']:,}円", "result_id": rid, "metric_key": "売上合計"})
                ans["facts"].append({"text": f"1日平均売上 {m['1日平均売上']:,}円", "result_id": rid, "metric_key": "1日平均売上"})
            elif t == "weekday_top_products":
                f = r["filters"]
                ans["summary"] += f" 期間 {f['開始日']}〜{f['終了日']} で売上合計が最も高い曜日は {m['最高売上の曜日']}、1日平均では {m['1日平均が最高の曜日']} です。"
                ans["facts"].append({"text": f"売上合計が最高の曜日：{m['最高売上の曜日']}", "result_id": rid, "metric_key": "最高売上の曜日"})
                ans["facts"].append({"text": f"1日平均が最高の曜日：{m['1日平均が最高の曜日']}", "result_id": rid, "metric_key": "1日平均が最高の曜日"})
                ans["limitations"] += ["曜日の出現回数が異なるため、合計と1日平均の両方を確認してください"]
            for w in r.get("warnings", []):
                ans["limitations"].append(w)
        if causal:
            ans["summary"] += " 原因についてはこのデータでは確認できないため、仮説として扱ってください。"
            if re.search(r"雨|天候|天気", q):
                ans["hypotheses"].append("天候の影響は仮説です。天候データがないため確認できません")
                ans["next_checks"].append("気象データ（日別の天候・気温）を入手して照合する")
            if re.search(r"在庫|欠品", q):
                ans["hypotheses"].append("欠品の可能性は仮説です。販売0の日があっても在庫データなしでは断定できません")
                ans["next_checks"].append("在庫履歴・欠品記録を確認する")
        ans["summary"] = ans["summary"].strip()
        return ans


# ---------------------------------------------------------------------------
# Copilot 本体
# ---------------------------------------------------------------------------
class SalesCopilot:
    PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}

    def __init__(self, df: pd.DataFrame, base_date: str = "2026-08-31", mode: str = "auto",
                 model: Optional[str] = None, store_name: str = "D店", client=None, provider: str = "auto"):
        """mode: 'api'（実接続）/ 'mock'（模擬）/ 'auto'（APIキーがあればapi、なければmock）
        provider: 'openai' / 'anthropic' / 'gemini' / 'auto'（環境変数のキーがある提供元を openai→anthropic→gemini の順で選ぶ）"""
        self.df = df
        self.base_date = base_date
        self.store_name = store_name
        self.overview = at.data_overview(df)
        self.tools = build_tool_specs(self.overview["カテゴリ"])
        if provider == "auto":
            provider = next((p for p, k in self.PROVIDER_KEYS.items() if os.environ.get(k)), "openai")
            if provider == "gemini" and not os.environ.get("GEMINI_API_KEY") and os.environ.get("GOOGLE_API_KEY"):
                provider = "gemini"
        self.provider = provider
        if mode == "auto":
            key = self.PROVIDER_KEYS.get(provider, "")
            has_key = bool(os.environ.get(key)) or (provider == "gemini" and bool(os.environ.get("GOOGLE_API_KEY")))
            mode = "api" if (has_key or client is not None) else "mock"
        self.mode = mode
        self.model = model
        self._client = client
        self.system_prompt = SYSTEM_PROMPT.replace("D店", store_name)
        self.history: list[dict] = []
        self._id_offset = 0

    def _make_llm(self):
        if self.mode == "api":
            if self.provider == "anthropic":
                from llm_providers import AnthropicLLM
                return AnthropicLLM(self.model, client=self._client, answer_schema=ANSWER_SCHEMA)
            if self.provider == "gemini":
                from llm_providers import GeminiLLM
                return GeminiLLM(self.model, client=self._client, answer_schema=ANSWER_SCHEMA)
            return OpenAILLM(self.model, client=self._client)
        return MockLLM(self.base_date, self.overview["カテゴリ"], self.overview["商品"], tuple(self.overview["期間"]))

    def context(self) -> str:
        lw = at.resolve_relative_period(self.base_date, "先週")
        pw = at.resolve_relative_period(self.base_date, "前週")
        return json.dumps({"店舗": self.store_name, "データ粒度": "1行＝1日の1商品の販売集計（取引IDなし・原価なし）",
                           "利用可能期間": self.overview["期間"], "基準日": self.base_date,
                           "先週": [lw["開始日"], lw["終了日"]], "前週": [pw["開始日"], pw["終了日"]],
                           "カテゴリ": self.overview["カテゴリ"], "商品": self.overview["商品"],
                           "使えるツール": [t["function"]["name"] for t in self.tools]}, ensure_ascii=False)

    def ask(self, question: str) -> dict:
        """1回の質問を処理して {answer, tool_log, results, verification, mode, ...} を返す"""
        store = ResultStore(offset=self._id_offset)
        guard = ToolGuard(self.df, store)
        llm = self._make_llm()
        llm.start(self.system_prompt, self.context(), question)
        tool_log = []
        clarification = ""
        try:
            for step in range(MAX_TOOL_CALLS + 1):
                plan = llm.plan(self.tools)
                if not plan["tool_calls"]:
                    clarification = plan["text"]
                    break
                for tc in plan["tool_calls"]:
                    result = guard.execute(tc["name"], tc["args"])
                    tool_log.append({"step": len(tool_log) + 1, "tool": tc["name"], "args": tc["args"],
                                     "status": result["status"], "result_id": result["result_id"],
                                     "error_code": result.get("error_code")})
                    llm.tool_result(tc["id"], result_for_ai(result))
                if guard.calls >= MAX_TOOL_CALLS:
                    break
            answer = llm.answer()
            verification = verify_answer(answer, store)
            retried = False
            if not verification["ok"] and self.mode == "api":
                retried = True
                answer = llm.answer(" 前回の回答には次の問題がありました。修正してください: " + "; ".join(verification["problems"]))
                verification = verify_answer(answer, store)
            if not verification["ok"]:
                answer = fallback_answer(question, store, "; ".join(verification["problems"][:2]))
        except Exception as e:   # API障害など：無限再試行せず状態を返す
            answer = {"summary": f"生成AIとの通信に失敗しました（{type(e).__name__}: {e}）。模擬モード（mode='mock'）で流れを確認するか、APIキー・接続を確認して再実行してください。",
                      "facts": [], "hypotheses": [], "next_checks": ["OPENAI_API_KEY と POS_COPILOT_MODEL の設定を確認する"],
                      "limitations": [], "needs_clarification": False, "clarification_question": "", "_error": str(e)}
            verification, retried = {"ok": False, "problems": [str(e)]}, False
        if clarification and not tool_log and not answer.get("needs_clarification"):
            answer["needs_clarification"] = True
            answer["clarification_question"] = answer.get("clarification_question") or clarification
        out = {"question": question, "mode": self.mode, "provider": self.provider, "base_date": self.base_date, "answer": answer,
               "model": getattr(llm, "model", None),
               "tool_log": tool_log, "results": store.all(), "verification": verification, "retried": retried,
               "usage": getattr(llm, "usage", None)}
        self._id_offset += store._n
        self.history.append(out)
        return out


# ---------------------------------------------------------------------------
# 表示用
# ---------------------------------------------------------------------------
def format_answer(out: dict) -> str:
    """回答をMarkdown文字列にする（Notebook・Streamlit共通）"""
    a = out["answer"]
    mode = "模擬モード（ルールベース・API未接続）" if out["mode"] == "mock" else f"API実接続（{out.get('provider', 'openai')} / {out.get('model') or ''}）"
    lines = [f"**[{mode}]** 基準日 {out['base_date']}", ""]
    if a.get("needs_clarification"):
        lines += ["### 確認が必要です", a.get("clarification_question", ""), ""]
    lines += ["### 結論", a.get("summary", ""), ""]
    if a.get("facts"):
        lines.append("### 事実（データで確認できること）")
        for f in a["facts"]:
            lines.append(f"- {f['text']}　`{f['result_id']}` / `{f['metric_key']}`")
        lines.append("")
    if a.get("hypotheses"):
        lines.append("### 仮説（データでは確認できないこと）")
        lines += [f"- {h}" for h in a["hypotheses"]] + [""]
    if a.get("next_checks"):
        lines.append("### 次に確認すること")
        lines += [f"- {n}" for n in a["next_checks"]] + [""]
    if a.get("limitations"):
        lines.append("### データの制約・注意")
        lines += [f"- {l}" for l in a["limitations"]] + [""]
    if out["tool_log"]:
        lines.append("### 使用ツールと集計条件")
        for t in out["tool_log"]:
            lines.append(f"- {t['step']}. `{t['tool']}` {json.dumps(t['args'], ensure_ascii=False)} → {t['status']} (`{t['result_id']}`)" + (f" {t['error_code']}" if t.get("error_code") else ""))
        lines.append("")
    v = out["verification"]
    lines.append("検証: " + ("OK" if v["ok"] else "NG → " + "; ".join(v["problems"][:3])) + ("（再生成1回）" if out.get("retried") else ""))
    if a.get("_fallback"):
        lines.append("※ 定型回答に切り替えました")
    return "\n".join(lines)


def result_tables(out: dict) -> list[tuple[str, pd.DataFrame]]:
    """根拠表の一覧 [(見出し, DataFrame), ...]"""
    tables = []
    for rid, r in out["results"].items():
        if r.get("status") != "ok":
            tables.append((f"{rid} {r['tool']} — エラー {r.get('error_code')}: {r.get('message')}", pd.DataFrame([r.get("available_options", {})])))
            continue
        tables.append((f"{rid} {r['tool']} {json.dumps(r['filters'], ensure_ascii=False)}", at.to_dataframe(r)))
        for k in r.get("extra_tables", {}):
            tables.append((f"{rid} {r['tool']} — {k}", at.to_dataframe(r, k)))
    return tables
