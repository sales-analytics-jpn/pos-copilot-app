# -*- coding: utf-8 -*-
"""
llm_providers.py — Claude（Anthropic）と Gemini（Google）向けの接続層

copilot.py の OpenAILLM と同じインターフェース（start / plan / tool_result / answer）を持つので、
SalesCopilot(provider="anthropic") / SalesCopilot(provider="gemini") で差し替えられる。
ToolGuard・ResultStore・verify_answer・MockLLM はそのまま共通。

OpenAI との主な違い（詳細は 03_session_2/02_pos_copilot_claude.ipynb / 02_pos_copilot_gemini.ipynb）

| 項目 | OpenAI（Chat Completions） | Claude（Messages API） | Gemini（google-genai） |
|---|---|---|---|
| ツール定義 | {"type":"function","function":{name,description,parameters}} | {name, description, input_schema} | FunctionDeclaration(name, description, parameters_json_schema) |
| システム指示 | messages の role="system" | 引数 system= | config.system_instruction |
| ツール要求 | message.tool_calls[i].function.{name,arguments(JSON文字列)} | content の type="tool_use" ブロック（name, input は dict） | response.function_calls[i].{name,args(dict)} |
| ツール結果の返し方 | role="tool" メッセージ（tool_call_id） | role="user" の tool_result ブロック（tool_use_id） | Part.from_function_response(name, response) |
| 構造化出力 | response_format={"type":"json_schema",...} | output_config={"format":{"type":"json_schema","schema":...}} | config.response_mime_type="application/json" + response_json_schema |
| APIキー環境変数 | OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY（または GOOGLE_API_KEY） |
| モデル名の環境変数 | POS_COPILOT_MODEL | POS_COPILOT_MODEL_ANTHROPIC | POS_COPILOT_MODEL_GEMINI |
"""
from __future__ import annotations

import copy
import json
import os
from typing import Optional

# 教材制作時（2026-09-13）に各社公式ドキュメントで確認した既定モデル。配布前に再確認して固定すること
DEFAULT_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
DEFAULT_MODEL_GEMINI = "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# ツール定義の変換（OpenAI形式 → 各社形式）
# ---------------------------------------------------------------------------
def openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """OpenAIの tools 配列を Anthropic の tools 配列に変換する（parameters → input_schema）"""
    out = []
    for t in tools:
        f = t["function"]
        out.append({"name": f["name"], "description": f["description"], "input_schema": copy.deepcopy(f["parameters"])})
    return out


def _nullable_for_gemini(schema):
    """Gemini向け：{"anyOf":[{...},{"type":"null"}]} を {"type":[...,"null"]} 形式に変換する（再帰）"""
    if isinstance(schema, dict):
        if "anyOf" in schema and any(x.get("type") == "null" for x in schema["anyOf"]):
            base = [x for x in schema["anyOf"] if x.get("type") != "null"][0]
            new = copy.deepcopy(base)
            new["type"] = [base["type"], "null"]
            if "description" in schema:
                new["description"] = schema["description"]
            return new
        return {k: _nullable_for_gemini(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_nullable_for_gemini(x) for x in schema]
    return schema


def openai_tools_to_gemini(tools: list[dict]) -> list[dict]:
    """OpenAIの tools 配列を Gemini の FunctionDeclaration 用 dict に変換する"""
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "parameters_json_schema": _nullable_for_gemini(t["function"]["parameters"])} for t in tools]


# ---------------------------------------------------------------------------
# Claude（Anthropic Messages API）
# ---------------------------------------------------------------------------
class AnthropicLLM:
    def __init__(self, model: Optional[str] = None, client=None, answer_schema: Optional[dict] = None, max_tokens: int = 2048):
        from anthropic import Anthropic   # 遅延import
        self.model = model or os.environ.get("POS_COPILOT_MODEL_ANTHROPIC", DEFAULT_MODEL_ANTHROPIC)
        self.client = client or Anthropic()   # ANTHROPIC_API_KEY を環境変数から読む
        self.answer_schema = answer_schema
        self.max_tokens = max_tokens
        self.system = ""
        self.messages: list[dict] = []
        self.usage = []
        self._tools_cache = None

    def start(self, system_prompt: str, context: str, question: str):
        self.system = system_prompt                      # Claude はシステム指示を messages ではなく system= で渡す
        self.messages = [{"role": "user", "content": f"# コンテキスト\n{context}\n\n# 質問\n{question}"}]

    def plan(self, tools: list[dict]) -> dict:
        self._tools_cache = openai_tools_to_anthropic(tools)
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=self.system,
                                           messages=self.messages, tools=self._tools_cache)
        self.usage.append(getattr(resp, "usage", None))
        # 会話履歴には応答ブロックをそのまま追加（tool_use ブロックを含む）
        self.messages.append({"role": "assistant", "content": resp.content})
        calls, texts = [], []
        for block in resp.content:
            if block.type == "tool_use":
                calls.append({"id": block.id, "name": block.name, "args": dict(block.input)})   # input は既に dict
            elif block.type == "text":
                texts.append(block.text)
        self._pending_results = []
        return {"tool_calls": calls, "text": "\n".join(texts)}

    def tool_result(self, call_id: str, content: str):
        # Claude は複数のツール結果を「1つの user メッセージ」にまとめて返す
        self._pending_results.append({"type": "tool_result", "tool_use_id": call_id, "content": content})
        if self.messages and self.messages[-1]["role"] == "user" and isinstance(self.messages[-1]["content"], list):
            self.messages[-1]["content"].append(self._pending_results[-1])
        else:
            self.messages.append({"role": "user", "content": [self._pending_results[-1]]})

    def answer(self, extra_instruction: str = "") -> dict:
        instr = "以上のツール結果だけを根拠に、指定のJSON形式で店長向けの回答を作ってください。" + extra_instruction
        if self.messages and self.messages[-1]["role"] == "user":
            content = self.messages[-1]["content"]
            if isinstance(content, list):
                content.append({"type": "text", "text": instr})
            else:
                self.messages[-1]["content"] = content + "\n\n" + instr
        else:
            self.messages.append({"role": "user", "content": instr})
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=self.system, messages=self.messages,
            output_config={"format": {"type": "json_schema", "schema": self.answer_schema["schema"]}})   # 構造化出力
        self.usage.append(getattr(resp, "usage", None))
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        self.messages.append({"role": "assistant", "content": text})
        return json.loads(text)


# ---------------------------------------------------------------------------
# Gemini（google-genai SDK, generate_content）
# ---------------------------------------------------------------------------
class GeminiLLM:
    def __init__(self, model: Optional[str] = None, client=None, answer_schema: Optional[dict] = None):
        from google import genai   # 遅延import
        from google.genai import types
        self.types = types
        self.model = model or os.environ.get("POS_COPILOT_MODEL_GEMINI", DEFAULT_MODEL_GEMINI)
        self.client = client or genai.Client()   # GEMINI_API_KEY（または GOOGLE_API_KEY）を環境変数から読む
        self.answer_schema = answer_schema
        self.system = ""
        self.contents: list = []
        self.usage = []
        self._call_names: dict[str, str] = {}

    def start(self, system_prompt: str, context: str, question: str):
        self.system = system_prompt                      # Gemini は config.system_instruction で渡す
        self.contents = [self.types.Content(role="user", parts=[self.types.Part.from_text(text=f"# コンテキスト\n{context}\n\n# 質問\n{question}")])]

    def plan(self, tools: list[dict]) -> dict:
        t = self.types
        decls = [t.FunctionDeclaration(**d) for d in openai_tools_to_gemini(tools)]
        config = t.GenerateContentConfig(
            system_instruction=self.system, tools=[t.Tool(function_declarations=decls)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True))   # 自動実行させず、Python側で検証してから実行する
        resp = self.client.models.generate_content(model=self.model, contents=self.contents, config=config)
        self.usage.append(getattr(resp, "usage_metadata", None))
        cand = resp.candidates[0] if resp.candidates else None
        if cand is not None and cand.content is not None:
            self.contents.append(cand.content)              # モデルの応答（function_call を含む）を履歴に追加
        calls = []
        for i, fc in enumerate(resp.function_calls or []):
            cid = getattr(fc, "id", None) or f"gemini_call_{len(self.contents)}_{i}"
            self._call_names[cid] = fc.name
            calls.append({"id": cid, "name": fc.name, "args": dict(fc.args or {})})
        text = ""
        try:
            text = resp.text or ""
        except Exception:
            text = ""
        return {"tool_calls": calls, "text": text}

    def tool_result(self, call_id: str, content: str):
        t = self.types
        name = self._call_names.get(call_id, "unknown")
        part = t.Part.from_function_response(name=name, response={"result": json.loads(content)})
        # 直前が function_response の user コンテンツならまとめる
        if self.contents and self.contents[-1].role == "user" and self.contents[-1].parts and getattr(self.contents[-1].parts[0], "function_response", None):
            self.contents[-1].parts.append(part)
        else:
            self.contents.append(t.Content(role="user", parts=[part]))

    def answer(self, extra_instruction: str = "") -> dict:
        t = self.types
        instr = "以上のツール結果だけを根拠に、指定のJSON形式で店長向けの回答を作ってください。" + extra_instruction
        self.contents.append(t.Content(role="user", parts=[t.Part.from_text(text=instr)]))
        config = t.GenerateContentConfig(system_instruction=self.system, response_mime_type="application/json",
                                         response_json_schema=_nullable_for_gemini(self.answer_schema["schema"]))   # 構造化出力（additionalProperties は除く）
        resp = self.client.models.generate_content(model=self.model, contents=self.contents, config=config)
        self.usage.append(getattr(resp, "usage_metadata", None))
        text = resp.text or "{}"
        self.contents.append(t.Content(role="model", parts=[t.Part.from_text(text=text)]))
        return json.loads(text)
