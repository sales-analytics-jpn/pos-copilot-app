# -*- coding: utf-8 -*-
"""
app.py — POS売上分析Copilot Streamlitアプリ

起動（ローカル）:  cd 04_app && streamlit run app.py
APIキー: 環境変数 OPENAI_API_KEY 等、または .streamlit/secrets.toml（未設定なら模擬モードで動く）
Community Cloud: 08_streamlit_cloud/ の同じ app.py を GitHub に置いて公開（README_deploy.md）
"""
import os
import sys

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
# 共通モジュールの場所：講座フォルダ構成（../src）か、公開用フォルダ（同じ階層）のどちらでも動く
for cand in (os.path.join(os.path.dirname(HERE), "src"), HERE):
    if os.path.exists(os.path.join(cand, "analysis_tools.py")):
        sys.path.insert(0, cand)
        break

import analysis_tools as at   # noqa: E402
import charts                 # noqa: E402
import copilot as cp          # noqa: E402

# APIキー：環境変数が無ければ st.secrets（.streamlit/secrets.toml または Community Cloud の Secrets）から読む
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "POS_COPILOT_MODEL", "POS_COPILOT_MODEL_ANTHROPIC", "POS_COPILOT_MODEL_GEMINI"):
    try:
        if not os.environ.get(_k) and _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def make_client(provider: str, api_key: str):
    """画面で入力されたキーからAPIクライアントを作る（環境変数は書き換えない）"""
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=api_key)
    from openai import OpenAI
    return OpenAI(api_key=api_key)


st.set_page_config(page_title="POS売上分析Copilot", layout="wide")
st.title("POS売上分析Copilot（学習用・最小構成）")
st.caption("計算はPython、説明は生成AI。回答の数値は必ず根拠表で確認できます。")

# ---- 1. CSVアップロード・データ確認 --------------------------------------------------------
with st.sidebar:
    st.header("1. データ")
    up = st.file_uploader("POS売上CSV（日付・曜日・商品名・カテゴリ・単価・数量・売上金額）", type="csv")
    default_path = next(p for p in (os.path.join(os.path.dirname(HERE), "01_data", "pos_sales_clean.csv"),
                                    os.path.join(HERE, "data", "pos_sales_clean.csv")) if os.path.exists(p))
    use_sample = st.checkbox("配布サンプル（D店）を使う", value=up is None)
    st.header("2. 条件")
    base_date = st.text_input("基準日（YYYY-MM-DD）", "2026-08-31")
    provider = st.selectbox("生成AIの提供元", ["auto", "openai", "anthropic", "gemini"], index=0,
                            help="auto: キーがある提供元を openai→anthropic→gemini の順で選ぶ")

    # 入力欄に出す提供元を決める（auto のときは、キーがある提供元 → 無ければ openai）
    eff_provider = provider
    if provider == "auto":
        eff_provider = next((p for p, k in PROVIDER_KEYS.items()
                             if os.environ.get(k) or st.session_state.get(f"apikey_{p}")), "openai")
    env_name = PROVIDER_KEYS[eff_provider]

    st.header("3. APIキー（任意）")
    if os.environ.get(env_name):
        st.success(f"{env_name} は設定済み（環境変数／secrets.toml）")
    else:
        _typed = st.text_input(f"{env_name}", type="password", key=f"apikey_{eff_provider}",
                               placeholder="sk-... （空のままなら模擬モード）",
                               help="入力したキーはこのブラウザのセッションの中だけで使われ、ファイルにも環境変数にも保存されません。"
                                    "タブを閉じると消えます。")
        if not _typed:
            st.caption("未入力のままでも、模擬モードで全機能を試せます。")

    mode = st.radio("接続", ["auto", "mock", "api"], index=0,
                    help="auto: APIキーがあれば実接続、なければ模擬モード")
    st.markdown("---")
    st.markdown("**注意**：自社データを外部APIへ送る前に、送信範囲（データ概要・集計結果のみ）と社内ルールを確認してください。")
    st.caption("インターネットに公開したアプリでは、ここにご自身のキーを入力しないでください（公開時は運営者のキーを Secrets に設定して使います）。")

src = up if (up is not None and not use_sample) else default_path
raw = at.load_pos_csv(src)
check = at.validate_pos_data(raw)
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("データ確認")
    st.write(f"ファイル: `{getattr(up, 'name', 'pos_sales_clean.csv') if not use_sample else 'pos_sales_clean.csv'}`　行数: {check.get('n_rows', len(raw))}")
    if check["status"] == "error":
        st.error(check["message"])
        st.stop()
    if check["n_issues"]:
        st.warning(f"検査で {check['n_issues']} 件の問題を検出しました。安全な修正だけを適用し、判断が必要な行は除外して分析します。")
        st.dataframe(pd.DataFrame(check["issues"]), height=200)
    df, log = at.prepare_pos_data(raw)
    for line in log:
        st.write("・" + line)
    ov = at.data_overview(df)
    st.write(f"期間 {ov['期間'][0]}〜{ov['期間'][1]}（{ov['日数']}日）　商品 {ov['商品数']}　売上合計 {ov['売上合計']:,}円")
    st.write("カテゴリ: " + "、".join(ov["カテゴリ"]))

# ---- 2. 質問 -------------------------------------------------------------------------
with col2:
    st.subheader("質問")
    ex = st.selectbox("質問例", ["（入力する）", "2026年8月の売上推移を見せて", "8月の曜日ごとの売れ筋を3つずつ", "8月の重点商品は？",
                                "先週から売上が落ちた商品は？", "最近、売上が落ちたのはなぜ？", "雨が原因ですよね？", "客単価は？"])
    q = st.text_input("質問", value="" if ex == "（入力する）" else ex)
    run = st.button("質問する", type="primary")

# 画面で入力されたキーは環境変数に入れず、そのキーで作ったクライアントを Copilot に渡す（セッションごとに独立）
session_key = (st.session_state.get(f"apikey_{eff_provider}") or "").strip()
have_key = bool(session_key or os.environ.get(env_name)
                or (eff_provider == "gemini" and os.environ.get("GOOGLE_API_KEY")))

if mode == "api" and not have_key:
    st.warning(f"接続を「api」にしていますが {env_name} が未入力です。左の「3. APIキー」に入力するか、接続を auto／mock にしてください。"
               "ここでは模擬モードで続けます。")
    mode = "mock"
eff_mode = "api" if (mode == "api" or (mode == "auto" and have_key)) else "mock"

client = None
if eff_mode == "api" and session_key:
    try:
        client = make_client(eff_provider, session_key)
    except Exception as e:
        st.error(f"APIクライアントを作成できませんでした：{type(e).__name__} {e}")
        eff_mode, client = "mock", None

cache_key = (base_date, eff_mode, eff_provider, id(src), bool(session_key), session_key[-4:])
if "copilot" not in st.session_state or st.session_state.get("_key") != cache_key:
    st.session_state.copilot = cp.SalesCopilot(df, base_date=base_date, mode=eff_mode,
                                               provider=eff_provider, client=client)
    st.session_state._key = cache_key
c = st.session_state.copilot
st.info(("模擬モード（ルールベース・API未接続）" if c.mode == "mock" else f"API実接続：{c.provider}") + f"　基準日 {base_date}")

if run and q.strip():
    with st.spinner("処理中…"):
        try:
            out = c.ask(q.strip())
        except Exception as e:
            out = None
            st.error(f"生成AIの呼び出しに失敗しました：{type(e).__name__}")
            st.code(str(e)[:1500])
            st.info("キー・モデル名・ネットワークをご確認ください。接続を「mock」にすれば、APIなしで同じ流れを試せます。")
    if out is not None:
        st.session_state.last = out

out = st.session_state.get("last")
if out:
    a = out["answer"]
    if a.get("needs_clarification"):
        st.warning("確認待ち：" + a.get("clarification_question", ""))
    # ---- 3. 回答 ----
    st.subheader("回答")
    st.markdown(cp.format_answer(out))
    # ---- 4. 根拠 ----
    st.subheader("根拠（使用ツール・条件・結果ID・表・グラフ）")
    for rid, r in out["results"].items():
        with st.expander(f"{rid}  {r['tool']}  {r.get('filters') or r.get('error_code')}", expanded=True):
            if r.get("status") != "ok":
                st.error(f"{r.get('error_code')}: {r.get('message')}")
                st.json(r.get("available_options", {}))
                continue
            st.json(r["metrics"])
            for wmsg in r.get("warnings", []):
                st.caption("⚠ " + wmsg)
            tdf = at.to_dataframe(r)
            st.dataframe(tdf, height=min(400, 40 + 28 * len(tdf)))
            # ---- 5. CSV出力（現在の条件と同じデータ） ----
            f = r["filters"]
            period = f"{f['開始日']}_{f['終了日']}" if "開始日" in f else f"{f['当期'][0]}_{f['当期'][1]}_vs_{f['前期'][0]}_{f['前期'][1]}"
            st.download_button(f"CSVを保存（{rid}_{r['tool']}_{period}.csv）", at.to_csv_bytes(r),
                               file_name=f"{rid}_{r['tool']}_{period}.csv", mime="text/csv", key=f"dl_{rid}")
            for k, rows in r.get("extra_tables", {}).items():
                st.markdown(f"**{k}**")
                st.dataframe(pd.DataFrame(rows), height=min(300, 40 + 28 * len(rows)))
                st.download_button(f"CSVを保存（{k}）", at.to_csv_bytes(r, k), file_name=f"{rid}_{r['tool']}_{k}_{period}.csv",
                                   mime="text/csv", key=f"dl_{rid}_{k}")
            fig = charts.plot_result(r)
            if fig is not None:
                st.pyplot(fig)
