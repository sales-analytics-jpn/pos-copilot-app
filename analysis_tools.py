# -*- coding: utf-8 -*-
"""
analysis_tools.py — POS売上分析の共通関数（第1弾 POS売上分析Copilot）

方針
  * 数値計算はすべてこのモジュール（Python）で行う。生成AIには計算させない
  * すべての分析関数は「結果辞書」を返す
        {"status": "ok", "tool": 関数名, "filters": {...}, "metrics": {...},
         "table": [行の辞書, ...], "extra_tables": {...}, "warnings": [...]}
    失敗時は
        {"status": "error", "tool": 関数名, "error_code": "...", "message": "...",
         "available_options": {...}}
  * 期間は開始日・終了日とも含む。データ期間から一部でも外れていれば黙って短縮せず error を返す
  * 不存在のカテゴリ・商品は候補を提示して再指定を求める
  * 増減率は前期売上が正のときだけ計算する（前期0なら None ＝ 算出不能）
  * ABC分類：当該商品を含める前の累積構成比で 70％未満=A、90％未満=B、それ以外=C
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

COLUMNS = ["日付", "曜日", "商品名", "カテゴリ", "単価", "数量", "売上金額"]
WD_JP = ["月", "火", "水", "木", "金", "土", "日"]
UNITS = ("day", "product", "category")

__all__ = [
    "COLUMNS", "WD_JP", "UNITS",
    "load_pos_csv", "validate_pos_data", "prepare_pos_data", "data_overview",
    "summarize_sales", "weekday_top_products", "abc_analysis", "compare_periods",
    "resolve_relative_period", "to_dataframe", "to_csv_bytes",
]


# ---------------------------------------------------------------------------
# 読み込み・検査・前処理
# ---------------------------------------------------------------------------
def load_pos_csv(path_or_buffer) -> pd.DataFrame:
    """CSVを「そのまま」読み込む（型変換は検査のあとで行う）。BOM付きUTF-8に対応"""
    return pd.read_csv(path_or_buffer, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def _to_int(s: str):
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def validate_pos_data(df: pd.DataFrame, expected_period: Optional[tuple[str, str]] = None) -> dict:
    """POSデータを検査し、問題行の一覧を返す。修正はしない。

    戻り値: {"status": "ok"|"error", "n_rows", "n_issues", "issues": [{"行","種類","内容"}...],
             "counts": {種類: 件数}, "warnings": [...]}
    行番号はExcelで開いたときの行番号（1行目がヘッダー、データは2行目から）
    """
    issues = []
    warnings = []

    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        return {"status": "error", "tool": "validate_pos_data", "error_code": "MISSING_COLUMNS",
                "message": f"必要な列がありません: {missing_cols}",
                "available_options": {"必要な列": COLUMNS, "実際の列": list(df.columns)}}

    def add(i, kind, text):
        issues.append({"行": int(i) + 2, "種類": kind, "内容": text})

    parsed_dates = []
    for i, r in df.iterrows():
        # 日付
        try:
            d = datetime.strptime(str(r["日付"]).strip(), "%Y-%m-%d").date()
        except ValueError:
            add(i, "日付の形式", f"日付「{r['日付']}」を YYYY-MM-DD として読めない")
            parsed_dates.append(None)
            continue
        parsed_dates.append(d)
        # 期間外
        if expected_period:
            s, e = (date.fromisoformat(x) for x in expected_period)
            if not (s <= d <= e):
                add(i, "期間外の日付", f"{d} はデータ定義の期間（{s}〜{e}）の外")
        # 曜日
        if str(r["曜日"]).strip() != WD_JP[d.weekday()]:
            add(i, "曜日不一致", f"{d} の曜日は「{WD_JP[d.weekday()]}」だがCSVは「{r['曜日']}」")
        # 商品名の空白
        if str(r["商品名"]) != str(r["商品名"]).strip():
            add(i, "名称の空白", f"商品名「{r['商品名']}」の前後に空白")
        # 数値
        price, qty, amt = _to_int(r["単価"]), _to_int(r["数量"]), _to_int(r["売上金額"])
        if qty is None or amt is None:
            add(i, "数量欠損", f"{d} {str(r['商品名']).strip()}：数量または売上金額が空欄・数値でない")
            continue
        if price is None:
            add(i, "単価欠損", f"{d} {str(r['商品名']).strip()}：単価が空欄・数値でない")
            continue
        if qty < 0 or amt < 0:
            add(i, "負の値", f"{d} {str(r['商品名']).strip()}：数量{qty}・売上金額{amt}（返品？定義を確認）")
        if price * qty != amt:
            add(i, "金額不整合", f"{d} {str(r['商品名']).strip()}：単価×数量={price * qty} だが売上金額={amt}")
        if price > 0 and price >= 10000:
            add(i, "異常な単価", f"{d} {str(r['商品名']).strip()}：単価{price}円（桁ずれ？商品マスタと照合）")

    # 重複（日付×商品名）
    key = df["日付"].str.strip() + "|" + df["商品名"].str.strip()
    dup_mask = key.duplicated(keep=False)
    if dup_mask.any():
        for k, grp in df[dup_mask].groupby(key[dup_mask]):
            rows = [int(x) + 2 for x in grp.index]
            same = grp.drop_duplicates().shape[0] == 1
            kind = "完全重複" if same else "内容の異なる重複"
            for j in grp.index[1:]:
                add(j, kind, f"{k.replace('|', ' ')}：行{rows}が同じ日付×商品（{'内容も同一' if same else '内容が異なる→どちらが正しいか確認'}）")

    counts = {}
    for it in issues:
        counts[it["種類"]] = counts.get(it["種類"], 0) + 1
    issues.sort(key=lambda x: x["行"])
    n_bad_rows = len({it["行"] for it in issues})
    if n_bad_rows:
        warnings.append(f"{n_bad_rows}行に問題があります。修正方針を決めてから prepare_pos_data() を実行してください")
    return {"status": "ok" if not issues else "issues_found", "tool": "validate_pos_data",
            "n_rows": int(len(df)), "n_issues": len(issues), "n_bad_rows": n_bad_rows,
            "issues": issues, "counts": counts, "warnings": warnings}


def prepare_pos_data(df: pd.DataFrame, expected_period: Optional[tuple[str, str]] = None,
                     drop_out_of_period: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """安全な修正だけを適用して分析用DataFrameを返す。

    行うこと: 前後空白の除去、曜日を日付から再計算、数値化、完全重複の統合
    行わないこと: 欠損の0埋め（欠測行は除外して記録）、金額不整合の上書き（除外して記録）、
                内容の異なる重複の自動選択（両方除外して記録）
    戻り値: (clean_df, log)
    """
    log = []
    d = df.copy()
    for c in ["日付", "曜日", "商品名", "カテゴリ"]:
        d[c] = d[c].astype(str).str.strip()
    d["日付"] = pd.to_datetime(d["日付"], format="%Y-%m-%d", errors="coerce")
    bad = d["日付"].isna()
    if bad.any():
        log.append(f"日付を読めない{int(bad.sum())}行を除外")
        d = d[~bad]
    if expected_period and drop_out_of_period:
        s, e = pd.Timestamp(expected_period[0]), pd.Timestamp(expected_period[1])
        out = (d["日付"] < s) | (d["日付"] > e)
        if out.any():
            log.append(f"期間外（{expected_period[0]}〜{expected_period[1]}の外）の{int(out.sum())}行を除外")
            d = d[~out]
    d["曜日"] = d["日付"].dt.weekday.map(lambda w: WD_JP[w])
    for c in ["単価", "数量", "売上金額"]:
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    na = d[["単価", "数量", "売上金額"]].isna().any(axis=1)
    if na.any():
        log.append(f"数量・単価・売上金額が欠損の{int(na.sum())}行を除外（0で埋めない）")
        d = d[~na]
    mism = d["単価"] * d["数量"] != d["売上金額"]
    if mism.any():
        log.append(f"単価×数量≠売上金額の{int(mism.sum())}行を除外（定義確認が必要。自動で上書きしない）")
        d = d[~mism]
    before = len(d)
    d = d.drop_duplicates()
    if len(d) < before:
        log.append(f"完全重複{before - len(d)}行を1行に統合")
    key = d["日付"].dt.strftime("%Y-%m-%d") + "|" + d["商品名"]
    conflict = key.duplicated(keep=False)
    if conflict.any():
        log.append(f"日付×商品が同じで内容が異なる{int(conflict.sum())}行を除外（どちらが正しいか要確認）")
        d = d[~conflict]
    d = d.sort_values(["日付", "カテゴリ", "商品名"]).reset_index(drop=True)
    for c in ["単価", "数量", "売上金額"]:
        d[c] = d[c].astype(int)
    if not log:
        log.append("問題なし。全行を分析対象にします")
    return d[COLUMNS], log


def data_overview(df: pd.DataFrame) -> dict:
    """AIに渡す・画面に出すデータ概要"""
    return {
        "期間": [df["日付"].min().strftime("%Y-%m-%d"), df["日付"].max().strftime("%Y-%m-%d")],
        "日数": int(df["日付"].nunique()),
        "行数": int(len(df)),
        "商品数": int(df["商品名"].nunique()),
        "カテゴリ": sorted(df["カテゴリ"].unique().tolist()),
        "商品": sorted(df["商品名"].unique().tolist()),
        "売上合計": int(df["売上金額"].sum()),
        "数量合計": int(df["数量"].sum()),
    }


# ---------------------------------------------------------------------------
# 共通フィルター
# ---------------------------------------------------------------------------
def _err(tool, code, message, options=None):
    return {"status": "error", "tool": tool, "error_code": code, "message": message,
            "available_options": options or {}}


def _parse_date(s, tool, label):
    try:
        return pd.Timestamp(datetime.strptime(str(s).strip(), "%Y-%m-%d").date()), None
    except (ValueError, TypeError):
        return None, _err(tool, "INVALID_DATE", f"{label}「{s}」は YYYY-MM-DD 形式で指定してください")


def _apply_filters(df, tool, start, end, category=None, product=None):
    """期間・カテゴリ・商品のフィルターをANDで適用する。戻り値 (filtered_df, filters, error)"""
    s, e = _parse_date(start, tool, "開始日")
    if e:
        return None, None, e
    t, e = _parse_date(end, tool, "終了日")
    if e:
        return None, None, e
    if s > t:
        return None, None, _err(tool, "INVALID_PERIOD", f"開始日{start}が終了日{end}より後です")
    dmin, dmax = df["日付"].min(), df["日付"].max()
    avail = {"利用可能期間": [dmin.strftime("%Y-%m-%d"), dmax.strftime("%Y-%m-%d")]}
    if t < dmin or s > dmax:
        return None, None, _err(tool, "PERIOD_OUT_OF_RANGE",
                                f"指定期間 {start}〜{end} はデータの期間外です", avail)
    if s < dmin or t > dmax:
        return None, None, _err(tool, "PERIOD_PARTIALLY_OUT_OF_RANGE",
                                f"指定期間 {start}〜{end} の一部がデータ期間の外です。利用可能期間内で再指定してください", avail)
    filters = {"開始日": s.strftime("%Y-%m-%d"), "終了日": t.strftime("%Y-%m-%d"),
               "カテゴリ": None, "商品名": None}
    out = df[(df["日付"] >= s) & (df["日付"] <= t)]
    if category not in (None, "", "all", "全体", "全カテゴリ"):
        cats = sorted(df["カテゴリ"].unique().tolist())
        if category not in cats:
            return None, None, _err(tool, "UNKNOWN_CATEGORY", f"カテゴリ「{category}」はデータにありません",
                                    {"カテゴリ候補": cats})
        out = out[out["カテゴリ"] == category]
        filters["カテゴリ"] = category
    if product not in (None, ""):
        prods = sorted(df["商品名"].unique().tolist())
        if product not in prods:
            cand = [p for p in prods if any(ch in p for ch in str(product)[:2])][:5]
            return None, None, _err(tool, "UNKNOWN_PRODUCT", f"商品「{product}」はデータにありません",
                                    {"商品候補": cand or prods[:10], "商品一覧": prods})
        out = out[out["商品名"] == product]
        filters["商品名"] = product
    return out, filters, None


def _warn_missing_days(sub, s, t):
    days_present = set(sub["日付"].dt.normalize().unique())
    expected = pd.date_range(s, t, freq="D")
    missing = [d.strftime("%Y-%m-%d") for d in expected if d not in days_present]
    if missing:
        return [f"期間内にデータのない日が{len(missing)}日あります（売上0とはみなしません）: {missing[:5]}{'…' if len(missing) > 5 else ''}"]
    return []


# ---------------------------------------------------------------------------
# 分析関数（AIに公開するのはこの4つ）
# ---------------------------------------------------------------------------
def summarize_sales(df: pd.DataFrame, start: str, end: str, category: Optional[str] = None,
                    unit: str = "day", product: Optional[str] = None) -> dict:
    """指定期間・条件の売上を集計する。unit: 'day'（日別推移）/'product'（商品別）/'category'（カテゴリ別）"""
    tool = "summarize_sales"
    if unit not in UNITS:
        return _err(tool, "INVALID_UNIT", f"集計単位「{unit}」は使えません", {"集計単位": list(UNITS)})
    sub, filters, err = _apply_filters(df, tool, start, end, category, product)
    if err:
        return err
    warnings = _warn_missing_days(sub, filters["開始日"], filters["終了日"])
    total_sales, total_qty = int(sub["売上金額"].sum()), int(sub["数量"].sum())
    if unit == "day":
        g = sub.groupby(sub["日付"].dt.strftime("%Y-%m-%d")).agg(数量=("数量", "sum"), 売上金額=("売上金額", "sum")).reset_index()
        g = g.rename(columns={"日付": "日付"})
        g["曜日"] = [WD_JP[pd.Timestamp(x).weekday()] for x in g["日付"]]
        g = g[["日付", "曜日", "数量", "売上金額"]]
    elif unit == "product":
        g = sub.groupby(["カテゴリ", "商品名"]).agg(数量=("数量", "sum"), 売上金額=("売上金額", "sum")).reset_index()
        g = g.sort_values(["売上金額", "商品名"], ascending=[False, True])
    else:
        g = sub.groupby("カテゴリ").agg(数量=("数量", "sum"), 売上金額=("売上金額", "sum")).reset_index()
        g = g.sort_values(["売上金額", "カテゴリ"], ascending=[False, True])
    if total_sales > 0:
        g["構成比"] = (g["売上金額"] / total_sales * 100).round(1)
    # 合計照合（内訳の合計＝全体）
    assert int(g["売上金額"].sum()) == total_sales, "内訳の合計が全体売上と一致しません"
    n_days = int(sub["日付"].nunique())
    metrics = {"売上合計": total_sales, "数量合計": total_qty, "日数": n_days,
               "1日平均売上": (round(total_sales / n_days, 1) if n_days else None)}
    return {"status": "ok", "tool": tool, "filters": {**filters, "集計単位": unit},
            "metrics": metrics, "table": g.to_dict(orient="records"), "extra_tables": {}, "warnings": warnings}


def weekday_top_products(df: pd.DataFrame, start: str, end: str, category: Optional[str] = None,
                         top_n: int = 3) -> dict:
    """曜日別の売上合計・収録日数・1日平均と、曜日ごとの売れ筋トップN（売上金額基準、同額は商品名昇順）"""
    tool = "weekday_top_products"
    try:
        top_n = int(top_n)
    except (TypeError, ValueError):
        return _err(tool, "INVALID_TOP_N", "上位件数は整数で指定してください")
    if not (1 <= top_n <= 10):
        return _err(tool, "INVALID_TOP_N", "上位件数は1〜10の範囲で指定してください")
    sub, filters, err = _apply_filters(df, tool, start, end, category)
    if err:
        return err
    warnings = _warn_missing_days(sub, filters["開始日"], filters["終了日"])
    wk = sub.groupby("曜日").agg(売上金額=("売上金額", "sum"), 数量=("数量", "sum"), 日数=("日付", "nunique")).reindex(WD_JP).dropna()
    wk = wk.astype(int).reset_index()
    wk["1日平均売上"] = (wk["売上金額"] / wk["日数"]).round(1)
    counts = wk["日数"].unique().tolist()
    if len(counts) > 1:
        warnings.append("曜日ごとの収録日数が異なります。合計だけでなく1日平均で比べてください")
    top = (sub.groupby(["曜日", "商品名"])["売上金額"].sum().reset_index())
    top = top.sort_values(["曜日", "売上金額", "商品名"], ascending=[True, False, True])
    top["順位"] = top.groupby("曜日").cumcount() + 1
    top = top[top["順位"] <= top_n]
    top["曜日順"] = top["曜日"].map({w: i for i, w in enumerate(WD_JP)})
    top = top.sort_values(["曜日順", "順位"]).drop(columns="曜日順")[["曜日", "順位", "商品名", "売上金額"]]
    metrics = {"売上合計": int(sub["売上金額"].sum()), "最高売上の曜日": (wk.sort_values("売上金額", ascending=False).iloc[0]["曜日"] if len(wk) else None),
               "1日平均が最高の曜日": (wk.sort_values("1日平均売上", ascending=False).iloc[0]["曜日"] if len(wk) else None)}
    return {"status": "ok", "tool": tool, "filters": {**filters, "上位件数": top_n, "順位の基準": "売上金額（同額は商品名昇順）"},
            "metrics": metrics, "table": wk.to_dict(orient="records"),
            "extra_tables": {"曜日別トップ商品": top.to_dict(orient="records")}, "warnings": warnings}


def abc_analysis(df: pd.DataFrame, start: str, end: str, category: Optional[str] = None) -> dict:
    """商品別売上の累積構成比でA・B・Cに分類する（当該商品を含める前の累積構成比で判定）"""
    tool = "abc_analysis"
    sub, filters, err = _apply_filters(df, tool, start, end, category)
    if err:
        return err
    warnings = _warn_missing_days(sub, filters["開始日"], filters["終了日"])
    by = sub.groupby(["カテゴリ", "商品名"])["売上金額"].sum().reset_index()
    total = int(by["売上金額"].sum())
    if (by["売上金額"] < 0).any():
        return _err(tool, "NEGATIVE_SALES", "売上が負の商品があります。返品・純売上の扱いを確認してから再実行してください",
                    {"負の商品": by[by["売上金額"] < 0]["商品名"].tolist()})
    if total <= 0:
        return _err(tool, "NO_SALES", "指定条件の総売上が0のため、ABC分類を算出できません", {"条件": filters})
    pos = by[by["売上金額"] > 0].sort_values(["売上金額", "商品名"], ascending=[False, True]).copy()
    zero = by[by["売上金額"] == 0].sort_values("商品名").copy()
    cum_before = pos["売上金額"].cumsum().shift(fill_value=0) / total * 100   # 途中で丸めない
    pos["ABC区分"] = ["A" if c < 70 else ("B" if c < 90 else "C") for c in cum_before]
    pos["構成比"] = (pos["売上金額"] / total * 100).round(1)
    pos["累積構成比"] = (pos["売上金額"].cumsum() / total * 100).round(1)
    zero["ABC区分"], zero["構成比"], zero["累積構成比"] = "C", 0.0, 100.0
    out = pd.concat([pos, zero])
    out["順位"] = range(1, len(out) + 1)
    out["備考"] = ["" ] * len(pos) + ["売上なし"] * len(zero)
    out = out[["順位", "カテゴリ", "商品名", "売上金額", "構成比", "累積構成比", "ABC区分", "備考"]]
    if len(zero):
        warnings.append(f"売上0の商品が{len(zero)}件あり、Cに分類しました")
    cls_sum = out.groupby("ABC区分")["売上金額"].agg(["count", "sum"]).reindex(["A", "B", "C"]).fillna(0).astype(int)
    metrics = {"売上合計": total, "商品数": int(len(out)),
               "A商品数": int(cls_sum.loc["A", "count"]), "B商品数": int(cls_sum.loc["B", "count"]), "C商品数": int(cls_sum.loc["C", "count"]),
               "A売上構成比": round(float(cls_sum.loc["A", "sum"]) / total * 100, 1),
               "A商品": out[out["ABC区分"] == "A"]["商品名"].tolist()}
    return {"status": "ok", "tool": tool,
            "filters": {**filters, "分類ルール": "含める前の累積構成比 <70%:A, <90%:B, それ以外:C"},
            "metrics": metrics, "table": out.to_dict(orient="records"), "extra_tables": {}, "warnings": warnings}


def compare_periods(df: pd.DataFrame, cur_start: str, cur_end: str, prev_start: str, prev_end: str,
                    category: Optional[str] = None, product: Optional[str] = None) -> dict:
    """当期と前期の売上を比較する（全体・商品別・カテゴリ別の増減額・増減率）"""
    tool = "compare_periods"
    cur, f_cur, err = _apply_filters(df, tool, cur_start, cur_end, category, product)
    if err:
        err["message"] = "当期: " + err["message"]
        return err
    prev, f_prev, err = _apply_filters(df, tool, prev_start, prev_end, category, product)
    if err:
        err["message"] = "前期: " + err["message"]
        return err
    warnings = []
    n_cur = (pd.Timestamp(f_cur["終了日"]) - pd.Timestamp(f_cur["開始日"])).days + 1
    n_prev = (pd.Timestamp(f_prev["終了日"]) - pd.Timestamp(f_prev["開始日"])).days + 1
    if n_cur != n_prev:
        warnings.append(f"当期{n_cur}日と前期{n_prev}日で日数が異なります。1日平均でも比べてください")
    elif pd.Timestamp(f_cur["開始日"]).weekday() != pd.Timestamp(f_prev["開始日"]).weekday():
        warnings.append("当期と前期で曜日構成が異なります（開始曜日が違う）")
    for lbl, sub, f in (("当期", cur, f_cur), ("前期", prev, f_prev)):
        warnings += [f"{lbl}: {w}" for w in _warn_missing_days(sub, f["開始日"], f["終了日"])]

    def _pct(a, b):
        return round((a - b) / b * 100, 1) if b > 0 else None

    def _table(key):
        a = cur.groupby(key)["売上金額"].sum()
        b = prev.groupby(key)["売上金額"].sum()
        idx = sorted(set(a.index) | set(b.index))
        rows = []
        for k in idx:
            x, y = int(a.get(k, 0)), int(b.get(k, 0))
            note = ""
            if y == 0 and x > 0:
                note = "前期売上なし（増減率は算出不能）"
            elif y == 0 and x == 0:
                note = "両期間とも売上なし"
            rows.append({key: k, "当期売上": x, "前期売上": y, "増減額": x - y, "増減率": _pct(x, y), "備考": note})
        rows.sort(key=lambda r: (r["増減額"], r[key]))
        return rows

    by_product = _table("商品名")
    cat_map = dict(zip(df["商品名"], df["カテゴリ"]))
    for r in by_product:
        r["カテゴリ"] = cat_map.get(r["商品名"], "")
    by_category = _table("カテゴリ")
    tc, tp = int(cur["売上金額"].sum()), int(prev["売上金額"].sum())
    diff = tc - tp
    assert sum(r["増減額"] for r in by_product) == diff, "商品別増減額の合計が全体増減額と一致しません"
    metrics = {"当期売上": tc, "前期売上": tp, "増減額": diff, "増減率": _pct(tc, tp),
               "当期日数": n_cur, "前期日数": n_prev,
               "減少額上位": [(r["商品名"], r["増減額"]) for r in by_product if r["増減額"] < 0][:3],
               "増加額上位": [(r["商品名"], r["増減額"]) for r in sorted(by_product, key=lambda r: (-r["増減額"], r["商品名"])) if r["増減額"] > 0][:3]}
    if tp == 0:
        warnings.append("前期売上が0のため、全体の増減率は算出できません")
    if metrics["増減率"] is None:
        metrics["増減率の注記"] = "前期売上0のため算出不能"
    return {"status": "ok", "tool": tool,
            "filters": {"当期": [f_cur["開始日"], f_cur["終了日"]], "前期": [f_prev["開始日"], f_prev["終了日"]],
                        "カテゴリ": f_cur["カテゴリ"], "商品名": f_cur["商品名"]},
            "metrics": metrics, "table": by_product, "extra_tables": {"カテゴリ別": by_category}, "warnings": warnings}


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------
def resolve_relative_period(base_date: str, phrase: str) -> Optional[dict]:
    """「先週」「前週」「今月」「先月」を基準日から日付に解決する。実行日には置き換えない"""
    b = date.fromisoformat(base_date)
    monday = b - timedelta(days=b.weekday())
    if phrase in ("先週", "last_week"):
        s = monday - timedelta(days=7)
        return {"開始日": s.isoformat(), "終了日": (s + timedelta(days=6)).isoformat()}
    if phrase in ("前週", "先々週", "week_before_last"):
        s = monday - timedelta(days=14)
        return {"開始日": s.isoformat(), "終了日": (s + timedelta(days=6)).isoformat()}
    if phrase in ("今週", "this_week"):
        return {"開始日": monday.isoformat(), "終了日": b.isoformat()}
    if phrase in ("今月", "this_month"):
        return {"開始日": b.replace(day=1).isoformat(), "終了日": b.isoformat()}
    if phrase in ("先月", "last_month"):
        first = b.replace(day=1)
        last_prev = first - timedelta(days=1)
        return {"開始日": last_prev.replace(day=1).isoformat(), "終了日": last_prev.isoformat()}
    return None


def to_dataframe(result: dict, table: str = "table") -> pd.DataFrame:
    """結果辞書の表をDataFrameにする。table='table' か extra_tables のキー"""
    if result.get("status") != "ok":
        return pd.DataFrame()
    rows = result["table"] if table == "table" else result["extra_tables"].get(table, [])
    return pd.DataFrame(rows)


def to_csv_bytes(result: dict, table: str = "table") -> bytes:
    """Excelで開けるBOM付きUTF-8のCSVバイト列（カンマ・改行を含む文字列は自動で引用される）"""
    buf = io.StringIO()
    to_dataframe(result, table).to_csv(buf, index=False)
    return ("\ufeff" + buf.getvalue()).encode("utf-8")
