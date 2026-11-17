# -*- coding: utf-8 -*-
"""
charts.py — 結果辞書からグラフを描く（Notebook・Streamlit共通）。数値は結果辞書の表をそのまま使う
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

import analysis_tools as at

COLORS = {"A": "#1f77b4", "B": "#7fb3d5", "C": "#c9d6df", "main": "#1f77b4", "sub": "#9aa5b1", "neg": "#d9534f", "pos": "#2a9d8f"}


def setup_japanese_font() -> str:
    """日本語フォントを設定する。japanize-matplotlib があればそれを使い、なければ環境内のCJKフォントを探す"""
    try:
        import japanize_matplotlib  # noqa: F401
        return "japanize-matplotlib"
    except Exception:
        pass
    candidates = ["Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic", "Meiryo", "Yu Gothic", "MS Gothic",
                  "Hiragino Sans", "Noto Sans CJK SC", "Noto Sans CJK TC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            matplotlib.rcParams["font.family"] = c
            matplotlib.rcParams["axes.unicode_minus"] = False
            return c
    return "(日本語フォントが見つかりません。pip install japanize-matplotlib を実行してください)"


setup_japanese_font()


def _fmt_yen(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def plot_daily(result: dict, ax=None, title=None):
    """summarize_sales(unit='day') の日別推移"""
    df = at.to_dataframe(result)
    if df.empty or "日付" not in df.columns:
        return None
    ax = ax or plt.subplots(figsize=(9, 3.2))[1]
    x = pd.to_datetime(df["日付"])
    ax.plot(x, df["売上金額"], marker="o", ms=3, color=COLORS["main"])
    wk = x.dt.weekday >= 5
    ax.scatter(x[wk], df["売上金額"][wk], color=COLORS["neg"], s=18, zorder=3, label="土日")
    f = result["filters"]
    ax.set_title(title or f"日別売上推移 {f['開始日']}〜{f['終了日']}" + (f"（{f['カテゴリ']}）" if f.get("カテゴリ") else ""))
    ax.set_ylabel("売上金額（円）")
    _fmt_yen(ax)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.figure.autofmt_xdate()
    ax.figure.tight_layout()
    return ax.figure


def plot_weekday(result: dict, ax=None):
    """weekday_top_products の曜日別合計と1日平均"""
    df = at.to_dataframe(result)
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2)) if ax is None else (ax.figure, [ax, ax.twinx()])
    axes[0].bar(df["曜日"], df["売上金額"], color=COLORS["main"])
    axes[0].set_title("曜日別 売上合計")
    _fmt_yen(axes[0])
    for i, (v, n) in enumerate(zip(df["売上金額"], df["日数"])):
        axes[0].text(i, v, f"{n}日", ha="center", va="bottom", fontsize=8)
    axes[1].bar(df["曜日"], df["1日平均売上"], color=COLORS["pos"])
    axes[1].set_title("曜日別 1日平均売上")
    _fmt_yen(axes[1])
    for a in axes:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_abc(result: dict, ax=None):
    """abc_analysis のパレート図（棒＝売上、線＝累積構成比）"""
    df = at.to_dataframe(result)
    if df.empty:
        return None
    ax = ax or plt.subplots(figsize=(9, 3.6))[1]
    colors = [COLORS[c] for c in df["ABC区分"]]
    ax.bar(df["商品名"], df["売上金額"], color=colors)
    ax.set_ylabel("売上金額（円）")
    _fmt_yen(ax)
    ax2 = ax.twinx()
    ax2.plot(df["商品名"], df["累積構成比"], color="#333", marker="o", ms=3)
    ax2.axhline(70, color="#999", ls="--", lw=0.8)
    ax2.axhline(90, color="#999", ls="--", lw=0.8)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("累積構成比（％）")
    f = result["filters"]
    ax.set_title(f"ABC分析 {f['開始日']}〜{f['終了日']}" + (f"（{f['カテゴリ']}）" if f.get("カテゴリ") else "") +
                 f"　A:{result['metrics']['A商品数']} B:{result['metrics']['B商品数']} C:{result['metrics']['C商品数']}")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.figure.tight_layout()
    return ax.figure


def plot_compare(result: dict, ax=None, top: int = 10):
    """compare_periods の商品別増減額（減少・増加の大きい順）"""
    df = at.to_dataframe(result)
    if df.empty:
        return None
    df = df.sort_values("増減額")
    show = pd.concat([df.head(top // 2), df.tail(top - top // 2)]).drop_duplicates("商品名")
    ax = ax or plt.subplots(figsize=(9, 3.6))[1]
    colors = [COLORS["neg"] if v < 0 else COLORS["pos"] for v in show["増減額"]]
    ax.barh(show["商品名"], show["増減額"], color=colors)
    ax.axvline(0, color="#333", lw=0.8)
    _fmt_yen(ax, "x")
    f, m = result["filters"], result["metrics"]
    ax.set_title(f"商品別 増減額　当期 {f['当期'][0]}〜{f['当期'][1]} vs 前期 {f['前期'][0]}〜{f['前期'][1]}　全体 {m['増減額']:+,}円")
    ax.grid(axis="x", alpha=0.3)
    ax.figure.tight_layout()
    return ax.figure


def plot_result(result: dict):
    """ツール名に応じて適切なグラフを描く"""
    if result.get("status") != "ok":
        return None
    t = result["tool"]
    if t == "summarize_sales" and result["filters"].get("集計単位") == "day":
        return plot_daily(result)
    if t == "summarize_sales":
        df = at.to_dataframe(result)
        key = "商品名" if "商品名" in df.columns else "カテゴリ"
        ax = plt.subplots(figsize=(9, 3.4))[1]
        ax.bar(df[key], df["売上金額"], color=COLORS["main"])
        _fmt_yen(ax)
        ax.set_title(f"{key}別 売上 {result['filters']['開始日']}〜{result['filters']['終了日']}")
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        ax.figure.tight_layout()
        return ax.figure
    if t == "weekday_top_products":
        return plot_weekday(result)
    if t == "abc_analysis":
        return plot_abc(result)
    if t == "compare_periods":
        return plot_compare(result)
    return None
