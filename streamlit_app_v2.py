from __future__ import annotations

import io
import os
import re
import time
import math
import datetime as dt
from dataclasses import dataclass
from html import unescape
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
CACHE_TTL_SECONDS = 60 * 60 * 24
REQUEST_TIMEOUT = 20


# -----------------------------------------------------------------------------
# Secrets / Settings
# -----------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


def get_finmind_token() -> str:
    return get_secret("FINMIND_TOKEN")


def get_google_api_key() -> str:
    return get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")


def get_active_etf_signal_csv_url() -> str:
    return get_secret("ACTIVE_ETF_SIGNAL_CSV_URL")


# -----------------------------------------------------------------------------
# FinMind API
# -----------------------------------------------------------------------------
class FinMindError(RuntimeError):
    pass


def finmind_get(params: Dict[str, Any], token: Optional[str] = None) -> List[Dict[str, Any]]:
    token = token if token is not None else get_finmind_token()
    payload = dict(params)
    headers = {}
    if token:
        payload["token"] = token
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(FINMIND_URL, headers=headers, params=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FinMindError(f"FinMind 連線失敗：{exc}") from exc

    try:
        body = resp.json()
    except ValueError as exc:
        raise FinMindError("FinMind 回傳不是合法 JSON。") from exc

    if body.get("status") != 200:
        raise FinMindError(f"FinMind 回應異常：{body.get('msg')}（status={body.get('status')}）")
    return body.get("data", []) or []


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_stock_info_cached(token_marker: str = "") -> List[Dict[str, str]]:
    _ = token_marker
    data = finmind_get({"dataset": "TaiwanStockInfo"})
    seen = set()
    rows: List[Dict[str, str]] = []
    for row in data:
        stock_id = str(row.get("stock_id", "")).strip()
        stock_name = str(row.get("stock_name", "")).strip()
        industry_category = str(row.get("industry_category", "")).strip()
        market = str(row.get("type", "")).strip() or str(row.get("market", "")).strip()
        if not stock_id or stock_id in seen:
            continue
        seen.add(stock_id)
        rows.append(
            {
                "stock_id": stock_id,
                "stock_name": stock_name,
                "industry_category": industry_category,
                "market": market,
            }
        )
    return rows


def resolve_stock(query: str, stock_info: List[Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], List[Dict[str, str]], str]:
    q = query.strip()
    if not q:
        return None, [], "請輸入股票名稱或代號。"

    if re.fullmatch(r"\d{4,6}", q):
        exact = [r for r in stock_info if r["stock_id"] == q]
        if exact:
            return exact[0], [], ""
        return None, [], f"找不到代號「{q}」。"

    exact_name = [r for r in stock_info if r["stock_name"] == q]
    if len(exact_name) == 1:
        return exact_name[0], [], ""

    candidates = [r for r in stock_info if q in r["stock_name"] or r["stock_id"].startswith(q)]
    if len(candidates) == 1:
        return candidates[0], [], ""
    if len(candidates) > 1:
        return None, candidates[:30], f"「{q}」對應到多檔，請從候選清單選一檔。"
    return None, [], f"找不到「{q}」，請確認名稱或改用股票代號。"


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------
def normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    return text[:10] if text else ""


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def safe_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}%"


def safe_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def safe_int(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{int(round(value)):,}"


def metric_delta_text(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    sign = "+" if value > 0 else ""
    if abs(value) >= 1000:
        return f"{sign}{value:,.0f}{suffix}"
    return f"{sign}{value:.2f}{suffix}"


def render_tag(text: str, kind: str = "neutral") -> str:
    cls = {
        "good": "pill-good",
        "bad": "pill-bad",
        "neutral": "pill-neutral",
        "warn": "pill-warn",
    }.get(kind, "pill-neutral")
    return f'<span class="pill {cls}">{text}</span>'


# -----------------------------------------------------------------------------
# News Module
# -----------------------------------------------------------------------------
@dataclass
class NewsItem:
    date: str
    title: str
    link: str
    source: str
    stock_id: str
    raw: Dict[str, Any]


POSITIVE_KEYWORDS = [
    "創高", "大增", "成長", "上修", "優於", "轉盈", "獲利", "接單", "訂單", "擴產",
    "漲價", "調漲", "法說", "利多", "突破", "買超", "合作", "投資", "併購", "出貨",
]
NEGATIVE_KEYWORDS = [
    "下修", "衰退", "虧損", "減產", "裁員", "違約", "調查", "起訴", "重挫", "跌停",
    "利空", "賣超", "取消", "延遲", "庫存", "匯損", "警示", "處置", "營收減", "年減",
]


def normalize_news_rows(rows: List[Dict[str, Any]], stock_id: str) -> List[NewsItem]:
    items: List[NewsItem] = []
    seen = set()
    for row in rows:
        title = str(row.get("title") or row.get("headline") or "").strip()
        link = str(row.get("link") or row.get("url") or "").strip()
        date = normalize_date(row.get("date") or row.get("publish_time") or row.get("time"))
        source = str(row.get("source") or row.get("publisher") or row.get("media") or "").strip()
        if not title:
            continue
        key = (date, title, link)
        if key in seen:
            continue
        seen.add(key)
        items.append(NewsItem(date=date, title=title, link=link, source=source, stock_id=stock_id, raw=row))
    items.sort(key=lambda x: (x.date, x.title), reverse=True)
    return items


@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_news_cached(stock_id: str, days: int, token_marker: str = "") -> List[Dict[str, Any]]:
    _ = token_marker
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = finmind_get({"dataset": "TaiwanStockNews", "data_id": stock_id, "start_date": start})
    items = normalize_news_rows(rows, stock_id)
    return [item.__dict__ for item in items]


def classify_headline(title: str) -> str:
    pos = sum(1 for word in POSITIVE_KEYWORDS if word in title)
    neg = sum(1 for word in NEGATIVE_KEYWORDS if word in title)
    if pos > neg:
        return "偏利多"
    if neg > pos:
        return "偏利空"
    return "中性 / 待判讀"


def news_to_dataframe(news_dicts: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in news_dicts:
        rows.append(
            {
                "日期": item.get("date", ""),
                "標題": item.get("title", ""),
                "來源": item.get("source", ""),
                "初步判讀": classify_headline(item.get("title", "")),
                "連結": item.get("link", ""),
            }
        )
    return pd.DataFrame(rows)


def make_news_brief(news_df: pd.DataFrame, stock_name: str, stock_id: str, limit: int = 10) -> str:
    if news_df.empty:
        return "這段期間沒有抓到新聞，無法整理。"
    subset = news_df.head(limit)
    lines = [f"- {row['日期']}｜{row['初步判讀']}｜{row['標題']}" for _, row in subset.iterrows()]
    sentiment_counts = news_df["初步判讀"].value_counts().to_dict()
    count_text = "、".join(f"{k} {v} 則" for k, v in sentiment_counts.items())
    return (
        f"{stock_name}（{stock_id}）近期待判讀新聞共 {len(news_df)} 則。"
        f"標題關鍵字初判：{count_text or '無'}。\n\n最新重點：\n" + "\n".join(lines)
    )


# -----------------------------------------------------------------------------
# PTT Module
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_ptt_cached(keyword: str, max_pages: int = 2) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    base = "https://www.ptt.cc"
    headers = {"User-Agent": "Mozilla/5.0", "cookie": "over18=1"}
    url = f"{base}/bbs/Stock/search?q={requests.utils.quote(keyword)}"
    try:
        for _ in range(max_pages):
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                break
            html = resp.text
            for match in re.finditer(r'<div class="title">\s*<a href="([^"]+)">(.*?)</a>', html, re.S):
                href = match.group(1)
                title = unescape(re.sub(r"<.*?>", "", match.group(2))).strip()
                if title:
                    results.append({"title": title, "link": base + href})
            prev = re.search(r'<a class="btn wide" href="([^"]+)">&lsaquo;', html)
            if not prev:
                break
            url = base + unescape(prev.group(1))
            time.sleep(0.3)
    except Exception:
        return []

    seen = set()
    uniq = []
    for row in results:
        if row["title"] in seen:
            continue
        seen.add(row["title"])
        uniq.append(row)
    return uniq[:30]


# -----------------------------------------------------------------------------
# AI summary / analysis
# -----------------------------------------------------------------------------
def ai_summarize_news(news_df: pd.DataFrame, stock_name: str, stock_id: str) -> str:
    api_key = get_google_api_key()
    if not api_key:
        return "尚未設定 GOOGLE_API_KEY 或 GEMINI_API_KEY，所以未啟用 AI 摘要。"
    if news_df.empty:
        return "沒有新聞資料可摘要。"

    try:
        from google import genai
    except Exception:
        return "尚未安裝 google-genai，請確認 requirements.txt 是否已部署。"

    headlines = [f"{row['日期']}｜{row['初步判讀']}｜{row['標題']}" for _, row in news_df.head(20).iterrows()]
    prompt = f"""
你是台股新聞整理助手。請根據以下新聞標題，整理 {stock_name}（{stock_id}）的新聞重點。

要求：
1. 用繁體中文。
2. 不要給買賣建議，不要喊多喊空。
3. 分成：核心事件、可能利多、可能利空、需要追蹤的數據、短線觀察重點。
4. 標題資料不足時，要明確說資料不足，不要腦補。
5. 請提醒：標題初判不能取代完整基本面與技術面分析。

新聞標題：
{chr(10).join(headlines)}
""".strip()
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return (getattr(response, "text", "") or "").strip() or "AI 沒有回傳內容。"
    except Exception as exc:
        return f"AI 摘要失敗：{exc}"


def ai_analyze_stock(analysis_payload: Dict[str, Any], stock_name: str, stock_id: str) -> str:
    api_key = get_google_api_key()
    if not api_key:
        return "尚未設定 GOOGLE_API_KEY 或 GEMINI_API_KEY，所以未啟用 AI 個股分析說明。"
    try:
        from google import genai
    except Exception:
        return "尚未安裝 google-genai，請確認 requirements.txt 是否已部署。"

    prompt = f"""
你是台股研究助理。請根據下列結構化資料，對 {stock_name}（{stock_id}）做一份精簡、客觀的 AI 個股分析。

輸出要求：
1. 使用繁體中文。
2. 明確分成 A.基本面 B.籌碼面 C.消息面 D.技術面。
3. 每一面向先說出資料結論，再指出資料不足之處。
4. 最後用 3 句話總結目前觀察重點。
5. 禁止直接下買進/賣出指令。
6. 不要重複原始 JSON，請整理成人可讀文字。

資料：
{analysis_payload}
""".strip()
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return (getattr(response, "text", "") or "").strip() or "AI 沒有回傳內容。"
    except Exception as exc:
        return f"AI 個股分析失敗：{exc}"


# -----------------------------------------------------------------------------
# Market data fetchers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_price_cached(stock_id: str, days: int, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = finmind_get({"dataset": "TaiwanStockPrice", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = ["Trading_Volume", "Trading_money", "open", "max", "min", "close", "spread", "Trading_turnover"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_per_cached(stock_id: str, days: int = 120, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = finmind_get({"dataset": "TaiwanStockPER", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ["PER", "PBR", "dividend_yield"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_month_revenue_cached(stock_id: str, years: int = 3, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = f"{dt.date.today().year - years}-01-01"
    rows = finmind_get({"dataset": "TaiwanStockMonthRevenue", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ["revenue", "revenue_month", "revenue_year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_dividend_cached(stock_id: str, years: int = 5, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = f"{dt.date.today().year - years}-01-01"
    rows = finmind_get({"dataset": "TaiwanStockDividend", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    num_cols = ["CashEarningsDistribution", "CashStatutorySurplus", "StockEarningsDistribution", "year"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_financial_statements_cached(stock_id: str, years: int = 3, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = f"{dt.date.today().year - years}-01-01"
    rows = finmind_get({"dataset": "TaiwanStockFinancialStatements", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
    return df.sort_values(["date", "type"]).reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_institutional_wide_cached(stock_id: str, days: int = 60, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = finmind_get({"dataset": "TaiwanStockInstitutionalInvestorsBuySellWide", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in df.columns:
        if c not in {"date", "stock_id"}:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_shareholding_cached(stock_id: str, days: int = 60, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = finmind_get({"dataset": "TaiwanStockShareholding", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["ForeignInvestmentSharesRatio", "ForeignInvestmentRemainRatio", "ForeignInvestmentShares", "NumberOfSharesIssued"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_margin_short_cached(stock_id: str, days: int = 60, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = finmind_get({"dataset": "TaiwanStockMarginPurchaseShortSale", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in df.columns:
        if c not in {"date", "stock_id", "Note"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_short_balance_cached(stock_id: str, days: int = 60, token_marker: str = "") -> pd.DataFrame:
    _ = token_marker
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    rows = finmind_get({"dataset": "TaiwanDailyShortSaleBalances", "data_id": stock_id, "start_date": start})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in df.columns:
        if c not in {"date", "stock_id"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Technical calculations
# -----------------------------------------------------------------------------
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["MA5"] = out["close"].rolling(5).mean()
    out["MA20"] = out["close"].rolling(20).mean()
    out["MA60"] = out["close"].rolling(60).mean()
    rolling_std = out["close"].rolling(20).std()
    out["BB_upper"] = out["MA20"] + 2 * rolling_std
    out["BB_lower"] = out["MA20"] - 2 * rolling_std

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["DIF"] = ema12 - ema26
    out["DEA"] = out["DIF"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["DIF"] - out["DEA"]
    return out


def build_technical_chart(df: pd.DataFrame, stock_name: str, stock_id: str) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.48, 0.18, 0.17, 0.17],
        subplot_titles=(f"{stock_name} {stock_id} 日 K 線", "成交量", "RSI", "MACD"),
    )
    colors = np.where(df["close"] >= df["open"], "#3f8f6b", "#d76b5b")

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["max"],
            low=df["min"],
            close=df["close"],
            name="K線",
            increasing_line_color="#3f8f6b",
            decreasing_line_color="#d76b5b",
            increasing_fillcolor="#d9efe3",
            decreasing_fillcolor="#f7ddd7",
        ),
        row=1,
        col=1,
    )

    for name, color in [("MA5", "#b88a3b"), ("MA20", "#6f8f8f"), ("MA60", "#9d7b73")]:
        fig.add_trace(go.Scatter(x=df["date"], y=df[name], mode="lines", name=name, line=dict(width=1.6, color=color)), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["date"], y=df["BB_upper"], mode="lines", name="BB上緣", line=dict(color="#c1ab94", dash="dot", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["BB_lower"], mode="lines", name="BB下緣", line=dict(color="#c1ab94", dash="dot", width=1), fill="tonexty", fillcolor="rgba(193,171,148,0.08)"), row=1, col=1)

    fig.add_trace(go.Bar(x=df["date"], y=df["Trading_Volume"], name="成交量", marker_color=colors, opacity=0.85), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["RSI14"], mode="lines", name="RSI", line=dict(color="#7b8f8f", width=1.8)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#d2a79e", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#b8d2c0", line_width=1, row=3, col=1)

    fig.add_trace(go.Bar(x=df["date"], y=df["MACD_hist"], name="MACD柱", marker_color=np.where(df["MACD_hist"] >= 0, "#d76b5b", "#5b9a70"), opacity=0.9), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["DIF"], mode="lines", name="DIF", line=dict(color="#6f8f8f", width=1.8)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["DEA"], mode="lines", name="DEA", line=dict(color="#b88a3b", width=1.6)), row=4, col=1)

    fig.update_layout(
        height=920,
        margin=dict(l=20, r=20, t=48, b=20),
        paper_bgcolor="#f6f2eb",
        plot_bgcolor="#f6f2eb",
        showlegend=True,
        legend_orientation="h",
        legend_y=1.04,
        font=dict(color="#594f46", size=12),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(89,79,70,0.08)")
    fig.update_xaxes(showgrid=False)
    return fig


# -----------------------------------------------------------------------------
# Active ETF signals (best-effort)
# -----------------------------------------------------------------------------
def normalize_active_etf_signal_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    col_map = {c.lower().strip(): c for c in df.columns}
    required_alias = {
        "date": ["date", "日期"],
        "stock_id": ["stock_id", "股票代號", "代號"],
        "etf_id": ["etf_id", "etf代號", "基金代號"],
        "action": ["action", "方向", "+/-", "plus_minus"],
        "shares_delta": ["shares_delta", "change_shares", "股數變動", "張數變動", "delta"],
    }
    rename: Dict[str, str] = {}
    for target, aliases in required_alias.items():
        for alias in aliases:
            if alias.lower() in col_map:
                rename[col_map[alias.lower()]] = target
                break
    df = df.rename(columns=rename)
    missing = [c for c in ["date", "stock_id", "etf_id", "action", "shares_delta"] if c not in df.columns]
    if missing:
        raise ValueError(f"主動ETF資料缺少欄位：{', '.join(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stock_id"] = out["stock_id"].astype(str).str.strip()
    out["etf_id"] = out["etf_id"].astype(str).str.strip()
    out["action"] = out["action"].astype(str).str.strip().str.lower()
    out["shares_delta"] = pd.to_numeric(out["shares_delta"], errors="coerce")

    action_sign = []
    for action, delta in zip(out["action"], out["shares_delta"]):
        if action in {"+", "plus", "buy", "add", "increase", "up", "加碼"}:
            action_sign.append(1)
        elif action in {"-", "minus", "sell", "reduce", "decrease", "down", "減碼"}:
            action_sign.append(-1)
        else:
            action_sign.append(1 if (pd.notna(delta) and delta >= 0) else -1)
    out["signal"] = action_sign
    return out.dropna(subset=["date"])


def load_active_etf_signal_data(uploaded_file: Any = None) -> Tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return normalize_active_etf_signal_df(df), "uploaded"
        except Exception as exc:
            raise ValueError(f"主動ETF上傳資料讀取失敗：{exc}") from exc

    url = get_active_etf_signal_csv_url()
    if url:
        try:
            df = pd.read_csv(url)
            return normalize_active_etf_signal_df(df), "url"
        except Exception as exc:
            raise ValueError(f"ACTIVE_ETF_SIGNAL_CSV_URL 讀取失敗：{exc}") from exc

    return pd.DataFrame(), "none"


def summarize_active_etf_signals(signal_df: pd.DataFrame, stock_id: str, lookback_days: int = 10) -> Dict[str, Any]:
    if signal_df.empty:
        return {
            "available": False,
            "latest_date": None,
            "plus_count": np.nan,
            "minus_count": np.nan,
            "net_signal": np.nan,
            "net_shares_delta": np.nan,
            "details": pd.DataFrame(),
        }
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    df = signal_df[(signal_df["stock_id"] == str(stock_id)) & (signal_df["date"] >= cutoff)].copy()
    if df.empty:
        return {
            "available": False,
            "latest_date": None,
            "plus_count": 0,
            "minus_count": 0,
            "net_signal": 0,
            "net_shares_delta": 0,
            "details": pd.DataFrame(),
        }
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()
    plus_count = int((latest_df["signal"] > 0).sum())
    minus_count = int((latest_df["signal"] < 0).sum())
    net_signal = plus_count - minus_count
    net_shares_delta = float(latest_df["shares_delta"].fillna(0).sum())
    return {
        "available": True,
        "latest_date": latest_date,
        "plus_count": plus_count,
        "minus_count": minus_count,
        "net_signal": net_signal,
        "net_shares_delta": net_shares_delta,
        "details": latest_df.sort_values(["signal", "shares_delta"], ascending=[False, False]),
    }


# -----------------------------------------------------------------------------
# Analysis builders
# -----------------------------------------------------------------------------
def extract_statement_value(df: pd.DataFrame, aliases: List[str]) -> float:
    if df.empty:
        return np.nan
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    if latest.empty:
        return np.nan
    mask = pd.Series(False, index=latest.index)
    for alias in aliases:
        mask = mask | latest["type"].astype(str).str.contains(alias, case=False, na=False)
        if "origin_name" in latest.columns:
            mask = mask | latest["origin_name"].astype(str).str.contains(alias, case=False, na=False)
    vals = latest.loc[mask, "value"].dropna().tolist()
    return float(vals[0]) if vals else np.nan


def build_fundamental_payload(revenue_df: pd.DataFrame, per_df: pd.DataFrame, dividend_df: pd.DataFrame, fs_df: pd.DataFrame) -> Dict[str, Any]:
    latest_revenue = revenue_df.iloc[-1] if not revenue_df.empty else pd.Series(dtype=float)
    latest_per = per_df.iloc[-1] if not per_df.empty else pd.Series(dtype=float)
    latest_dividend = dividend_df.sort_values(["year", "date"]).iloc[-1] if not dividend_df.empty else pd.Series(dtype=float)

    revenue_yoy = to_float(latest_revenue.get("revenue_year"))
    revenue_mom = to_float(latest_revenue.get("revenue_month"))
    latest_revenue_val = to_float(latest_revenue.get("revenue"))
    latest_revenue_date = latest_revenue.get("date")

    eps = extract_statement_value(fs_df, ["BasicEarningsPerShare", "EarningsPerShare", "每股盈餘", "基本每股盈餘"])
    revenue_q = extract_statement_value(fs_df, ["Revenue", "營業收入", "營收"])
    gross_profit = extract_statement_value(fs_df, ["GrossProfit", "營業毛利", "毛利"])
    operating_income = extract_statement_value(fs_df, ["OperatingIncome", "營業利益", "營業淨利"])
    net_income = extract_statement_value(fs_df, ["ProfitLoss", "本期淨利", "稅後淨利"])

    gross_margin = (gross_profit / revenue_q * 100) if pd.notna(gross_profit) and pd.notna(revenue_q) and revenue_q not in [0, np.nan] else np.nan
    operating_margin = (operating_income / revenue_q * 100) if pd.notna(operating_income) and pd.notna(revenue_q) and revenue_q not in [0, np.nan] else np.nan
    net_margin = (net_income / revenue_q * 100) if pd.notna(net_income) and pd.notna(revenue_q) and revenue_q not in [0, np.nan] else np.nan

    latest_per_val = to_float(latest_per.get("PER"))
    latest_pbr_val = to_float(latest_per.get("PBR"))
    latest_div_yield = to_float(latest_per.get("dividend_yield"))
    latest_cash_div = to_float(latest_dividend.get("CashEarningsDistribution")) + to_float(latest_dividend.get("CashStatutorySurplus"))

    score = 50.0
    if pd.notna(revenue_yoy):
        score += np.interp(revenue_yoy, [-30, 0, 15, 30], [-16, -4, 10, 16])
    if pd.notna(revenue_mom):
        score += np.interp(revenue_mom, [-20, 0, 10, 20], [-8, -2, 4, 8])
    if pd.notna(latest_per_val):
        if 8 <= latest_per_val <= 25:
            score += 8
        elif latest_per_val < 8:
            score += 5
        elif latest_per_val > 35:
            score -= 8
        else:
            score -= 2
    if pd.notna(latest_div_yield):
        score += np.interp(latest_div_yield, [0, 2, 5, 8], [0, 3, 8, 10])
    if pd.notna(eps):
        score += np.interp(eps, [-2, 0, 5, 15], [-10, -2, 8, 12])
    if pd.notna(gross_margin):
        score += np.interp(gross_margin, [10, 20, 35, 50], [-4, 2, 6, 10])
    score = clamp(score)

    return {
        "score": round(score, 1),
        "latest_revenue": latest_revenue_val,
        "latest_revenue_date": str(latest_revenue_date.date()) if isinstance(latest_revenue_date, pd.Timestamp) else "—",
        "revenue_yoy": revenue_yoy,
        "revenue_mom": revenue_mom,
        "per": latest_per_val,
        "pbr": latest_pbr_val,
        "dividend_yield": latest_div_yield,
        "cash_dividend": latest_cash_div,
        "eps": eps,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
    }


def build_chip_payload(inst_df: pd.DataFrame, shareholding_df: pd.DataFrame, margin_df: pd.DataFrame, short_df: pd.DataFrame, active_etf_summary: Dict[str, Any]) -> Dict[str, Any]:
    inst_df = inst_df.copy()
    if not inst_df.empty:
        inst_df["foreign_net"] = inst_df.get("Foreign_Investor_buy", 0) - inst_df.get("Foreign_Investor_sell", 0) + inst_df.get("Foreign_Dealer_Self_buy", 0) - inst_df.get("Foreign_Dealer_Self_sell", 0)
        inst_df["trust_net"] = inst_df.get("Investment_Trust_buy", 0) - inst_df.get("Investment_Trust_sell", 0)
        inst_df["dealer_net"] = inst_df.get("Dealer_buy", 0) - inst_df.get("Dealer_sell", 0) + inst_df.get("Dealer_self_buy", 0) - inst_df.get("Dealer_self_sell", 0) + inst_df.get("Dealer_Hedging_buy", 0) - inst_df.get("Dealer_Hedging_sell", 0)
        inst_df["three_total_net"] = inst_df[["foreign_net", "trust_net", "dealer_net"]].sum(axis=1)

    inst_5d = inst_df.tail(5)["three_total_net"].sum() if not inst_df.empty else np.nan
    foreign_5d = inst_df.tail(5)["foreign_net"].sum() if not inst_df.empty else np.nan
    trust_5d = inst_df.tail(5)["trust_net"].sum() if not inst_df.empty else np.nan

    latest_share = shareholding_df.iloc[-1] if not shareholding_df.empty else pd.Series(dtype=float)
    prev_share = shareholding_df.iloc[-6] if len(shareholding_df) >= 6 else pd.Series(dtype=float)
    foreign_ratio = to_float(latest_share.get("ForeignInvestmentSharesRatio"))
    foreign_ratio_delta_5d = foreign_ratio - to_float(prev_share.get("ForeignInvestmentSharesRatio")) if not latest_share.empty and not prev_share.empty else np.nan

    latest_margin = margin_df.iloc[-1] if not margin_df.empty else pd.Series(dtype=float)
    prev_margin = margin_df.iloc[-6] if len(margin_df) >= 6 else pd.Series(dtype=float)
    margin_balance = to_float(latest_margin.get("MarginPurchaseTodayBalance"))
    margin_delta_5d = margin_balance - to_float(prev_margin.get("MarginPurchaseTodayBalance")) if not latest_margin.empty and not prev_margin.empty else np.nan
    short_sale_balance = to_float(latest_margin.get("ShortSaleTodayBalance"))
    short_sale_delta_5d = short_sale_balance - to_float(prev_margin.get("ShortSaleTodayBalance")) if not latest_margin.empty and not prev_margin.empty else np.nan

    latest_short = short_df.iloc[-1] if not short_df.empty else pd.Series(dtype=float)
    prev_short = short_df.iloc[-6] if len(short_df) >= 6 else pd.Series(dtype=float)
    sbl_balance = to_float(latest_short.get("SBLShortSalesCurrentDayBalance"))
    sbl_delta_5d = sbl_balance - to_float(prev_short.get("SBLShortSalesCurrentDayBalance")) if not latest_short.empty and not prev_short.empty else np.nan

    score = 50.0
    if pd.notna(inst_5d):
        score += np.interp(inst_5d, [-50000, -10000, 0, 10000, 50000], [-16, -6, 0, 8, 16])
    if pd.notna(foreign_ratio_delta_5d):
        score += np.interp(foreign_ratio_delta_5d, [-2, -0.5, 0, 0.5, 2], [-10, -4, 0, 4, 10])
    if pd.notna(margin_delta_5d):
        score += np.interp(margin_delta_5d, [-10000, -1000, 0, 1000, 10000], [6, 2, 0, -2, -6])
    if pd.notna(sbl_delta_5d):
        score += np.interp(sbl_delta_5d, [-10000, -1000, 0, 1000, 10000], [8, 3, 0, -3, -8])
    if active_etf_summary.get("available"):
        score += np.interp(active_etf_summary.get("net_signal", 0), [-5, -2, 0, 2, 5], [-8, -4, 0, 4, 8])
    score = clamp(score)

    return {
        "score": round(score, 1),
        "institutional_5d_net": inst_5d,
        "foreign_5d_net": foreign_5d,
        "trust_5d_net": trust_5d,
        "foreign_shareholding_ratio": foreign_ratio,
        "foreign_shareholding_delta_5d": foreign_ratio_delta_5d,
        "margin_balance": margin_balance,
        "margin_delta_5d": margin_delta_5d,
        "short_sale_balance": short_sale_balance,
        "short_sale_delta_5d": short_sale_delta_5d,
        "sbl_balance": sbl_balance,
        "sbl_delta_5d": sbl_delta_5d,
        "active_etf_plus": active_etf_summary.get("plus_count"),
        "active_etf_minus": active_etf_summary.get("minus_count"),
        "active_etf_net": active_etf_summary.get("net_signal"),
        "active_etf_shares_delta": active_etf_summary.get("net_shares_delta"),
        "active_etf_available": active_etf_summary.get("available", False),
    }


def build_news_payload(news_df: pd.DataFrame, ptt_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    if news_df.empty:
        return {
            "score": 50.0,
            "news_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "ptt_count": len(ptt_rows),
            "headline_bias": "資料不足",
        }

    counts = news_df["初步判讀"].value_counts().to_dict()
    positive = counts.get("偏利多", 0)
    negative = counts.get("偏利空", 0)
    neutral = counts.get("中性 / 待判讀", 0)
    total = len(news_df)
    sentiment_balance = (positive - negative) / max(total, 1)
    score = clamp(50 + sentiment_balance * 35 + min(total, 15) * 0.7)

    if positive > negative:
        bias = "偏多"
    elif negative > positive:
        bias = "偏空"
    else:
        bias = "中性"

    return {
        "score": round(score, 1),
        "news_count": total,
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "ptt_count": len(ptt_rows),
        "headline_bias": bias,
    }


def build_technical_payload(price_df: pd.DataFrame) -> Dict[str, Any]:
    if price_df.empty:
        return {
            "score": 50.0,
            "close": np.nan,
            "ma20": np.nan,
            "ma60": np.nan,
            "rsi": np.nan,
            "macd_hist": np.nan,
            "bb_position": np.nan,
            "pct_from_ma20": np.nan,
            "pct_from_ma60": np.nan,
        }
    latest = price_df.iloc[-1]
    close = to_float(latest.get("close"))
    ma20 = to_float(latest.get("MA20"))
    ma60 = to_float(latest.get("MA60"))
    rsi = to_float(latest.get("RSI14"))
    macd_hist = to_float(latest.get("MACD_hist"))
    bb_upper = to_float(latest.get("BB_upper"))
    bb_lower = to_float(latest.get("BB_lower"))
    bb_position = ((close - bb_lower) / (bb_upper - bb_lower) * 100) if pd.notna(close) and pd.notna(bb_upper) and pd.notna(bb_lower) and (bb_upper - bb_lower) != 0 else np.nan
    pct_from_ma20 = ((close - ma20) / ma20 * 100) if pd.notna(ma20) and ma20 != 0 else np.nan
    pct_from_ma60 = ((close - ma60) / ma60 * 100) if pd.notna(ma60) and ma60 != 0 else np.nan

    score = 50.0
    if pd.notna(pct_from_ma20):
        score += np.interp(pct_from_ma20, [-15, -5, 0, 5, 15], [-15, -5, 2, 8, 12])
    if pd.notna(pct_from_ma60):
        score += np.interp(pct_from_ma60, [-20, -5, 0, 5, 20], [-12, -5, 2, 8, 12])
    if pd.notna(rsi):
        if 45 <= rsi <= 68:
            score += 10
        elif rsi > 80 or rsi < 25:
            score -= 8
        elif 35 <= rsi < 45 or 68 < rsi <= 75:
            score += 2
        else:
            score -= 2
    if pd.notna(macd_hist):
        score += np.interp(macd_hist, [-10, -2, 0, 2, 10], [-10, -4, 0, 4, 10])
    score = clamp(score)

    return {
        "score": round(score, 1),
        "close": close,
        "ma20": ma20,
        "ma60": ma60,
        "rsi": rsi,
        "macd_hist": macd_hist,
        "bb_position": bb_position,
        "pct_from_ma20": pct_from_ma20,
        "pct_from_ma60": pct_from_ma60,
    }


def build_overall_score(fundamental: Dict[str, Any], chip: Dict[str, Any], news: Dict[str, Any], technical: Dict[str, Any]) -> Dict[str, Any]:
    weights = {"fundamental": 0.30, "chip": 0.25, "news": 0.20, "technical": 0.25}
    total = (
        fundamental["score"] * weights["fundamental"]
        + chip["score"] * weights["chip"]
        + news["score"] * weights["news"]
        + technical["score"] * weights["technical"]
    )
    return {
        "overall": round(total, 1),
        "weights": weights,
    }


# -----------------------------------------------------------------------------
# UI rendering helpers
# -----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f4ef 0%, #f5f2ec 100%);
            color: #3e3730;
        }
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1360px;}
        h1, h2, h3, h4 {color: #453f38; letter-spacing: -0.02em;}
        .apple-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(89,79,70,0.08);
            border-radius: 24px;
            padding: 18px 20px;
            box-shadow: 0 14px 38px rgba(86,76,68,0.07);
            backdrop-filter: blur(18px);
            margin-bottom: 14px;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,255,255,0.74));
            border-radius: 28px;
            padding: 20px 22px;
            border: 1px solid rgba(89,79,70,0.07);
            box-shadow: 0 18px 40px rgba(86,76,68,0.09);
            margin-bottom: 18px;
        }
        .small-label {font-size: 0.84rem; color: #8d8074; margin-bottom: 4px;}
        .big-value {font-size: 2.1rem; line-height: 1.1; font-weight: 700; color: #403932;}
        .section-title {font-size: 1.15rem; font-weight: 700; margin-bottom: 6px; color: #4b433c;}
        .section-sub {font-size: 0.88rem; color: #8b7d70; margin-bottom: 14px;}
        .score-row {display:flex; align-items:center; gap:12px; margin: 2px 0 12px 0;}
        .score-pill {
            min-width: 64px; text-align:center; padding: 8px 10px; border-radius: 999px;
            background:#f2ece4; color:#3e3730; font-weight:700; font-size: 1rem; border:1px solid rgba(89,79,70,0.08);
        }
        .score-bar {width:100%; height:10px; background:#ece5dc; border-radius:999px; overflow:hidden;}
        .score-fill {height:100%; border-radius:999px; background: linear-gradient(90deg, #d9a389, #b88a3b, #7f9b8e);}
        .metric-grid {display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px;}
        .metric-item {padding: 10px 12px; border-radius: 16px; background:#fbf9f5; border:1px solid rgba(89,79,70,0.06);}
        .metric-name {font-size: 0.82rem; color:#8d8074;}
        .metric-value {font-size: 1.05rem; font-weight: 650; margin-top: 2px; color:#453f38;}
        .metric-delta {font-size: 0.82rem; margin-top: 3px; color:#8b7d70;}
        .pill {display:inline-flex; align-items:center; padding:4px 10px; border-radius:999px; font-size:0.78rem; font-weight:600; margin-right:6px;}
        .pill-good {background:#e3f1ea; color:#3f7356;}
        .pill-bad {background:#f9e5e2; color:#b15e51;}
        .pill-neutral {background:#efe8df; color:#726459;}
        .pill-warn {background:#f5edd8; color:#9a7d2e;}
        .summary-list {padding-left: 18px; margin: 0;}
        .summary-list li {margin-bottom: 6px; color:#5c534b;}
        .muted {color:#8d8074; font-size:0.88rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_secret_status() -> None:
    finmind_ok = bool(get_finmind_token())
    google_ok = bool(get_google_api_key())
    col1, col2 = st.columns(2)
    with col1:
        st.caption("FinMind Token")
        st.write("✅ 已設定" if finmind_ok else "⚠️ 未設定，會用較低免費額度")
    with col2:
        st.caption("Google AI Key")
        st.write("✅ 已設定" if google_ok else "⚠️ 未設定，AI 功能停用")


def render_score_box(title: str, subtitle: str, score: float, metrics: List[Tuple[str, str, str]], tags: Optional[List[str]] = None) -> None:
    html = [
        '<div class="apple-card">',
        f'<div class="section-title">{title}</div>',
        f'<div class="section-sub">{subtitle}</div>',
        '<div class="score-row">',
        f'<div class="score-pill">{score:.1f}</div>',
        '<div class="score-bar"><div class="score-fill" style="width:{:.1f}%"></div></div>'.format(score),
        '</div>',
    ]
    if tags:
        html.append('<div style="margin-bottom:10px;">' + " ".join(tags) + "</div>")
    html.append('<div class="metric-grid">')
    for name, value, delta in metrics:
        html.append(
            f'''<div class="metric-item">
                    <div class="metric-name">{name}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-delta">{delta or '&nbsp;'}</div>
                </div>'''
        )
    html.append('</div></div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="台股新聞 + AI 個股分析", page_icon="📈", layout="wide")
    inject_css()

    st.title("📈 台股新聞模組 ＋ AI 個股分析")
    st.caption("Apple 風格儀表板｜新聞、籌碼、基本面、技術面整合。評分是量化輔助，不是投資建議。")

    with st.sidebar:
        st.header("查詢設定")
        query = st.text_input("股票名稱或代號", value="台積電", placeholder="例：台積電 / 2330 / 聯發科")
        news_days = st.slider("新聞天數", min_value=1, max_value=30, value=7)
        price_days = st.slider("技術面天數", min_value=90, max_value=365, value=180, step=10)
        include_ptt = st.checkbox("抓 PTT 股票板討論", value=True)
        include_news_ai = st.checkbox("新聞 AI 摘要", value=True)
        include_stock_ai = st.checkbox("AI 個股分析說明", value=True)
        st.divider()
        st.markdown("**主動 ETF +/- 碼**")
        st.caption("這塊如果你沒有固定資料源，程式只能做 best-effort。可上傳 CSV，或在 Secrets 設定 ACTIVE_ETF_SIGNAL_CSV_URL。")
        active_etf_file = st.file_uploader("上傳主動 ETF 加減碼 CSV", type=["csv"])
        st.divider()
        render_secret_status()
        st.divider()
        st.caption("Secrets 建議：FINMIND_TOKEN、GOOGLE_API_KEY、ACTIVE_ETF_SIGNAL_CSV_URL")

    token_marker = "token-on" if get_finmind_token() else "token-off"

    try:
        stock_info = load_stock_info_cached(token_marker=token_marker)
    except Exception as exc:
        st.error(f"股票清單載入失敗：{exc}")
        st.stop()

    match, candidates, message = resolve_stock(query, stock_info)
    if candidates:
        st.warning(message)
        display = [f"{r['stock_name']}（{r['stock_id']}）" for r in candidates]
        selected = st.selectbox("請選擇標的", display)
        idx = display.index(selected)
        match = candidates[idx]
    elif message and not match:
        st.warning(message)
        st.stop()

    assert match is not None
    stock_id = match["stock_id"]
    stock_name = match["stock_name"]

    # Load data
    with st.spinner("讀取市場資料中..."):
        news_dicts = fetch_news_cached(stock_id, news_days, token_marker=token_marker)
        news_df = news_to_dataframe(news_dicts)
        ptt_rows = fetch_ptt_cached(stock_name, max_pages=2) if include_ptt else []
        price_df = add_technical_indicators(fetch_price_cached(stock_id, price_days, token_marker=token_marker))
        per_df = fetch_per_cached(stock_id, token_marker=token_marker)
        revenue_df = fetch_month_revenue_cached(stock_id, token_marker=token_marker)
        dividend_df = fetch_dividend_cached(stock_id, token_marker=token_marker)
        fs_df = fetch_financial_statements_cached(stock_id, token_marker=token_marker)
        inst_df = fetch_institutional_wide_cached(stock_id, token_marker=token_marker)
        shareholding_df = fetch_shareholding_cached(stock_id, token_marker=token_marker)
        margin_df = fetch_margin_short_cached(stock_id, token_marker=token_marker)
        short_df = fetch_short_balance_cached(stock_id, token_marker=token_marker)

    active_etf_warning = None
    try:
        active_signal_df, signal_source = load_active_etf_signal_data(active_etf_file)
    except Exception as exc:
        active_signal_df, signal_source = pd.DataFrame(), "error"
        active_etf_warning = str(exc)

    active_etf_summary = summarize_active_etf_signals(active_signal_df, stock_id)

    fundamental = build_fundamental_payload(revenue_df, per_df, dividend_df, fs_df)
    chip = build_chip_payload(inst_df, shareholding_df, margin_df, short_df, active_etf_summary)
    news_payload = build_news_payload(news_df, ptt_rows)
    technical = build_technical_payload(price_df)
    overall = build_overall_score(fundamental, chip, news_payload, technical)

    latest_close = price_df.iloc[-1]["close"] if not price_df.empty else np.nan
    latest_spread = price_df.iloc[-1]["spread"] if not price_df.empty and "spread" in price_df.columns else np.nan
    spread_text = metric_delta_text(latest_spread) if pd.notna(latest_spread) else "—"
    score_kind = "good" if overall["overall"] >= 65 else ("warn" if overall["overall"] >= 45 else "bad")

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="small-label">{match.get('industry_category') or '—'}｜{match.get('market') or '—'}｜近 {news_days} 天新聞｜近 {price_days} 天技術面</div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:18px; flex-wrap:wrap;">
                <div>
                    <div class="section-title" style="font-size:1.6rem;">{stock_name}（{stock_id}）</div>
                    <div class="muted">AI 綜合評分依據 A~D 四面向加權試算，目的是幫你快速篩檢，不是替你做交易決策。</div>
                </div>
                <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
                    <div>
                        <div class="small-label">最新收盤</div>
                        <div class="big-value">{safe_num(latest_close)}</div>
                        <div class="muted">日漲跌：{spread_text}</div>
                    </div>
                    <div>
                        <div class="small-label">AI 綜合評分</div>
                        <div class="big-value">{overall['overall']:.1f}</div>
                        <div>{render_tag('強勢結構' if score_kind == 'good' else ('中性觀察' if score_kind == 'warn' else '弱勢/保守看待'), score_kind)}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if active_etf_warning:
        st.warning(active_etf_warning)

    # 4 block dashboard
    row1_col1, row1_col2 = st.columns(2, gap="large")
    row2_col1, row2_col2 = st.columns(2, gap="large")

    with row1_col1:
        render_score_box(
            "A. 基本面",
            "月營收、估值、殖利率、EPS、獲利率等快速彙整",
            fundamental["score"],
            [
                ("月營收", safe_int(fundamental["latest_revenue"]), f"{fundamental['latest_revenue_date']}") ,
                ("營收 YoY", safe_pct(fundamental["revenue_yoy"]), f"MoM {safe_pct(fundamental['revenue_mom'])}"),
                ("PER / PBR", f"{safe_num(fundamental['per'])} / {safe_num(fundamental['pbr'])}", f"殖利率 {safe_pct(fundamental['dividend_yield'])}"),
                ("EPS", safe_num(fundamental["eps"]), f"現金股利 {safe_num(fundamental['cash_dividend'])}"),
                ("毛利率", safe_pct(fundamental["gross_margin"]), f"營益率 {safe_pct(fundamental['operating_margin'])}"),
                ("淨利率", safe_pct(fundamental["net_margin"]), "最新財報口徑"),
            ],
            tags=[
                render_tag("營收動能" if (pd.notna(fundamental["revenue_yoy"]) and fundamental["revenue_yoy"] > 0) else "營收偏弱", "good" if (pd.notna(fundamental["revenue_yoy"]) and fundamental["revenue_yoy"] > 0) else "bad"),
                render_tag("估值適中" if (pd.notna(fundamental["per"]) and 8 <= fundamental["per"] <= 25) else "估值需再判斷", "neutral"),
            ],
        )

    with row1_col2:
        render_score_box(
            "B. 籌碼面",
            "三大法人、外資持股、融資融券、借券與主動 ETF +/- 碼",
            chip["score"],
            [
                ("三大法人 5 日淨額", safe_int(chip["institutional_5d_net"]), f"外資 {safe_int(chip['foreign_5d_net'])} / 投信 {safe_int(chip['trust_5d_net'])}"),
                ("外資持股比", safe_pct(chip["foreign_shareholding_ratio"]), f"5日變化 {metric_delta_text(chip['foreign_shareholding_delta_5d'], '%')}"),
                ("融資餘額", safe_int(chip["margin_balance"]), f"5日變化 {metric_delta_text(chip['margin_delta_5d'])}"),
                ("融券餘額", safe_int(chip["short_sale_balance"]), f"5日變化 {metric_delta_text(chip['short_sale_delta_5d'])}"),
                ("借券餘額", safe_int(chip["sbl_balance"]), f"5日變化 {metric_delta_text(chip['sbl_delta_5d'])}"),
                ("主動 ETF +/- 碼", (f"+{int(chip['active_etf_plus'])} / -{int(chip['active_etf_minus'])}" if pd.notna(chip['active_etf_plus']) and pd.notna(chip['active_etf_minus']) else "—"), f"淨值 {metric_delta_text(chip['active_etf_net'])}；股數 {metric_delta_text(chip['active_etf_shares_delta'])}"),
            ],
            tags=[
                render_tag("法人偏多" if (pd.notna(chip["institutional_5d_net"]) and chip["institutional_5d_net"] > 0) else "法人偏空", "good" if (pd.notna(chip["institutional_5d_net"]) and chip["institutional_5d_net"] > 0) else "bad"),
                render_tag("主動ETF已接入" if chip["active_etf_available"] else "主動ETF待資料源", "good" if chip["active_etf_available"] else "warn"),
            ],
        )

    with row2_col1:
        render_score_box(
            "C. 消息面",
            "FinMind 新聞 + PTT 熱度，先做情緒分布再交給 AI 說明",
            news_payload["score"],
            [
                ("新聞總數", safe_int(news_payload["news_count"]), f"近 {news_days} 天"),
                ("利多 / 利空", f"{news_payload['positive_count']} / {news_payload['negative_count']}", f"中性 {news_payload['neutral_count']} 則"),
                ("標題傾向", news_payload["headline_bias"], "標題關鍵字初判"),
                ("PTT 討論", safe_int(news_payload["ptt_count"]), "僅供熱度參考"),
            ],
            tags=[
                render_tag("新聞偏正向" if news_payload["headline_bias"] == "偏多" else ("新聞偏負向" if news_payload["headline_bias"] == "偏空" else "新聞中性"), "good" if news_payload["headline_bias"] == "偏多" else ("bad" if news_payload["headline_bias"] == "偏空" else "neutral")),
            ],
        )
        if not news_df.empty:
            top_rows = news_df.head(4)
            html = ['<div class="apple-card"><div class="section-title">新聞重點</div><ul class="summary-list">']
            for _, row in top_rows.iterrows():
                html.append(f"<li>{row['日期']}｜{row['初步判讀']}｜{row['標題']}</li>")
            html.append('</ul></div>')
            st.markdown("".join(html), unsafe_allow_html=True)

    with row2_col2:
        render_score_box(
            "D. 技術面",
            "趨勢、強弱、均線、布林通道與 MACD 狀態",
            technical["score"],
            [
                ("收盤價", safe_num(technical["close"]), f"MA20 {safe_num(technical['ma20'])} / MA60 {safe_num(technical['ma60'])}"),
                ("距 MA20", safe_pct(technical["pct_from_ma20"]), f"距 MA60 {safe_pct(technical['pct_from_ma60'])}"),
                ("RSI14", safe_num(technical["rsi"]), "45~68 視為偏健康區"),
                ("MACD 柱", safe_num(technical["macd_hist"]), "正值代表短期動能較強"),
                ("布林位置", safe_pct(technical["bb_position"]), "越靠近 100 越接近上緣"),
            ],
            tags=[
                render_tag("站上均線" if (pd.notna(technical["pct_from_ma20"]) and technical["pct_from_ma20"] > 0 and pd.notna(technical["pct_from_ma60"]) and technical["pct_from_ma60"] > 0) else "均線偏弱", "good" if (pd.notna(technical["pct_from_ma20"]) and technical["pct_from_ma20"] > 0 and pd.notna(technical["pct_from_ma60"]) and technical["pct_from_ma60"] > 0) else "bad"),
                render_tag("MACD 翻正" if (pd.notna(technical["macd_hist"]) and technical["macd_hist"] > 0) else "MACD 偏弱", "good" if (pd.notna(technical["macd_hist"]) and technical["macd_hist"] > 0) else "bad"),
            ],
        )

    st.markdown("### 技術面圖表")
    if price_df.empty:
        st.info("抓不到價格資料，技術圖表無法顯示。")
    else:
        st.plotly_chart(build_technical_chart(price_df.tail(120), stock_name, stock_id), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(["AI 個股分析", "新聞模組", "PTT / 主動ETF 明細", "原始資料摘要"])

    with tab1:
        st.markdown("### AI 個股分析")
        analysis_payload = {
            "stock": {"stock_id": stock_id, "stock_name": stock_name, "industry": match.get("industry_category"), "market": match.get("market")},
            "overall": overall,
            "fundamental": fundamental,
            "chip": chip,
            "news": news_payload,
            "technical": technical,
        }
        if include_stock_ai:
            with st.spinner("AI 正在整理四大面向..."):
                st.write(ai_analyze_stock(analysis_payload, stock_name, stock_id))
        else:
            st.info("你已關閉 AI 個股分析說明。")

        st.markdown("### 評分結構")
        score_df = pd.DataFrame(
            [
                ["A. 基本面", fundamental["score"], overall["weights"]["fundamental"]],
                ["B. 籌碼面", chip["score"], overall["weights"]["chip"]],
                ["C. 消息面", news_payload["score"], overall["weights"]["news"]],
                ["D. 技術面", technical["score"], overall["weights"]["technical"]],
            ],
            columns=["面向", "分數", "權重"],
        )
        score_df["加權分數"] = score_df["分數"] * score_df["權重"]
        st.dataframe(score_df, use_container_width=True, hide_index=True)
        st.caption("盲點提醒：這個分數偏向量化快篩。你有沒有想過，產業循環、法說指引、一次性業外與政策風險都可能讓分數失真？")

    with tab2:
        st.markdown("### 新聞模組")
        if news_df.empty:
            st.info("這段期間沒有抓到新聞。")
        else:
            left, right = st.columns([1.2, 1])
            with left:
                st.dataframe(
                    news_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={"連結": st.column_config.LinkColumn("連結", display_text="開啟")},
                )
                csv = news_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下載新聞 CSV", data=csv, file_name=f"{stock_id}_{stock_name}_news.csv", mime="text/csv")
            with right:
                st.write(make_news_brief(news_df, stock_name, stock_id))

        if include_news_ai:
            st.markdown("### 新聞 AI 摘要")
            with st.spinner("產生新聞 AI 摘要中..."):
                st.write(ai_summarize_news(news_df, stock_name, stock_id))

    with tab3:
        left, right = st.columns(2)
        with left:
            st.markdown("### PTT 股票板討論")
            if not ptt_rows:
                st.info("沒有抓到 PTT 討論，或你已關閉。")
            else:
                for row in ptt_rows[:15]:
                    st.markdown(f"- [{row['title']}]({row['link']})")
        with right:
            st.markdown("### 主動 ETF +/- 碼明細")
            if not active_etf_summary.get("available"):
                st.info("目前沒有主動 ETF 加減碼資料。你可以上傳 CSV，或在 Secrets 指定 ACTIVE_ETF_SIGNAL_CSV_URL。")
                st.caption("CSV 最少欄位：date, stock_id, etf_id, action, shares_delta")
            else:
                st.write(
                    f"最新資料日：{active_etf_summary['latest_date'].date()}｜+碼 {active_etf_summary['plus_count']} 檔 ETF｜-碼 {active_etf_summary['minus_count']} 檔 ETF｜淨值 {active_etf_summary['net_signal']}"
                )
                st.dataframe(active_etf_summary["details"], use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### 原始資料摘要")
        mini_tabs = st.tabs(["基本面", "籌碼面", "技術面"])
        with mini_tabs[0]:
            st.write("月營收")
            st.dataframe(revenue_df.tail(12), use_container_width=True, hide_index=True)
            st.write("PER / PBR")
            st.dataframe(per_df.tail(10), use_container_width=True, hide_index=True)
            st.write("股利")
            st.dataframe(dividend_df.tail(10), use_container_width=True, hide_index=True)
        with mini_tabs[1]:
            st.write("三大法人（寬表）")
            st.dataframe(inst_df.tail(10), use_container_width=True, hide_index=True)
            st.write("外資持股")
            st.dataframe(shareholding_df.tail(10), use_container_width=True, hide_index=True)
            st.write("融資融券")
            st.dataframe(margin_df.tail(10), use_container_width=True, hide_index=True)
        with mini_tabs[2]:
            st.dataframe(price_df.tail(20), use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "風險提醒：這個頁面把很多資料拼在一起，但不代表結論一定對。最大的 3 個風險是："
        "(1) 主動 ETF 加減碼若沒有穩定官方資料源，結果只會是 best-effort；"
        "(2) 財報欄位抓取受公司科目命名影響，部分指標可能缺值；"
        "(3) 新聞情緒與量化評分不能取代法說、產業趨勢與風險管理。"
    )


if __name__ == "__main__":
    main()
