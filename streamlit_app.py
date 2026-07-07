"""
streamlit_app.py — 台股個股新聞 / PTT 討論 / AI 摘要模組

部署方式：
1. GitHub 放這個檔案 + requirements.txt
2. Streamlit Community Cloud 部署時，Secrets 填入：
   FINMIND_TOKEN = "你的 FinMind token"
   GOOGLE_API_KEY = "你的 Google AI / Gemini API key"

注意：不要把真實 API key 寫死在 GitHub 程式碼裡。
"""

from __future__ import annotations

import os
import re
import json
import time
import datetime as dt
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
CACHE_TTL_SECONDS = 60 * 60 * 24
REQUEST_TIMEOUT = 20


# -----------------------------------------------------------------------------
# Secrets / Settings
# -----------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    """優先讀 Streamlit Secrets，再讀環境變數。"""
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
    # Google 新 SDK 官方偏好 GEMINI_API_KEY；使用者習慣常叫 GOOGLE_API_KEY。
    return get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")


# -----------------------------------------------------------------------------
# FinMind API
# -----------------------------------------------------------------------------
class FinMindError(RuntimeError):
    pass


def finmind_get(params: Dict[str, Any], token: Optional[str] = None) -> List[Dict[str, Any]]:
    """呼叫 FinMind v4 data endpoint，回傳 data 陣列。"""
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
    """抓台股股票清單。token_marker 只用來讓 cache 隨 token 狀態區隔。"""
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
        rows.append({
            "stock_id": stock_id,
            "stock_name": stock_name,
            "industry_category": industry_category,
            "market": market,
        })
    return rows


def resolve_stock(query: str, stock_info: List[Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], List[Dict[str, str]], str]:
    """
    回傳：
    - match: 唯一匹配股票，找不到或多筆時為 None
    - candidates: 多筆候選
    - message: 給 UI 顯示的訊息
    """
    q = query.strip()
    if not q:
        return None, [], "請輸入股票名稱或代號。"

    # 1) 精準代號
    if re.fullmatch(r"\d{4,6}", q):
        exact = [r for r in stock_info if r["stock_id"] == q]
        if exact:
            return exact[0], [], ""
        return None, [], f"找不到代號「{q}」。"

    # 2) 精準中文名
    exact_name = [r for r in stock_info if r["stock_name"] == q]
    if len(exact_name) == 1:
        return exact_name[0], [], ""

    # 3) 部分中文名 / 代號前綴
    candidates = [
        r for r in stock_info
        if q in r["stock_name"] or r["stock_id"].startswith(q)
    ]

    if len(candidates) == 1:
        return candidates[0], [], ""
    if len(candidates) > 1:
        return None, candidates[:30], f"「{q}」對應到多檔，請從候選清單選一檔。"

    return None, [], f"找不到「{q}」，請確認名稱或改用股票代號。"


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


def normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # FinMind 常見格式：YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS
    return text[:10]


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
    rows = finmind_get({
        "dataset": "TaiwanStockNews",
        "data_id": stock_id,
        "start_date": start,
    })
    items = normalize_news_rows(rows, stock_id)
    return [item.__dict__ for item in items]


POSITIVE_KEYWORDS = [
    "創高", "大增", "成長", "上修", "優於", "轉盈", "獲利", "接單", "訂單", "擴產",
    "漲價", "調漲", "法說", "利多", "突破", "買超", "合作", "投資", "併購", "出貨",
]
NEGATIVE_KEYWORDS = [
    "下修", "衰退", "虧損", "減產", "裁員", "違約", "調查", "起訴", "重挫", "跌停",
    "利空", "賣超", "取消", "延遲", "庫存", "匯損", "警示", "處置", "營收減", "年減",
]


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
        rows.append({
            "日期": item.get("date", ""),
            "標題": item.get("title", ""),
            "來源": item.get("source", ""),
            "初步判讀": classify_headline(item.get("title", "")),
            "連結": item.get("link", ""),
        })
    return pd.DataFrame(rows)


def make_news_brief(news_df: pd.DataFrame, stock_name: str, stock_id: str, limit: int = 12) -> str:
    if news_df.empty:
        return "這段期間沒有抓到新聞，無法整理。"

    subset = news_df.head(limit)
    lines = []
    for _, row in subset.iterrows():
        lines.append(f"- {row['日期']}｜{row['初步判讀']}｜{row['標題']}")

    sentiment_counts = news_df["初步判讀"].value_counts().to_dict()
    count_text = "、".join(f"{k} {v} 則" for k, v in sentiment_counts.items())

    return (
        f"{stock_name}（{stock_id}）近期待判讀新聞共 {len(news_df)} 則。"
        f"標題關鍵字初判：{count_text or '無'}。\n\n"
        "最新重點：\n" + "\n".join(lines)
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
            time.sleep(0.4)
    except Exception:
        return []

    # 去重
    seen = set()
    uniq = []
    for row in results:
        key = row["title"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq[:30]


# -----------------------------------------------------------------------------
# Gemini Summary
# -----------------------------------------------------------------------------
def ai_summarize_news(news_df: pd.DataFrame, stock_name: str, stock_id: str, risk_mode: bool = True) -> str:
    api_key = get_google_api_key()
    if not api_key:
        return "尚未設定 GOOGLE_API_KEY 或 GEMINI_API_KEY，所以未啟用 AI 摘要。"
    if news_df.empty:
        return "沒有新聞資料可摘要。"

    try:
        from google import genai
    except Exception:
        return "尚未安裝 google-genai，請確認 requirements.txt 是否已部署。"

    headlines = []
    for _, row in news_df.head(20).iterrows():
        headlines.append(f"{row['日期']}｜{row['初步判讀']}｜{row['標題']}")

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
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        return text.strip() or "AI 沒有回傳內容。"
    except Exception as exc:
        if risk_mode:
            return f"AI 摘要失敗：{exc}"
        return "AI 摘要失敗。"


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def render_secret_status() -> None:
    finmind_ok = bool(get_finmind_token())
    google_ok = bool(get_google_api_key())

    col1, col2 = st.columns(2)
    with col1:
        st.caption("FinMind Token")
        st.write("✅ 已設定" if finmind_ok else "⚠️ 未設定，會用較低免費額度")
    with col2:
        st.caption("Google AI Key")
        st.write("✅ 已設定" if google_ok else "⚠️ 未設定，AI 摘要停用")


def main() -> None:
    st.set_page_config(page_title="台股新聞模組", page_icon="📰", layout="wide")

    st.title("📰 台股個股新聞 / 社群討論模組")
    st.caption("FinMind 新聞 + PTT 討論 + Gemini AI 摘要。新聞標題只作初步整理，不等於投資建議。")

    with st.sidebar:
        st.header("設定")
        query = st.text_input("股票名稱或代號", value="台積電", placeholder="例：台積電 / 2330 / 聯發科")
        days = st.slider("新聞天數", min_value=1, max_value=30, value=7)
        include_ptt = st.checkbox("抓 PTT 股票板討論", value=True)
        include_ai = st.checkbox("產生 AI 摘要", value=True)
        st.divider()
        render_secret_status()
        st.divider()
        st.caption("部署時請到 Streamlit Cloud 的 Secrets 填入 FINMIND_TOKEN 與 GOOGLE_API_KEY。")

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

    st.subheader(f"{stock_name}（{stock_id}）")
    meta_cols = st.columns(3)
    meta_cols[0].metric("產業", match.get("industry_category") or "—")
    meta_cols[1].metric("市場", match.get("market") or "—")
    meta_cols[2].metric("查詢區間", f"近 {days} 天")

    with st.spinner("抓取新聞中..."):
        try:
            news_dicts = fetch_news_cached(stock_id=stock_id, days=days, token_marker=token_marker)
            news_df = news_to_dataframe(news_dicts)
        except Exception as exc:
            st.error(f"新聞抓取失敗：{exc}")
            news_df = pd.DataFrame()

    if news_df.empty:
        st.info("這段期間沒有抓到新聞。可以把天數拉長，或確認 FinMind token 是否可用。")
    else:
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### 新聞列表")
            st.dataframe(
                news_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "連結": st.column_config.LinkColumn("連結", display_text="開啟"),
                },
            )
            csv = news_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下載 CSV",
                data=csv,
                file_name=f"{stock_id}_{stock_name}_news.csv",
                mime="text/csv",
            )

        with right:
            st.markdown("### 初步摘要")
            st.write(make_news_brief(news_df, stock_name, stock_id))

    if include_ai:
        st.markdown("### Gemini AI 新聞摘要")
        with st.spinner("產生 AI 摘要中..."):
            st.write(ai_summarize_news(news_df, stock_name, stock_id))

    if include_ptt:
        st.markdown("### PTT 股票板討論")
        ptt_rows = fetch_ptt_cached(stock_name, max_pages=2)
        if not ptt_rows:
            st.info("沒有抓到 PTT 討論，或 PTT 版面結構暫時無法解析。")
        else:
            for row in ptt_rows[:15]:
                st.markdown(f"- [{row['title']}]({row['link']})")

    st.divider()
    st.caption("風險提醒：新聞標題與社群討論容易有雜訊，請搭配財報、籌碼、技術面與重大公告交叉驗證。")


if __name__ == "__main__":
    main()
