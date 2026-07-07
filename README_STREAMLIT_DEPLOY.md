# 台股新聞模組 Streamlit 部署說明

## 1. GitHub 要放的檔案

請把以下檔案放到同一個 GitHub repo：

- `streamlit_app.py`
- `requirements.txt`
- `.gitignore`
- `.streamlit/secrets.toml.example`（只能放範例，不要放真實 key）

## 2. Streamlit Community Cloud 部署

1. 到 Streamlit Community Cloud 建立 App
2. 選擇你的 GitHub repo
3. Main file path 填：`streamlit_app.py`
4. 到 App settings / Secrets 貼上：

```toml
FINMIND_TOKEN = "你的 FinMind token"
GOOGLE_API_KEY = "你的 Google AI / Gemini API key"
```

## 3. 本機測試

```bash
pip install -r requirements.txt
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 編輯 .streamlit/secrets.toml，貼上真實 key
streamlit run streamlit_app.py
```

## 4. 功能

- 中文名稱或股票代號解析，例如：台積電 / 2330
- FinMind `TaiwanStockNews` 新聞抓取
- 新聞標題去重、日期排序、CSV 下載
- 標題關鍵字初步判讀：偏利多 / 偏利空 / 中性
- PTT 股票板搜尋
- Google Gemini AI 摘要

## 5. 重要風險

不要把真實 API key commit 到 GitHub。即使 repo 是 private，也不建議把 key 寫死；未來轉 public、協作者外流、部署 log 或 commit history 都可能造成憑證外洩。
