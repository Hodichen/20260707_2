# Streamlit 升級版：新聞模組 + AI 個股分析

## 這次新增什麼
1. 新增 **AI 個股分析模組**
2. 介面改成 **Apple 風格平整化儀表板**
3. 以 **4 大區塊** 顯示重點：
   - A. 基本面
   - B. 籌碼面
   - C. 消息面
   - D. 技術面
4. 新增 **AI 四面向量化評分**
5. 技術面加入 **K 線 / MA / 布林通道 / RSI / MACD**
6. 籌碼面支援 **主動 ETF +/- 碼**（透過 CSV 上傳或 URL）

## 你要注意的 3 個盲點
1. **主動 ETF +/- 碼資料源** 如果沒有穩定來源，無法自動且長期穩定運作。
2. **財報欄位** 用 FinMind 原始報表抓值，不同公司科目命名可能不同，少數欄位可能抓不到。
3. **AI 分數** 是快篩工具，不是交易訊號。

## GitHub / Streamlit Cloud 要放哪些檔案
- `streamlit_app.py`（請用這次的 `streamlit_app_v2.py` 內容覆蓋）
- `requirements.txt`（請用 `requirements_v2.txt` 內容覆蓋）
- `ACTIVE_ETF_SIGNAL_SAMPLE.csv`（可選，做格式參考）

## Secrets 建議
```toml
FINMIND_TOKEN = "你的 FinMind token"
GOOGLE_API_KEY = "你的 Google AI key"
# 如果你有固定的主動ETF加減碼CSV網址，可加這個
ACTIVE_ETF_SIGNAL_CSV_URL = "https://.../active_etf_signal.csv"
```

## 主動 ETF +/- 碼 CSV 格式
最少需要這 5 個欄位：
- `date`
- `stock_id`
- `etf_id`
- `action`  （可用 `+` / `-` / `buy` / `sell` / `加碼` / `減碼`）
- `shares_delta`

## 直接上線測試步驟
1. 把 `streamlit_app_v2.py` 改名成 `streamlit_app.py`
2. 把 `requirements_v2.txt` 改名成 `requirements.txt`
3. Push 到 GitHub
4. Streamlit Cloud 選 repo，Main file path 填 `streamlit_app.py`
5. Secrets 填入 `FINMIND_TOKEN`、`GOOGLE_API_KEY`
6. 如果主動 ETF 也要上線，就再加 `ACTIVE_ETF_SIGNAL_CSV_URL`
7. Deploy

## 如果你現在就要最短路徑
- 用新的 `streamlit_app_v2.py`
- 用新的 `requirements_v2.txt`
- 先不接主動 ETF 真實資料源，先用內建上傳 CSV 測流程
- 新聞 / 基本面 / 籌碼 / 技術面會先完整跑起來
