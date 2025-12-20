# 📈 Stock Price Forecasting System

A production-oriented stock price forecasting system using a deep learning model served via **FastAPI** and visualized with an interactive **Streamlit frontend**.

The system provides **fast deterministic forecasts by default** and supports **optional Monte Carlo–based confidence intervals** for uncertainty estimation.

---

## ✨ Features

- Deep learning–based stock price forecasting
- Fast predictions (no Monte Carlo by default)
- Optional confidence intervals using Monte Carlo sampling
- Candlestick charts for historical prices
- Side-by-side historical and forecast visualization
- FastAPI backend for scalable inference
- Streamlit frontend with clean UX and error handling
- Cached model loading for low latency


## 🔁 Forecasting Modes

### 1️⃣ Fast Mode (Default)
- Single forward pass of the trained model
- Very fast and suitable for dashboards
- No confidence interval

### 2️⃣ Confidence Interval Mode (Optional)
- Uses Monte Carlo sampling with dropout enabled
- Provides uncertainty bounds (upper & lower CI)
- Slower due to multiple forward passes
- User-controlled from the Streamlit UI

---

## 🧠 Model & Data Pipeline

- **Data Source**: Yahoo Finance (`yfinance`)
- **Indicators**: SMA, EMA, RSI, MACD, ATR, Bollinger Bands, and more
- **Scaling**: MinMaxScaler
- **Model Type**: LSTM-based deep learning models
- **Inference Window**: 30 timesteps
- **Caching**: Models cached per ticker using LRU cache

---

## 🚀 Getting Started

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows
```

### 2️⃣ Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
### 2️⃣ Install Frontend Dependencies

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## 🔌 API Reference

### GET `/forecast`

#### Query Parameters

| Parameter    | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `ticker`    | string | ✅       | Stock ticker (AAPL, MSFT, etc.) |
| `horizon`   | int    | ✅       | Number of days to forecast |
| `use_ci`    | bool   | ❌       | Enable confidence interval |
| `mc_samples`| int    | ❌       | Monte Carlo samples (only if CI enabled) |

---

### Sample Response — Fast Mode

```json
{
  "ticker": "AAPL",
  "last_close": 263.12,
  "predictions": [263.5, 263.8, 264.1]
}
```

### Sample Response — Confidence Interval Mode
```json
{
  "ticker": "AAPL",
  "last_close": 263.12,
  "predictions": [263.5, 263.8, 264.1],
  "lower": [261.9, 262.1, 262.4],
  "upper": [265.2, 265.8, 266.3]
}
```

## 🖥️ Frontend Highlights

- One-month historical candlestick chart  
- Separate, zoomed forecast chart  
- Confidence interval toggle with performance warning  
- Graceful backend connection handling  
- Retry option when backend is unavailable  
- Auto-scaled y-axis to avoid flat-line illusion  
