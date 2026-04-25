import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta
from requests.exceptions import ConnectionError, Timeout

API_URL = "http://127.0.0.1:8000/forecast"

st.set_page_config(
    page_title="Stock Forecasting",
    layout="wide"
)

st.title("📈 Multi-Day Stock Price Forecasting")

# -------------------------------
# STOCK LIST
# -------------------------------
stocks = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Intel (INTC)": "INTC",
    "IBM": "IBM",
    "AMD": "AMD",
    "Microsoft (MSFT)": "MSFT",
    "Ford (F)": "F",
    "Walmart (WMT)": "WMT",
    "JPMorgan (JPM)": "JPM",
    "BA": "BA"
}

company = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[company]

# -------------------------------
# MODEL AVAILABILITY (🔥 NEW)
# -------------------------------
MODEL_AVAILABILITY = {
    "lstm-gru": ["AAPL", "GOOGL", "AMZN", "AMD", "MSFT", "INTC", "IBM"],
    "transformer": ["F", "IBM", "INTC", "JPM", "WMT"]
}

# -------------------------------
# MODEL SELECTION
# -------------------------------
model_option = st.sidebar.selectbox(
    "Model Selection",
    ["Auto (Recommended)", "LSTM-GRU", "Transformer"]
)

model_map = {
    "Auto (Recommended)": None,
    "LSTM-GRU": "lstm-gru",
    "Transformer": "transformer"
}

selected_model = model_map[model_option]

# -------------------------------
# AUTO MODEL LOGIC (🔥 NEW)
# -------------------------------
if selected_model is None:
    if ticker in MODEL_AVAILABILITY["transformer"]:
        selected_model = "transformer"
    elif ticker in MODEL_AVAILABILITY["lstm-gru"]:
        selected_model = "lstm-gru"

# -------------------------------
# VALIDATION (🔥 NEW)
# -------------------------------
if selected_model and ticker not in MODEL_AVAILABILITY.get(selected_model, []):
    st.warning(f"{selected_model} model not available for {ticker}")
    st.stop()

# -------------------------------
# FORECAST SETTINGS
# -------------------------------
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 30, 7)

use_ci = st.sidebar.checkbox(
    "Show Confidence Interval (slower)",
    value=False
)

mc_samples = None
if use_ci:
    mc_samples = st.sidebar.slider(
        "Monte Carlo Samples",
        10, 100, 30, step=10
    )

# -------------------------------
# LOAD HISTORICAL DATA
# -------------------------------
@st.cache_data(show_spinner=False)
def load_historical_data(ticker):
    df = yf.Ticker(ticker).history(period="1mo")
    df.reset_index(inplace=True)
    return df

hist = load_historical_data(ticker)

# -------------------------------
# FETCH FORECAST
# -------------------------------
def fetch_forecast(params):
    try:
        r = requests.get(API_URL, params=params)
        if r.status_code != 200:
            return None, f"Backend error (HTTP {r.status_code})"
        return r.json(), None

    except ConnectionError:
        return None, "❌ Backend not running"

    except Timeout:
        return None, "⏱️ Backend timed out"

    except Exception as e:
        return None, str(e)

# -------------------------------
# CALL API
# -------------------------------
with st.spinner("🔮 Generating forecast..."):
    params = {
        "ticker": ticker,
        "horizon": horizon,
        "use_ci": use_ci,
        "model_type": selected_model
    }

    if use_ci:
        params["mc_samples"] = mc_samples

    result, error = fetch_forecast(params)

if error:
    st.warning(error)
    st.stop()

# -------------------------------
# RESPONSE HANDLING (SAFE)
# -------------------------------
predictions = result.get("predictions", [])
sentiment = result.get("sentiment", {})
if not predictions:
    st.error("No predictions available for this ticker.")
    st.stop()

last_close = result.get("last_close")
if last_close is None:
    last_close = float(hist["Close"].iloc[-1])

lower = result.get("lower")
upper = result.get("upper")

# -------------------------------
# AVAILABLE MODELS DISPLAY (FIXED)
# -------------------------------
available_models = [
    m for m, tickers in MODEL_AVAILABILITY.items()
    if ticker in tickers
]

st.info(
    f"🤖 Model Used: {selected_model.upper()} | Available: {', '.join(available_models)}"
)
# -------------------------------
# SENTIMENT DISPLAY (🔥 NEW)
# -------------------------------
if sentiment:
    st.subheader("🧠 Market Sentiment")

    score = sentiment.get("score", 0)
    label = sentiment.get("label", "Neutral")
    warning = sentiment.get("warning", False)

    c1, c2 = st.columns(2)

    c1.metric("Sentiment", label)
    c2.metric("Score", f"{score:.2f}")

    if warning:
        st.warning("⚠️ Market sentiment is volatile. Use predictions cautiously.")
# -------------------------------
# FORECAST DATES
# -------------------------------
forecast_dates = pd.date_range(
    start=hist["Date"].iloc[-1] + timedelta(days=1),
    periods=horizon,
    freq="B"
)

# -------------------------------
# CHARTS
# -------------------------------
left, right = st.columns(2)

with left:
    st.subheader("🕯️ Historical Price")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist["Date"],
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"]
    ))

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🔮 Forecast")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode="lines+markers"
    ))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TABLE
# -------------------------------
st.subheader("📅 Forecast Table")

table = pd.DataFrame({
    "Date": forecast_dates,
    "Predicted Price": predictions
})

st.dataframe(table, use_container_width=True)

# -------------------------------
# METRICS
# -------------------------------
st.subheader("📌 Summary")

c1, c2, c3 = st.columns(3)

c1.metric("Last Close", f"${last_close:.2f}")
c2.metric("Next Day", f"${predictions[0]:.2f}")
c3.metric("End Value", f"${predictions[-1]:.2f}")