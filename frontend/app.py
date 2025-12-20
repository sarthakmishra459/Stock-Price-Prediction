import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import timedelta
from requests.exceptions import ConnectionError, Timeout

# ---------------------------------
# Config
# ---------------------------------
API_URL = "http://127.0.0.1:8000/forecast"

st.set_page_config(
    page_title="Stock Forecasting",
    layout="wide"
)

st.title("📈 Multi-Day Stock Price Forecasting")

# ---------------------------------
# Sidebar
# ---------------------------------
stocks = {
    "Apple (AAPL)": "AAPL",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Intel (INTC)": "INTC",
    "IBM": "IBM",
    "AMD": "AMD",
    "Microsoft (MSFT)": "MSFT"
}

company = st.sidebar.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[company]

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

# ---------------------------------
# Load Historical Data (1 month)
# ---------------------------------
@st.cache_data(show_spinner=False)
def load_historical_data(ticker):
    df = yf.Ticker(ticker).history(period="1mo")
    df.reset_index(inplace=True)
    return df

hist = load_historical_data(ticker)

# ---------------------------------
# Backend Call (Safe)
# ---------------------------------
def fetch_forecast(params):
    try:
        r = requests.get(API_URL, params=params)
        if r.status_code != 200:
            return None, f"Backend error (HTTP {r.status_code})"
        return r.json(), None

    except ConnectionError:
        return None, "❌ Backend not running"

    except Timeout:
        return None, "⏱️ Backend timed out (disable CI)"

    except Exception as e:
        return None, str(e)


with st.spinner("🔮 Generating forecast..."):
    params = {
        "ticker": ticker,
        "horizon": horizon,
        "use_ci": use_ci
    }
    if use_ci:
        params["mc_samples"] = mc_samples

    result, error = fetch_forecast(params)

if error:
    st.warning(error)
    st.info("Start FastAPI backend and retry.")
    if st.button("🔄 Retry"):
        st.rerun()
    st.stop()

# ---------------------------------
# Extract Forecast
# ---------------------------------
predictions = result["predictions"]
last_close = result["last_close"]
lower = result.get("lower")
upper = result.get("upper")

forecast_dates = pd.date_range(
    start=hist["Date"].iloc[-1] + timedelta(days=1),
    periods=horizon,
    freq="B"
)

# ---------------------------------
# Layout: Two Columns
# ---------------------------------
left, right = st.columns(2)

# =================================
# LEFT: Historical Chart
# =================================
with left:
    st.subheader("🕯️ Historical Price (Last Month)")

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Candlestick(
        x=hist["Date"],
        open=hist["Open"],
        high=hist["High"],
        low=hist["Low"],
        close=hist["Close"],
        name="Historical"
    ))

    hist_fig.update_layout(
        height=450,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(hist_fig, use_container_width=True)

# =================================
# RIGHT: Prediction Chart (Zoomed)
# =================================
with right:
    st.subheader("🔮 Forecast")

    pred_fig = go.Figure()

    pred_fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="red", width=3)
    ))
    
    # ---------------------------------
    # Dynamic Y-axis scaling
    # ---------------------------------
    if use_ci and lower is not None and upper is not None:
        y_min = min(lower)
        y_max = max(upper)

        padding = (y_max - y_min) * 0.15  # 15% padding for CI
        pred_fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(150,150,150,0.25)",  # semi-transparent
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Confidence Interval"
        ))
    else:
        y_min = min(predictions)
        y_max = max(predictions)

        padding = (y_max - y_min) * 0.05  # tighter zoom

    pred_fig.update_yaxes(
        range=[y_min - padding, y_max + padding]
    )


    pred_fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(pred_fig, use_container_width=True)

# ---------------------------------
# Forecast Table
# ---------------------------------
st.subheader("📅 Forecast Table")

table = {
    "Date": forecast_dates,
    "Predicted Price": predictions
}

if use_ci and lower is not None and upper is not None:
    table["Lower CI"] = lower
    table["Upper CI"] = upper

st.dataframe(pd.DataFrame(table), use_container_width=True)

# ---------------------------------
# Summary Metrics
# ---------------------------------
st.subheader("📌 Summary")

c1, c2, c3 = st.columns(3)

c1.metric("Last Close", f"${last_close:.2f}")
c2.metric(
    "Next Day Forecast",
    f"${predictions[0]:.2f}",
    delta=f"{predictions[0] - last_close:.2f}"
)
c3.metric(
    "End of Horizon",
    f"${predictions[-1]:.2f}",
    delta=f"{predictions[-1] - predictions[0]:.2f}"
)
