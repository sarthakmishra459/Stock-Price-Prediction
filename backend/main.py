from fastapi import FastAPI
from functools import lru_cache
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os

from stockstats import StockDataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

app = FastAPI(title="Stock Forecast API")

WINDOW = 30
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -------------------------------
# Cache initialization
# -------------------------------
@app.on_event("startup")
def startup():
    FastAPICache.init(InMemoryBackend(), prefix="forecast-cache")

# -------------------------------
# Data Collection (helper function)
# -------------------------------
def data_collection(ticker: str) -> pd.DataFrame:
    start_date = datetime.datetime(2013, 8, 24)
    end_date = datetime.datetime.today()

    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)

    data = data.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    stock = StockDataFrame.retype(data)

    # Add indicators
    data['sma_10']   = stock['close_10_sma']
    data['ema_10']   = stock['close_10_ema']
    data['vwma']     = stock['vwma']
    data['lrma']     = stock['close_10_lrma']
    data['kama']     = stock['close_10_kama']
    data['ichimoku'] = stock['ichimoku']
    data['eri_bull'] = stock['eribull']
    data['eri_bear'] = stock['eribear']
    data['cti']      = stock['cti']
    data['rsv']      = stock['rsv_9']
    data['rsi_14']   = stock['rsi_14']
    data['kdjk']     = stock['kdjk']
    data['macd']     = stock['macd']
    data['wr_14']    = stock['wr_14']
    data['cci_14']   = stock['cci_14']
    data['trix']     = stock['trix']
    data['ppo']      = stock['ppo']
    data['stochrsi'] = stock['stochrsi']
    data['wt1']      = stock['wt1']
    data['aroon_14'] = stock['aroon_14']
    data['ao']       = stock['ao']
    data['roc_10']   = stock['close_10_roc']
    data['coppock']  = stock['coppock']
    data['kst']      = stock['kst']
    data['rvgi']     = stock['rvgi']
    data['pgo']      = stock['pgo']
    data['psl']      = stock['psl']
    data['pvo']      = stock['pvo']
    data['mstd_20']  = stock['close_20_mstd']
    data['mvar_20']  = stock['close_20_mvar']
    data['boll']     = stock['boll']
    data['tr']       = stock['tr']
    data['atr_14']   = stock['atr_14']
    data['chop']     = stock['chop']
    data['ker']      = stock['ker']
    data['vr']       = stock['vr']
    data['mfi_14']   = stock['mfi_14']
    data['bop']      = stock['bop']
    data['dma']      = stock['dma']
    data['dmi']      = stock['adx']
    data['adxr']     = stock['adxr']
    data['mad_20']   = stock['close_20_mad']

    data.dropna(inplace=True)
    return data

# -------------------------------
# Cached model loading
# -------------------------------
@lru_cache(maxsize=10)
def load_model_cached(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_best_model.h5")
    return load_model(model_path, compile=False)

# -------------------------------
# Forecast helpers
# -------------------------------
def monte_carlo_forecast(model, scaled_data, close_idx, close_scaler, days, samples):
    window = scaled_data[-WINDOW:].copy()
    paths = []

    for _ in range(samples):
        w = window.copy()
        preds = []

        for _ in range(days):
            x = np.expand_dims(w, axis=0)
            pred = model(x, training=True).numpy()[0, 0]
            preds.append(pred)

            next_row = w[-1].copy()
            next_row[close_idx] = pred
            w = np.vstack([w[1:], next_row])

        paths.append(preds)

    paths = np.array(paths)
    paths = close_scaler.inverse_transform(paths.reshape(-1, 1))
    return paths.reshape(samples, days)

def simple_forecast(model, scaled_data, close_idx, close_scaler, days):
    window = scaled_data[-WINDOW:].copy()
    preds = []

    for _ in range(days):
        x = np.expand_dims(window, axis=0)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)

        next_row = window[-1].copy()
        next_row[close_idx] = pred
        window = np.vstack([window[1:], next_row])

    preds = close_scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()

    return preds.tolist()

# -------------------------------
# API Endpoint (with cache)
# -------------------------------
@app.get("/forecast")
@cache(expire=300)
def forecast(
    ticker: str,
    horizon: int = 7,
    use_ci: bool = False,
    mc_samples: int = 30
):
    data = data_collection(ticker)

    values = data.values
    close_idx = data.columns.get_loc("close")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    close_scaler = MinMaxScaler()
    close_scaler.fit(data[['close']])

    model = load_model_cached(ticker)

    preds = simple_forecast(model, scaled, close_idx, close_scaler, horizon)

    response = {
        "ticker": str(ticker),
        "last_close": float(data['close'].iloc[-1]),
        "predictions": [float(p) for p in preds]
    }

    if use_ci:
        mc = monte_carlo_forecast(model, scaled, close_idx, close_scaler, horizon, mc_samples)
        response["lower"] = [float(x) for x in np.percentile(mc, 5, axis=0).tolist()]
        response["upper"] = [float(x) for x in np.percentile(mc, 95, axis=0).tolist()]

    return response