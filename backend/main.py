# ================================
# IMPORTS
# ================================
from fastapi import FastAPI
from functools import lru_cache
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import os
import tensorflow as tf
print("TF VERSION:", tf.__version__)
from tensorflow.keras.layers import (
    Layer, Dense, Dropout, LayerNormalization, Conv1D
)
from tensorflow import keras

from stockstats import StockDataFrame
from sklearn.preprocessing import MinMaxScaler

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache


# ================================
# FASTAPI LIFESPAN (FIXED CACHE)
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend(), prefix="forecast-cache")
    yield


app = FastAPI(
    title="Stock Forecast API",
    lifespan=lifespan
)

# ================================
# CUSTOM LAYERS (ALL REQUIRED)
# ================================

class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

    def build(self, input_shape):
        self.wl = self.add_weight(shape=(self.seq_len,), initializer='uniform')
        self.bl = self.add_weight(shape=(self.seq_len,), initializer='uniform')
        self.wp = self.add_weight(shape=(self.seq_len,), initializer='uniform')
        self.bp = self.add_weight(shape=(self.seq_len,), initializer='uniform')

    def call(self, x):
        x = tf.reduce_mean(x[:, :, :4], axis=-1)
        linear = tf.expand_dims(self.wl * x + self.bl, -1)
        periodic = tf.expand_dims(tf.sin(x * self.wp + self.bp), -1)
        return tf.concat([linear, periodic], axis=-1)

    def get_config(self):
        return {"seq_len": self.seq_len}


class SingleAttention(Layer):
    def __init__(self, d_k, d_v, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k)
        self.key = Dense(self.d_k)
        self.value = Dense(self.d_v)

    def call(self, inputs):
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn / tf.sqrt(tf.cast(self.d_k, tf.float32))
        attn = tf.nn.softmax(attn)

        v = self.value(inputs[2])
        return tf.matmul(attn, v)


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.heads = [SingleAttention(self.d_k, self.d_v) for _ in range(self.n_heads)]
        self.linear = Dense(input_shape[0][-1])

    def call(self, inputs):
        attn = [h(inputs) for h in self.heads]
        concat = tf.concat(attn, axis=-1)
        return self.linear(concat)


class TransformerEncoder(Layer):
    def __init__(
        self,
        d_k,
        d_v,
        n_heads,
        ff_dim,
        dropout_rate=0.1,
        attn_heads=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # IMPORTANT: accept but ignore attn_heads (rebuild later)
        self.attn_heads = attn_heads

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)

        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_norm = LayerNormalization(epsilon=1e-6)

        self.ff1 = Conv1D(self.ff_dim, 1, activation="relu")
        self.ff2 = Conv1D(input_shape[0][-1], 1)

        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = inputs[0]

        attn = self.attn_multi(inputs)
        x = self.attn_norm(x + self.attn_dropout(attn))

        ff = self.ff1(x)
        ff = self.ff2(ff)
        return self.ff_norm(x + self.ff_dropout(ff))

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_k": self.d_k,
            "d_v": self.d_v,
            "n_heads": self.n_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
            "attn_heads": []  # keep for compatibility
        })
        return config


# ================================
# PATHS
# ================================
WINDOW = 30
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)


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
# ================================
# MODEL LOADER (FIXED FOR SAVEDMODEL)
# ================================
@lru_cache(maxsize=20)
def load_model_cached(ticker, model_type):
    base = os.path.join(MODEL_DIR, model_type)

    # -------------------------
    # TRANSFORMER (SavedModel)
    # -------------------------
    if model_type == "transformer":
        path = os.path.join(base, f"{ticker}_savedmodel")

        if os.path.exists(path):
            print("Loading Transformer:", path)
            return keras.Sequential([
                keras.layers.TFSMLayer(path, call_endpoint="serving_default")
            ])

    # -------------------------
    # LSTM-GRU (.h5)
    # -------------------------
    elif model_type == "lstm-gru":
        path = os.path.join(base, f"{ticker}.h5")

        if os.path.exists(path):
            print("Loading LSTM:", path)
            return keras.models.load_model(path, compile=False)

    raise FileNotFoundError(f"No model for {ticker} ({model_type})")


def lstm_prepare(data):
    values = data.values
    close_idx = data.columns.get_loc("close")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    close_scaler = MinMaxScaler()
    close_scaler.fit(data[['close']])

    return scaled, close_idx, close_scaler


# ================================
# TRANSFORMER PIPELINE (FIXED)
# ================================
def transformer_pipeline(data):
    df = data[['open','high','low','close','volume']].copy()

    df = df.rolling(10).mean().dropna()

    for col in df.columns:
        df[col] = df[col].pct_change()

    df.dropna(inplace=True)

    return df.values


# ================================
# FORECAST LOOP
# ================================
def forecast_loop(model, data, close_idx, days, window):
    window_data = data[-window:].copy()
    preds = []

    for _ in range(days):
        x = np.expand_dims(window_data, axis=0)

        output = model(x)

        # 🔥 FIX for TFSMLayer
        if isinstance(output, dict):
            output = list(output.values())[0]

        pred = output.numpy()[0][0]

        preds.append(pred)

        new_row = window_data[-1].copy()
        new_row[close_idx] = pred
        window_data = np.vstack([window_data[1:], new_row])

    return preds


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

# ================================
# API
# ================================
@app.get("/forecast")
def forecast(
    ticker: str,
    horizon: int = 7,
    model_type: str = None,
    use_ci: bool = False,
    mc_samples: int = 30
):
    ticker = ticker.upper()

    # -------------------------------
    # MODEL AVAILABILITY
    # -------------------------------
    MODEL_AVAILABILITY = {
        "lstm-gru": ["AAPL", "GOOGL", "AMZN", "AMD", "MSFT", "INTC", "IBM"],
        "transformer": ["F", "IBM", "INTC", "JPM", "WMT"]
    }

    # -------------------------------
    # AUTO SELECT
    # -------------------------------
    if model_type is None:
        if ticker in MODEL_AVAILABILITY["transformer"]:
            model_type = "transformer"
        elif ticker in MODEL_AVAILABILITY["lstm-gru"]:
            model_type = "lstm-gru"
        else:
            raise ValueError(f"No model for {ticker}")

    if ticker not in MODEL_AVAILABILITY.get(model_type, []):
        raise ValueError(f"{model_type} not available for {ticker}")

    # -------------------------------
    # TRANSFORMER FLOW
    # -------------------------------
    if model_type == "transformer":

        data = yf.Ticker(ticker).history(period="10y")
        data = data.rename(columns={
            'Open':'open','High':'high','Low':'low',
            'Close':'close','Volume':'volume'
        })

        values = transformer_pipeline(data)

        model = load_model_cached(ticker, "transformer")

        preds = forecast_loop(model, values, 3, horizon, 128)

        last_close = float(data["close"].iloc[-1])

        return {
            "ticker": ticker,
            "model_used": "transformer",
            "available_models": ["transformer", "lstm-gru"] if ticker in ["IBM","INTC"] else ["transformer"],
            "predictions": [float(p) for p in preds],
            "last_close": last_close
        }

    # -------------------------------
    # LSTM FLOW (ORIGINAL LOGIC)
    # -------------------------------
    else:

        data = data_collection(ticker)

        scaled, close_idx, close_scaler = lstm_prepare(data)

        model = load_model_cached(ticker, "lstm-gru")

        preds = simple_forecast(model, scaled, close_idx, close_scaler, horizon)

        response = {
            "ticker": ticker,
            "model_used": "lstm-gru",
            "available_models": ["lstm-gru", "transformer"] if ticker in ["IBM","INTC"] else ["lstm-gru"],
            "predictions": [float(p) for p in preds],
            "last_close": float(data['close'].iloc[-1])
        }

        if use_ci:
            mc = monte_carlo_forecast(model, scaled, close_idx, close_scaler, horizon, mc_samples)
            response["lower"] = np.percentile(mc, 5, axis=0).tolist()
            response["upper"] = np.percentile(mc, 95, axis=0).tolist()

        return response