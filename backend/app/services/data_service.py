import datetime
import pandas as pd
import yfinance as yf

from stockstats import StockDataFrame
from sklearn.preprocessing import MinMaxScaler
from transformers import pipeline


# -------------------------------
# Lazy-loaded FinBERT
# -------------------------------
sentiment_model = None


def get_sentiment_model():
    global sentiment_model

    if sentiment_model is None:
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )

    return sentiment_model

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


def lstm_prepare(data):
    values = data.values
    close_idx = data.columns.get_loc("close")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    close_scaler = MinMaxScaler()
    close_scaler.fit(data[['close']])

    return scaled, close_idx, close_scaler


def transformer_pipeline(data):
    df = data[['open','high','low','close','volume']].copy()
    df = df.rolling(10).mean().dropna()

    for col in df.columns:
        df[col] = df[col].pct_change()

    df.dropna(inplace=True)
    return df.values


def get_sentiment(ticker):
    news = yf.Ticker(ticker).news

    if not news:
        return {
            "score": 0,
            "label": "Neutral",
            "warning": False,
            "headlines": [],
            "details": []
        }

    # 🔥 More stable (use 10 instead of 5)
    headlines = [
        n["content"]["title"]
        for n in news[:10]
        if "content" in n and "title" in n["content"]
    ]

    if not headlines:
        return {
            "score": 0,
            "label": "Neutral",
            "warning": False,
            "headlines": [],
            "details": []
        }

    results = get_sentiment_model()(headlines)

    # 🔥 Correct FinBERT scoring (confidence weighted)
    score = sum([
        (r["score"] if r["label"].lower() == "positive" else
         -r["score"] if r["label"].lower() == "negative" else
         0)
        for r in results
    ]) / len(results)

    # 🔥 Better thresholds
    if score > 0.1:
        label = "Positive"
    elif score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    warning = abs(score) > 0.6

    return {
        "score": round(score, 3),
        "label": label,
        "warning": warning,
        "headlines": headlines,
        "details": results
    }