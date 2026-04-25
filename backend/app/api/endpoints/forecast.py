from fastapi import APIRouter, HTTPException
import yfinance as yf
import numpy as np
from app.core.config import MODEL_AVAILABILITY
from app.services.data_service import (
    data_collection, 
    transformer_pipeline, 
    lstm_prepare, 
    get_sentiment
)
from app.services.model_service import load_model_cached
from app.services.forecast_service import (
    forecast_loop, 
    monte_carlo_forecast, 
    simple_forecast
)

router = APIRouter()

@router.get("/forecast")
def forecast(
    ticker: str,
    horizon: int = 7,
    model_type: str = None,
    use_ci: bool = False,
    mc_samples: int = 30
):
    ticker = ticker.upper()

    # -------------------------------
    # AUTO SELECT
    # -------------------------------
    if model_type is None:
        if ticker in MODEL_AVAILABILITY["transformer"]:
            model_type = "transformer"
        elif ticker in MODEL_AVAILABILITY["lstm-gru"]:
            model_type = "lstm-gru"
        else:
            raise HTTPException(status_code=404, detail=f"No model for {ticker}")

    if ticker not in MODEL_AVAILABILITY.get(model_type, []):
        raise HTTPException(status_code=404, detail=f"{model_type} not available for {ticker}")

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
            
        sentiment = get_sentiment(ticker)
        response["sentiment"] = sentiment   

        return response
