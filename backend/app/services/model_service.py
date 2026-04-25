import os
from functools import lru_cache
from tensorflow import keras
from app.core.config import MODEL_DIR

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
            # Note: Custom layers might be needed if loading .h5 with custom layers
            # but usually for .h5 with custom layers one provides custom_objects.
            # However, the original code didn't use custom_objects here.
            return keras.models.load_model(path, compile=False)

    raise FileNotFoundError(f"No model for {ticker} ({model_type})")
