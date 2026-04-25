import os

WINDOW = 30
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

MODEL_AVAILABILITY = {
    "lstm-gru": ["AAPL", "GOOGL", "AMZN", "AMD", "MSFT", "INTC", "IBM"],
    "transformer": ["F", "IBM", "INTC", "JPM", "WMT"]
}
