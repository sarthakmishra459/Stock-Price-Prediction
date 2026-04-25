import numpy as np
from app.core.config import WINDOW

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
