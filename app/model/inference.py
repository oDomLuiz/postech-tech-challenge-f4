from __future__ import annotations

import numpy as np

from app.core.config import settings
from app.model.loader import load_model_bundle


def predict_next_close(prices: list[float]) -> tuple[float, int, str]:
    bundle = load_model_bundle()
    window_size = int(bundle.metadata.get("window_size", settings.window_size))
    model_version = str(bundle.metadata.get("model_version", settings.model_version))

    if len(prices) < window_size:
        raise ValueError(f"At least {window_size} prices are required for prediction.")

    recent_prices = np.array(prices[-window_size:], dtype=np.float32).reshape(-1, 1)
    scaled_prices = bundle.scaler.transform(recent_prices)
    model_input = scaled_prices.reshape(1, window_size, 1)
    prediction_scaled = bundle.model.predict(model_input, verbose=0)
    prediction = bundle.scaler.inverse_transform(prediction_scaled)[0][0]
    return float(prediction), window_size, model_version
