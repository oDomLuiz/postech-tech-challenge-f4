from __future__ import annotations

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from app.core.config import PROCESSED_DATA_DIR, ensure_project_dirs, settings
from src.preprocess import prepare_lstm_dataset
from src.utils import load_raw_data, save_json


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero = y_true != 0
    if not non_zero.any():
        return 0.0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)


def evaluate_model(ticker: str) -> dict[str, float]:
    from tensorflow.keras.models import load_model

    ensure_project_dirs()
    if not settings.model_path.exists() or not settings.scaler_path.exists():
        raise FileNotFoundError("Model or scaler not found. Run src/train.py first.")

    metadata = {}
    if settings.metadata_path.exists():
        metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
    window_size = int(metadata.get("window_size", settings.window_size))

    df = load_raw_data(ticker)
    dataset = prepare_lstm_dataset(df, window_size=window_size)
    model = load_model(settings.model_path)
    scaler = joblib.load(settings.scaler_path)

    predictions_scaled = model.predict(dataset.x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled).ravel()
    actual = scaler.inverse_transform(dataset.y_test.reshape(-1, 1)).ravel()

    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    mape = mean_absolute_percentage_error(actual, predictions)

    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title(f"{ticker.upper()} actual vs predicted close price")
    plt.xlabel("Time")
    plt.ylabel("Close price")
    plt.legend()
    output_plot = PROCESSED_DATA_DIR / f"{ticker.upper()}_actual_vs_predicted.png"
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    metrics = {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}
    save_json(PROCESSED_DATA_DIR / f"{ticker.upper()}_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model.")
    parser.add_argument("--ticker", default=settings.default_ticker)
    args = parser.parse_args()

    metrics = evaluate_model(args.ticker)
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"MAPE: {metrics['mape']}%")


if __name__ == "__main__":
    main()
