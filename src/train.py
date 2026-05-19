from __future__ import annotations

import argparse

import joblib

from app.core.config import ensure_project_dirs, settings
from src.preprocess import prepare_lstm_dataset
from src.utils import load_raw_data, metadata_payload, save_json


def build_lstm_model(window_size: int):
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(), loss="mean_squared_error")
    return model


def train_model(ticker: str, window_size: int, epochs: int, batch_size: int) -> None:
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    ensure_project_dirs()
    df = load_raw_data(ticker)
    dataset = prepare_lstm_dataset(df, window_size=window_size)
    model = build_lstm_model(window_size)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ModelCheckpoint(
            filepath=str(settings.model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(settings.model_path)
    joblib.dump(dataset.scaler, settings.scaler_path)
    metadata = metadata_payload(ticker, window_size)
    metadata["epochs"] = epochs
    metadata["batch_size"] = batch_size
    metadata["final_train_loss"] = float(history.history["loss"][-1])
    metadata["final_validation_loss"] = float(history.history["val_loss"][-1])
    save_json(settings.metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM stock price model.")
    parser.add_argument("--ticker", default=settings.default_ticker)
    parser.add_argument("--window-size", type=int, default=settings.window_size)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_model(args.ticker, args.window_size, args.epochs, args.batch_size)
    print(f"Model saved to {settings.model_path}")
    print(f"Scaler saved to {settings.scaler_path}")


if __name__ == "__main__":
    main()
