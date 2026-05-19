from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.core.config import settings


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    scaled_values: np.ndarray


def clean_price_data(
    df: pd.DataFrame, target_column: str = settings.target_column
) -> pd.DataFrame:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    cleaned = df.copy()
    if "Date" in cleaned.columns:
        cleaned["Date"] = pd.to_datetime(cleaned["Date"])
        cleaned = cleaned.sort_values("Date")

    cleaned = cleaned.dropna(subset=[target_column])
    cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="coerce")
    cleaned = cleaned.dropna(subset=[target_column])
    return cleaned.reset_index(drop=True)


def scale_prices(values: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    return scaled, scaler


def create_sequences(
    scaled_values: np.ndarray, window_size: int = settings.window_size
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be greater than zero.")
    if len(scaled_values) <= window_size:
        raise ValueError(
            f"At least {window_size + 1} values are required to create sequences."
        )

    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    for index in range(window_size, len(scaled_values)):
        x_values.append(scaled_values[index - window_size : index])
        y_values.append(float(scaled_values[index][0]))

    return np.array(x_values, dtype=np.float32), np.array(y_values, dtype=np.float32)


def split_sequences(
    x_values: np.ndarray,
    y_values: np.ndarray,
    train_size: float = settings.train_size,
    validation_size: float = settings.validation_size,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = len(x_values)
    train_end = int(total * train_size)
    val_end = train_end + int(total * validation_size)

    if train_end == 0 or val_end <= train_end or val_end >= total:
        raise ValueError("Not enough samples for train/validation/test split.")

    return (
        x_values[:train_end],
        y_values[:train_end],
        x_values[train_end:val_end],
        y_values[train_end:val_end],
        x_values[val_end:],
        y_values[val_end:],
    )


def prepare_lstm_dataset(
    df: pd.DataFrame,
    target_column: str = settings.target_column,
    window_size: int = settings.window_size,
) -> DatasetSplit:
    cleaned = clean_price_data(df, target_column)
    prices = cleaned[target_column].to_numpy(dtype=np.float32)
    scaled_values, scaler = scale_prices(prices)
    x_values, y_values = create_sequences(scaled_values, window_size)
    x_train, y_train, x_val, y_val, x_test, y_test = split_sequences(x_values, y_values)
    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        scaler=scaler,
        scaled_values=scaled_values,
    )
