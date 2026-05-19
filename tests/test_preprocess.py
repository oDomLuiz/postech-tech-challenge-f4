from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import clean_price_data, create_sequences, prepare_lstm_dataset


def test_create_sequences_uses_previous_window_to_predict_next_value() -> None:
    values = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)

    x_values, y_values = create_sequences(values, window_size=2)

    assert x_values.shape == (2, 2, 1)
    assert y_values.shape == (2,)
    np.testing.assert_array_equal(x_values[0].ravel(), np.array([0.1, 0.2], dtype=np.float32))
    assert y_values[0] == pytest.approx(0.3)


def test_create_sequences_rejects_short_series() -> None:
    values = np.array([[0.1], [0.2]], dtype=np.float32)

    with pytest.raises(ValueError, match="At least 4 values"):
        create_sequences(values, window_size=3)


def test_clean_price_data_sorts_dates_and_removes_missing_target() -> None:
    df = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "Close": [30.0, 10.0, None],
        }
    )

    cleaned = clean_price_data(df)

    assert cleaned["Close"].tolist() == [10.0, 30.0]


def test_prepare_lstm_dataset_returns_temporal_splits() -> None:
    df = pd.DataFrame({"Close": np.arange(1, 101, dtype=np.float32)})

    dataset = prepare_lstm_dataset(df, window_size=5)

    assert dataset.x_train.shape[1:] == (5, 1)
    assert len(dataset.x_train) > len(dataset.x_val) > 0
    assert len(dataset.x_test) > 0
