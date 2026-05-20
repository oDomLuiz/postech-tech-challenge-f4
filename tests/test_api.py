from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["model"] == "LSTM"


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_ready" in body


def test_predict_returns_503_when_model_assets_are_missing(monkeypatch) -> None:
    from app.model import inference
    from app.model.loader import ModelNotReadyError

    def fake_predict(_prices: list[float]):
        raise ModelNotReadyError("model is not ready")

    monkeypatch.setattr(inference, "predict_next_close", fake_predict)
    monkeypatch.setattr("app.api.routes.predict_next_close", fake_predict)

    response = client.post("/predict", json={"prices": [1.0] * 60})

    assert response.status_code == 503
    assert response.json()["detail"] == "model is not ready"


def test_predict_rejects_short_price_list(monkeypatch) -> None:
    from app.model import inference

    def fake_predict(_prices: list[float]):
        raise ValueError("At least 60 prices are required for prediction.")

    monkeypatch.setattr(inference, "predict_next_close", fake_predict)
    monkeypatch.setattr("app.api.routes.predict_next_close", fake_predict)

    response = client.post("/predict", json={"prices": [1.0, 2.0]})

    assert response.status_code == 422
    assert "At least 60 prices" in response.json()["detail"]


def test_predict_success_with_mocked_inference(monkeypatch) -> None:
    from app.model import inference

    def fake_predict(_prices: list[float]):
        return 194.251, 60, "1.0.0"

    monkeypatch.setattr(inference, "predict_next_close", fake_predict)
    monkeypatch.setattr("app.api.routes.predict_next_close", fake_predict)

    response = client.post("/predict", json={"prices": [1.0] * 60})

    assert response.status_code == 200
    assert response.json() == {
        "predicted_close_price": 194.251,
        "window_size": 60,
        "model_version": "1.0.0",
    }


def test_predict_by_ticker_success_with_mocked_download_and_inference(monkeypatch) -> None:
    from app.model import inference

    def fake_download(_ticker: str, start_date: str, end_date: str):
        assert start_date < end_date
        return pd.DataFrame({"Close": [100.0] * 60})

    def fake_predict(_prices: list[float]):
        return 194.251, 60, "1.0.0"

    monkeypatch.setattr("app.api.routes.download_stock_data", fake_download)
    monkeypatch.setattr(inference, "predict_next_close", fake_predict)
    monkeypatch.setattr("app.api.routes.predict_next_close", fake_predict)

    response = client.post(
        "/predict-by-ticker", json={"ticker": " aapl ", "period": "6mo"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "ticker": "AAPL",
        "predicted_close_price": 194.251,
        "window_size": 60,
        "model_version": "1.0.0",
    }


def test_predict_by_ticker_returns_503_when_download_fails(monkeypatch) -> None:
    def fake_download(_ticker: str, start_date: str, end_date: str):
        raise ValueError("invalid json")

    monkeypatch.setattr("app.api.routes.download_stock_data", fake_download)

    response = client.post("/predict-by-ticker", json={"ticker": "AAPL", "period": "6mo"})

    assert response.status_code == 503
    assert "Yahoo chart API, yfinance and Stooq all failed" in response.json()["detail"]


def test_predict_by_ticker_rejects_empty_download(monkeypatch) -> None:
    def fake_download(_ticker: str, start_date: str, end_date: str):
        return pd.DataFrame()

    monkeypatch.setattr("app.api.routes.download_stock_data", fake_download)

    response = client.post("/predict-by-ticker", json={"ticker": "AAPL", "period": "6mo"})

    assert response.status_code == 422
    assert "No usable price data returned for AAPL" in response.json()["detail"]


def test_predict_by_ticker_rejects_invalid_period() -> None:
    response = client.post(
        "/predict-by-ticker", json={"ticker": "AAPL", "period": "recent"}
    )

    assert response.status_code == 422
    assert "Invalid period" in response.json()["detail"]
