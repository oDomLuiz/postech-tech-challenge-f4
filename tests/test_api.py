from __future__ import annotations

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
