from __future__ import annotations

import yfinance as yf
from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    HealthResponse,
    MetricsResponse,
    PredictByTickerRequest,
    PredictByTickerResponse,
    PredictRequest,
    PredictResponse,
)
from app.core.config import settings
from app.core.monitoring import logger, metrics_store
from app.model.inference import predict_next_close
from app.model.loader import ModelNotReadyError, model_assets_available


router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {
        "project": settings.project_name,
        "status": "running",
        "model": "LSTM",
    }


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_ready=model_assets_available())


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    snapshot = metrics_store.snapshot()
    return MetricsResponse(
        total_requests=snapshot.total_requests,
        total_errors=snapshot.total_errors,
        average_response_time_ms=snapshot.average_response_time_ms,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if len(payload.prices) < settings.window_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"At least {settings.window_size} prices are required for prediction.",
        )

    try:
        predicted_price, window_size, model_version = predict_next_close(payload.prices)
        logger.info(
            "prediction window_size=%s model_version=%s predicted_close_price=%.4f",
            window_size,
            model_version,
            predicted_price,
        )
        return PredictResponse(
            predicted_close_price=round(predicted_price, 4),
            window_size=window_size,
            model_version=model_version,
        )
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/predict-by-ticker", response_model=PredictByTickerResponse)
def predict_by_ticker(payload: PredictByTickerRequest) -> PredictByTickerResponse:
    try:
        data = yf.download(payload.ticker.upper(), period=payload.period, progress=False)
        if data.empty or settings.target_column not in data.columns:
            raise ValueError(f"No usable price data returned for {payload.ticker}.")

        prices = data[settings.target_column].dropna().astype(float).tolist()
        predicted_price, window_size, model_version = predict_next_close(prices)
        return PredictByTickerResponse(
            ticker=payload.ticker.upper(),
            predicted_close_price=round(predicted_price, 4),
            window_size=window_size,
            model_version=model_version,
        )
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
