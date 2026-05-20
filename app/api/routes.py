from __future__ import annotations

import re
from datetime import date, timedelta

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
from src.collect_data import download_stock_data


router = APIRouter()


def _period_to_start_date(period: str, end_date: date | None = None) -> str:
    final_end_date = end_date or date.today()
    normalized_period = period.strip().lower()

    if normalized_period == "ytd":
        return date(final_end_date.year, 1, 1).isoformat()
    if normalized_period == "max":
        return "1900-01-01"

    match = re.fullmatch(r"(\d+)(d|wk|mo|y)", normalized_period)
    if not match:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Invalid period. Use values like 5d, 1mo, 6mo, 1y, ytd, or max."
            ),
        )

    quantity = int(match.group(1))
    unit = match.group(2)
    days_by_unit = {
        "d": 1,
        "wk": 7,
        "mo": 31,
        "y": 366,
    }
    start_date = final_end_date - timedelta(days=quantity * days_by_unit[unit])
    return start_date.isoformat()


def _download_ticker_prices(ticker: str, period: str) -> list[float]:
    end_date = date.today().isoformat()
    start_date = _period_to_start_date(period)

    try:
        data = download_stock_data(ticker, start_date=start_date, end_date=end_date)
    except Exception as exc:
        logger.warning(
            "ticker_download_failed ticker=%s period=%s error=%s",
            ticker,
            period,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Could not fetch market data for {ticker}. "
                "Yahoo chart API, yfinance and Stooq all failed. "
                "Check your internet connection, proxy/VPN/firewall, ticker symbol, "
                "or try again later."
            ),
        ) from exc

    if data.empty or settings.target_column not in data.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No usable price data returned for {ticker}. "
                "Check the ticker symbol and requested period."
            ),
        )

    prices = data[settings.target_column].dropna().astype(float).tolist()
    if not prices:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No closing prices returned for {ticker}. "
                "Check the ticker symbol and requested period."
            ),
        )
    return prices


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
    ticker = payload.ticker.strip().upper()
    period = payload.period.strip()
    if not ticker or not period:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Ticker and period are required.",
        )

    try:
        prices = _download_ticker_prices(ticker, period)
        predicted_price, window_size, model_version = predict_next_close(prices)
        return PredictByTickerResponse(
            ticker=ticker,
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
