from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    prices: list[float] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predicted_close_price: float
    window_size: int
    model_version: str


class PredictByTickerRequest(BaseModel):
    ticker: str = Field(default="AAPL", min_length=1)
    period: str = Field(default="6mo", min_length=1)


class PredictByTickerResponse(BaseModel):
    ticker: str
    predicted_close_price: float
    window_size: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_ready: bool


class MetricsResponse(BaseModel):
    total_requests: int
    total_errors: int
    average_response_time_ms: float
