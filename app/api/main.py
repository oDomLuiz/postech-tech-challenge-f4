from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import ensure_project_dirs, settings
from app.core.monitoring import monitoring_middleware


ensure_project_dirs()

app = FastAPI(
    title=settings.project_name,
    version=settings.model_version,
    description="REST API for LSTM-based stock close price prediction.",
)
app.middleware("http")(monitoring_middleware)
app.include_router(router)
