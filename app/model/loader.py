from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import joblib

from app.core.config import settings


class ModelNotReadyError(RuntimeError):
    """Raised when inference assets do not exist yet."""


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    scaler: Any
    metadata: dict[str, Any]


def model_assets_available() -> bool:
    return settings.model_path.exists() and settings.scaler_path.exists()


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    if not model_assets_available():
        raise ModelNotReadyError(
            "Trained model assets were not found. Run src/train.py before prediction."
        )

    from tensorflow.keras.models import load_model

    metadata: dict[str, Any] = {}
    if settings.metadata_path.exists():
        metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))

    model = load_model(settings.model_path)
    scaler = joblib.load(settings.scaler_path)
    return ModelBundle(model=model, scaler=scaler, metadata=metadata)
