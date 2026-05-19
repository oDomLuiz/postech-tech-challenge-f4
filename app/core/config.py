from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"


@dataclass(frozen=True)
class Settings:
    project_name: str = "Tech Challenge Fase 4 - LSTM Stock Prediction"
    default_ticker: str = "AAPL"
    target_column: str = "Close"
    window_size: int = 60
    train_size: float = 0.80
    validation_size: float = 0.10
    test_size: float = 0.10
    model_version: str = "1.0.0"
    model_path: Path = MODELS_DIR / "lstm_model.keras"
    scaler_path: Path = MODELS_DIR / "scaler.pkl"
    metadata_path: Path = MODELS_DIR / "metadata.json"


settings = Settings()


def ensure_project_dirs() -> None:
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
