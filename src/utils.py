from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import RAW_DATA_DIR, ensure_project_dirs, settings


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def raw_data_path(ticker: str) -> Path:
    ensure_project_dirs()
    return RAW_DATA_DIR / f"{normalize_ticker(ticker)}.csv"


def load_raw_data(ticker: str) -> pd.DataFrame:
    path = raw_data_path(ticker)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run src/collect_data.py first."
        )
    return pd.read_csv(path)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def metadata_payload(ticker: str, window_size: int) -> dict[str, Any]:
    return {
        "ticker": normalize_ticker(ticker),
        "target_column": settings.target_column,
        "window_size": window_size,
        "model_version": settings.model_version,
        "trained_at": utc_now_iso(),
    }
