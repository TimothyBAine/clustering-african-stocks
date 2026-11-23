"""Paths and settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_BASE_URL = "https://mansaapi.com/api/v1"


def api_key() -> str:
    key = os.getenv("MANSA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MANSA_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def base_url() -> str:
    return os.getenv("MANSA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
