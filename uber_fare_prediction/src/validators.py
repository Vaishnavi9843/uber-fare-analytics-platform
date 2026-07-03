"""validators.py

Lightweight validation helpers.

These are intentionally small and focused so they can be reused both in the
Streamlit app and later in a FastAPI service.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")

