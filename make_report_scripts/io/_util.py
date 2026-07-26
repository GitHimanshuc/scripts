"""Shared implementation helpers for report data readers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

TIME_COLUMN = "t(M)"
OverlapPolicy = Literal["preserve", "first", "last", "error"]

_TIME_ALIASES = {
    "t": TIME_COLUMN,
    "time": TIME_COLUMN,
    "Time": TIME_COLUMN,
    "time after step": TIME_COLUMN,
}


def natural_sort_key(value: str | Path) -> tuple[object, ...]:
    """Return a deterministic key that sorts embedded numbers numerically."""

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    )


def decode_labels(values: Iterable[object]) -> list[str]:
    """Decode byte-valued HDF5 legends into ordinary strings."""

    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def as_float64(data: object) -> np.ndarray:
    """Convert ordinary or SpEC double-double arrays to float64."""

    array = np.asarray(data)
    fields = array.dtype.fields
    if fields and "hi" in fields:
        return np.asarray(array["hi"], dtype=np.float64)
    return np.asarray(array, dtype=np.float64)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the library's canonical column naming."""

    normalized = frame.copy()
    normalized.rename(columns=_TIME_ALIASES, inplace=True)
    normalized.columns = [
        column.replace("math_utils::", "") if isinstance(column, str) else column
        for column in normalized.columns
    ]
    return normalized


def combine_frames(
    frames: Sequence[pd.DataFrame],
    *,
    time_column: str = TIME_COLUMN,
    overlap: OverlapPolicy = "preserve",
    sort: bool = True,
) -> pd.DataFrame:
    """Combine segment frames with an explicit repeated-time policy.

    Frames must be supplied in chronological segment order. For ``last`` the
    later segment wins at an overlapping time; for ``first`` the earlier one
    wins. ``preserve`` retains every row, which is useful for event diagnostics
    where repeated times are meaningful.
    """

    if not frames:
        raise FileNotFoundError("No data files matched the requested source")
    if overlap not in {"preserve", "first", "last", "error"}:
        raise ValueError(f"Unknown overlap policy: {overlap!r}")

    normalized = [normalize_columns(frame) for frame in frames]
    combined = pd.concat(normalized, ignore_index=True, sort=False)

    if time_column not in combined.columns:
        return combined

    repeated = combined.duplicated(subset=time_column, keep=False)
    if overlap == "error" and repeated.any():
        values = combined.loc[repeated, time_column].drop_duplicates().tolist()
        raise ValueError(f"Repeated {time_column} values: {values[:10]}")
    if overlap in {"first", "last"}:
        combined = combined.drop_duplicates(subset=time_column, keep=overlap)
    if sort:
        combined = combined.sort_values(time_column, kind="stable")
    return combined.reset_index(drop=True)

