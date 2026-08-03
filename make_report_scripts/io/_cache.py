"""Versioned JSON caching for bulk run-data loads."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CACHE_FORMAT = "make_report_scripts.runs_data_dict"
_CACHE_VERSION = 1
_TAG = "__make_report_scripts_type__"


class CacheFormatError(ValueError):
    """Raised when a cache file cannot be safely reconstructed."""


def make_cache_request(
    runs: list[tuple[str, tuple[str, ...]]],
    data_file_path: str,
) -> dict[str, object]:
    """Build the canonical request stored in and hashed for a cache file."""

    return {
        "data_file_path": data_file_path,
        "runs": [
            {"name": run_name, "roots": list(run_roots)}
            for run_name, run_roots in runs
        ],
    }


def cache_path(
    cache_folder: str | Path,
    request: dict[str, object],
) -> Path:
    """Return a readable, collision-resistant path for one load request."""

    canonical = json.dumps(
        request,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    data_file_path = str(request["data_file_path"]).split("@", 1)[0]
    stem = Path(data_file_path).stem or "run-data"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-.")
    return (
        Path(cache_folder)
        / f"make-report-cache-{safe_stem or 'run-data'}-{digest}.json"
    )


def _encode_float(value: float) -> float | dict[str, str]:
    if math.isnan(value):
        return {_TAG: "float", "value": "nan"}
    if math.isinf(value):
        return {_TAG: "float", "value": "inf" if value > 0 else "-inf"}
    return value


def _encode_value(value: Any) -> object:
    if value is pd.NA:
        return {_TAG: "pd.NA"}
    if value is pd.NaT:
        return {_TAG: "pd.NaT"}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, complex):
        return {
            _TAG: "complex",
            "real": _encode_float(float(value.real)),
            "imag": _encode_float(float(value.imag)),
        }
    if isinstance(value, pd.Timestamp):
        return {_TAG: "timestamp", "value": value.isoformat()}
    if isinstance(value, pd.Timedelta):
        return {_TAG: "timedelta", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {_TAG: "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {_TAG: "date", "value": value.isoformat()}
    if isinstance(value, bytes):
        return {
            _TAG: "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return {_TAG: "list", "items": [_encode_value(item) for item in value]}
    if isinstance(value, dict):
        return {
            _TAG: "dict",
            "items": [
                [_encode_value(key), _encode_value(item)]
                for key, item in value.items()
            ],
        }
    raise TypeError(f"Cannot encode cache value of type {type(value).__name__}")


def _decode_float(value: object) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if not isinstance(value, dict) or value.get(_TAG) != "float":
        raise CacheFormatError("Malformed encoded floating-point value")
    encoded = value.get("value")
    values = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}
    if encoded not in values:
        raise CacheFormatError(f"Unknown encoded floating-point value: {encoded!r}")
    return values[str(encoded)]


def _decode_value(value: object) -> Any:
    if not isinstance(value, dict) or _TAG not in value:
        return value

    kind = value[_TAG]
    if kind == "float":
        return _decode_float(value)
    if kind == "complex":
        return complex(
            _decode_float(value.get("real")),
            _decode_float(value.get("imag")),
        )
    if kind == "pd.NA":
        return pd.NA
    if kind == "pd.NaT":
        return pd.NaT
    if kind == "timestamp":
        return pd.Timestamp(value["value"])
    if kind == "timedelta":
        return pd.Timedelta(value["value"])
    if kind == "datetime":
        return datetime.fromisoformat(str(value["value"]))
    if kind == "date":
        return date.fromisoformat(str(value["value"]))
    if kind == "bytes":
        return base64.b64decode(str(value["value"]).encode("ascii"))
    if kind in {"tuple", "list"}:
        items = [_decode_value(item) for item in value.get("items", [])]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {
            _decode_value(item[0]): _decode_value(item[1])
            for item in value.get("items", [])
        }
    raise CacheFormatError(f"Unknown encoded cache value type: {kind!r}")


def _frame_to_payload(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": [_encode_value(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "rows": [
            [_encode_value(value) for value in row]
            for row in frame.itertuples(index=False, name=None)
        ],
    }


def _frame_from_payload(payload: object) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise CacheFormatError("Cached DataFrame is not an object")
    try:
        columns = [_decode_value(column) for column in payload["columns"]]
        dtypes = payload["dtypes"]
        rows = [
            [_decode_value(value) for value in row]
            for row in payload["rows"]
        ]
    except (KeyError, TypeError) as error:
        raise CacheFormatError("Cached DataFrame is missing required fields") from error
    if not isinstance(dtypes, list) or len(columns) != len(dtypes):
        raise CacheFormatError("Cached DataFrame has inconsistent column metadata")
    if any(len(row) != len(columns) for row in rows):
        raise CacheFormatError("Cached DataFrame has an inconsistent row width")

    restored_columns: list[pd.Series] = []
    for position, dtype in enumerate(dtypes):
        try:
            restored_columns.append(
                pd.Series(
                    [row[position] for row in rows],
                    dtype=str(dtype),
                )
            )
        except (TypeError, ValueError) as error:
            raise CacheFormatError(
                f"Cannot restore cached column {columns[position]!r} as {dtype!r}"
            ) from error
    frame = (
        pd.concat(restored_columns, axis="columns")
        if restored_columns
        else pd.DataFrame(index=range(len(rows)))
    )
    frame.columns = columns
    return frame


def write_cache(
    path: Path,
    request: dict[str, object],
    runs_data: dict[str, pd.DataFrame],
) -> None:
    """Atomically write one bulk-loader result."""

    payload = {
        "format": _CACHE_FORMAT,
        "version": _CACHE_VERSION,
        "request": request,
        "runs": [
            {"name": run_name, "frame": _frame_to_payload(frame)}
            for run_name, frame in runs_data.items()
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_cache(
    path: Path,
    expected_request: dict[str, object],
) -> dict[str, pd.DataFrame]:
    """Read and validate one bulk-loader cache file."""

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise CacheFormatError("Cache root is not an object")
    if payload.get("format") != _CACHE_FORMAT:
        raise CacheFormatError("Cache format marker is missing or unsupported")
    if payload.get("version") != _CACHE_VERSION:
        raise CacheFormatError(f"Unsupported cache version: {payload.get('version')!r}")
    if payload.get("request") != expected_request:
        raise CacheFormatError("Cache request metadata does not match")
    entries = payload.get("runs")
    if not isinstance(entries, list):
        raise CacheFormatError("Cache run collection is not a list")

    runs_data: dict[str, pd.DataFrame] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise CacheFormatError("Cache contains an invalid run entry")
        run_name = entry["name"]
        if run_name in runs_data:
            raise CacheFormatError(f"Cache contains duplicate run {run_name!r}")
        runs_data[run_name] = _frame_from_payload(entry.get("frame"))
    expected_names = [entry["name"] for entry in expected_request.get("runs", [])]
    if list(runs_data) != expected_names:
        raise CacheFormatError("Cached runs do not match the requested run order")
    return runs_data


__all__ = [
    "CacheFormatError",
    "cache_path",
    "make_cache_request",
    "read_cache",
    "write_cache",
]
