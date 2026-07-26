"""Readers for apparent-horizon HDF5 diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import pandas as pd

from ._util import TIME_COLUMN, as_float64, combine_frames, natural_sort_key


def make_col_names(value_name: str, value_size: int) -> list[str]:
    """Build stable columns for a scalar or vector-valued dataset."""

    if value_size == 1:
        return [value_name]
    return [f"{value_name}_{index}" for index in range(value_size)]


def append_to_df(data, col_names, df) -> None:
    """Compatibility helper that appends array columns to a DataFrame."""

    for index, column_name in enumerate(col_names):
        df[column_name] = data[:, index]


def make_Bh_pandas(h5_dir: h5py.Group) -> pd.DataFrame:
    """Convert one horizon HDF5 group into a time-aligned DataFrame."""

    frames: list[pd.DataFrame] = []
    dataset_names: list[str] = []

    def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset) and item.ndim == 2 and item.shape[1] >= 1:
            dataset_names.append(name)

    h5_dir.visititems(collect)
    for name in sorted(dataset_names, key=natural_sort_key):
        dataset = h5_dir[name]
        values = as_float64(dataset[...])
        value_name = name.removesuffix(".dat")
        columns = [
            TIME_COLUMN,
            *make_col_names(value_name, values.shape[1] - 1),
        ]
        frames.append(pd.DataFrame(values, columns=columns))

    if not frames:
        raise ValueError(f"{h5_dir.name} contains no tabular horizon datasets")

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=TIME_COLUMN, how="outer", sort=True)
    return result.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


def horizon_to_pandas(horizon_path: str | Path) -> dict[str, pd.DataFrame]:
    """Read every horizon object from a ``Horizons.h5`` file."""

    path = Path(horizon_path)
    if not path.exists():
        raise FileNotFoundError(path)

    frames: dict[str, pd.DataFrame] = {}
    with h5py.File(path, "r") as h5_file:
        for key, item in h5_file.items():
            if key == "VersionHist.ver" or not isinstance(item, h5py.Group):
                continue
            frames[key.removesuffix(".dir")] = make_Bh_pandas(item)
    return frames


def read_horizonh5(
    horizon_path: str | Path,
    horizon_name: str,
) -> pd.DataFrame | None:
    """Read one horizon object, returning ``None`` when absent."""

    with h5py.File(horizon_path, "r") as h5_file:
        key = f"{horizon_name}.dir"
        return make_Bh_pandas(h5_file[key]) if key in h5_file else None


def read_horizon_across_Levs(
    path_list: Sequence[str | Path],
    *,
    overlap: str = "last",
) -> dict[str, pd.DataFrame]:
    """Stitch all available horizon objects across ordered segments."""

    per_horizon: dict[str, list[pd.DataFrame]] = {}
    for path in sorted((Path(item) for item in path_list), key=natural_sort_key):
        for horizon_name, frame in horizon_to_pandas(path).items():
            per_horizon.setdefault(horizon_name, []).append(frame)
    return {
        name: combine_frames(frames, overlap=overlap)
        for name, frames in per_horizon.items()
    }


def load_horizon_data_from_levs(
    base_path: str | Path,
    runs_path: Mapping[str, str | Path],
    *,
    overlap: str = "last",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load horizon files for several named run patterns."""

    root = Path(base_path)
    return {
        run_name: read_horizon_across_Levs(
            list(root.glob(str(pattern))),
            overlap=overlap,
        )
        for run_name, pattern in runs_path.items()
    }


def flatten_dict(
    horizon_data_dict: Mapping[str, Mapping[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Flatten ``run -> horizon -> frame`` into ``run_horizon -> frame``."""

    return {
        f"{run_name}_{horizon_name}": frame
        for run_name, horizon_frames in horizon_data_dict.items()
        for horizon_name, frame in horizon_frames.items()
    }

