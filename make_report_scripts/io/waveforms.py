"""Readers for waveform, finite-radius, and orbit HDF5 products."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ._util import TIME_COLUMN, as_float64, decode_labels, natural_sort_key


def get_worldtube_extraction_radii(folder_path: str | Path) -> list[str]:
    """List radii available as ``BondiCceR*.h5`` files."""

    root = Path(folder_path)
    return sorted(
        (
            path.name.split("BondiCceR", 1)[1].split(".", 1)[0]
            for path in root.glob("BondiCceR*.h5")
        ),
        key=natural_sort_key,
    )


def get_finite_radius_variables(folder_path: str | Path) -> list[str]:
    """List variables available as ``*_CodeUnits.h5`` files."""

    return sorted(
        path.name.split("_", 1)[0]
        for path in Path(folder_path).glob("*_CodeUnits.h5")
    )


def get_finite_radius_extractions(file_path: str | Path) -> list[str]:
    """List extraction radii in one finite-radius HDF5 file."""

    with h5py.File(file_path, "r") as h5_file:
        return sorted(
            (
                key.removeprefix("R").removesuffix(".dir")
                for key in h5_file
                if key.startswith("R") and key.endswith(".dir")
            ),
            key=natural_sort_key,
        )


def find_min_max_l(keys) -> tuple[int, int]:
    """Find the smallest and largest spherical-harmonic degree in keys."""

    degrees = [
        int(key.split("_")[1].removeprefix("l"))
        for key in keys
        if "Y_l" in key
    ]
    if not degrees:
        raise ValueError("No Y_l<degree> datasets found")
    return min(degrees), max(degrees)


def read_finite_radius_quantities(
    file_path: str | Path,
    radius: str | int,
) -> pd.DataFrame:
    """Read complex spherical-harmonic modes at a finite radius."""

    group_name = f"R{radius}.dir"
    with h5py.File(file_path, "r") as h5_file:
        if group_name not in h5_file:
            available = get_finite_radius_extractions(file_path)
            raise KeyError(
                f"{group_name} not found in {file_path}; available radii: {available}"
            )
        group = h5_file[group_name]
        minimum_l, maximum_l = find_min_max_l(group.keys())

        time_dataset = group.get("Y_l2_m0.dat")
        if time_dataset is None:
            first_mode = next(
                (group[key] for key in group if key.startswith("Y_l")),
                None,
            )
            if first_mode is None:
                raise ValueError(f"{group.name} contains no mode datasets")
            time_dataset = first_mode

        result: dict[str, np.ndarray] = {
            TIME_COLUMN: as_float64(time_dataset[:, 0])
        }
        for ell in range(minimum_l, maximum_l + 1):
            for mode in range(-ell, ell + 1):
                key = f"Y_l{ell}_m{mode}.dat"
                if key not in group:
                    continue
                dataset = group[key]
                result[f"{ell},{mode}"] = (
                    as_float64(dataset[:, 1])
                    + 1j * as_float64(dataset[:, 2])
                )
        return pd.DataFrame(result)


def read_worldtube_data(file_path: str | Path, variable: str) -> pd.DataFrame:
    """Read one complex Bondi worldtube variable."""

    with h5py.File(file_path, "r") as h5_file:
        if variable not in h5_file:
            raise KeyError(f"{variable!r} not found in {file_path}")
        dataset = h5_file[variable]
        if "Legend" not in dataset.attrs:
            raise ValueError(f"{dataset.name} has no Legend attribute")
        labels = decode_labels(dataset.attrs["Legend"])
        labels[0] = TIME_COLUMN
        index = {name: position for position, name in enumerate(labels)}
        result: dict[str, np.ndarray] = {
            TIME_COLUMN: as_float64(dataset[:, index[TIME_COLUMN]])
        }

        mode_pairs: set[tuple[int, int]] = set()
        for label in labels:
            if label.startswith(("Re(", "Im(")) and label.endswith(")"):
                ell, mode = label[3:-1].split(",")
                mode_pairs.add((int(ell), int(mode)))

        for ell, mode in sorted(mode_pairs):
            real_name = f"Re({ell},{mode})"
            imaginary_name = f"Im({ell},{mode})"
            values = np.zeros(dataset.shape[0], dtype=np.complex128)
            if real_name in index:
                values.real = as_float64(dataset[:, index[real_name]])
            if imaginary_name in index:
                values.imag = as_float64(dataset[:, index[imaginary_name]])
            result[f"{ell},{mode}"] = values
        return pd.DataFrame(result)


def read_orbit_diagnostics(file_name: str | Path) -> pd.DataFrame:
    """Read and time-align all tabular datasets in OrbitDiagnostics HDF5."""

    frames: list[pd.DataFrame] = []
    with h5py.File(file_name, "r") as h5_file:
        datasets: list[h5py.Dataset] = []

        def collect(_name: str, item: h5py.Group | h5py.Dataset) -> None:
            if (
                isinstance(item, h5py.Dataset)
                and item.ndim == 2
                and "Legend" in item.attrs
            ):
                datasets.append(item)

        h5_file.visititems(collect)
        for dataset in datasets:
            labels = decode_labels(dataset.attrs["Legend"])
            labels[0] = TIME_COLUMN
            frames.append(pd.DataFrame(as_float64(dataset[...]), columns=labels))

    if not frames:
        raise ValueError(f"No tabular OrbitDiagnostics datasets found in {file_name}")
    result = frames[0]
    for frame in frames[1:]:
        duplicate_columns = [
            column for column in frame.columns if column in result and column != TIME_COLUMN
        ]
        frame = frame.drop(columns=duplicate_columns)
        result = result.merge(frame, on=TIME_COLUMN, how="outer", sort=True)
    return result.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


# Notebook-era compatibility names.
GetWTDataExtracRadii = get_worldtube_extraction_radii
GetFiniteRadiiDataVars = get_finite_radius_variables
GetFiniteRadiusExtractionList = get_finite_radius_extractions
FindMinMaxL = find_min_max_l
read_finite_radius_quantaties = read_finite_radius_quantities
read_WT_data = read_worldtube_data
read_OrbitDiagnostics_file = read_orbit_diagnostics
