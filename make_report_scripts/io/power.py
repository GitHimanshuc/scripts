"""Power-diagnostic readers for joined HDF5 and extracted directory layouts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pandas as pd

from ._util import (
    TIME_COLUMN,
    OverlapPolicy,
    as_float64,
    combine_frames,
    natural_sort_key,
    normalize_columns,
)
from .dat import read_dat_file, read_dat_file_single_bh

PowerField = Literal["psi", "kappa"]


def join_str_with_underscore(values: Sequence[str]) -> str:
    """Join name components using the notebook's established separator."""

    return "_".join(values)


def find_subdomains(path: str | Path) -> list[str]:
    """Discover extracted ``<subdomain>.dir`` directories."""

    return sorted(
        (
            item.name.removesuffix(".dir")
            for item in Path(path).iterdir()
            if item.is_dir() and item.name.endswith(".dir")
        ),
        key=natural_sort_key,
    )


def find_topologies(path: str | Path) -> list[str]:
    """Discover topology prefixes from extracted root-level DAT files."""

    return sorted(
        {
            item.stem.split("_", 1)[0]
            for item in Path(path).iterdir()
            if item.is_file() and "_" in item.stem
        },
        key=natural_sort_key,
    )


def find_dat_file_names(path: str | Path) -> list[str]:
    """Discover diagnostic-name suffixes from extracted DAT files."""

    return sorted(
        {
            item.stem.split("_", 1)[1]
            for item in Path(path).iterdir()
            if item.is_file() and "_" in item.stem
        },
        key=natural_sort_key,
    )


def get_top_name_and_mode(name: str) -> tuple[str, int]:
    """Parse ``Bf0I1(12 modes).dat`` into topology and mode count."""

    file_name = Path(name).name
    match = re.fullmatch(r"(.+)\((\d+) modes\)[.]dat", file_name)
    if not match:
        raise ValueError(f"Unrecognized power-spectrum file name: {name!r}")
    return match.group(1), int(match.group(2))


def find_highest_modes_for_topologies(path: str | Path) -> dict[str, int]:
    """Find the largest coefficient count stored for each topology."""

    result: dict[str, int] = {}
    for item in Path(path).glob("*.dat"):
        try:
            topology, modes = get_top_name_and_mode(item.name)
        except ValueError:
            continue
        result[topology] = max(result.get(topology, 0), modes)
    return result


def make_mode_dataframe(path: str | Path) -> dict[str, pd.DataFrame]:
    """Stitch extracted spectrum files into one DataFrame per topology."""

    grouped: dict[str, list[pd.DataFrame]] = {}
    for item in sorted(Path(path).glob("*.dat"), key=natural_sort_key):
        try:
            topology, _ = get_top_name_and_mode(item.name)
        except ValueError:
            continue
        grouped.setdefault(topology, []).append(read_dat_file(item))
    result: dict[str, pd.DataFrame] = {}
    for topology, frames in grouped.items():
        combined = combine_frames(frames, overlap="last")
        result[topology] = combined.reindex(
            columns=sort_by_coefs_numbers(combined.columns.to_list())
        )
    return result


def topology_names(subdomain_name: str) -> tuple[str, ...]:
    """Return the unique spectral topology names used by a subdomain."""

    if re.match(r"Sphere", subdomain_name):
        return ("Bf0I1", "Bf1S2")
    if re.match(r"Cylinder", subdomain_name):
        return ("Bf0I1", "Bf1S1", "Bf2I1")
    if re.match(r"FilledCylinder", subdomain_name):
        return ("Bf0I1", "Bf1B2Radial", "Bf1B2")
    raise ValueError(f"Unrecognized subdomain name: {subdomain_name!r}")


def get_top_name_from_number(top_number: int, subdomain_name: str) -> str:
    """Map a notebook-era topology number to its SpEC name.

    The historical API treated topology 2 on spheres as another ``Bf1S2``.
    That compatibility behavior is retained here. New code should iterate
    :func:`topology_names`, which returns unique topology names.
    """

    names = topology_names(subdomain_name)
    if len(names) == 2 and top_number == 2:
        return names[1]
    try:
        return names[top_number]
    except IndexError as error:
        raise ValueError(
            f"Topology {top_number} is unavailable for {subdomain_name}"
        ) from error


def filter_columns(
    columns: Iterable[str],
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[str]:
    """Filter names that match any include and no exclude pattern."""

    included = include_patterns or (r".*",)
    excluded = exclude_patterns or ()
    return sorted(
        (
            column
            for column in columns
            if any(re.search(pattern, column) for pattern in included)
            and not any(re.search(pattern, column) for pattern in excluded)
        ),
        key=natural_sort_key,
    )


def chain_filter_columns(
    columns: Iterable[str],
    include_patterns: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
) -> list[str]:
    """Apply include and exclude regexes sequentially."""

    filtered = list(columns)
    for pattern in include_patterns or (r".*",):
        filtered = [column for column in filtered if re.search(pattern, column)]
    for pattern in exclude_patterns or ():
        filtered = [column for column in filtered if not re.search(pattern, column)]
    return sorted(filtered, key=natural_sort_key)


def sort_by_coefs_numbers(columns: Sequence[str]) -> list[str]:
    """Place non-coefficient columns first and coefficient columns numerically."""

    ordinary = [column for column in columns if "coef" not in column]
    coefficients = [column for column in columns if "coef" in column]
    return ordinary + sorted(
        coefficients,
        key=lambda column: int(re.search(r"coef(\d+)$", column).group(1)),
    )


def get_extreme_coef_for_each_domain(
    frame: pd.DataFrame,
    min_or_max: Literal["min", "max"] = "min",
) -> pd.DataFrame:
    """Reduce all coefficient columns to one extreme series per subdomain."""

    if min_or_max not in {"min", "max"}:
        raise ValueError("min_or_max must be 'min' or 'max'")
    result: dict[str, pd.Series] = {TIME_COLUMN: frame[TIME_COLUMN]}
    subdomains = sorted(
        {
            column.split("_", 1)[0]
            for column in frame.columns
            if column != TIME_COLUMN and "_" in column
        },
        key=natural_sort_key,
    )
    for subdomain in subdomains:
        columns = [
            column for column in frame.columns if column.startswith(f"{subdomain}_")
        ]
        reducer = frame[columns].min if min_or_max == "min" else frame[columns].max
        result[subdomain] = reducer(axis=1)
    return pd.DataFrame(result)


def _validate_field(field: str) -> PowerField:
    if field not in {"psi", "kappa"}:
        raise ValueError("field must be 'psi' or 'kappa'")
    return field


@dataclass(slots=True)
class PowerSpectrumCube:
    """A dense power spectrum with explicit coordinate arrays."""

    time: np.ndarray
    subdomains: tuple[str, ...]
    coefficients: tuple[int, ...]
    values: np.ndarray

    def coefficient(self, number: int) -> pd.DataFrame:
        """Return one coefficient as a time-by-subdomain DataFrame."""

        if number not in self.coefficients:
            maximum = max(self.coefficients, default=-1)
            raise ValueError(f"Coefficient {number} is unavailable; maximum is {maximum}")
        coefficient_index = self.coefficients.index(number)
        data: dict[str, np.ndarray] = {TIME_COLUMN: self.time}
        data.update(
            {
                subdomain: self.values[:, subdomain_index, coefficient_index]
                for subdomain_index, subdomain in enumerate(self.subdomains)
            }
        )
        return pd.DataFrame(data)

    def __getitem__(self, number: int) -> pd.DataFrame:
        return self.coefficient(number)

    def to_xarray(self):
        """Convert to xarray when that optional dependency is installed."""

        try:
            import xarray as xr
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Install xarray to use PowerSpectrumCube.to_xarray()"
            ) from error
        return xr.DataArray(
            self.values,
            coords={
                "time": self.time,
                "subdomain": self.subdomains,
                "coefficient": self.coefficients,
            },
            dims=("time", "subdomain", "coefficient"),
            name="power",
        )


class PowerDiagnosticsLoader:
    """Load and stitch one or more ``PowerDiagnostics.h5`` files."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        overlap: OverlapPolicy = "last",
    ):
        self.paths = tuple(
            sorted((Path(path) for path in paths), key=natural_sort_key)
        )
        if not self.paths:
            raise FileNotFoundError("No PowerDiagnostics.h5 files were supplied")
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        self.overlap = overlap
        self._subdomains: tuple[str, ...] | None = None

    @classmethod
    def from_ev(
        cls,
        ev_path: str | Path,
        segment_pattern: str,
        *,
        overlap: OverlapPolicy = "last",
    ) -> "PowerDiagnosticsLoader":
        root = Path(ev_path)
        paths = list(root.glob(f"{segment_pattern}/Run/PowerDiagnostics.h5"))
        if not paths:
            raise FileNotFoundError(
                f"No PowerDiagnostics.h5 files under {root}/{segment_pattern}/Run"
            )
        return cls(paths, overlap=overlap)

    @property
    def subdomains(self) -> tuple[str, ...]:
        """Return the union of subdomains present in all segments."""

        if self._subdomains is None:
            names: set[str] = set()
            for path in self.paths:
                with h5py.File(path, "r") as h5_file:
                    names.update(
                        key.removesuffix(".dir")
                        for key, item in h5_file.items()
                        if key.endswith(".dir") and isinstance(item, h5py.Group)
                    )
            self._subdomains = tuple(sorted(names, key=natural_sort_key))
        return self._subdomains

    def spectrum(
        self,
        subdomain: str,
        topology: str,
        field: PowerField = "psi",
    ) -> pd.DataFrame:
        """Load a power spectrum across all matching segments and mode counts."""

        _validate_field(field)
        frames: list[pd.DataFrame] = []
        group_path = f"{subdomain}.dir/Power{field}.dir"
        for path in self.paths:
            with h5py.File(path, "r") as h5_file:
                if group_path not in h5_file:
                    continue
                group = h5_file[group_path]
                for dataset_name in sorted(group, key=natural_sort_key):
                    if not dataset_name.startswith(f"{topology}("):
                        continue
                    dataset = group[dataset_name]
                    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
                        continue
                    values = as_float64(dataset[...])
                    columns = [
                        TIME_COLUMN,
                        *[f"coef{index}" for index in range(values.shape[1] - 1)],
                    ]
                    frames.append(pd.DataFrame(values, columns=columns))

        if not frames:
            raise KeyError(
                f"No {field} spectrum for {subdomain}/{topology} in {self.paths}"
            )
        combined = combine_frames(frames, overlap=self.overlap)
        return combined.reindex(
            columns=sort_by_coefs_numbers(combined.columns.to_list())
        )

    def diagnostic(
        self,
        file_name: str,
        top_number: int,
        field: PowerField,
    ) -> pd.DataFrame:
        """Load a non-spectrum diagnostic for every available subdomain."""

        _validate_field(field)
        value_index = 1 if field == "psi" else 2
        segment_frames: list[pd.DataFrame] = []

        for path in self.paths:
            series_frames: list[pd.DataFrame] = []
            with h5py.File(path, "r") as h5_file:
                for group_name, group in h5_file.items():
                    if not group_name.endswith(".dir") or not isinstance(group, h5py.Group):
                        continue
                    subdomain = group_name.removesuffix(".dir")
                    topology = get_top_name_from_number(top_number, subdomain)
                    dataset_name = f"{topology}_{file_name}"
                    if dataset_name not in group:
                        continue
                    dataset = group[dataset_name]
                    if dataset.ndim != 2 or dataset.shape[1] <= value_index:
                        continue
                    column = (
                        f"{field}_{Path(file_name).stem}_{topology} on {subdomain}"
                    )
                    series_frames.append(
                        pd.DataFrame(
                            {
                                TIME_COLUMN: as_float64(dataset[:, 0]),
                                column: as_float64(dataset[:, value_index]),
                            }
                        )
                    )

            if not series_frames:
                continue
            segment = series_frames[0]
            for frame in series_frames[1:]:
                segment = segment.merge(
                    frame,
                    on=TIME_COLUMN,
                    how="outer",
                    sort=True,
                )
            segment_frames.append(segment)

        if not segment_frames:
            raise KeyError(
                f"No {file_name} data for topology {top_number} in {self.paths}"
            )
        return combine_frames(segment_frames, overlap=self.overlap)

    def sphere_spectrum(
        self,
        field: PowerField = "psi",
        *,
        topology: str = "Bf1S2",
        subdomain_pattern: str = r"SphereC\d+",
    ) -> PowerSpectrumCube:
        """Load a coefficient cube for all matching spherical subdomains."""

        _validate_field(field)
        regex = re.compile(subdomain_pattern)
        subdomains = tuple(
            name for name in self.subdomains if regex.fullmatch(name)
        )
        if not subdomains:
            raise KeyError(f"No subdomains matched {subdomain_pattern!r}")

        spectra = {
            subdomain: self.spectrum(subdomain, topology, field)
            for subdomain in subdomains
        }
        time = np.unique(
            np.concatenate(
                [frame[TIME_COLUMN].to_numpy() for frame in spectra.values()]
            )
        )
        maximum_coefficients = max(
            sum(column.startswith("coef") for column in frame.columns)
            for frame in spectra.values()
        )
        coefficients = tuple(range(maximum_coefficients))
        values = np.full(
            (len(time), len(subdomains), maximum_coefficients),
            np.nan,
            dtype=np.float64,
        )

        for subdomain_index, subdomain in enumerate(subdomains):
            frame = spectra[subdomain]
            time_indices = np.searchsorted(time, frame[TIME_COLUMN].to_numpy())
            for coefficient in coefficients:
                column = f"coef{coefficient}"
                if column in frame:
                    values[time_indices, subdomain_index, coefficient] = frame[
                        column
                    ].to_numpy()
        return PowerSpectrumCube(time, subdomains, coefficients, values)


class ExtractedPowerDiagnosticsLoader:
    """Load an ``extracted-PowerDiagnostics`` directory tree."""

    def __init__(
        self,
        root: str | Path,
        *,
        overlap: OverlapPolicy = "last",
    ):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.overlap = overlap

    @property
    def subdomains(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    path.name.removesuffix(".dir")
                    for path in self.root.glob("*.dir")
                    if path.is_dir()
                ),
                key=natural_sort_key,
            )
        )

    def _subdomain_path(self, subdomain: str) -> Path:
        path = self.root / f"{subdomain}.dir"
        if not path.is_dir():
            raise KeyError(f"Unknown subdomain: {subdomain}")
        return path

    def topologies(self, subdomain: str) -> tuple[str, ...]:
        path = self._subdomain_path(subdomain)
        names = {
            file_path.stem.split("_", 1)[0]
            for file_path in path.glob("*.dat")
            if "_" in file_path.stem
        }
        for field in ("psi", "kappa"):
            mode_path = path / f"Power{field}.dir"
            if mode_path.is_dir():
                names.update(
                    file_path.name.split("(", 1)[0]
                    for file_path in mode_path.glob("*.dat")
                )
        return tuple(sorted(names, key=natural_sort_key))

    def diagnostic_names(self, subdomain: str) -> tuple[str, ...]:
        path = self._subdomain_path(subdomain)
        return tuple(
            sorted(
                {
                    file_path.stem.split("_", 1)[1]
                    for file_path in path.glob("*.dat")
                    if "_" in file_path.stem
                },
                key=natural_sort_key,
            )
        )

    def spectrum(
        self,
        subdomain: str,
        topology: str,
        field: PowerField = "psi",
    ) -> pd.DataFrame:
        _validate_field(field)
        directory = self._subdomain_path(subdomain) / f"Power{field}.dir"
        paths = sorted(directory.glob(f"{topology}(* modes).dat"), key=natural_sort_key)
        if not paths:
            raise KeyError(f"No {field} spectrum for {subdomain}/{topology}")
        return combine_frames(
            [read_dat_file(path) for path in paths],
            overlap=self.overlap,
        )

    def diagnostic(
        self,
        subdomain: str,
        topology: str,
        name: str,
    ) -> pd.DataFrame:
        path = self._subdomain_path(subdomain) / f"{topology}_{name}.dat"
        if not path.is_file():
            raise KeyError(f"No diagnostic {subdomain}/{topology}/{name}")
        return normalize_columns(read_dat_file(path))

    def load_nested(self) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
        """Return the notebook-era nested representation."""

        result: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
        for subdomain in self.subdomains:
            result[subdomain] = {}
            for topology in self.topologies(subdomain):
                values: dict[str, pd.DataFrame] = {}
                for field in ("psi", "kappa"):
                    try:
                        values[f"{field}_ps"] = self.spectrum(
                            subdomain,
                            topology,
                            field,
                        )
                    except KeyError:
                        pass
                for name in self.diagnostic_names(subdomain):
                    try:
                        values[name] = self.diagnostic(subdomain, topology, name)
                    except KeyError:
                        pass
                if values:
                    result[subdomain][topology] = values
        return result

    def load_flat(
        self,
        *,
        return_df: bool = True,
        diagnostic_patterns: Sequence[str] | None = None,
        include_spectra: bool = True,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Load a flat mapping, optionally merged on the actual time coordinate."""

        values: dict[str, pd.DataFrame] = {}
        for subdomain in self.subdomains:
            for topology in self.topologies(subdomain):
                for name in self.diagnostic_names(subdomain):
                    if diagnostic_patterns and not any(
                        re.search(pattern, name) for pattern in diagnostic_patterns
                    ):
                        continue
                    try:
                        values[f"{subdomain}_{topology}_{name}"] = self.diagnostic(
                            subdomain,
                            topology,
                            name,
                        )
                    except KeyError:
                        pass
                if include_spectra:
                    for field in ("psi", "kappa"):
                        try:
                            values[f"{subdomain}_{topology}_{field}_ps"] = self.spectrum(
                                subdomain,
                                topology,
                                field,
                            )
                        except KeyError:
                            pass

        if not return_df:
            return values
        if not values:
            raise FileNotFoundError(f"No power diagnostics found in {self.root}")

        merged: pd.DataFrame | None = None
        for key, frame in values.items():
            renamed = normalize_columns(frame).rename(
                columns={
                    column: f"{key}_{column}"
                    for column in frame.columns
                    if column != TIME_COLUMN
                }
            )
            merged = (
                renamed
                if merged is None
                else merged.merge(renamed, on=TIME_COLUMN, how="outer", sort=True)
            )
        return merged.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


def convert_series_to_coeff_df(data: pd.Series, top_num: int) -> pd.DataFrame:
    """Convert one flat power row to coefficient-by-subdomain form."""

    patterns = {
        0: r"Bf0",
        1: r"Bf1(S\d|B2R)",
        2: r"((Bf1S2|Bf1B2_)|Bf2)",
    }
    if top_num not in patterns:
        raise ValueError("top_num must be 0, 1, or 2")
    indices = [
        str(index)
        for index in data.index
        if index != TIME_COLUMN and re.search(patterns[top_num], str(index))
    ]
    subdomains = sorted(
        {index.split("_", 1)[0] for index in indices},
        key=natural_sort_key,
    )
    result: dict[str, dict[int, float]] = {name: {} for name in subdomains}
    for index in indices:
        value = data[index]
        if pd.isna(value):
            continue
        subdomain = index.split("_", 1)[0]
        match = re.search(r"coef(\d+)$", index)
        if match:
            result[subdomain][int(match.group(1))] = value
    return pd.DataFrame(
        {name: pd.Series(values) for name, values in result.items()}
    )


def series_closest_to_time(
    time: float,
    frame: pd.DataFrame,
) -> tuple[float, pd.Series]:
    """Return the first row at or after a target time."""

    candidates = np.flatnonzero(frame[TIME_COLUMN].to_numpy() >= time)
    if len(candidates) == 0:
        raise ValueError(f"No sample at or after t={time}")
    index = int(candidates[0])
    return float(frame.iloc[index][TIME_COLUMN]), frame.iloc[index].copy()


# Notebook-era compatibility API.
def load_power_diagonistics(path: str | Path):
    return ExtractedPowerDiagnosticsLoader(path).load_nested()


def load_power_diagonistics_flat(
    path: str | Path,
    reload: bool = False,
    return_df: bool = True,
    load_dat_files_only: Sequence[str] | None = None,
):
    del reload  # The unsafe fixed pickle cache was intentionally removed.
    return ExtractedPowerDiagnosticsLoader(path).load_flat(
        return_df=return_df,
        diagnostic_patterns=load_dat_files_only,
        include_spectra=load_dat_files_only is None,
    )


# Correctly spelled aliases for new callers.
load_power_diagnostics = load_power_diagonistics
load_power_diagnostics_flat = load_power_diagonistics_flat


class LoadPowerDiagnostics(PowerDiagnosticsLoader):
    """Backward-compatible constructor for the notebook class."""

    def __init__(self, Ev_path: str | Path, lev_regex: str):
        loaded = PowerDiagnosticsLoader.from_ev(Ev_path, lev_regex)
        super().__init__(loaded.paths, overlap=loaded.overlap)
        self.Ev_path = Path(Ev_path)
        self.lev_regex = lev_regex
        self.power_diag_paths = list(self.paths)
        self.info_dict: dict[str, dict[str, object]] = {}
        for path in self.paths:
            segment_name = path.relative_to(self.Ev_path).parts[0]
            entry: dict[str, object] = {"path": path}
            with h5py.File(path, "r") as h5_file:
                entry.update(
                    {
                        key.removesuffix(".dir"): {}
                        for key in h5_file
                        if key.endswith(".dir")
                    }
                )
            self.info_dict[segment_name] = entry

    def get_df_power_spec(
        self,
        subdomain: str,
        top_name: str,
        psi_or_kappa: PowerField = "psi",
    ) -> pd.DataFrame:
        return self.spectrum(subdomain, top_name, psi_or_kappa)

    def get_df_non_ps(
        self,
        file_name: str,
        top_number: int,
        psi_or_kappa: PowerField,
    ) -> pd.DataFrame:
        return self.diagnostic(file_name, top_number, psi_or_kappa)


class SphereCPowerData:
    """Backward-compatible view over :class:`PowerSpectrumCube`."""

    def __init__(
        self,
        Ev_path: str | Path,
        lev_regex: str,
        psi_or_kappa: PowerField = "psi",
    ):
        self.pd_obj = LoadPowerDiagnostics(Ev_path, lev_regex)
        self.topology = "Bf1S2"
        self._cube = self.pd_obj.sphere_spectrum(
            psi_or_kappa,
            topology=self.topology,
        )
        self.sphereC_keys = list(self._cube.subdomains)
        self.time = self._cube.time
        self.max_coefs = len(self._cube.coefficients)
        self._coef_data = np.moveaxis(self._cube.values, 2, 0)

    def get_coef(self, coef_num: int) -> pd.DataFrame:
        return self._cube.coefficient(coef_num)

    def __getitem__(self, coef_num: int) -> pd.DataFrame:
        return self.get_coef(coef_num)


def get_top_df_power_spec(
    Ev_path: str | Path,
    lev_regex: str,
    subdomain: str,
    psi_or_kappa: PowerField,
) -> dict[str, pd.DataFrame]:
    loader = LoadPowerDiagnostics(Ev_path, lev_regex)
    return {
        topology: loader.spectrum(subdomain, topology, psi_or_kappa)
        for topology in topology_names(subdomain)
    }


__all__ = [
    "ExtractedPowerDiagnosticsLoader",
    "LoadPowerDiagnostics",
    "PowerDiagnosticsLoader",
    "PowerSpectrumCube",
    "SphereCPowerData",
    "chain_filter_columns",
    "convert_series_to_coeff_df",
    "filter_columns",
    "find_dat_file_names",
    "find_highest_modes_for_topologies",
    "find_subdomains",
    "find_topologies",
    "get_extreme_coef_for_each_domain",
    "get_top_df_power_spec",
    "get_top_name_and_mode",
    "get_top_name_from_number",
    "join_str_with_underscore",
    "load_power_diagnostics",
    "load_power_diagnostics_flat",
    "load_power_diagonistics",
    "load_power_diagonistics_flat",
    "make_mode_dataframe",
    "read_dat_file_single_bh",
    "series_closest_to_time",
    "sort_by_coefs_numbers",
    "topology_names",
]
