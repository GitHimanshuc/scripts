"""High-level run loading and notebook-compatibility dispatch."""

from __future__ import annotations

import glob
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from string import ascii_uppercase
from typing import Literal

import pandas as pd

from ._cache import (
    CacheFormatError,
    cache_path,
    make_cache_request,
    read_cache,
    write_cache,
)
from ._util import OverlapPolicy, combine_frames, natural_sort_key
from .dat import (
    get_last_time_from_tstepper_diag,
    hist_files_to_dataframe,
    read_dat_file,
    read_dat_file_uneq_cols,
    read_point_interpolation_file,
)
from .horizons import read_horizonh5
from .power import PowerDiagnosticsLoader
from .profiler import read_profiler
from .waveforms import (
    get_worldtube_extraction_radii,
    read_finite_radius_quantities,
    read_orbit_diagnostics,
    read_worldtube_data,
)

DataKind = Literal[
    "auto",
    "dat",
    "unequal_dat",
    "history",
    "horizon",
    "profiler",
    "power_diagnostic",
    "worldtube",
    "finite_radius",
    "point_interpolation",
    "orbit",
]


def _join_pattern(base: str | Path, relative_path: str | Path) -> str:
    return f"{str(base).rstrip('/')}/{str(relative_path).lstrip('/')}"


def _is_failed_path(path: Path) -> bool:
    return any("failed" in part.casefold() for part in path.parts)


def all_words_upto(value: int | str) -> list[str]:
    """Generate segment suffixes ``AA, AB, ...`` through an index or word."""

    if isinstance(value, str):
        if len(value) < 2 or any(character not in ascii_uppercase for character in value):
            raise ValueError("Segment words must contain at least two uppercase letters")
        maximum_index = sum(26**length for length in range(2, len(value)))
        maximum_index += sum(
            (ord(character) - ord("A")) * 26**power
            for power, character in enumerate(reversed(value))
        )
    else:
        if value < 0:
            raise ValueError("value must be non-negative")
        maximum_index = value

    words: list[str] = []
    length = 2
    while len(words) <= maximum_index:
        words.extend(
            "".join(characters)
            for characters in product(ascii_uppercase, repeat=length)
        )
        length += 1
    return words[: maximum_index + 1]


@dataclass(frozen=True, slots=True)
class RunSource:
    """Location of one simulation's segmented ``Run`` directories."""

    run_patterns: tuple[str, ...]

    @classmethod
    def from_ev(
        cls,
        ev_path: str | Path,
        segment_pattern: str,
        *,
        run_directory: str = "Run",
    ) -> "RunSource":
        return cls(
            (
                str(Path(ev_path) / segment_pattern / run_directory),
            )
        )

    @classmethod
    def from_run_patterns(
        cls,
        patterns: str | Path | Sequence[str | Path],
    ) -> "RunSource":
        if isinstance(patterns, (str, Path)):
            patterns = (patterns,)
        return cls(tuple(str(pattern) for pattern in patterns))


class RunDataLoader:
    """Facade for deterministic, time-normalized loading across segments."""

    def __init__(
        self,
        source: RunSource | str | Path | Sequence[str | Path],
        *,
        overlap: OverlapPolicy = "last",
        exclude_failed: bool = True,
    ):
        self.source = (
            source
            if isinstance(source, RunSource)
            else RunSource.from_run_patterns(source)
        )
        self.overlap = overlap
        self.exclude_failed = exclude_failed

    @classmethod
    def from_ev(
        cls,
        ev_path: str | Path,
        segment_pattern: str,
        *,
        run_directory: str = "Run",
        overlap: OverlapPolicy = "last",
        exclude_failed: bool = True,
    ) -> "RunDataLoader":
        return cls(
            RunSource.from_ev(
                ev_path,
                segment_pattern,
                run_directory=run_directory,
            ),
            overlap=overlap,
            exclude_failed=exclude_failed,
        )

    def paths(self, relative_path: str | Path) -> tuple[Path, ...]:
        """Resolve a relative path, preserving run-pattern priority.

        Matches are naturally sorted within each pattern. When several roots
        are supplied, their caller-provided order determines which root is
        considered later by overlap policies such as ``last``.
        """

        seen: set[Path] = set()
        matches: list[Path] = []
        for run_pattern in self.source.run_patterns:
            pattern = _join_pattern(run_pattern, relative_path)
            pattern_matches = sorted(
                (Path(match) for match in glob.iglob(pattern, recursive=True)),
                key=natural_sort_key,
            )
            for path in pattern_matches:
                if path in seen or not (path.is_file() or path.is_dir()):
                    continue
                if self.exclude_failed and _is_failed_path(path):
                    continue
                seen.add(path)
                matches.append(path)
        return tuple(matches)

    def _read_many(self, paths: Sequence[Path], reader) -> pd.DataFrame:
        if not paths:
            raise FileNotFoundError("No files matched this run source")
        return combine_frames(
            [reader(path) for path in paths],
            overlap=self.overlap,
        )

    def dat(
        self,
        relative_path: str | Path,
        *,
        unequal_columns: bool = False,
    ) -> pd.DataFrame:
        """Load an ordinary or variable-width DAT diagnostic."""

        reader = read_dat_file_uneq_cols if unequal_columns else read_dat_file
        return self._read_many(self.paths(relative_path), reader)

    def history(self, relative_path: str | Path) -> pd.DataFrame:
        """Load one history diagnostic across segments."""

        return self._read_many(
            self.paths(relative_path),
            hist_files_to_dataframe,
        )

    def horizon(
        self,
        horizon_name: str,
        *,
        relative_path: str | Path = "ApparentHorizons/Horizons.h5",
    ) -> pd.DataFrame:
        """Load one apparent horizon across segments."""

        frames = [
            frame
            for path in self.paths(relative_path)
            if (frame := read_horizonh5(path, horizon_name)) is not None
        ]
        if not frames:
            raise KeyError(
                f"{horizon_name!r} was absent from every {relative_path} file"
            )
        return combine_frames(frames, overlap=self.overlap)

    def profiler(self, relative_path: str | Path) -> pd.DataFrame:
        """Load HDF5 profiler output across segments."""

        return self._read_many(self.paths(relative_path), read_profiler)

    def worldtube(
        self,
        relative_path: str | Path,
        variable: str,
    ) -> pd.DataFrame:
        """Load one Bondi worldtube variable across segments."""

        return self._read_many(
            self.paths(relative_path),
            lambda path: read_worldtube_data(path, variable),
        )

    def finite_radius(
        self,
        relative_path: str | Path,
        radius: str | int,
    ) -> pd.DataFrame:
        """Load one finite-radius waveform extraction across segments."""

        return self._read_many(
            self.paths(relative_path),
            lambda path: read_finite_radius_quantities(path, radius),
        )

    def point_interpolation(
        self,
        relative_path: str | Path,
        *,
        load_all: bool = False,
    ) -> pd.DataFrame:
        """Load point-interpolation output across segments."""

        paths = self.paths(relative_path)
        if load_all:
            # One representative path per parent/variable avoids loading the
            # same family once for every glob match.
            representatives: dict[tuple[Path, str], Path] = {}
            for path in paths:
                variable = path.stem.removeprefix("Int_").split("_", 1)[0]
                representatives.setdefault((path.parent, variable), path)
            paths = tuple(representatives.values())
        return self._read_many(
            paths,
            lambda path: read_point_interpolation_file(path, load_all=load_all),
        )

    def orbit(self, relative_path: str | Path) -> pd.DataFrame:
        """Load OrbitDiagnostics HDF5 output across segments."""

        return self._read_many(
            self.paths(relative_path),
            read_orbit_diagnostics,
        )

    def power(
        self,
        *,
        relative_path: str | Path = "PowerDiagnostics.h5",
    ) -> PowerDiagnosticsLoader:
        """Return the specialized loader for joined power diagnostics."""

        paths = self.paths(relative_path)
        if not paths:
            raise FileNotFoundError(f"No {relative_path} files matched")
        return PowerDiagnosticsLoader(paths, overlap=self.overlap)

    def load(
        self,
        relative_path: str | Path,
        *,
        kind: DataKind = "auto",
        **options,
    ) -> pd.DataFrame:
        """Load a diagnostic with explicit or conservative automatic dispatch."""

        path_text = str(relative_path)
        if kind == "auto":
            if "Coefs.dat" in path_text:
                kind = "unequal_dat"
            elif "Hist-" in path_text:
                kind = "history"
            elif "Profiler" in path_text:
                kind = "profiler"
            elif "OrbitDiagnostics" in path_text:
                kind = "orbit"
            elif "Int_" in path_text:
                kind = "point_interpolation"
            else:
                kind = "dat"

        methods = {
            "dat": self.dat,
            "unequal_dat": lambda path, **kwargs: self.dat(
                path,
                unequal_columns=True,
                **kwargs,
            ),
            "history": self.history,
            "profiler": self.profiler,
            "point_interpolation": self.point_interpolation,
            "orbit": self.orbit,
        }
        if kind not in methods:
            raise ValueError(
                f"Use the typed loader method for kind {kind!r}; "
                "it requires format-specific arguments"
            )
        return methods[kind](relative_path, **options)

    def segment_last_times(
        self,
        *,
        relative_path: str | Path = "TStepperDiag.dat",
    ) -> dict[str, float]:
        """Return the final recorded time for every matching segment."""

        return {
            path.parent.parent.name if path.parent.name == "Run" else path.parent.name:
            get_last_time_from_tstepper_diag(path)
            for path in self.paths(relative_path)
        }


@dataclass(frozen=True, slots=True)
class _LegacyRequest:
    pattern: str
    kind: DataKind
    options: dict[str, object]


def _parse_legacy_pattern(pattern: str | Path) -> _LegacyRequest:
    text = str(pattern)
    parts = text.split("@")
    path_pattern = parts[0]

    if "Horizons.h5" in path_pattern and len(parts) == 2:
        return _LegacyRequest(
            path_pattern,
            "horizon",
            {"horizon_name": parts[1]},
        )
    if "PowerDiagnostics" in path_pattern and len(parts) == 4:
        return _LegacyRequest(
            path_pattern,
            "power_diagnostic",
            {
                "file_name": parts[1],
                "field": parts[2],
                "top_number": int(parts[3]),
            },
        )
    if "BondiCceR" in path_pattern and len(parts) == 2:
        return _LegacyRequest(
            path_pattern,
            "worldtube",
            {"variable": parts[1]},
        )
    if "_FiniteRadii_CodeUnits" in path_pattern and len(parts) == 2:
        return _LegacyRequest(
            path_pattern,
            "finite_radius",
            {"radius": parts[1]},
        )
    if len(parts) == 2 and parts[1] == "ALL" and "Int_" in path_pattern:
        return _LegacyRequest(
            path_pattern,
            "point_interpolation",
            {"load_all": True},
        )
    if len(parts) != 1:
        raise ValueError(f"Unrecognized legacy data pattern: {text!r}")

    if "Coefs.dat" in path_pattern:
        kind: DataKind = "unequal_dat"
    elif "Hist-" in path_pattern:
        kind = "history"
    elif "Profiler" in path_pattern:
        kind = "profiler"
    elif "OrbitDiagnostics" in path_pattern:
        kind = "orbit"
    elif "Int_" in path_pattern:
        kind = "point_interpolation"
    else:
        kind = "dat"
    return _LegacyRequest(path_pattern, kind, {})


def _legacy_paths(request: _LegacyRequest) -> list[Path]:
    paths = sorted(
        (
            Path(match)
            for match in glob.iglob(request.pattern, recursive=True)
            if Path(match).is_file() and not _is_failed_path(Path(match))
        ),
        key=natural_sort_key,
    )
    if paths:
        return paths
    if request.kind == "worldtube":
        available = get_worldtube_extraction_radii(
            Path(request.pattern.split("BondiCceR", 1)[0])
        )
        raise FileNotFoundError(
            f"No files matched {request.pattern!r}; "
            f"available extraction radii: {available}"
        )
    raise FileNotFoundError(f"No files matched {request.pattern!r}")


def _read_legacy_path(request: _LegacyRequest, path: Path) -> pd.DataFrame:
    if request.kind == "dat":
        return read_dat_file(path)
    if request.kind == "unequal_dat":
        return read_dat_file_uneq_cols(path)
    if request.kind == "history":
        return hist_files_to_dataframe(path)
    if request.kind == "horizon":
        frame = read_horizonh5(path, str(request.options["horizon_name"]))
        if frame is None:
            raise KeyError(
                f"{request.options['horizon_name']} was not found in {path}"
            )
        return frame
    if request.kind == "profiler":
        return read_profiler(path)
    if request.kind == "power_diagnostic":
        return PowerDiagnosticsLoader([path], overlap="preserve").diagnostic(
            str(request.options["file_name"]),
            int(request.options["top_number"]),
            str(request.options["field"]),
        )
    if request.kind == "worldtube":
        return read_worldtube_data(path, str(request.options["variable"]))
    if request.kind == "finite_radius":
        return read_finite_radius_quantities(path, str(request.options["radius"]))
    if request.kind == "point_interpolation":
        return read_point_interpolation_file(
            path,
            load_all=bool(request.options.get("load_all", False)),
        )
    if request.kind == "orbit":
        return read_orbit_diagnostics(path)
    raise ValueError(f"Unsupported legacy request kind: {request.kind}")


def read_dat_file_across_AA(
    file_patterns: str | Path | Sequence[str | Path],
) -> pd.DataFrame:
    """Compatibility facade for the notebook's encoded path API.

    Unlike the notebook implementation, each pattern retains its own format
    options, so heterogeneous pattern lists cannot accidentally reuse the final
    pattern's HDF5 key or extraction radius.
    """

    if isinstance(file_patterns, (str, Path)):
        file_patterns = (file_patterns,)

    frames: list[pd.DataFrame] = []
    loaded_point_families: set[tuple[Path, str]] = set()
    for pattern in file_patterns:
        request = _parse_legacy_pattern(pattern)
        for path in _legacy_paths(request):
            if request.options.get("load_all"):
                variable = path.stem.removeprefix("Int_").split("_", 1)[0]
                family = (path.parent, variable)
                if family in loaded_point_families:
                    continue
                loaded_point_families.add(family)
            frames.append(_read_legacy_path(request, path))
    return combine_frames(frames, overlap="preserve")


def load_data_from_levs(
    runs_path: Mapping[str, str | Path | Sequence[str | Path]],
    data_file_path: str | Path,
    *,
    cache_folder: str | Path | None = None,
    reload_cache: bool = False,
) -> tuple[pd.Index, dict[str, pd.DataFrame]]:
    """Load one diagnostic for several runs, with optional JSON caching.

    A configured cache is reused until ``reload_cache=True``. Cache identity
    includes the ordered run mapping and diagnostic path. Missing, malformed,
    or incompatible cache files are warned about and rebuilt from the source
    data.
    """

    normalized_runs: list[tuple[str, tuple[str, ...]]] = []
    for run_name, run_roots in runs_path.items():
        if isinstance(run_roots, (str, Path)):
            run_roots = (run_roots,)
        normalized_runs.append(
            (str(run_name), tuple(str(run_root) for run_root in run_roots))
        )

    request = make_cache_request(normalized_runs, str(data_file_path))
    target_cache: Path | None = None
    if cache_folder is not None:
        cache_directory = Path(cache_folder)
        cache_directory.mkdir(parents=True, exist_ok=True)
        target_cache = cache_path(cache_directory, request)
        if target_cache.is_file() and not reload_cache:
            try:
                cached_data = read_cache(target_cache, request)
            except (
                CacheFormatError,
                IndexError,
                KeyError,
                OSError,
                TypeError,
                ValueError,
            ) as error:
                warnings.warn(
                    f"Could not use cache {target_cache}; loading source data "
                    f"and rebuilding it ({error})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                last_run = next(reversed(cached_data)) if cached_data else None
                columns = (
                    cached_data[last_run].columns
                    if last_run is not None
                    else pd.Index([])
                )
                return columns, cached_data
        elif not reload_cache:
            warnings.warn(
                f"Cache {target_cache} is missing; loading source data and "
                "creating it",
                RuntimeWarning,
                stacklevel=2,
            )

    data: dict[str, pd.DataFrame] = {}
    columns = pd.Index([])
    for run_name, run_roots in normalized_runs:
        patterns = [_join_pattern(root, data_file_path) for root in run_roots]
        data[run_name] = read_dat_file_across_AA(patterns)
        columns = data[run_name].columns
    if target_cache is not None:
        write_cache(target_cache, request, data)
    return columns, data


def read_AH_files(ev_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two inspiral apparent-horizon DAT files."""

    root = str(ev_path).rstrip("/")
    return (
        read_dat_file_across_AA(f"{root}/Run/ApparentHorizons/AhA.dat"),
        read_dat_file_across_AA(f"{root}/Run/ApparentHorizons/AhB.dat"),
    )


def get_segment_vs_last_step_dict(
    ev_path: str | Path,
    *,
    include_missing: bool = False,
) -> dict[str, float | None]:
    """Return each segment's final TStepperDiag time."""

    result: dict[str, float | None] = {}
    for segment in sorted(Path(ev_path).iterdir(), key=natural_sort_key):
        if not segment.is_dir():
            continue
        path = segment / "Run/TStepperDiag.dat"
        if path.is_file():
            result[segment.name] = get_last_time_from_tstepper_diag(path)
        elif include_missing:
            result[segment.name] = None
    return result


__all__ = [
    "RunDataLoader",
    "RunSource",
    "all_words_upto",
    "get_segment_vs_last_step_dict",
    "load_data_from_levs",
    "read_AH_files",
    "read_dat_file_across_AA",
]
