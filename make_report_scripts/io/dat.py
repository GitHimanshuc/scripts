"""Readers for SpEC text diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ._util import TIME_COLUMN

_COLUMN_DECLARATION = re.compile(
    r"\[\s*\d+\s*\]\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^#\s]+))"
)


def read_dat_file(file_name: str | Path) -> pd.DataFrame:
    """Read a SpEC ``.dat`` file whose columns are declared in comments."""

    path = Path(file_name)
    column_names: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.lstrip().startswith("#"):
                break
            match = _COLUMN_DECLARATION.search(line)
            if match:
                column_names.append(next(value for value in match.groups() if value))

    if not column_names:
        raise ValueError(f"No column declarations found in {path}")
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=column_names,
        engine="python",
    )


def read_dat_file_single_bh(file_name: str | Path) -> pd.DataFrame:
    """Read a variable-width single-BH diagnostic without a column legend."""

    path = Path(file_name)
    with path.open(encoding="utf-8") as stream:
        widths = [
            len(line.split())
            for line in stream
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not widths:
        raise ValueError(f"No data rows found in {path}")
    maximum_columns = max(widths)
    columns = [TIME_COLUMN, *[str(index) for index in range(maximum_columns - 1)]]
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=columns,
        engine="python",
    )


def read_dat_file_uneq_cols(file_name: str | Path) -> pd.DataFrame:
    """Read coefficient files such as ``AhACoefs.dat`` with growing rows."""

    path = Path(file_name)
    with path.open(encoding="utf-8") as stream:
        widths = [
            len(line.split())
            for line in stream
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not widths:
        raise ValueError(f"No data rows found in {path}")

    maximum_columns = max(widths)
    coefficient_count = maximum_columns - 4
    maximum_l = int(np.sqrt(coefficient_count) - 1)
    if (maximum_l + 1) ** 2 != coefficient_count:
        raise ValueError(
            f"{path} has {coefficient_count} coefficient columns, "
            "which is not a complete set of spherical-harmonic modes"
        )

    coefficient_names = [
        f"{ell},{mode}"
        for ell in range(maximum_l + 1)
        for mode in range(-ell, ell + 1)
    ]
    columns = [TIME_COLUMN, "Center-x", "Center-y", "Center-z", *coefficient_names]
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=columns,
        engine="python",
    )


def hist_files_to_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Parse a ``Hist-*`` file into one row per history entry."""

    def parse_line(line: str) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for variable, raw_value in re.findall(r"([^;=\s]+)=\s*([^;]+)", line):
            value = raw_value.strip()
            if "ResizeTheseSubdomains" not in variable:
                parsed[variable] = (
                    float(value) if re.fullmatch(r"[\d.eE+-]+", value) else value
                )
                continue

            entries = value.strip().removeprefix("(").removesuffix(")").split("),")
            for entry in entries:
                name, separator, dimensions = entry.strip().partition("(")
                if not separator:
                    raise ValueError(f"Cannot parse subdomain history entry: {entry!r}")
                radius, ell, mode = dimensions.rstrip(")").split(",")
                parsed[f"{name}_R"] = int(radius)
                parsed[f"{name}_L"] = int(ell)
                parsed[f"{name}_M"] = int(mode)
        return parsed

    path = Path(file_path)
    with path.open(encoding="utf-8") as stream:
        return pd.DataFrame(
            parse_line(line.strip()) for line in stream if line.strip()
        )


def read_point_interpolation_file(
    file_path: str | Path,
    load_all: bool = False,
) -> pd.DataFrame:
    """Read one or all point-interpolation components sharing a variable."""

    path = Path(file_path)

    def column_names(dat_path: Path) -> list[str]:
        variable_name = dat_path.stem.removeprefix("Int_")
        with dat_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("# Points"):
                    points = line.partition("=")[2].strip().split(", ")
                    return [f"{variable_name}_{point.strip()}" for point in points]
        raise ValueError(f"No '# Points =' declaration found in {dat_path}")

    paths = [path]
    if load_all:
        stem_parts = path.stem.removeprefix("Int_").split("_")
        if not stem_parts:
            raise ValueError(f"Cannot determine interpolation variable from {path}")
        paths = sorted(path.parent.glob(f"Int_{stem_parts[0]}_*.dat"))
        if not paths:
            raise FileNotFoundError(f"No interpolation files matched {path.parent}")

    frames = [
        pd.read_csv(
            dat_path,
            sep=r"\s+",
            comment="#",
            names=[TIME_COLUMN, *column_names(dat_path)],
            engine="python",
        )
        for dat_path in paths
    ]
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=TIME_COLUMN, how="outer", sort=True)
    return result.sort_values(TIME_COLUMN, kind="stable").reset_index(drop=True)


def read_last_line(path: str | Path) -> str:
    """Return the final non-empty text line, including for one-line files."""

    file_path = Path(path)
    with file_path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        if position == 0:
            raise ValueError(f"{file_path} is empty")
        buffer = bytearray()
        while position:
            position -= 1
            stream.seek(position)
            byte = stream.read(1)
            if byte == b"\n" and buffer:
                break
            if byte not in {b"\n", b"\r"}:
                buffer.extend(byte)
        return bytes(reversed(buffer)).decode().rstrip()


def get_last_time_from_tstepper_diag(path: str | Path) -> float:
    """Read the time value from the final ``TStepperDiag.dat`` row."""

    return float(read_last_line(path).split()[0])

