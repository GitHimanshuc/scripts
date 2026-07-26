"""Readers for SpEC profiler diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import pandas as pd

from ._util import natural_sort_key

_STEP = re.compile(r"^Step(.+)[.]dir$")
_PROC = re.compile(r"^Proc(.+)[.]txt$")


def _parse_profile_lines(lines: list[str]) -> tuple[float, dict[tuple[str, str], float | int]]:
    if len(lines) < 7:
        raise ValueError("Profiler entry is too short")
    time = float(lines[0].split("=")[-1].rstrip(" ;\n"))
    header = lines[4]
    positions = [
        header.find("Event"),
        header.find("cum(%)"),
        header.find("exc(%)"),
        header.find("inc(%)"),
    ]
    if any(position < 0 for position in positions):
        raise ValueError("Profiler entry has an unrecognized column header")
    event_end = positions[0] + len("Event")
    cumulative_end = positions[1] + len("cum(%)")
    exclusive_end = positions[2] + len("exc(%)")
    inclusive_end = positions[3] + len("inc(%)")

    values: dict[tuple[str, str], float | int] = {}
    for line in lines[6:-2]:
        if not line.strip():
            continue
        event = line[:event_end].strip()
        values[("cum", event)] = float(line[event_end:cumulative_end].strip())
        values[("exc", event)] = float(line[cumulative_end:exclusive_end].strip())
        values[("inc", event)] = float(line[exclusive_end:inclusive_end].strip())
        values[("N", event)] = int(line[inclusive_end:].strip())
    return time, values


def read_profiler(file_name: str | Path) -> pd.DataFrame:
    """Read an HDF5 profiler archive into a flat DataFrame."""

    rows: list[dict[str, object]] = []
    with h5py.File(file_name, "r") as h5_file:
        for step_name in sorted(h5_file, key=natural_sort_key):
            step_match = _STEP.fullmatch(step_name)
            if not step_match or not isinstance(h5_file[step_name], h5py.Group):
                continue
            group = h5_file[step_name]
            for process_name in sorted(group, key=natural_sort_key):
                process_match = _PROC.fullmatch(process_name)
                if not process_match:
                    continue
                raw = group[process_name][0]
                text = raw.decode() if isinstance(raw, bytes) else str(raw)
                time, values = _parse_profile_lines(text.splitlines())
                row: dict[str, object] = {
                    "t(M)": time,
                    "step": step_match.group(1),
                    "proc": process_match.group(1),
                }
                row.update(
                    {
                        f"{event}_{metric}": value
                        for (metric, event), value in values.items()
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def read_profiler_multiindex(folder_path: str | Path) -> pd.DataFrame:
    """Read an extracted profiler directory with indexed rows and columns."""

    root = Path(folder_path)
    step_directories = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and _STEP.fullmatch(path.name)
        ),
        key=natural_sort_key,
    )
    if not step_directories:
        raise FileNotFoundError(f"No Step*.dir directories found in {root}")

    rows: list[dict[tuple[str, str], float | int]] = []
    row_names: list[tuple[str, float]] = []
    for step_directory in step_directories:
        for process_path in sorted(step_directory.glob("Proc*.txt"), key=natural_sort_key):
            if "Summary" in process_path.name:
                continue
            match = _PROC.fullmatch(process_path.name)
            if not match:
                continue
            lines = process_path.read_text(encoding="utf-8").splitlines()
            time, values = _parse_profile_lines(lines)
            rows.append(values)
            row_names.append((match.group(1), time))

    index = pd.MultiIndex.from_tuples(row_names, names=["proc", "t(M)"])
    frame = pd.DataFrame(rows, index=index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["metric", "process"])
    return frame.sort_index()
