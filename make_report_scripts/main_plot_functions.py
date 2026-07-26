"""Compatibility facade for notebook-era loading and plotting functions.

New code should import loaders from :mod:`make_report_scripts.io` and plotting
functions from :mod:`make_report_scripts.plotting`.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .io import (
    load_data_from_levs,
    read_AH_files,
    read_dat_file_across_AA,
    read_horizonh5,
)
from .plotting.diagnostics import (
    plot_damping_times,
    plot_min_grid_spacing as _plot_min_grid_spacing,
    save_column_plots,
)
from .plotting.timeseries import (
    legacy_plot_graph_for_runs,
    legacy_plot_graph_for_runs_wrapper,
)


def add_diff_columns(runs_data_dict, x_axis, y_axis, diff_base):
    """Add interpolated difference columns in place for compatibility."""

    if diff_base not in runs_data_dict:
        raise KeyError(f"{diff_base!r} is not present")
    reference = runs_data_dict[diff_base].sort_values(x_axis, kind="stable")
    unique_x, indices = np.unique(reference[x_axis], return_index=True)
    if len(unique_x) < 2:
        raise ValueError("The difference reference needs two unique x values")
    interpolator = CubicSpline(
        unique_x,
        reference[y_axis].to_numpy()[indices],
        extrapolate=False,
    )
    for run_name, frame in runs_data_dict.items():
        if run_name == diff_base:
            continue
        difference = frame[y_axis] - interpolator(frame[x_axis])
        frame[f"diff_{y_axis}"] = difference
        frame[f"diff_abs_{y_axis}"] = np.abs(difference)


def plot_graph_for_runs(*args, **kwargs):
    """Compatibility wrapper for the original notebook plotting function."""

    return legacy_plot_graph_for_runs(*args, **kwargs)


def plot_graph_for_runs_wrapper(*args, **kwargs):
    """Compatibility wrapper for plotting several y columns."""

    return legacy_plot_graph_for_runs_wrapper(*args, **kwargs)


def find_file(pattern):
    """Return the first naturally sorted match or raise FileNotFoundError."""

    matches = sorted(glob.glob(str(pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def plots_for_a_folder(things_to_plot, plot_folder_path, data_folder_path):
    """Load configured diagnostics and save one plot per requested y column."""

    output_root = Path(plot_folder_path)
    saved = []
    for plot_info in things_to_plot:
        file_name = plot_info["file_name"]
        columns = list(plot_info["columns"])
        if len(columns) < 2:
            raise ValueError(f"{file_name} needs an x column and at least one y column")
        frame = read_dat_file_across_AA(
            f"{str(data_folder_path).rstrip('/')}/**/{file_name}"
        )
        saved.extend(
            save_column_plots(
                frame,
                output_root / Path(file_name).stem,
                x_column=columns[0],
                columns=columns[1:],
            )
        )
    return saved


def is_the_current_run_going_on(run_folder):
    """Return false once any TerminationReason.txt is present."""

    root = Path(run_folder)
    return not any(root.glob("**/TerminationReason.txt"))


def plot_min_grid_spacing(runs_data_dict, **kwargs):
    """Compatibility wrapper returning the explicit Matplotlib axes."""

    return _plot_min_grid_spacing(runs_data_dict, **kwargs)


def plot_GrAdjustSubChunksToDampingTimes(runs_data_dict, **kwargs):
    """Compatibility wrapper for the notebook damping-time plot."""

    if len(runs_data_dict) != 1:
        raise ValueError("Exactly one run is required")
    run_name, frame = next(iter(runs_data_dict.items()))
    kwargs.setdefault("title", run_name)
    return plot_damping_times(frame, **kwargs)


__all__ = [
    "add_diff_columns",
    "find_file",
    "is_the_current_run_going_on",
    "load_data_from_levs",
    "plot_GrAdjustSubChunksToDampingTimes",
    "plot_graph_for_runs",
    "plot_graph_for_runs_wrapper",
    "plot_min_grid_spacing",
    "plots_for_a_folder",
    "read_AH_files",
    "read_dat_file_across_AA",
    "read_horizonh5",
]
