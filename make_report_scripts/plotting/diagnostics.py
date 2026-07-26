"""Specialized plots shared by report-generation notebooks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..io._util import TIME_COLUMN


def _time_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        if requested not in frame:
            raise KeyError(requested)
        return requested
    for candidate in (TIME_COLUMN, "t", "time", "Time"):
        if candidate in frame:
            return candidate
    raise KeyError("No time column was found")


def plot_min_grid_spacing(
    runs: Mapping[str, pd.DataFrame],
    *,
    ax: Axes | None = None,
    time_column: str | None = None,
    domain_columns: Sequence[str] | None = None,
) -> Axes:
    """Plot the minimum grid spacing across domains for every run."""

    if not runs:
        raise ValueError("runs must not be empty")
    if ax is None:
        _, ax = plt.subplots()
    for run_name, frame in runs.items():
        time_name = _time_column(frame, time_column)
        columns = list(domain_columns) if domain_columns is not None else [
            column
            for column in frame.columns
            if column != time_name and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not columns:
            raise ValueError(f"{run_name} has no numeric domain columns")
        ax.plot(
            frame[time_name],
            frame[columns].min(axis="columns"),
            label=run_name,
        )
    ax.set_xlabel(TIME_COLUMN)
    ax.set_ylabel("minimum grid spacing")
    ax.set_title("Minimum grid spacing across all domains")
    ax.legend()
    return ax


def plot_damping_times(
    frame: pd.DataFrame,
    *,
    ax: Axes | None = None,
    time_column: str | None = None,
    column_pattern: str = "Tdamp",
    title: str | None = None,
) -> Axes:
    """Plot all damping-time columns and their row-wise minimum."""

    time_name = _time_column(frame, time_column)
    columns = [column for column in frame if column_pattern in str(column)]
    if not columns:
        raise ValueError(f"No columns contain {column_pattern!r}")
    if ax is None:
        _, ax = plt.subplots()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(columns)))
    for index, (column, color) in enumerate(zip(columns, colors)):
        ax.plot(
            frame[time_name],
            frame[column],
            label=column,
            color=color,
            linestyle="-" if index % 2 == 0 else "--",
        )
    ax.plot(
        frame[time_name],
        frame[columns].min(axis="columns"),
        label="minimum damping time",
        linewidth=3,
        linestyle=":",
        color="red",
    )
    ax.set_xlabel(TIME_COLUMN)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_column_grid(
    frame: pd.DataFrame,
    *,
    x_column: str = TIME_COLUMN,
    columns: Sequence[str] | None = None,
    ncols: int = 3,
    time_range: tuple[float | None, float | None] = (None, None),
    take_absolute: bool = False,
    yscale: str | None = None,
    sharex: bool = True,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot every requested y column on its own small-multiple axis."""

    if x_column not in frame:
        raise KeyError(x_column)
    columns = list(columns) if columns is not None else [
        column for column in frame if column != x_column
    ]
    if not columns:
        raise ValueError("No y columns were selected")
    if ncols < 1:
        raise ValueError("ncols must be positive")

    minimum, maximum = time_range
    selected = frame
    if minimum is not None:
        selected = selected[selected[x_column] >= minimum]
    if maximum is not None:
        selected = selected[selected[x_column] <= maximum]

    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        sharex=sharex,
        figsize=figsize or (5 * ncols, 3.5 * nrows),
    )
    for axis, column in zip(axes.flat, columns):
        values = np.abs(selected[column]) if take_absolute else selected[column]
        axis.plot(selected[x_column], values)
        axis.set_title(str(column))
        if yscale is not None:
            axis.set_yscale(yscale)
        axis.grid(False)
    for axis in axes.flat[len(columns) :]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel(x_column)
    fig.tight_layout()
    return fig, axes


def save_column_plots(
    frame: pd.DataFrame,
    output_directory: str | Path,
    *,
    x_column: str = TIME_COLUMN,
    columns: Sequence[str] | None = None,
    time_range: tuple[float | None, float | None] = (None, None),
    take_absolute: bool = False,
    yscale: str | None = None,
    dpi: int = 200,
) -> list[Path]:
    """Save one self-contained figure for every selected DataFrame column."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    columns = list(columns) if columns is not None else [
        column for column in frame if column != x_column
    ]
    minimum, maximum = time_range
    selected = frame
    if minimum is not None:
        selected = selected[selected[x_column] >= minimum]
    if maximum is not None:
        selected = selected[selected[x_column] <= maximum]

    saved: list[Path] = []
    for column in columns:
        fig, ax = plt.subplots()
        values = np.abs(selected[column]) if take_absolute else selected[column]
        ax.plot(selected[x_column], values)
        ax.set_xlabel(x_column)
        ax.set_ylabel(str(column))
        if yscale is not None:
            ax.set_yscale(yscale)
        ax.grid(False)
        fig.tight_layout()
        safe_name = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(column)
        )
        path = output / f"{safe_name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)
    return saved


__all__ = [
    "plot_column_grid",
    "plot_damping_times",
    "plot_min_grid_spacing",
    "save_column_plots",
]
