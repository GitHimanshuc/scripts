"""Reusable plotting for one diagnostic across several runs."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.interpolate import CubicSpline

ModificationFunction = Callable[
    [pd.Series, pd.Series, pd.DataFrame, str],
    tuple[Sequence[float], Sequence[float], str],
]


@dataclass(slots=True)
class PreparedSeries:
    """Plot-ready arrays and their effective labels."""

    run_name: str
    x: np.ndarray
    y: np.ndarray
    label: str
    x_name: str
    y_name: str


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("moving_average must be non-negative")
    if window == 1:
        return values
    if window > len(values):
        return np.asarray([], dtype=np.result_type(values, float))
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _interpolated_value(x: np.ndarray, y: np.ndarray, at: float):
    finite = np.isfinite(x) & np.isfinite(y)
    unique_x, indices = np.unique(x[finite], return_index=True)
    if len(unique_x) < 2:
        raise ValueError("At least two finite, unique x values are required")
    if at < unique_x.min() or at > unique_x.max():
        raise ValueError(
            f"Shift time {at} is outside [{unique_x.min()}, {unique_x.max()}]"
        )
    return CubicSpline(unique_x, y[finite][indices], extrapolate=False)(at)


def _reference_values(
    reference: pd.DataFrame,
    x_column: str,
    y_column: str,
    target_x: np.ndarray,
) -> np.ndarray:
    ordered = reference[[x_column, y_column]].dropna().sort_values(x_column)
    unique_x, indices = np.unique(ordered[x_column].to_numpy(), return_index=True)
    unique_y = ordered[y_column].to_numpy()[indices]
    if len(unique_x) < 2:
        raise ValueError("A reference run needs at least two unique x values")
    return CubicSpline(unique_x, unique_y, extrapolate=False)(target_x)


def _accepts_keyword(function: Callable, keyword: str) -> bool:
    """Return whether a callback can accept a keyword argument.

    Some extension-backed callables do not expose an inspectable signature.
    In that case, use the conservative call form without the optional keyword.
    """

    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == keyword
        for parameter in parameters
    )


def prepare_run_series(
    runs: Mapping[str, pd.DataFrame],
    x_column: str,
    y_column: str,
    *,
    time_range: tuple[float | None, float | None] | None = None,
    labels: Mapping[str, str | None] | None = None,
    sort_by: str | None = None,
    moving_average: int = 0,
    reference: str | None = None,
    absolute_difference: bool = False,
    shift_time: float | None = None,
    take_absolute: bool = False,
    modification_function: ModificationFunction | None = None,
    modified_data: MutableMapping[str, pd.DataFrame] | None = None,
) -> list[PreparedSeries]:
    """Prepare run series without mutating the input DataFrames."""

    if not runs:
        raise ValueError("runs must contain at least one DataFrame")
    if reference is not None and reference not in runs:
        raise KeyError(f"Reference run {reference!r} is not present")
    if labels is not None:
        missing_labels = set(runs) - set(labels)
        if missing_labels:
            raise KeyError(f"Missing labels for runs: {sorted(missing_labels)}")

    prepared_frames: dict[str, tuple[pd.DataFrame, str]] = {}
    for run_name, original in runs.items():
        if x_column not in original or y_column not in original:
            missing = [column for column in (x_column, y_column) if column not in original]
            raise KeyError(f"{run_name} is missing columns {missing}")
        frame = original.copy()
        effective_y = y_column
        if modification_function is not None:
            new_x, new_y, effective_y = modification_function(
                frame[x_column],
                frame[y_column],
                frame,
                y_column,
            )
            frame[x_column] = np.asarray(new_x)
            frame[effective_y] = np.asarray(new_y)
            if modified_data is not None:
                modified_data[run_name] = frame[[x_column, effective_y]].copy()
        frame = frame.sort_values(sort_by or x_column, kind="stable")
        prepared_frames[run_name] = (frame, effective_y)

    effective_y_names = {
        effective_y for _, effective_y in prepared_frames.values()
    }
    if len(effective_y_names) != 1:
        # Modification functions should return the same semantic y name.
        raise ValueError("Modification function returned inconsistent y names")

    minimum, maximum = time_range or (None, None)
    result: list[PreparedSeries] = []
    for run_name, (frame, effective_y) in prepared_frames.items():
        if reference is not None and run_name == reference:
            continue

        selection = np.ones(len(frame), dtype=bool)
        if minimum is not None:
            selection &= frame[x_column].to_numpy() >= minimum
        if maximum is not None:
            selection &= frame[x_column].to_numpy() <= maximum
        selected = frame.loc[selection, [x_column, effective_y]].dropna(
            subset=[x_column]
        )
        x = selected[x_column].to_numpy()
        y = selected[effective_y].to_numpy()
        output_y_name = effective_y

        if reference is not None:
            reference_frame, reference_y = prepared_frames[reference]
            y = y - _reference_values(reference_frame, x_column, reference_y, x)
            output_y_name = (
                f"abs_diff_{effective_y}"
                if absolute_difference
                else f"diff_{effective_y}"
            )
            if absolute_difference:
                y = np.abs(y)

        if shift_time is not None and len(x):
            y = y - _interpolated_value(x, y, shift_time)
        if take_absolute:
            y = np.abs(y)
            output_y_name = f"abs_{output_y_name}"
        if moving_average:
            y = _moving_average(np.asarray(y), moving_average)
            x = x[moving_average - 1 :]

        finite_or_nan = np.asarray(y)
        if len(x) == 0 or np.all(pd.isna(finite_or_nan)):
            continue
        label = run_name
        if labels is not None and labels[run_name] is not None:
            label = str(labels[run_name])
        result.append(
            PreparedSeries(
                run_name=run_name,
                x=np.asarray(x),
                y=np.asarray(y),
                label=label,
                x_name=x_column,
                y_name=output_y_name,
            )
        )
    return result


def plot_runs(
    runs: Mapping[str, pd.DataFrame],
    x_column: str,
    y_column: str,
    *,
    ax: Axes | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    labels: Mapping[str, str | None] | None = None,
    sort_by: str | None = None,
    moving_average: int = 0,
    reference: str | None = None,
    absolute_difference: bool = False,
    shift_time: float | None = None,
    take_absolute: bool = False,
    modification_function: ModificationFunction | None = None,
    modified_data: MutableMapping[str, pd.DataFrame] | None = None,
    line_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    title: str | None = None,
    legend: bool = True,
) -> Axes:
    """Plot a diagnostic across runs and return the Matplotlib axes."""

    if ax is None:
        _, ax = plt.subplots()
    series = prepare_run_series(
        runs,
        x_column,
        y_column,
        time_range=time_range,
        labels=labels,
        sort_by=sort_by,
        moving_average=moving_average,
        reference=reference,
        absolute_difference=absolute_difference,
        shift_time=shift_time,
        take_absolute=take_absolute,
        modification_function=modification_function,
        modified_data=modified_data,
    )
    for zorder, item in enumerate(series, start=2):
        kwargs = dict((line_kwargs or {}).get(item.run_name, {}))
        kwargs.setdefault("zorder", zorder)
        ax.plot(item.x, item.y, label=item.label, **kwargs)

    effective_y = series[0].y_name if series else y_column
    ax.set_xlabel(x_column)
    ax.set_ylabel(effective_y)
    if shift_time is not None:
        ax.axvline(shift_time, linestyle=":", color="red")
    if title is None:
        title = f'"{effective_y}" vs "{x_column}"'
        if moving_average:
            title += f" (moving average {moving_average})"
    ax.set_title(title)
    if legend and series:
        ax.legend()
    return ax


def plot_frame_columns(
    frame: pd.DataFrame,
    *,
    x_column: str = "t(M)",
    columns: Sequence[str] | None = None,
    ax: Axes | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    take_absolute: bool = False,
    legend: bool = True,
    **plot_kwargs,
) -> Axes:
    """Plot selected DataFrame columns against a common x coordinate."""

    if ax is None:
        _, ax = plt.subplots()
    if x_column not in frame:
        raise KeyError(x_column)
    columns = list(columns) if columns is not None else [
        column for column in frame.columns if column != x_column
    ]
    minimum, maximum = time_range or (None, None)
    selected = frame
    if minimum is not None:
        selected = selected[selected[x_column] >= minimum]
    if maximum is not None:
        selected = selected[selected[x_column] <= maximum]
    for column in columns:
        values = np.abs(selected[column]) if take_absolute else selected[column]
        ax.plot(selected[x_column], values, label=column, **plot_kwargs)
    ax.set_xlabel(x_column)
    if legend and columns:
        ax.legend()
    return ax


def _safe_plot_name(
    x_column: str,
    y_column: str,
    minimum: float,
    maximum: float,
    run_names: Sequence[str],
    *,
    moving_average: int = 0,
    reference: str | None = None,
) -> str:
    sanitize = lambda value: re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value))
    name = (
        f"{sanitize(y_column)}_vs_{sanitize(x_column)}"
        f"_minT={minimum}_maxT={maximum}"
    )
    if moving_average:
        name += f"_moving_avg_len={moving_average}"
    if reference:
        name += f"_diff_base={sanitize(reference)}"
    suffix = "__".join(sanitize(run) for run in run_names)
    if suffix:
        name += f"__{suffix}"
    return name[:240]


def legacy_plot_graph_for_runs(
    runs_data_dict_original,
    x_axis,
    y_axis,
    minT,
    maxT,
    legend_dict=None,
    save_path=None,
    moving_avg_len=0,
    plot_fun=None,
    sort_by=None,
    diff_base=None,
    modified_data_dict_storage=None,
    title=None,
    append_to_title="",
    plot_abs_diff=False,
    constant_shift_val_time=None,
    modification_function=None,
    take_abs=False,
):
    """Compatibility implementation for the original notebook signature."""

    ax = plt.gca()
    prepared = prepare_run_series(
        runs_data_dict_original,
        x_axis,
        y_axis,
        time_range=(minT, maxT),
        labels=legend_dict,
        sort_by=sort_by,
        moving_average=moving_avg_len,
        reference=diff_base,
        absolute_difference=plot_abs_diff,
        shift_time=constant_shift_val_time,
        take_absolute=take_abs,
        modification_function=modification_function,
        modified_data=modified_data_dict_storage,
    )

    if plot_fun is None:
        plot_fun = lambda x, y, label, **kwargs: ax.plot(
            x,
            y,
            label=label,
            **kwargs,
        )
    accepts_zorder = _accepts_keyword(plot_fun, "zorder")
    for zorder, item in enumerate(prepared, start=2):
        if accepts_zorder:
            plot_fun(item.x, item.y, item.label, zorder=zorder)
        else:
            plot_fun(item.x, item.y, item.label)
        if constant_shift_val_time is not None:
            ax.axhline(item.y[-1], linestyle=":")
            ax.annotate(
                f"{item.y[-1]:.2e}",
                xy=(item.x[-1], item.y[-1]),
                xytext=(4, 4),
                textcoords="offset points",
            )

    effective_y = prepared[0].y_name if prepared else y_axis
    ax.set_xlabel(x_axis)
    ax.set_ylabel(effective_y)
    if constant_shift_val_time is not None:
        ax.axvline(constant_shift_val_time, linestyle=":", color="red")
    if title is None:
        title = f'"{effective_y}" vs "{x_axis}"'
    ax.set_title(title + append_to_title)
    if prepared:
        ax.legend()

    if save_path is not None:
        output_directory = Path(save_path)
        output_directory.mkdir(parents=True, exist_ok=True)
        name = _safe_plot_name(
            x_axis,
            effective_y,
            minT,
            maxT,
            [item.run_name for item in prepared],
            moving_average=moving_avg_len,
            reference=diff_base,
        )
        ax.figure.savefig(output_directory / f"{name}.png")
    return ax


def legacy_plot_graph_for_runs_wrapper(
    runs_data_dict,
    x_axis,
    y_axis_list,
    minT,
    maxT,
    **kwargs,
):
    """Overlay several y columns using the original wrapper signature."""

    y_columns = list(y_axis_list)
    if not y_columns:
        raise ValueError("y_axis_list must not be empty")
    save_path = kwargs.pop("save_path", None)
    caller_labels = kwargs.pop("legend_dict", None)
    for index, y_axis in enumerate(y_columns):
        labels = caller_labels or {
            run_name: f"{run_name}_{y_axis}" for run_name in runs_data_dict
        }
        legacy_plot_graph_for_runs(
            runs_data_dict,
            x_axis,
            y_axis,
            minT,
            maxT,
            legend_dict=labels,
            save_path=save_path if index == len(y_columns) - 1 else None,
            **kwargs,
        )
    return plt.gca()


__all__ = [
    "PreparedSeries",
    "legacy_plot_graph_for_runs",
    "legacy_plot_graph_for_runs_wrapper",
    "plot_frame_columns",
    "plot_runs",
    "prepare_run_series",
]
