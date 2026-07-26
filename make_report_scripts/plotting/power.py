"""Plots for power spectra and spectral coefficients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..io._util import TIME_COLUMN, natural_sort_key
from ..io.power import PowerSpectrumCube, sort_by_coefs_numbers


def _coefficient_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in sort_by_coefs_numbers(frame.columns.to_list())
        if column.startswith("coef")
    ]


def _log10(values):
    array = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(array > 0, np.log10(array), np.nan)


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    midpoints = (values[:-1] + values[1:]) / 2
    return np.concatenate(
        (
            [values[0] - (midpoints[0] - values[0])],
            midpoints,
            [values[-1] + (values[-1] - midpoints[-1])],
        )
    )


def plot_power_heatmap(
    data: pd.DataFrame | PowerSpectrumCube,
    coefficient: int | None = None,
    *,
    ax: Axes | None = None,
    time_range: tuple[float | None, float | None] | None = None,
    log10: bool = True,
    cmap: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = True,
    title: str | None = None,
):
    """Plot one coefficient across time and ordered subdomains."""

    if isinstance(data, PowerSpectrumCube):
        if coefficient is None:
            raise ValueError("coefficient is required for a PowerSpectrumCube")
        frame = data.coefficient(coefficient)
    else:
        frame = data.copy()

    if TIME_COLUMN not in frame:
        raise KeyError(TIME_COLUMN)
    minimum, maximum = time_range or (None, None)
    if minimum is not None:
        frame = frame[frame[TIME_COLUMN] >= minimum]
    if maximum is not None:
        frame = frame[frame[TIME_COLUMN] <= maximum]
    frame = frame.sort_values(TIME_COLUMN, kind="stable")
    if frame.empty:
        raise ValueError("No samples remain in the requested time range")

    subdomains = sorted(
        (column for column in frame.columns if column != TIME_COLUMN),
        key=natural_sort_key,
    )
    if not subdomains:
        raise ValueError("The power frame has no subdomain columns")
    values = frame[subdomains].to_numpy(dtype=float)
    plotted = _log10(values) if log10 else values
    times = frame[TIME_COLUMN].to_numpy(dtype=float)

    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(
        np.arange(len(subdomains) + 1),
        _coordinate_edges(times),
        plotted,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(subdomains)) + 0.5, subdomains, rotation=90)
    ax.set_ylabel(TIME_COLUMN)
    label = f"log10(coef{coefficient})" if log10 else f"coef{coefficient}"
    if coefficient is None:
        label = "log10(power)" if log10 else "power"
    if title is not None:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=label)
    return ax, mesh


def plot_power_spectrum(
    frame: pd.DataFrame,
    time: float,
    *,
    ax: Axes | None = None,
    selection: str = "nearest",
    log_scale: bool = True,
    label: str | None = None,
    **plot_kwargs,
) -> tuple[Axes, float]:
    """Plot coefficient power at the sample nearest to or after a time."""

    if TIME_COLUMN not in frame:
        raise KeyError(TIME_COLUMN)
    times = frame[TIME_COLUMN].to_numpy(dtype=float)
    if selection == "nearest":
        row_index = int(np.nanargmin(np.abs(times - time)))
    elif selection == "after":
        candidates = np.flatnonzero(times >= time)
        if not len(candidates):
            raise ValueError(f"No sample at or after t={time}")
        row_index = int(candidates[0])
    else:
        raise ValueError("selection must be 'nearest' or 'after'")

    columns = _coefficient_columns(frame)
    coefficients = [int(column.removeprefix("coef")) for column in columns]
    values = frame.iloc[row_index][columns].to_numpy(dtype=float)
    actual_time = float(times[row_index])
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(
        coefficients,
        values,
        marker="o",
        label=label or f"t={actual_time:g}",
        **plot_kwargs,
    )
    ax.set_xlabel("coefficient")
    ax.set_ylabel("power")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    return ax, actual_time


def _plot_topology_axis(
    ax: Axes,
    spectra_by_run: Mapping[str, Mapping[str, pd.DataFrame]],
    topology: str,
    *,
    field: str,
    time_range: tuple[float | None, float | None],
    coefficient_range: tuple[int | None, int | None],
    log10: bool,
    line_styles: Sequence[str],
) -> None:
    minimum_time, maximum_time = time_range
    minimum_coefficient, maximum_coefficient = coefficient_range
    for run_index, (run_name, spectra) in enumerate(spectra_by_run.items()):
        if topology not in spectra:
            continue
        frame = spectra[topology].sort_values(TIME_COLUMN, kind="stable")
        if minimum_time is not None:
            frame = frame[frame[TIME_COLUMN] >= minimum_time]
        if maximum_time is not None:
            frame = frame[frame[TIME_COLUMN] <= maximum_time]

        for column in _coefficient_columns(frame):
            number = int(column.removeprefix("coef"))
            if minimum_coefficient is not None and number < minimum_coefficient:
                continue
            if maximum_coefficient is not None and number > maximum_coefficient:
                continue
            values = _log10(frame[column]) if log10 else frame[column]
            ax.plot(
                frame[TIME_COLUMN],
                values,
                label=f"{run_name}: {number}",
                linestyle=line_styles[run_index % len(line_styles)],
            )
    ax.set_title(topology)
    ax.set_xlabel(TIME_COLUMN)
    ax.set_ylabel(f"{'log10 ' if log10 else ''}Power {field}")
    ax.grid(False)


def plot_power_topologies(
    spectra_by_run: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    field: str,
    time_range: tuple[float | None, float | None] = (None, None),
    coefficient_range: tuple[int | None, int | None] = (None, None),
    log10: bool = True,
    line_styles: Sequence[str] = ("-", ":", "--", "-."),
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot every available topology for one or more runs."""

    if not spectra_by_run:
        raise ValueError("spectra_by_run must not be empty")
    topologies = sorted(
        {
            topology
            for spectra in spectra_by_run.values()
            for topology in spectra
        },
        key=natural_sort_key,
    )
    if not topologies:
        raise ValueError("No topologies were supplied")
    fig, axes = plt.subplots(
        1,
        len(topologies),
        squeeze=False,
        figsize=figsize or (5 * len(topologies), 5),
    )
    for axis, topology in zip(axes[0], topologies):
        _plot_topology_axis(
            axis,
            spectra_by_run,
            topology,
            field=field,
            time_range=time_range,
            coefficient_range=coefficient_range,
            log10=log10,
            line_styles=line_styles,
        )
        if legend:
            axis.legend(ncols=2, fontsize="small")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300)
    return fig, axes


def plot_power_field_comparison(
    spectra_by_field: Mapping[
        str,
        Mapping[str, Mapping[str, pd.DataFrame]],
    ],
    *,
    time_range: tuple[float | None, float | None] = (None, None),
    coefficient_range: tuple[int | None, int | None] = (None, None),
    log10: bool = True,
    line_styles: Sequence[str] = ("-", ":", "--", "-."),
    title: str | None = None,
    legend: bool = True,
    save_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray]:
    """Compare fields such as kappa and psi using topology-by-field axes."""

    fields = list(spectra_by_field)
    topologies = sorted(
        {
            topology
            for spectra_by_run in spectra_by_field.values()
            for spectra in spectra_by_run.values()
            for topology in spectra
        },
        key=natural_sort_key,
    )
    if not fields or not topologies:
        raise ValueError("At least one field and topology are required")

    fig, axes = plt.subplots(
        len(topologies),
        len(fields),
        squeeze=False,
        figsize=(5 * len(fields), 4 * len(topologies)),
    )
    for row, topology in enumerate(topologies):
        for column, field in enumerate(fields):
            axis = axes[row, column]
            _plot_topology_axis(
                axis,
                spectra_by_field[field],
                topology,
                field=field,
                time_range=time_range,
                coefficient_range=coefficient_range,
                log10=log10,
                line_styles=line_styles,
            )
            if legend:
                axis.legend(ncols=2, fontsize="x-small")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300)
    return fig, axes


def plot_all_tops(
    top_data_dict,
    subdomain,
    psi_or_kappa,
    save_folder=None,
    *,
    time_range=(None, None),
    coefficient_range=(None, None),
):
    """Compatibility wrapper for the power notebook's topology plot."""

    save_path = None
    if save_folder is not None:
        save_path = Path(save_folder) / f"{subdomain}_power_{psi_or_kappa}.png"
    return plot_power_topologies(
        top_data_dict,
        field=psi_or_kappa,
        time_range=time_range,
        coefficient_range=coefficient_range,
        title=subdomain,
        save_path=save_path,
    )


def plot_all_tops_both(
    top_data_dict_kappa,
    top_data_dict_psi,
    subdomain,
    save_folder=None,
    *,
    time_range=(None, None),
    coefficient_range=(None, None),
):
    """Compatibility wrapper comparing kappa and psi topology spectra."""

    save_path = None
    if save_folder is not None:
        save_path = Path(save_folder) / f"{subdomain}_power_kappa_psi.png"
    return plot_power_field_comparison(
        {
            "kappa": top_data_dict_kappa,
            "psi": top_data_dict_psi,
        },
        time_range=time_range,
        coefficient_range=coefficient_range,
        title=subdomain,
        save_path=save_path,
    )


__all__ = [
    "plot_all_tops",
    "plot_all_tops_both",
    "plot_power_field_comparison",
    "plot_power_heatmap",
    "plot_power_spectrum",
    "plot_power_topologies",
]
