"""Reproduce the HighAccuracy1025 constraint plots with cached loading.

Run this script from the directory where cache JSON files and the ``plots``
folder should live. Set ``RELOAD_CACHE`` to ``True`` for one run whenever the
source diagnostics should replace the cached data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make the repository-local package importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from make_report_scripts import load_data_from_levs, plot_runs


CACHE_FOLDER = Path.cwd()
RELOAD_CACHE = False
SAVE_FOLDER = Path("./plots/").resolve()

MIN_TIME = 1205.0
MAX_TIME = 4000.0


L15_MAIN_LEGEND = {
    "high_accuracy_main_L1": "Old Level 1",
    "high_accuracy_main_L2": "Old Level 2",
    "high_accuracy_main_L3": "Old Level 3",
    "high_accuracy_main_L4": "Old Level 4",
    "high_accuracy_main_L5": "Old Level 5",
}

L15_MAIN_RUNS = {
    "high_accuracy_main_L1": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev1_A?/Run/",
    "high_accuracy_main_L2": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev2_A?/Run/",
    "high_accuracy_main_L3": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev3_A?/Run/",
    "high_accuracy_main_L4": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev4_A?/Run/",
    "high_accuracy_main_L5": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev5_A?/Run/",
}

L15_ODE_FIX_LEGEND = {
    "high_accuracy_L1": "Ode Fix Level 1",
    "high_accuracy_L2": "Ode Fix Level 2",
    "high_accuracy_L3": "Ode Fix Level 3",
    "high_accuracy_L4": "Ode Fix Level 4",
    "high_accuracy_L5": "Ode Fix Level 5",
}

L15_ODE_FIX_RUNS = {
    "high_accuracy_L1": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev1_A?/Run/",
    "high_accuracy_L2": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev2_A?/Run/",
    "high_accuracy_L3": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev3_A?/Run/",
    "high_accuracy_L4": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev4_A?/Run/",
    "high_accuracy_L5": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev5_A?/Run/",
}

L16_SET1_LEGEND = {
    "6_set1_L6s1": "Set1 Level 1",
    "6_set1_L6s2": "Set1 Level 2",
    "6_set1_L6s3": "Set1 Level 3",
    "6_set1_L6s4": "Set1 Level 4",
    "6_set1_L6s5": "Set1 Level 5",
    "6_set1_L6s6": "Set1 Level 6",
}

L16_SET1_RUNS = {
    "6_set1_L6s1": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev1_A?/Run/",
    "6_set1_L6s2": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev2_A?/Run/",
    "6_set1_L6s3": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev3_A?/Run/",
    "6_set1_L6s4": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev4_A?/Run/",
    "6_set1_L6s5": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev5_A?/Run/",
    "6_set1_L6s6": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev6_A?/Run/",
}

JOINED_RUNS = {**L15_MAIN_RUNS, **L15_ODE_FIX_RUNS, **L16_SET1_RUNS}
JOINED_LEGEND = {**L15_MAIN_LEGEND, **L15_ODE_FIX_LEGEND, **L16_SET1_LEGEND}

RUN_GROUPS = (
    ("L15_main", L15_MAIN_RUNS, L15_MAIN_LEGEND),
    ("L15_ode_fix", L15_ODE_FIX_RUNS, L15_ODE_FIX_LEGEND),
    ("L16_set1", L16_SET1_RUNS, L16_SET1_LEGEND),
)

PLOTS_BY_DIAGNOSTIC = {
    "ConstraintNorms/GhCe_Linf.dat": (
        ("Linf(GhCe) on SphereA0", "SphereA0_Linf_GhCe"),
        ("Linf(GhCe) on SphereC6", "SphereC6_Linf_GhCe"),
    ),
    "ConstraintNorms/NormalizedGhCe_Linf.dat": (
        ("Linf(NormalizedGhCe) on SphereA0", "SphereA0_Linf_NormalizedGhCe"),
        ("Linf(NormalizedGhCe) on SphereC6", "SphereC6_Linf_NormalizedGhCe"),
    ),
    "ConstraintNorms/GhCe_Norms.dat": (
        ("L2(GhCe)", "L2(GhCe)"),
        ("Linf(GhCe)", "Linf(GhCe)"),
        ("VolLp(GhCe)", "VolLp(GhCe)"),
    ),
    "ConstraintNorms/NormalizedGhCe_Norms.dat": (
        ("L2(NormalizedGhCe)", "L2(NormalizedGhCe)"),
        ("Linf(NormalizedGhCe)", "Linf(NormalizedGhCe)"),
        ("VolLp(NormalizedGhCe)", "VolLp(NormalizedGhCe)"),
    ),
}


def _select_runs(runs_data, run_paths):
    """Return a run-ordered subset of an already loaded diagnostic."""

    return {run_name: runs_data[run_name] for run_name in run_paths}


def save_constraint_plot(
    runs_data,
    legend,
    y_axis,
    save_name,
    *,
    figsize=(5, 4),
):
    """Save one plot using the formatting from the original script."""

    with plt.style.context(["ggplot"]):
        fig, ax = plt.subplots(figsize=figsize)
        plot_runs(
            runs_data,
            "t(M)",
            y_axis,
            ax=ax,
            time_range=(MIN_TIME, np.nextafter(MAX_TIME, -np.inf)),
            labels=legend,
            title="",
        )
        ax.set_yscale("log")
        ax.set_title("")
        ax.set_ylabel(y_axis)
        ax.set_xlabel("t(M)")
        ax.grid(False)
        fig.tight_layout()
        output_path = SAVE_FOLDER / save_name
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    print(f"Saved {output_path}!\n")


def load_diagnostic(runs, data_file_path):
    """Load one diagnostic using this script's shared cache configuration."""

    _, runs_data = load_data_from_levs(
        runs,
        data_file_path,
        cache_folder=CACHE_FOLDER,
        reload_cache=RELOAD_CACHE,
    )
    return runs_data


def make_run_set_plots():
    """Create the 30 plots spanning all three run sets."""

    normalized_linf = None
    for data_file_path, plot_specs in PLOTS_BY_DIAGNOSTIC.items():
        runs_data = load_diagnostic(JOINED_RUNS, data_file_path)
        if data_file_path == "ConstraintNorms/NormalizedGhCe_Linf.dat":
            normalized_linf = runs_data

        for runs_set_name, run_paths, legend in RUN_GROUPS:
            selected_runs = _select_runs(runs_data, run_paths)
            for y_axis, filename_suffix in plot_specs:
                save_constraint_plot(
                    selected_runs,
                    legend,
                    y_axis,
                    f"{runs_set_name}_{filename_suffix}.pdf",
                )
    return normalized_linf


def make_individual_plots(normalized_linf=None):
    """Create the three Level-5 comparison plots."""

    individual_run_paths = {
        "high_accuracy_L5": JOINED_RUNS["high_accuracy_L5"],
        "6_set1_L6s5": JOINED_RUNS["6_set1_L6s5"],
    }
    if normalized_linf is None:
        normalized_linf = load_diagnostic(
            individual_run_paths,
            "ConstraintNorms/NormalizedGhCe_Linf.dat",
        )
    individual_runs = _select_runs(normalized_linf, individual_run_paths)
    individual_legend = {
        run_name: JOINED_LEGEND[run_name] for run_name in individual_run_paths
    }
    individual_plots = (
        (
            "Linf(NormalizedGhCe) on SphereC28",
            "joined_ML_5_S1_L5_SphereC28_Linf_NormalizedGhCe.pdf",
            (8, 5),
        ),
        (
            "Linf(NormalizedGhCe) on SphereC0",
            "joined_ML_5_S1_L5_SphereC0_Linf_NormalizedGhCe.pdf",
            (5, 4),
        ),
        (
            "Linf(NormalizedGhCe) on SphereC1",
            "joined_ML_5_S1_L5_SphereC1_Linf_NormalizedGhCe.pdf",
            (5, 4),
        ),
    )
    for y_axis, filename, figsize in individual_plots:
        save_constraint_plot(
            individual_runs,
            individual_legend,
            y_axis,
            filename,
            figsize=figsize,
        )


def main() -> None:
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    normalized_linf = make_run_set_plots()
    make_individual_plots(normalized_linf)


if __name__ == "__main__":
    main()
