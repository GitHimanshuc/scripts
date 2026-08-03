"""Reproduce all active, non-CCE spec-accuracy plots with JSON caching.

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
from make_report_scripts.io import get_top_name_from_number


CACHE_FOLDER = Path.cwd()
RELOAD_CACHE = False
SAVE_FOLDER = Path("./plots/").resolve()

MIN_TIME = 1205.0
MAX_TIME = 4000.0
POWER_MIN_TIME = 1210.0
POWER_MAX_TIME = 4000.0


L15_MAIN_LEGEND = {
    "high_accuracy_main_L1": "Old Level 1",
    "high_accuracy_main_L2": "Old Level 2",
    "high_accuracy_main_L3": "Old Level 3",
    "high_accuracy_main_L4": "Old Level 4",
    "high_accuracy_main_L5": "Old Level 5",
}

L15_MAIN_RUNS = {
    "high_accuracy_main_L1": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35_master/Ev/Lev1_A?/Run/",
    "high_accuracy_main_L2": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35_master/Ev/Lev2_A?/Run/",
    "high_accuracy_main_L3": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35_master/Ev/Lev3_A?/Run/",
    "high_accuracy_main_L4": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35_master/Ev/Lev4_A?/Run/",
    "high_accuracy_main_L5": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35_master/Ev/Lev5_A?/Run/",
}

L15_ODE_FIX_LEGEND = {
    "high_accuracy_L1": "Ode Fix Level 1",
    "high_accuracy_L2": "Ode Fix Level 2",
    "high_accuracy_L3": "Ode Fix Level 3",
    "high_accuracy_L4": "Ode Fix Level 4",
    "high_accuracy_L5": "Ode Fix Level 5",
}

L15_ODE_FIX_RUNS = {
    "high_accuracy_L1": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35/Ev/Lev1_A?/Run/",
    "high_accuracy_L2": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35/Ev/Lev2_A?/Run/",
    "high_accuracy_L3": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35/Ev/Lev3_A?/Run/",
    "high_accuracy_L4": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35/Ev/Lev4_A?/Run/",
    "high_accuracy_L5": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/high_accuracy_L35/Ev/Lev5_A?/Run/",
}

L16_SET1_LEGEND = {
    "6_set1_L6s1": "New Level 1",
    "6_set1_L6s2": "New Level 2",
    "6_set1_L6s3": "New Level 3",
    "6_set1_L6s4": "New Level 4",
    "6_set1_L6s5": "New Level 5",
    "6_set1_L6s6": "New Level 6",
}

L16_SET1_RUNS = {
    "6_set1_L6s1": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/6_segs/6_set1_L6/Ev/Lev1_A?/Run/",
    "6_set1_L6s2": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/6_segs/6_set1_L6/Ev/Lev2_A?/Run/",
    "6_set1_L6s3": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/6_segs/6_set1_L6/Ev/Lev3_A?/Run/",
    "6_set1_L6s4": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/6_segs/6_set1_L6/Ev/Lev4_A?/Run/",
    "6_set1_L6s5": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/6_segs/6_set1_L6/Ev/Lev5_A?/Run/",
    "6_set1_L6s6": "/resnick/groups/sxs/hchaudha/from_central/spec_runs/6_segs/6_set1_L6/Ev/Lev6_A?/Run/",
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
        (
            "Linf(GhCe) on SphereA0",
            r"$L_\infty(\mathcal{E}_{\mathrm{GH}})$ on SphereA0",
            "SphereA0_Linf_GhCe",
        ),
        (
            "Linf(GhCe) on SphereC6",
            r"$L_\infty(\mathcal{E}_{\mathrm{GH}})$ on SphereC6",
            "SphereC6_Linf_GhCe",
        ),
        (
            "Linf(GhCe) on SphereC22",
            r"$L_\infty(\mathcal{E}_{\mathrm{GH}})$ on SphereC22",
            "SphereC22_Linf_GhCe",
        ),
    ),
    "ConstraintNorms/NormalizedGhCe_Linf.dat": (
        (
            "Linf(NormalizedGhCe) on SphereA0",
            r"$L_\infty(\widehat{\mathcal{E}}_{\mathrm{GH}})$ on SphereA0",
            "SphereA0_Linf_NormalizedGhCe",
        ),
        (
            "Linf(NormalizedGhCe) on SphereC6",
            r"$L_\infty(\widehat{\mathcal{E}}_{\mathrm{GH}})$ on SphereC6",
            "SphereC6_Linf_NormalizedGhCe",
        ),
        (
            "Linf(NormalizedGhCe) on SphereC22",
            r"$L_\infty(\widehat{\mathcal{E}}_{\mathrm{GH}})$ on SphereC22",
            "SphereC22_Linf_NormalizedGhCe",
        ),
    ),
    "ConstraintNorms/GhCe_Norms.dat": (
        ("L2(GhCe)", r"$L_2(\mathcal{E}_{\mathrm{GH}})$", "L2(GhCe)"),
        ("Linf(GhCe)", r"$L_\infty(\mathcal{E}_{\mathrm{GH}})$", "Linf(GhCe)"),
        ("VolLp(GhCe)", r"$V_{\ell^p}(\mathcal{E}_{\mathrm{GH}})$", "VolLp(GhCe)"),
    ),
    "ConstraintNorms/NormalizedGhCe_Norms.dat": (
        (
            "L2(NormalizedGhCe)",
            r"$L_2(\widehat{\mathcal{E}}_{\mathrm{GH}})$",
            "L2(NormalizedGhCe)",
        ),
        (
            "Linf(NormalizedGhCe)",
            r"$L_\infty(\widehat{\mathcal{E}}_{\mathrm{GH}})$",
            "Linf(NormalizedGhCe)",
        ),
        (
            "VolLp(NormalizedGhCe)",
            r"$V_{\ell^p}(\widehat{\mathcal{E}}_{\mathrm{GH}})$",
            "VolLp(NormalizedGhCe)",
        ),
    ),
}


L15_MAIN_POWER_ROOTS = {
    f"high_accuracy_main_L{level}": Path(
        "/resnick/groups/sxs/hchaudha/from_central/spec_runs/"
        f"high_accuracy_L35_master/h5_files_Lev{level}"
    )
    for level in range(1, 6)
}

L15_ODE_FIX_POWER_ROOTS = {
    f"high_accuracy_L{level}": Path(
        "/resnick/groups/sxs/hchaudha/from_central/spec_runs/"
        f"high_accuracy_L35/h5_files_Lev{level}"
    )
    for level in range(1, 6)
}

L16_SET1_POWER_ROOTS = {
    f"6_set1_L6s{level}": Path(
        "/resnick/groups/sxs/hchaudha/from_central/spec_runs/6_segs/"
        f"6_set1_L6/h5_files_Lev{level}"
    )
    for level in range(1, 7)
}

POWER_RUN_GROUPS = (
    ("L15_main", L15_MAIN_POWER_ROOTS, L15_MAIN_LEGEND),
    ("L15_ode_fix", L15_ODE_FIX_POWER_ROOTS, L15_ODE_FIX_LEGEND),
    ("L16_set1", L16_SET1_POWER_ROOTS, L16_SET1_LEGEND),
)

POWER_DOMAINS = ("SphereA0", "SphereC6")
POWER_FIELDS = ("psi", "kappa")
POWER_TOPOLOGIES = (0, 1)


def _select_runs(runs_data, run_paths):
    return {run_name: runs_data[run_name] for run_name in run_paths}


def load_diagnostic(runs, data_file_path):
    _, runs_data = load_data_from_levs(
        runs,
        data_file_path,
        cache_folder=CACHE_FOLDER,
        reload_cache=RELOAD_CACHE,
    )
    return runs_data


def save_constraint_plot(
    runs_data,
    legend,
    y_axis,
    y_label,
    save_name,
    *,
    figsize=(5, 4),
):
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
        ax.set_ylabel(y_label)
        ax.set_xlabel("t(M)")
        ax.grid(False)
        fig.tight_layout()
        output_path = SAVE_FOLDER / save_name
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    print(f"Saved {output_path}!\n")


def make_constraint_plots():
    normalized_linf = None
    for data_file_path, plot_specs in PLOTS_BY_DIAGNOSTIC.items():
        runs_data = load_diagnostic(JOINED_RUNS, data_file_path)
        if data_file_path == "ConstraintNorms/NormalizedGhCe_Linf.dat":
            normalized_linf = runs_data

        for runs_set_name, run_paths, legend in RUN_GROUPS:
            selected_runs = _select_runs(runs_data, run_paths)
            for y_axis, y_label, filename_suffix in plot_specs:
                save_constraint_plot(
                    selected_runs,
                    legend,
                    y_axis,
                    y_label,
                    f"{runs_set_name}_{filename_suffix}.pdf",
                )
    return normalized_linf


def make_individual_plots(normalized_linf):
    individual_run_paths = {
        "high_accuracy_L5": JOINED_RUNS["high_accuracy_L5"],
        "6_set1_L6s5": JOINED_RUNS["6_set1_L6s5"],
    }
    individual_runs = _select_runs(normalized_linf, individual_run_paths)
    individual_legend = {
        run_name: JOINED_LEGEND[run_name] for run_name in individual_run_paths
    }
    individual_plots = (
        (
            "Linf(NormalizedGhCe) on SphereC28",
            r"$L_\infty(\widehat{\mathcal{E}}_{\mathrm{GH}})$ on SphereC28",
            "joined_ML_5_S1_L5_SphereC28_Linf_NormalizedGhCe.pdf",
            (8, 5),
        ),
        (
            "Linf(NormalizedGhCe) on SphereC0",
            r"$L_\infty(\widehat{\mathcal{E}}_{\mathrm{GH}})$ on SphereC0",
            "joined_ML_5_S1_L5_SphereC0_Linf_NormalizedGhCe.pdf",
            (5, 4),
        ),
        (
            "Linf(NormalizedGhCe) on SphereC1",
            r"$L_\infty(\widehat{\mathcal{E}}_{\mathrm{GH}})$ on SphereC1",
            "joined_ML_5_S1_L5_SphereC1_Linf_NormalizedGhCe.pdf",
            (5, 4),
        ),
    )
    for y_axis, y_label, filename, figsize in individual_plots:
        save_constraint_plot(
            individual_runs,
            individual_legend,
            y_axis,
            y_label,
            filename,
            figsize=figsize,
        )


def power_runs_to_plot():
    """Return every configured power-diagnostic run."""

    runs = {}
    metadata = {}
    for runs_set_name, roots, legend in POWER_RUN_GROUPS:
        for run_name, root in roots.items():
            level = int(root.name.removeprefix("h5_files_Lev"))
            runs[run_name] = root
            metadata[run_name] = (runs_set_name, level, legend[run_name])
    return runs, metadata


def save_power_plot(
    frame,
    *,
    domain,
    field,
    run_label,
    save_name,
):
    """Save one coefficient-over-time power plot in the original style."""

    selected = frame[
        (frame["t(M)"] >= POWER_MIN_TIME)
        & (frame["t(M)"] <= POWER_MAX_TIME)
    ]
    coefficient_columns = [
        column
        for column in selected.columns
        if isinstance(column, str)
        and column.startswith("coef")
        and column.removeprefix("coef").isdigit()
        and int(column.removeprefix("coef")) <= 100
        and not selected[column].isna().all()
    ]
    coefficient_columns.sort(key=lambda column: int(column.removeprefix("coef")))

    with plt.style.context(["ggplot"]):
        fig, ax = plt.subplots(figsize=(5, 4))
        for column in coefficient_columns:
            coefficient = column.removeprefix("coef")
            ax.plot(
                selected["t(M)"],
                selected[column],
                label=rf"$a_{{{coefficient}}}$",
            )
        if len(coefficient_columns) > 15:
            ax.legend(
                ncol=int(np.ceil(len(coefficient_columns) / 15)),
                loc="upper right",
            )
        else:
            ax.legend(loc="upper right")
        ax.set_title(f"{domain} of {run_label}")
        ax.set_xlabel("t(M)")
        ax.set_ylabel(f"Power {field}")
        ax.set_yscale("log")
        ax.set_ylim(5e-17, 5)
        ax.grid(False)
        fig.tight_layout()
        output_path = SAVE_FOLDER / save_name
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    print(f"Saved {output_path}!\n")


def make_power_plots():
    power_runs, metadata = power_runs_to_plot()
    for domain in POWER_DOMAINS:
        for top_number in POWER_TOPOLOGIES:
            topology = get_top_name_from_number(top_number, domain)
            for field in POWER_FIELDS:
                data_file_path = (
                    f"extracted-PowerDiagnostics/{domain}.dir/"
                    f"Power{field}.dir/{topology}(* modes).dat"
                )
                runs_data = load_diagnostic(power_runs, data_file_path)
                for run_name, frame in runs_data.items():
                    runs_set_name, level, run_label = metadata[run_name]
                    save_power_plot(
                        frame,
                        domain=domain,
                        field=field,
                        run_label=run_label,
                        save_name=(
                            f"{runs_set_name}_L{level}_PS_{field}_"
                            f"{domain}_{top_number}.pdf"
                        ),
                    )


def main() -> None:
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    normalized_linf = make_constraint_plots()
    make_individual_plots(normalized_linf)
    make_power_plots()


if __name__ == "__main__":
    main()
