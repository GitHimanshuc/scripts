import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cycler
from pathlib import Path

from helper_functions import (
    add_norm_constraints,
    num_points_per_subdomain,
    limit_by_col_val,
    filter_by_regex,
)
from main_plot_functions import (
    load_data_from_levs,
    plot_graph_for_runs,
    plot_graph_for_runs_wrapper,
)
from make_report_scripts.heatmap_related_functions import return_sorted_domain_names

# =========================================================================================================================================================
# ==================================================== GENERAL CONFIG ==========================================================
# =========================================================================================================================================================

save_report_base_folder = Path(
    "/work2/08330/tg876294/stampede3/reports/HighAccuracy1025/"
)
dirs_to_plot = []

# for testing change paths if we are on caltech hpc
if Path("/resnick/groups/sxs/hchaudha/scripts/make_report_scripts").exists():
    print("Running on Caltech HPC, changing paths for testing")
    save_report_base_folder = Path(
        "/resnick/groups/sxs/hchaudha/HighAccuracy1025/test_reports/"
    )
    BFI_main_folder = Path("/resnick/groups/sxs/hchaudha/HighAccuracy1025/")

    dirs_to_plot = [
        Path("/resnick/groups/sxs/hchaudha/HighAccuracy1025/HighAccuracy102501")
    ]

else:
    # My BFI main folder
    BFI_main_folder = Path("/scratch/projects/sxs/hchaudha/HighAccuracy1025/")
    dirs_to_plot = [d for d in BFI_main_folder.iterdir() if d.is_dir()]

    # Nils BFI main folder
    BFI_main_folder = Path("/scratch/projects/sxs/nfischer/HighAccuracy1025/")
    dirs_to_plot += [d for d in BFI_main_folder.iterdir() if d.is_dir()]

# ==================================================== Load data and plot ==========================================================

for sim_folder in dirs_to_plot:
    print(f"Simulation folder: {sim_folder}\n")

# %%
# =========================================== Copy the part below for each plot type ===========================================

    print("---- Plotting dt/dT vs t(M)\n")

    save_name = "wall_time.json"
    base_save_folder = save_report_base_folder / sim_folder.name
    if not base_save_folder.exists():
        base_save_folder.mkdir(parents=False, exist_ok=True)
    save_name = Path(f"{base_save_folder}/{save_name}")

    runs_to_plot = {}
    for lev in range(2, 7):
        # !!!! TODO note that we are using Ecc0 here, change if needed
        # Only add levs that exist
        if not (sim_folder / f"Ecc0/Ev/Lev{lev}_AA/Run/").exists():
            print(f"Skipping Lev{lev} as it does not exist")
            continue

        # Eventaully we need to clean up and add path support, for now convert to str
        runs_to_plot[f"{sim_folder.name}_{lev}"] = str(
            sim_folder / f"Ecc0/Ev/Lev{lev}_??/Run/"
        )

    ringdown_as_well = True
    # ringdown_as_well = False
    if ringdown_as_well:
        for key in list(runs_to_plot.keys()):
            runs_to_plot[key] = runs_to_plot[key].replace("/Ev/", "/Ev/**/")

    # psi_or_kappa = "psi"
    psi_or_kappa = "kappa"
    psi_or_kappa = "both"
    top_number = 0

    # data_file_path = "/TStepperDiag.dat"
    data_file_path = "/TimeInfo.dat"

    runs_data_dict = {}
    if psi_or_kappa == "both" and "PowerDiagnostics" in data_file_path:
        column_names_kappa, runs_data_dict_kappa = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@kappa@{top_number}"
        )
        column_names_psi, runs_data_dict_psi = load_data_from_levs(
            runs_to_plot, f"{data_file_path}@psi@{top_number}"
        )

        for key in runs_data_dict_psi:
            runs_data_dict[key] = pd.merge(
                runs_data_dict_kappa[key],
                runs_data_dict_psi[key],
                on="t(M)",
                how="outer",
            )
        column_names = runs_data_dict[key].columns.tolist()

    elif data_file_path == "ComputeCOM":
        bhA_cols, bhA_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhA"
        )
        bhB_cols, bhB_data = load_data_from_levs(
            runs_to_plot, "ApparentHorizons/Horizons.h5@AhB"
        )
        for key in bhA_data:
            bhA_df = bhA_data[key]
            bhB_df = bhB_data[key]
            mA = bhA_df["ChristodoulouMass"]
            mB = bhB_df["ChristodoulouMass"]
            com_x = (
                mA * bhA_df["CoordCenterInertial_0"]
                + mB * bhB_df["CoordCenterInertial_0"]
            ) / (mA + mB)
            com_y = (
                mA * bhA_df["CoordCenterInertial_1"]
                + mB * bhB_df["CoordCenterInertial_1"]
            ) / (mA + mB)
            com_z = (
                mA * bhA_df["CoordCenterInertial_2"]
                + mB * bhB_df["CoordCenterInertial_2"]
            ) / (mA + mB)
            com_df = pd.DataFrame(
                {"t(M)": bhA_df["t(M)"], "COM_X": com_x, "COM_Y": com_y, "COM_Z": com_z}
            )
            runs_data_dict[key] = com_df
            column_names = com_df.columns.tolist()
    else:
        if "PowerDiagnostics" in data_file_path:
            data_file_path = f"{data_file_path}@{psi_or_kappa}@{top_number}"
        column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    # load both psi and kappa

    # print(column_names)
    # print(runs_data_dict.keys())

    vars = set()
    domains = set()
    for run in runs_data_dict:
        for col in runs_data_dict[run]:
            vars.add(col.split(" on ")[0])
            domains.add(col.split(" on ")[1] if " on " in col else "All Domains")
    # list(sorted(vars)),list(sorted(domains))

    # #### Modify dict

    if "Constraints_Linf" in data_file_path:
        new_indices, runs_data_dict = add_norm_constraints(
            runs_data_dict,
            index_num=[1, 2, 3],
            norm=["Linf"],
            subdomains_seperately=True,
            replace_runs_data_dict=True,
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "GrDomain" in data_file_path:
        new_indices, runs_data_dict = num_points_per_subdomain(
            runs_data_dict, replace_runs_data_dict=False
        )
        print(runs_data_dict.keys())
        print(new_indices)

    if "ComputeCOM" in data_file_path:
        for key in runs_data_dict:
            df = runs_data_dict[key]
            df["COM_mag"] = np.sqrt(
                df["COM_X"] ** 2 + df["COM_Y"] ** 2 + df["COM_Z"] ** 2
            )
            runs_data_dict[key] = df
        print(runs_data_dict.keys())
        print(new_indices)

    # ### dat files plot

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = False
    modification_function = None
    append_to_title = ""
    y_axis_list = None
    x_axis = "t(M)"
    take_abs = False

    # take_abs = True

    # y_axis = 'MPI::MPwait_cum'
    # x_axis = 't'
    y_axis = 'T [hours]'
    # y_axis = "dt/dT"

    # y_axis = 'dt'

    minT = -1000

    maxT = 400000

    # if "GhCe" in y_axis:
    plot_fun = lambda x, y, label: plt.plot(x, y, label=label)

    legend_dict = {}
    for key in runs_data_dict.keys():
        legend_dict[key] = None

    # if 'Horizons.h5@' in data_file_path:
    #   append_to_title += " HorizonBH="+data_file_path.split('@')[-1]
    if "AhA" in data_file_path:
        append_to_title += " AhA"
    if "AhB" in data_file_path:
        append_to_title += " AhB"

    # Save data to JSON instead of plotting
    output_dict = {}
    for run_name, df in runs_data_dict.items():
        if x_axis in df.columns and y_axis in df.columns:
            # Filter by time range
            mask = (df[x_axis] >= minT) & (df[x_axis] <= maxT)
            filtered_df = df.loc[mask, [x_axis, y_axis]]
            output_dict[run_name] = {
                x_axis: filtered_df[x_axis].tolist(),
                y_axis: filtered_df[y_axis].tolist(),
            }

    if output_dict:
        # Find the branching point where runs diverge
        run_names = list(output_dict.keys())
        if len(run_names) > 1:
            # Build time->y_value dicts for each run
            run_time_to_y = {}
            for rn in run_names:
                run_time_to_y[rn] = dict(zip(
                    output_dict[rn][x_axis], output_dict[rn][y_axis]
                ))

            # Find common times across all runs
            common_times = set(output_dict[run_names[0]][x_axis])
            for rn in run_names[1:]:
                common_times &= set(output_dict[rn][x_axis])
            common_times = sorted(common_times)

            # Find the last common time where all runs have the same y value
            pbj_time = common_times[0] if common_times else 0
            for t in common_times:
                ref_val = run_time_to_y[run_names[0]][t]
                all_match = all(run_time_to_y[rn][t] == ref_val for rn in run_names)
                if all_match:
                    pbj_time = t
                else:
                    break

            # Find index in reference run
            ref_run = run_names[0]
            pbj_idx = output_dict[ref_run][x_axis].index(pbj_time)
            output_dict["pbj_info"] = {
                "pbj_time": pbj_time,
                "pbj_index": pbj_idx,
            }

        with open(save_name, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"Saved data to {save_name}")


# %% code for generating and saving the wall_time_summary.json
def load_wall_time_json(path):
    """Load one wall_time.json file."""
    path = Path(path)
    return json.loads(path.read_text())


def load_all_wall_time_jsons(base_dir="HighAccuracy1025"):
    """Return {simulation_name: wall_time_dict} for every wall_time.json below base_dir."""
    base_dir = Path(base_dir)
    return {
        path.parent.name: load_wall_time_json(path)
        for path in sorted(base_dir.glob("HighAccuracy*/wall_time.json"))
    }


def nearest_index(values, target):
    """Index of the sample whose value is closest to target."""
    if not values:
        raise ValueError("cannot choose a nearest index from an empty list")
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def first_reset_index(values):
    """Return the first index where the counter resets, or None if it never resets."""
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            return i
    return None


def iter_level_items(wall_time_data, simulation_name=None):
    """Yield (level_key, level_number, level_data) entries for one simulation."""
    for key, value in wall_time_data.items():
        if not isinstance(value, dict) or "t(M)" not in value or "T [hours]" not in value:
            continue
        if simulation_name is not None and not key.startswith(f"{simulation_name}_"):
            continue
        level_text = key.rsplit("_", 1)[-1]
        try:
            level = int(level_text)
        except ValueError:
            level = level_text
        yield key, level, value


def wall_time_used_by_level(wall_time_data, simulation_name=None, start_time=None, use_pbj_time=True):
    """
    Compute phase-1-after-PBJ and phase-2 wall-clock hours for each level.

    `T [hours]` is assumed to reset at the phase-2 boundary. The first decrease in
    `T [hours]` is used as the phase-2 start.
    """
    if start_time is None and use_pbj_time:
        start_time = wall_time_data.get("pbj_info", {}).get("pbj_time")

    rows = []
    for level_key, level, level_data in iter_level_items(wall_time_data, simulation_name):
        times = level_data["t(M)"]
        wall_hours = level_data["T [hours]"]
        if len(times) != len(wall_hours):
            raise ValueError(f"{level_key} has {len(times)} times but {len(wall_hours)} wall-time values")
        if not times:
            continue

        start_idx = nearest_index(times, start_time) if start_time is not None else 0
        reset_idx = first_reset_index(wall_hours)
        final_idx = len(times) - 1

        if reset_idx is None:
            phase1_after_pbj_hours = wall_hours[final_idx] - wall_hours[start_idx]
            phase2_hours = 0.0
            phase2_start_t = None
        elif start_idx < reset_idx:
            phase1_after_pbj_hours = wall_hours[reset_idx - 1] - wall_hours[start_idx]
            phase2_hours = wall_hours[final_idx]
            phase2_start_t = times[reset_idx]
        else:
            phase1_after_pbj_hours = 0.0
            phase2_hours = wall_hours[final_idx] - wall_hours[start_idx]
            phase2_start_t = times[reset_idx]

        phase1_after_pbj_hours = max(phase1_after_pbj_hours, 0.0)
        phase2_hours = max(phase2_hours, 0.0)

        rows.append(
            {
                "simulation": level_key.rsplit("_", 1)[0],
                "level_key": level_key,
                "level": level,
                "pbj_index": start_idx,
                "pbj_time_M": times[start_idx],
                "phase2_start_index": reset_idx,
                "phase2_start_t_M": phase2_start_t,
                "t_final_M": times[final_idx],
                "phase1_after_pbj_hours": phase1_after_pbj_hours,
                "phase2_hours": phase2_hours,
                "total_hours_after_pbj": phase1_after_pbj_hours + phase2_hours,
            }
        )

    return pd.DataFrame(rows).sort_values("level").reset_index(drop=True)


def plot_wall_time_by_level(summary, title=None):
    """Stacked bar plot of phase-1-after-PBJ and phase-2 wall hours."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required only for plotting; the summary table works without it"
        ) from exc

    ax = summary.plot.bar(
        x="level",
        y=["phase1_after_pbj_hours", "phase2_hours"],
        stacked=True,
        figsize=(7, 4),
    )
    ax.set_xlabel("Level")
    ax.set_ylabel("Wall time used [hours]")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


# %%
# Load every simulation available in this local report folder.
wall_time_by_simulation = load_all_wall_time_jsons(save_report_base_folder)
sorted(wall_time_by_simulation)


# %%
# Generate the compact dictionary used by downstream scripts.
# Lists are ordered by level: Lev2, Lev3, Lev4, Lev5, Lev6.
wall_time_summary_dict = {}

for simulation_name, wall_time_data in wall_time_by_simulation.items():
    run_number = simulation_name[-2:]
    summary = wall_time_used_by_level(wall_time_data, simulation_name=simulation_name)

    lev2_key = f"{simulation_name}_2"
    lev2_pbj_index = int(summary.loc[summary["level"] == 2, "pbj_index"].iloc[0])
    pbj_wall_time = wall_time_data[lev2_key]["T [hours]"][lev2_pbj_index]

    wall_time_summary_dict[run_number] = {
        "post_pbj_wall_time": summary["phase1_after_pbj_hours"].round().astype(int).tolist(),
        "ringdown_wall_time": summary["phase2_hours"].round().astype(int).tolist(),
        "pbj_wall_time": int(round(pbj_wall_time)),
    }

# save the dict as json
output_path = save_report_base_folder / "wall_time_summary.json"
output_path.write_text(json.dumps(wall_time_summary_dict, indent=2))
print(f"Saved wall time summary to {output_path}")
