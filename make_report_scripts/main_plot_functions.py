# %%
import copy
import glob
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import CubicSpline

from data_readers import (
    read_dat_file,
    read_dat_file_uneq_cols,
    hist_files_to_dataframe,
    read_profiler,
    read_power_diagnostics_non_power_spectrum,
    read_WT_data,
    read_finite_radius_quantaties,
    read_point_interpolation_file,
    read_OrbitDiagnostics_file,
    make_Bh_pandas,
)
from helper_functions import (
    sort_run_data_dict,
    moving_average_valid,
)


def read_horizonh5(horizonh5_path, h5_key):
    with h5py.File(horizonh5_path, "r") as hf:
        # h5_key = ['AhA','AhB','AhC']
        # Horizons.h5 has keys 'AhA.dir'
        key = h5_key + ".dir"
        # 'AhC' will not be all the horizons.h5
        if key in hf.keys():
            return make_Bh_pandas(hf[key])
        else:
            return None


def read_AH_files(Ev_path):
    fileA = Ev_path + "Run/ApparentHorizons/AhA.dat"
    fileB = Ev_path + "Run/ApparentHorizons/AhB.dat"

    dataA = read_dat_file_across_AA(fileA)
    dataB = read_dat_file_across_AA(fileB)

    return dataA, dataB


def read_dat_file_across_AA(file_pattern):
    # ApparentHorizons/Horizons.h5@AhA
    if "Horizons.h5@" in file_pattern:
        file_pattern, h5_key = file_pattern.split("@")
    if "PowerDiagnostics" in file_pattern:
        file_pattern, dat_file_name, psi_or_kappa, top_num = file_pattern.split("@")
    if "BondiCceR" in file_pattern:
        file_pattern, WT_var = file_pattern.split("@")
        if "Index:" in file_pattern:
            raise Exception("Not Implemented yet!")
    if "_FiniteRadii_CodeUnits" in file_pattern:
        file_pattern, radius = file_pattern.split("@")
        if "Index:" in file_pattern:
            raise Exception("Not Implemented yet!")
    if "@ALL" in file_pattern:
        file_pattern, _ = file_pattern.split("@")
        loadALL = True
    else:
        loadALL = False

    path_pattern = file_pattern
    path_collection = []

    print(path_pattern)
    for folder_name in glob.iglob(path_pattern, recursive=True):
        if os.path.isdir(folder_name) or os.path.isfile(folder_name):
            path_collection.append(folder_name)

    # If we gave wrong radii for the BondiData then return the correct radii list
    if "BondiCceR" in file_pattern and len(path_collection) == 0:
        general_pattern_for_Bondi = file_pattern.split("BondiCceR")[0] + "BondiCceR*.h5"
        all_Bondi_files = glob.glob(general_pattern_for_Bondi)
        radii_list = []
        for file in all_Bondi_files:
            radii_list.append(file.split("BondiCceR")[1].split(".h5")[0])
        raise Exception(
            f"No files found for pattern {file_pattern}!\nAvailable extraction radii are: {sorted(set(radii_list))}"
        )
    path_collection.sort()

    read_data_collection = []
    for path in path_collection:
        print(path)
        # AhACoefs.dat has uneq cols
        if "Coefs.dat" in path:
            read_data_collection.append(read_dat_file_uneq_cols(path))
        elif "Hist-" in path:
            read_data_collection.append(hist_files_to_dataframe(path))
        elif "Profiler" in path:
            read_data_collection.append(read_profiler(path))
        elif "Horizons.h5" in path:
            returned_data = read_horizonh5(path, h5_key)
            if returned_data is not None:
                read_data_collection.append(returned_data)
        elif "PowerDiagnostics" in path:
            returned_data = read_power_diagnostics_non_power_spectrum(
                path, dat_file_name, psi_or_kappa, top_num
            )
            if returned_data is not None:
                read_data_collection.append(returned_data)
        elif "BondiCceR" in path:
            returned_data = read_WT_data(path, WT_var)
            if returned_data is not None:
                read_data_collection.append(returned_data)
        elif "_FiniteRadii_CodeUnits" in path:
            returned_data = read_finite_radius_quantaties(path, radius)
            if returned_data is not None:
                read_data_collection.append(returned_data)
        elif "Int_" in path:
            read_data_collection.append(read_point_interpolation_file(path, loadALL))
        elif "OrbitDiagnostics" in path:
            read_data_collection.append(read_OrbitDiagnostics_file(path))
        else:
            read_data_collection.append(read_dat_file(path))

    data = pd.concat(read_data_collection)
    rename_dict = {
        "t": "t(M)",
        "time": "t(M)",
        "Time": "t(M)",
        "time after step": "t(M)",
    }
    data.rename(columns=rename_dict, inplace=True)

    # print(data.columns)
    return data


# Give a dict of {"run_name" = runs_path} and data_file_path to get {"run_name" = dat_file_data}
def load_data_from_levs(runs_path, data_file_path):
    data_dict = {}
    column_list = ""
    for run_name in runs_path.keys():
        data_dict[run_name] = read_dat_file_across_AA(
            runs_path[run_name] + data_file_path
        )
        column_list = data_dict[run_name].columns
    return column_list, data_dict


def add_diff_columns(runs_data_dict, x_axis, y_axis, diff_base):
    if diff_base not in runs_data_dict.keys():
        raise Exception(f"{diff_base} not in {runs_data_dict.keys()}")

    unique_x_data, unique_indices = np.unique(
        runs_data_dict[diff_base][x_axis], return_index=True
    )
    # sorted_indices = np.sort(unique_indices)
    unique_y_data = runs_data_dict[diff_base][y_axis].iloc[unique_indices]
    interpolated_data = sp.interpolate.CubicSpline(
        unique_x_data, unique_y_data, extrapolate=False
    )
    # interpolated_data = sp.interpolate.PchipInterpolator(unique_x_data, unique_y_data, extrapolate=False)

    for key in runs_data_dict:
        if key == diff_base:
            continue
        df = runs_data_dict[key]
        df["diff_abs_" + y_axis] = np.abs(df[y_axis] - interpolated_data(df[x_axis]))
        df["diff_" + y_axis] = df[y_axis] - interpolated_data(df[x_axis])


def plot_graph_for_runs_wrapper(
    runs_data_dict,
    x_axis,
    y_axis_list,
    minT,
    maxT,
    legend_dict=None,
    save_path=None,
    moving_avg_len=0,
    plot_fun=lambda x, y, label: plt.plot(x, y, label=label),
    sort_by=None,
    diff_base=None,
    title=None,
    append_to_title="",
    plot_abs_diff=False,
    constant_shift_val_time=None,
    modification_function=None,
    take_abs=False,
):
    # Do this better using columns of a pandas dataframe
    for y_axis in y_axis_list[:-1]:
        legend_dict = {}
        for key in runs_data_dict:
            legend_dict[key] = key + "_" + str(y_axis)
        plot_graph_for_runs(
            runs_data_dict,
            x_axis,
            y_axis,
            minT,
            maxT,
            legend_dict=legend_dict,
            save_path=None,
            moving_avg_len=moving_avg_len,
            plot_fun=plot_fun,
            sort_by=sort_by,
            diff_base=diff_base,
            title=title,
            append_to_title=append_to_title,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            modification_function=modification_function,
            take_abs=take_abs,
        )

    # Save when plotting the last y_axis.
    y_axis = y_axis_list[-1]
    legend_dict = {}
    for key in runs_data_dict:
        legend_dict[key] = key + "_" + str(y_axis)
    plot_graph_for_runs(
        runs_data_dict,
        x_axis,
        y_axis,
        minT,
        maxT,
        legend_dict=legend_dict,
        save_path=save_path,
        moving_avg_len=moving_avg_len,
        plot_fun=plot_fun,
        sort_by=sort_by,
        diff_base=diff_base,
        title=title,
        append_to_title=append_to_title,
        plot_abs_diff=plot_abs_diff,
        constant_shift_val_time=constant_shift_val_time,
        modification_function=modification_function,
        take_abs=take_abs,
    )

    plt.ylabel("")
    plt.title("" + append_to_title)
    if moving_avg_len > 0:
        plt.title(
            f"{title} (moving avg len = {moving_avg_len})" + append_to_title,
        )

    if save_path is not None:
        fig_x_label = ""
        fig_y_label = ""

        for y_axis in y_axis_list:
            fig_x_label = fig_x_label + x_axis.replace("/", "_").replace(".", "_")
            fig_y_label = fig_y_label + y_axis.replace("/", "_").replace(".", "_")
        save_file_name = (
            f"{fig_y_label}_vs_{fig_x_label}_minT={minT}_maxT={maxT}".replace(".", "_")
        )
        if moving_avg_len > 0:
            save_file_name = save_file_name + f"_moving_avg_len={moving_avg_len}"
        if diff_base is not None:
            save_file_name = save_file_name + f"_diff_base={diff_base}"

        if len(save_file_name) >= 251:  # <save_file_name>.png >=255
            save_file_name = save_file_name[:245] + str(random.randint(10000, 99999))
            print(f"The filename was too long!! New filename is {save_file_name}")

        plt.savefig(save_path + save_file_name)


def plot_graph_for_runs(
    runs_data_dict_original,
    x_axis,
    y_axis,
    minT,
    maxT,
    legend_dict=None,
    save_path=None,
    moving_avg_len=0,
    plot_fun=lambda x, y, label: plt.plot(x, y, label=label),
    sort_by=None,
    diff_base=None,
    title=None,
    append_to_title="",
    plot_abs_diff=False,
    constant_shift_val_time=None,
    modification_function=None,
    take_abs=False,
):
    runs_data_dict = runs_data_dict_original
    if modification_function is not None:
        runs_data_dict = copy.deepcopy(runs_data_dict_original)
        for key in runs_data_dict:
            new_x, new_y, new_y_axis = modification_function(
                runs_data_dict[key][x_axis],
                runs_data_dict[key][y_axis],
                runs_data_dict[key],
                y_axis,
            )
            runs_data_dict[key][new_y_axis] = new_y
            runs_data_dict[key][x_axis] = new_x
        y_axis = new_y_axis

    sort_run_data_dict(runs_data_dict, sort_by=sort_by)
    current_runs_data_dict_keys = list(runs_data_dict.keys())

    if diff_base is not None:
        add_diff_columns(runs_data_dict, x_axis, y_axis, diff_base)
        current_runs_data_dict_keys = []
        for key in runs_data_dict:
            if key == diff_base:
                continue
            else:
                current_runs_data_dict_keys.append(key)
        if plot_abs_diff:
            y_axis = "diff_abs_" + y_axis
        else:
            y_axis = "diff_" + y_axis

    # Find the indices corresponding to maxT and minT
    minT_indx_list = {}
    maxT_indx_list = {}

    if legend_dict is None:
        legend_dict = {}
        for run_name in current_runs_data_dict_keys:
            legend_dict[run_name] = None
    else:
        for run_name in current_runs_data_dict_keys:
            if run_name not in legend_dict:
                raise ValueError(f"{run_name} not in {legend_dict=}")

    for run_name in current_runs_data_dict_keys:
        minT_indx_list[run_name] = len(
            runs_data_dict[run_name][x_axis][runs_data_dict[run_name][x_axis] < minT]
        )
        maxT_indx_list[run_name] = len(
            runs_data_dict[run_name][x_axis][runs_data_dict[run_name][x_axis] < maxT]
        )

    if moving_avg_len == 0:
        for run_name in current_runs_data_dict_keys:
            x_data = runs_data_dict[run_name][x_axis][
                minT_indx_list[run_name] : maxT_indx_list[run_name]
            ]
            y_data = runs_data_dict[run_name][y_axis][
                minT_indx_list[run_name] : maxT_indx_list[run_name]
            ]

            if constant_shift_val_time is not None:
                shift_label_val = np.abs(x_data.iloc[-1] - x_data.iloc[0]) / 4
                unique_x_data, unique_indices = np.unique(x_data, return_index=True)
                # sorted_indices = np.sort(unique_indices)
                unique_y_data = y_data.iloc[unique_indices]
                try:
                    interpolated_data = CubicSpline(
                        unique_x_data, unique_y_data, extrapolate=False
                    )
                except Exception:
                    print(run_name, unique_y_data)
                y_data = y_data - interpolated_data(constant_shift_val_time)
                if plot_abs_diff:
                    y_data = np.abs(y_data)

            #   print(f"{len(x_data)=},{len(y_data)=},{len(np.argsort(x_data))=},{type(x_data)=}")

            #   sorted_indices = x_data.argsort()
            #   x_data = x_data.iloc[sorted_indices]
            #   y_data = y_data.iloc[sorted_indices]
            legend = legend_dict[run_name]
            if legend is None:
                legend = run_name
            if take_abs:
                y_data = np.abs(y_data)

            if np.all(np.isnan(y_data)):
                print(
                    f"Warning: {run_name} has no data {y_axis=} for {x_axis}={minT} to {maxT}. Skipping."
                )
                continue
            plot_fun(x_data, y_data, legend)

            if constant_shift_val_time is not None:
                plt.axhline(y=y_data.iloc[-1], linestyle=":")
                plt.text(
                    x=np.random.rand() * shift_label_val + x_data.iloc[0],
                    y=y_data.iloc[-1],
                    s=f"{y_data.iloc[-1]:.2e}",
                    verticalalignment="bottom",
                )

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if constant_shift_val_time is not None:
            plt.axvline(x=constant_shift_val_time, linestyle=":", color="red")
        if title is None:
            title = '"' + y_axis + '" vs "' + x_axis + '"'
            if constant_shift_val_time is not None:
                title = title + f" constant_shift_val_time={constant_shift_val_time}"
            if diff_base is not None:
                title = title + f" diff_base={diff_base}"
            if plot_abs_diff:
                title = title + " (abs_diff)"
            if take_abs:
                title = title + " (abs)"
        plt.title(title + append_to_title)
        plt.legend()

    else:
        for run_name in current_runs_data_dict_keys:
            x_data = np.array(
                runs_data_dict[run_name][x_axis][
                    minT_indx_list[run_name] + moving_avg_len - 1 : maxT_indx_list[
                        run_name
                    ]
                ]
            )
            y_data = np.array(
                moving_average_valid(
                    runs_data_dict[run_name][y_axis][
                        minT_indx_list[run_name] : maxT_indx_list[run_name]
                    ],
                    moving_avg_len,
                )
            )

            if constant_shift_val_time is not None:
                shift_label_val = np.abs(x_data.iloc[-1] - x_data.iloc[0]) / 4
                unique_x_data, unique_indices = np.unique(x_data, return_index=True)
                # sorted_indices = np.sort(unique_indices)
                unique_y_data = y_data.iloc[unique_indices]

                interpolated_data = CubicSpline(
                    unique_x_data, unique_y_data, extrapolate=False
                )
                y_data = y_data - interpolated_data(constant_shift_val_time)
                if plot_abs_diff:
                    y_data = np.abs(y_data)

            #   sorted_indices = np.argsort(x_data)
            #   x_data = x_data[sorted_indices]
            #   y_data = y_data[sorted_indices]
            legend = legend_dict[run_name]
            if legend is None:
                legend = run_name
            if take_abs:
                y_data = np.abs(y_data)

            if np.all(np.isnan(y_data)):
                print(
                    f"Warning: {run_name} has no data {y_axis=} for {x_axis}={minT} to {maxT}. Skipping."
                )
                continue
            plot_fun(x_data, y_data, legend)

            if constant_shift_val_time is not None:
                plt.axhline(y=y_data.iloc[-1], linestyle=":")
                plt.text(
                    x=np.random.rand() * shift_label_val + x_data.iloc[0],
                    y=y_data.iloc[-1],
                    s=f"{y_data.iloc[-1]:.1f}",
                    verticalalignment="bottom",
                )

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if constant_shift_val_time is not None:
            plt.axvline(x=constant_shift_val_time, linestyle=":", color="red")
        if title is None:
            title = (
                '"'
                + y_axis
                + '" vs "'
                + x_axis
                + '"  '
                + f"avg_window_len={moving_avg_len}"
            )
            if constant_shift_val_time is not None:
                title = title + f" constant_shift_val_time={constant_shift_val_time}"
            if diff_base is not None:
                title = title + f" diff_base={diff_base}"
            if plot_abs_diff:
                title = title + " (abs_diff)"
            if take_abs:
                title = title + " (abs)"
        plt.title(title + append_to_title)
        plt.legend()

    if save_path is not None:
        fig_x_label = x_axis.replace("/", "_").replace(".", "_")
        fig_y_label = y_axis.replace("/", "_").replace(".", "_")
        save_file_name = (
            f"{fig_y_label}_vs_{fig_x_label}_minT={minT}_maxT={maxT}".replace(".", "_")
        )
        if moving_avg_len > 0:
            save_file_name = save_file_name + f"_moving_avg_len={moving_avg_len}"
        if diff_base is not None:
            save_file_name = save_file_name + f"_diff_base={diff_base}"

        for run_name in current_runs_data_dict_keys:
            save_file_name = (
                save_file_name + "__" + run_name.replace("/", "_").replace(".", "_")
            )

        if len(save_file_name) >= 251:  # <save_file_name>.png >=255
            save_file_name = save_file_name[:245] + str(random.randint(10000, 99999))
            print(f"The filename was too long!! New filename is {save_file_name}")

        plt.savefig(save_path + save_file_name)

