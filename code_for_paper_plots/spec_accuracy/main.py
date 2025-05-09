import glob
import itertools
import os
import pickle
import random
import re
import string
from pathlib import Path
from random import choice as rc
from typing import Dict, List

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 10)

spec_home = "/home/himanshu/spec/my_spec"
matplotlib.matplotlib_fname()


# =================================================================================================
# =================================================================================================
# FUNCTION DEFINITIONS
# =================================================================================================
# =================================================================================================

# =================================================================================================
# make_report_and_plots.ipynb
# =================================================================================================


def make_Bh_pandas(h5_dir):
    # Empty dataframe
    df = pd.DataFrame()

    # List of all the vars in the h5 file
    var_list = []
    h5_dir.visit(var_list.append)

    for var in var_list:
        # This means there is no time column
        # print(f"{var} : {h5_dir[var].shape}")
        if df.shape == (0, 0):
            # data[:,0] is time and then we have the data
            data = h5_dir[var]

            # vars[:-4] to remove the .dat at the end
            col_names = make_col_names(var[:-4], data.shape[1] - 1)
            col_names.append("t")
            # Reverse the list so that we get ["t","var_name"]
            col_names.reverse()
            append_to_df(data[:], col_names, df)

        else:
            data = h5_dir[var]
            col_names = make_col_names(var[:-4], data.shape[1] - 1)
            append_to_df(data[:, 1:], col_names, df)

    return df


def append_to_df(data, col_names, df):
    for i, col_name in enumerate(col_names):
        df[col_name] = data[:, i]


def make_col_names(val_name: str, val_size: int):
    col_names = []
    if val_size == 1:
        col_names.append(val_name)
    else:
        for i in range(val_size):
            col_names.append(val_name + f"_{i}")
    return col_names


def horizon_to_pandas(horizon_path: Path):
    assert horizon_path.exists()
    df_dict = {}
    with h5py.File(horizon_path, "r") as hf:
        # Not all horizon files may have AhC
        for key in hf.keys():
            if key == "VersionHist.ver":
                # Newer runs have this
                continue
            df_dict[key[:-4]] = make_Bh_pandas(hf[key])

    return df_dict


def read_horizon_across_Levs(path_list: List[Path]):
    df_listAB = []
    df_listC = []
    final_dict = {}
    for path in path_list:
        df_lev = horizon_to_pandas(path)
        # Either [AhA,AhB] or [AhA,AhB,AhC]
        if len(df_lev.keys()) > 1:
            df_listAB.append(df_lev)
        # Either [AhC] or [AhA,AhB,AhC]
        if (len(df_lev.keys()) == 1) or (len(df_lev.keys()) == 3):
            df_listC.append(df_lev)
    if len(df_listAB) == 1:
        # There was only one lev
        final_dict = df_listAB[0]
    else:
        final_dict["AhA"] = pd.concat([df["AhA"] for df in df_listAB])
        final_dict["AhB"] = pd.concat([df["AhB"] for df in df_listAB])
        if len(df_listC) > 0:
            final_dict["AhC"] = pd.concat([df["AhC"] for df in df_listC])

    return final_dict


def load_horizon_data_from_levs(base_path: Path, runs_path: Dict[str, Path]):
    data_dict = {}
    for run_name in runs_path.keys():
        path_list = list(base_path.glob(runs_path[run_name]))
        print(path_list)
        data_dict[run_name] = read_horizon_across_Levs(path_list)
    return data_dict


def flatten_dict(horizon_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    flattened_data = {}
    for run_name in horizon_data_dict.keys():
        for horizons in horizon_data_dict[run_name]:
            flattened_data[run_name + "_" + horizons] = horizon_data_dict[run_name][
                horizons
            ]
            # print(run_name+"_"+horizons)
    return flattened_data


def read_profiler(file_name):
    with h5py.File(file_name, "r") as f:
        steps = set()
        procs = set()
        names = []
        f.visit(names.append)
        for name in names:
            step = name.split(".")[0][4:]
            steps.add(step)
            if "Proc" in name:
                procs.add(name.split("/")[-1][4:-4])

        dict_list = []
        for step in steps:
            for proc in procs:
                data = f[f"Step{step}.dir/Proc{proc}.txt"][0].decode()

                lines = data.split("\n")
                time = float((lines[0].split("=")[-1])[:-1])

                curr_dict = {"t(M)": time, "step": step, "proc": proc}
                # Find where the columns end
                a = lines[4]
                event_end = a.find("Event") + 5
                cum_end = a.find("cum(%)") + 6
                exc_end = a.find("exc(%)") + 6
                inc_end = a.find("inc(%)") + 6

                for line in lines[6:-2]:
                    Event = line[:event_end].strip()
                    cum = float(line[event_end:cum_end].strip())
                    exc = float(line[cum_end:exc_end].strip())
                    inc = float(line[exc_end:inc_end].strip())
                    N = int(line[inc_end:].strip())
                    # print(a)
                    # a = line.split("  ")
                    # Event,cum,exc,inc,N = [i.strip() for i in a if i!= '']
                    curr_dict[f"{Event}_cum"] = cum
                    curr_dict[f"{Event}_exc"] = exc
                    curr_dict[f"{Event}_inc"] = inc
                    curr_dict[f"{Event}_N"] = N

                dict_list.append(curr_dict)
    return pd.DataFrame(dict_list)


def read_dat_file(file_name):
    cols_names = []
    # Read column names
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "#" not in line:
                # From now onwards it will be all data
                break
            elif "=" in line:
                if ("[" not in line) and ("]" not in line):
                    continue
                cols_names.append(line.split("=")[-1][1:-1].strip())
            else:
                continue

    return pd.read_csv(file_name, sep="\s+", comment="#", names=cols_names)


def hist_files_to_dataframe(file_path):
    # Function to parse a single line and return a dictionary of values
    def parse_line(line):
        data = {}
        # Find all variable=value pairs
        pairs = re.findall(r"([^;=\s]+)=\s*([^;]+)", line)
        for var, val in pairs:
            # Hist-GrDomain.txt should be parsed a little differently
            if "ResizeTheseSubdomains" in var:
                items = val.split("),")
                items[-1] = items[-1][:-1]
                for item in items:
                    name, _, vals = item.split("(")
                    r, l, m = vals[:-1].split(",")
                    data[f"{name}_R"] = int(r)
                    data[f"{name}_L"] = int(l)
                    data[f"{name}_M"] = int(m)
            else:
                data[var] = float(val) if re.match(r"^[\d.e+-]+$", val) else val
        return data

    with open(file_path, "r") as file:
        # Parse the lines
        data = []
        for line in file.readlines():
            data.append(parse_line(line.strip()))

        # Create a DataFrame
        df = pd.DataFrame(data)

    return df


# Files like AhACoefs.dat have unequal number of columns
def read_dat_file_uneq_cols(file_name):
    cols_names = []

    temp_file = "./temp.csv"
    col_length = 0
    with open(file_name, "r") as f:
        with open(temp_file, "w") as w:
            lines = f.readlines()
            for line in lines:
                if line[0] != "#":  # This is data
                    w.writelines(" ".join(line.split()[:col_length]) + "\n")
                if (
                    line[0:3] == "# [" or line[0:4] == "#  ["
                ):  # Some dat files have comments on the top
                    cols_names.append(line.split("=")[-1][1:-1].strip())
                    col_length = col_length + 1

    return pd.read_csv(temp_file, delim_whitespace=True, names=cols_names)


def read_dat_file_across_AA(file_pattern):
    # ApparentHorizons/Horizons.h5@AhA
    if "Horizons.h5@" in file_pattern:
        file_pattern, h5_key = file_pattern.split("@")

    path_pattern = file_pattern
    path_collection = []

    for folder_name in glob.iglob(path_pattern, recursive=True):
        if os.path.isdir(folder_name) or os.path.isfile(folder_name):
            path_collection.append(folder_name)
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


# Combines all the pvd files into a single file and save it in the base folder
def combine_pvd_files(base_folder: Path, file_pattern: str, output_path=None):
    pvd_start = """<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n"""
    pvd_end = "  </Collection>\n</VTKFile>"

    vis_folder_name = file_pattern.split("/")[-1][:-4]
    Lev = file_pattern[0:4]

    if output_path is None:
        output_path = f"{base_folder}/{vis_folder_name}_{Lev}.pvd"

    pvd_files = list(base_folder.glob(file_pattern))
    pvd_folders = list(base_folder.glob(file_pattern[:-4]))

    with open(output_path, "w") as write_file:
        write_file.writelines(pvd_start)
        for files in pvd_files:
            print(files)
            with files.open("r") as f:
                for line in f.readlines():
                    line = line.replace(vis_folder_name, str(files)[:-4])
                    if "DataSet" in line:
                        write_file.writelines(line)
        write_file.writelines(pvd_end)

    print(output_path)


def moving_average(array, avg_len):
    return np.convolve(array, np.ones(avg_len)) / avg_len


def moving_average_valid(array, avg_len):
    return np.convolve(array, np.ones(avg_len), "valid") / avg_len


def path_to_folder_name(folder_name):
    return folder_name.replace("/", "_")


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
    interpolated_data = CubicSpline(unique_x_data, unique_y_data, extrapolate=False)

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
    )

    plt.ylabel("")
    plt.title("" + append_to_title)

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
    runs_data_dict,
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
):
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
                except Exception as e:
                    print(run_name, unique_y_data)
                y_data = y_data - interpolated_data(constant_shift_val_time)

            #   print(f"{len(x_data)=},{len(y_data)=},{len(np.argsort(x_data))=},{type(x_data)=}")

            #   sorted_indices = x_data.argsort()
            #   x_data = x_data.iloc[sorted_indices]
            #   y_data = y_data.iloc[sorted_indices]
            legend = legend_dict[run_name]
            if legend is None:
                legend = run_name
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

            #   sorted_indices = np.argsort(x_data)
            #   x_data = x_data[sorted_indices]
            #   y_data = y_data[sorted_indices]
            legend = legend_dict[run_name]
            if legend is None:
                legend = run_name
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


def find_file(pattern):
    return glob.glob(pattern, recursive=True)[0]


def is_the_current_run_going_on(run_folder):
    if len(find_file(run_folder + "/**/" + "TerminationReason.txt")) > 0:
        return False
    else:
        return True


def plot_min_grid_spacing(runs_data_dict):
    """
    runs_data_dict should have dataframes with MinimumGridSpacing.dat data.
    The function will compute the min grid spacing along all domains and plot it.
    """
    keys = runs_data_dict.keys()
    if len(keys) == 0:
        print("There are no dataframes in the dict")

    for key in keys:
        t_step = runs_data_dict[key]["t"]
        min_val = runs_data_dict[key].drop(columns=["t"]).min(axis="columns")
        plt.plot(t_step, min_val, label=key)

    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Min Grid Spacing")
    plt.title("Min grid spacing in all domains")
    plt.show()


def plot_GrAdjustSubChunksToDampingTimes(runs_data_dict):
    keys = runs_data_dict.keys()
    if len(keys) > 1:
        print(
            "To plot the Tdamp for various quantities only put one dataframe in the runs_data_dict"
        )

    data: pd.DataFrame = runs_data_dict[list(keys)[0]]
    tdamp_keys = []
    for key in data.keys():
        if "Tdamp" in key:
            tdamp_keys.append(key)

    # Get a colormap
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(tdamp_keys)))

    t_vals = data["time"]
    for i, color, key in zip(range(len(tdamp_keys)), colors, tdamp_keys):
        if i % 2 == 0:
            plt.plot(t_vals, data[key], label=key, color=color)
        else:
            plt.plot(t_vals, data[key], label=key, color=color, linestyle="--")

    min_tdamp = data[tdamp_keys].min(axis="columns")
    plt.plot(
        t_vals,
        min_tdamp,
        label="min_tdamp",
        linewidth=3,
        linestyle="dotted",
        color="red",
    )

    plt.legend()
    plt.xlabel("time")
    plt.title(list(keys)[0])
    plt.show()


def add_max_and_min_val(runs_data_dict):
    # If we load a file with 5 columns with first being time, then find max and min values for all the other columns, at all times and add it to the dataframe.
    # Useful when you want to find like Linf across all domains at all times
    for run_name in runs_data_dict.keys():
        data_frame = runs_data_dict[run_name]
        t = data_frame.iloc[:, 0]
        max_val = np.zeros_like(t)
        min_val = np.zeros_like(t)
        for i in range(len(t)):
            max_val[i] = data_frame.iloc[i, 1:].max()
            min_val[i] = data_frame.iloc[i, 1:].max()

        # Add the values to the dataframe
        data_frame["max_val"] = max_val
        data_frame["min_val"] = min_val


def sort_run_data_dict(runs_data_dict: dict, sort_by=None):
    for run_name in runs_data_dict.keys():
        run_df = runs_data_dict[run_name]
        if sort_by is None:
            sort_by = run_df.keys()[0]
        runs_data_dict[run_name] = run_df.sort_values(by=sort_by)


save_folder_path = Path("./plots/").resolve()
if not save_folder_path.exists():
    raise Exception(f"Save folder {save_folder_path} does not exist")


# =================================================================================================
# power_diag.ipynb
# =================================================================================================


def join_str_with_underscore(str_list):
    a = str_list[0]
    for i in str_list[1:]:
        a = a + f"_{i}"
    return a


def get_top_name_from_number(top_number: int, subdomain_name: str) -> str:
    if re.match(r"Sphere", subdomain_name):
        return ["Bf0I1", "Bf1S2", "Bf1S2"][top_number]
    elif re.match(r"Cylinder", subdomain_name):
        return ["Bf0I1", "Bf1S1", "Bf2I1"][top_number]
    elif re.match(r"FilledCylinder", subdomain_name):
        return ["Bf0I1", "Bf1B2Radial", "Bf1B2"][top_number]
    else:
        raise Exception(f"{subdomain_name=} not recognized!")


def get_domain_name(col_name):
    def AMR_domains_to_decimal(subdoamin_name):
        # SphereC28.0.1
        a = subdoamin_name.split(".")
        # a = [SphereC28,0,1]
        decimal_rep = a[0] + "."
        # decimal_rep = SphereC28.
        for i in a[1:]:
            decimal_rep = decimal_rep + i
        # decimal_rep = SphereC28.01
        return decimal_rep

    if "on" in col_name:
        return AMR_domains_to_decimal(col_name.split(" ")[-1])
    if "." in col_name:
        return AMR_domains_to_decimal(col_name.split(" ")[-1])
    elif "_" in col_name:
        return col_name.split("_")[0]
    elif "MinimumGridSpacing" in col_name:
        return col_name.split("[")[-1][:-1]
    else:
        return col_name
        # raise Exception(f"{col_name} type not implemented in return_sorted_domain_names")


def filtered_domain_names(domain_names, filter):
    return [i for i in domain_names if re.match(filter, get_domain_name(i))]


def sort_spheres(sphere_list, reverse=False):
    if len(sphere_list) == 0:
        return []
    if "SphereA" in sphere_list[0]:
        return sorted(
            sphere_list,
            key=lambda x: float(get_domain_name(x).lstrip("SphereA")),
            reverse=reverse,
        )
    elif "SphereB" in sphere_list[0]:
        return sorted(
            sphere_list,
            key=lambda x: float(get_domain_name(x).lstrip("SphereB")),
            reverse=reverse,
        )
    elif "SphereC" in sphere_list[0]:
        return sorted(
            sphere_list,
            key=lambda x: float(get_domain_name(x).lstrip("SphereC")),
            reverse=reverse,
        )
    elif "SphereD" in sphere_list[0]:
        return sorted(
            sphere_list,
            key=lambda x: float(get_domain_name(x).lstrip("SphereD")),
            reverse=reverse,
        )
    elif "SphereE" in sphere_list[0]:
        return sorted(
            sphere_list,
            key=lambda x: float(get_domain_name(x).lstrip("SphereE")),
            reverse=reverse,
        )


def return_sorted_domain_names(domain_names):
    FilledCylinderCA = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}CA")
    CylinderCA = filtered_domain_names(domain_names, r"Cylinder.{0,2}CA")
    FilledCylinderEA = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}EA")
    CylinderEA = filtered_domain_names(domain_names, r"Cylinder.{0,2}EA")
    SphereA = sort_spheres(filtered_domain_names(domain_names, "SphereA"), reverse=True)
    CylinderSMA = filtered_domain_names(domain_names, r"CylinderS.{0,2}MA")
    FilledCylinderMA = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}MA")

    FilledCylinderMB = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}MB")
    CylinderSMB = filtered_domain_names(domain_names, r"CylinderS.{0,2}MB")
    SphereB = sort_spheres(filtered_domain_names(domain_names, "SphereB"), reverse=True)
    CylinderEB = filtered_domain_names(domain_names, r"Cylinder.{0,2}EB")
    FilledCylinderEB = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}EB")
    CylinderCB = filtered_domain_names(domain_names, r"Cylinder.{0,2}CB")
    FilledCylinderCB = filtered_domain_names(domain_names, r"FilledCylinder.{0,2}CB")

    SphereC = sort_spheres(
        filtered_domain_names(domain_names, "SphereC"), reverse=False
    )
    SphereD = sort_spheres(
        filtered_domain_names(domain_names, "SphereD"), reverse=False
    )
    SphereE = sort_spheres(
        filtered_domain_names(domain_names, "SphereE"), reverse=False
    )

    combined_columns = [
        FilledCylinderCA,
        CylinderCA,
        FilledCylinderEA,
        CylinderEA,
        SphereA,
        CylinderSMA,
        FilledCylinderMA,
        FilledCylinderMB,
        CylinderSMB,
        SphereB,
        CylinderEB,
        FilledCylinderEB,
        CylinderCB,
        FilledCylinderCB,
        SphereC,
        SphereD,
        SphereE,
    ]
    combined_columns = [item for sublist in combined_columns for item in sublist]

    # Just append the domains not following any patterns in the front. Mostly domains surrounding sphereA for high spin and mass ratios
    combined_columns_set = set(combined_columns)
    domain_names_set = set()
    for i in domain_names:
        domain_names_set.add(i)
    subdomains_not_sorted = list(domain_names_set - combined_columns_set)
    return subdomains_not_sorted + combined_columns


def limit_by_col_val(min_val, max_val, col_name, df):
    filter = (df[col_name] >= min_val) & (df[col_name] <= max_val)
    return df[filter]


def read_dat_file_single_bh(file_name):
    # Find the max number of columns
    with open(file_name, "r") as f:
        max_columns = max(len(line.split()) for line in f if not line.startswith("#"))
    return pd.read_csv(
        file_name,
        sep="\s+",
        comment="#",
        header=None,
        names=[str(i) for i in np.arange(-1, max_columns)],
    ).rename(columns={"-1": "t"})


def find_subdomains(path: Path):
    subdomain_set = set()
    for i in path.iterdir():
        if i.is_dir():
            subdomain_set.add(i.stem)

    return list(subdomain_set)


def find_topologies(path: Path):
    topologies_set = set()
    for i in path.iterdir():
        if i.is_file():
            topologies_set.add(i.stem.split("_")[0])

    return list(topologies_set)


def find_dat_file_names(path: Path):
    file_name_set = set()
    for i in path.iterdir():
        if i.is_file():
            file_name_set.add(i.stem.split("_")[1])

    return list(file_name_set)


def get_top_name_and_mode(name):
    # Bf0I1(12 modes).dat -> Bf0I1, 12
    top_name = name.split("(")[0]
    mode = int(name.split("(")[-1].split(" ")[0])
    return top_name, mode


def find_highest_modes_for_topologies(path: Path):
    highest_mode_dict = {}
    for i in path.iterdir():
        if i.is_file():
            top_name, mode = get_top_name_and_mode(i.stem)
            if top_name in highest_mode_dict:
                if highest_mode_dict[top_name] < mode:
                    highest_mode_dict[top_name] = mode
            else:
                highest_mode_dict[top_name] = mode

    return highest_mode_dict


def make_mode_dataframe(path: Path):
    highest_mode_dict = find_highest_modes_for_topologies(path)
    top_dataframe_list = {i: [] for i in highest_mode_dict}

    for i in path.iterdir():
        for top_name in highest_mode_dict:
            if (top_name + "(") in i.stem:
                top_dataframe_list[top_name].append(read_dat_file(i))

    top_mode_df_dict = {}
    for i, df_list in top_dataframe_list.items():
        result = pd.concat(df_list, ignore_index=True)

        # Remove duplicates based on 't' column (keep first occurrence)
        # result = result.drop_duplicates(subset='t', keep='first')

        # Sort by 't' and reset index
        top_mode_df_dict[i] = result.sort_values("t").reset_index(drop=True)
    return top_mode_df_dict


def filter_columns(
    cols: List[str],
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> List[str]:
    """
    Filter a list of column names using include and exclude regex patterns.

    Args:
        cols: List of column names to filter
        include_patterns: List of regex patterns to include (if None, includes all)
        exclude_patterns: List of regex patterns to exclude (if None, excludes none)

    Returns:
        List of filtered column names

    Examples:
        >>> cols = ['age_2020', 'age_2021', 'height_2020', 'weight_2021']
        >>> filter_columns(cols, ['age_.*'], ['.*2021'])
        ['age_2020']
    """
    # Handle None inputs
    include_patterns = include_patterns or [".*"]
    exclude_patterns = exclude_patterns or []

    # First, get columns that match any include pattern
    included_cols = set()
    for pattern in include_patterns:
        included_cols.update(col for col in cols if re.search(pattern, col))

    # Then remove any columns that match exclude patterns
    for pattern in exclude_patterns:
        included_cols = {col for col in included_cols if not re.search(pattern, col)}

    return sorted(list(included_cols))


def chain_filter_columns(
    cols: List[str],
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> List[str]:
    """
    Filter columns sequentially using chained include and exclude regex patterns.
    Each pattern filters from the result of the previous pattern.

    Args:
        cols: List of column names to filter
        include_patterns: List of regex patterns to include (if None, includes all)
        exclude_patterns: List of regex patterns to exclude (if None, excludes none)

    Returns:
        List of filtered column names

    Examples:
        >>> cols = ['age_2020_q1', 'age_2020_q2', 'age_2021_q1', 'height_2020_q1']
        >>> chain_filter_columns(cols, ['age_.*', '.*q1'], ['.*2021.*'])
        ['age_2020_q1']
    """
    # Handle None inputs
    include_patterns = include_patterns or [".*"]
    exclude_patterns = exclude_patterns or []

    # Start with all columns
    filtered_cols = set(cols)

    # Apply include patterns sequentially
    for pattern in include_patterns:
        filtered_cols = {col for col in filtered_cols if re.search(pattern, col)}

    # Apply exclude patterns sequentially
    for pattern in exclude_patterns:
        filtered_cols = {col for col in filtered_cols if not re.search(pattern, col)}

    return sorted(list(filtered_cols))


def sort_by_coefs_numbers(col_list: List[str]):
    with_coef_list = []
    without_coef_list = []
    for col in col_list:
        if "coef" not in col:
            without_coef_list.append(col)
        else:
            with_coef_list.append(col)
    return without_coef_list + sorted(
        with_coef_list, key=lambda x: int(x.split("_")[-1][4:])
    )


def get_extreme_coef_for_each_domain(df, min_or_max="min"):
    col_names = df.columns
    subdomains = set([i.split("_")[0] for i in col_names]) - set(["t(M)"])
    exterme_coef = {"t(M)": df["t(M)"]}
    for sd in subdomains:
        sd_cols = [i for i in col_names if f"{sd}_" in i]
        if min_or_max == "max":
            exterme_coef[sd] = df[sd_cols].max(axis=1)
        elif min_or_max == "min":
            exterme_coef[sd] = df[sd_cols].min(axis=1)
        else:
            raise Exception(
                f"Only supported values of min_or_max are min and max and not {min_or_max=}"
            )

    return pd.DataFrame(exterme_coef)


def load_power_diagonistics(PowDiag_path: Path):
    pow_diag_dict = {}
    for sd in find_subdomains(PowDiag_path):
        pow_diag_dict[sd] = {}
        sd_path = PowDiag_path / f"{sd}.dir"

        psi_pd = make_mode_dataframe(sd_path / f"Powerpsi.dir")
        kappa_pd = make_mode_dataframe(sd_path / f"Powerkappa.dir")
        # For each subdomain save things by topology
        for top in find_topologies(sd_path):
            pow_diag_dict[sd][top] = {}
            psi_pd_sorted_cols = sort_by_coefs_numbers(psi_pd[top].columns.to_list())
            pow_diag_dict[sd][top][f"psi_ps"] = psi_pd[top][psi_pd_sorted_cols]

            kappa_pd_sorted_cols = sort_by_coefs_numbers(
                kappa_pd[top].columns.to_list()
            )
            pow_diag_dict[sd][top][f"kappa_ps"] = kappa_pd[top][kappa_pd_sorted_cols]

            for dat_file in find_dat_file_names(sd_path):
                pow_diag_dict[sd][top][f"{dat_file}"] = read_dat_file(
                    sd_path / f"{top}_{dat_file}.dat"
                )

    return pow_diag_dict


def load_power_diagonistics_flat(
    PowDiag_path: Path,
    reload: bool = False,
    return_df: bool = True,
    load_dat_files_only: list[str] = None,
):
    cache_data = PowDiag_path / "pandas.pkl"
    if cache_data.exists():
        if not reload:
            with open(cache_data, "rb") as f:
                pow_diag_dict = pickle.load(f)
                print(f"Loaded from cache {cache_data}")
            return pow_diag_dict

    # Same as load_power_diagonistics but no nested dicts. This makes it easy to filter
    pow_diag_dict = {}
    for sd in find_subdomains(PowDiag_path):
        sd_path = PowDiag_path / f"{sd}.dir"

        for top in find_topologies(sd_path):
            for dat_file in find_dat_file_names(sd_path):
                if load_dat_files_only is not None:
                    for allowed_dat_files in load_dat_files_only:
                        if re.search(allowed_dat_files, dat_file) is None:
                            continue
                        else:
                            print(dat_file)
                pow_diag_dict[f"{sd}_{top}_{dat_file}"] = read_dat_file(
                    sd_path / f"{top}_{dat_file}.dat"
                )
        if load_dat_files_only is not None:
            print(sd_path)
            continue

        psi_pd = make_mode_dataframe(sd_path / f"Powerpsi.dir")
        kappa_pd = make_mode_dataframe(sd_path / f"Powerkappa.dir")
        # For each subdomain save things by topology
        for top in find_topologies(sd_path):
            psi_pd_sorted_cols = sort_by_coefs_numbers(psi_pd[top].columns.to_list())
            pow_diag_dict[f"{sd}_{top}_psi_ps"] = psi_pd[top][psi_pd_sorted_cols]

            kappa_pd_sorted_cols = sort_by_coefs_numbers(
                kappa_pd[top].columns.to_list()
            )
            pow_diag_dict[f"{sd}_{top}_kappa_ps"] = kappa_pd[top][kappa_pd_sorted_cols]

        print(sd_path)

    if return_df:
        # This can be definitely merged with the stuff above but it's fast enough anyways
        flat_dict = {}
        flat_dict["t"] = pow_diag_dict[rc(list(pow_diag_dict.keys()))]["t"]
        for key, item in pow_diag_dict.items():
            for col in item.columns:
                if "t" == col:
                    continue
                else:
                    flat_dict[f"{key}_{col}"] = item[col]

        flat_df = pd.DataFrame(flat_dict)
        with open(cache_data, "wb") as f:
            pickle.dump(flat_df, f)
            print(f"Cached at {cache_data}")

        return flat_df

    return pow_diag_dict


def convert_series_to_coeff_df(data, top_num):
    irr_top_regex = 0
    match top_num:
        case 0:
            irr_top_regex = r"Bf0"  # First top
        case 1:
            irr_top_regex = (
                r"Bf1(S\d|B2R)"  # Second top, S2 for spheres, B2 for filled cylinders
            )
        case 2:
            irr_top_regex = r"((Bf1S2|Bf1B2_)|Bf2)"  # Thrid top S2 for spheres, B2 radial for filled cylinders
        case _:
            raise Exception(f"{top_num=} should be one of [0,1,2]")

    indices = set(data.index) - set(["t(M)"])
    indices = set([i for i in indices if re.search(irr_top_regex, i)])

    subdomains = set([i.split("_")[0] for i in indices])
    # The coefs are sorted?? Do not assume but they are

    data_dict = {sd: {} for sd in subdomains}

    def coef_number(x):
        return int(x.split("_")[-1][4:])

    max_coef = 0
    for sd in subdomains:
        for idx in indices:
            if f"{sd}_" in idx:
                if pd.notna(data[idx]):
                    data_dict[sd][coef_number(idx)] = data[idx]
        data_dict[sd] = dict(sorted(data_dict[sd].items()))
        if len(data_dict[sd]) > max_coef:
            max_coef = len(data_dict[sd])

    pd_data = pd.DataFrame(data_dict)
    return pd_data


def series_closest_to_time(t, df):
    time_index = np.where(df["t(M)"] > t)[0][0]
    time = df["t(M)"][time_index]
    return time, df.iloc[time_index].copy()


# =================================================================================================
# =================================================================================================
# PLOTTING
# =================================================================================================
# =================================================================================================

# =================================================================================================
# Constraints
# =================================================================================================

L15_main_legend = {
    "high_accuracy_main_L1": "Old Level 1",
    "high_accuracy_main_L2": "Old Level 2",
    "high_accuracy_main_L3": "Old Level 3",
    "high_accuracy_main_L4": "Old Level 4",
    "high_accuracy_main_L5": "Old Level 5",
}

L15_main_runs = {
    "high_accuracy_main_L1": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev1_A?/Run/",
    "high_accuracy_main_L2": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev2_A?/Run/",
    "high_accuracy_main_L3": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev3_A?/Run/",
    "high_accuracy_main_L4": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev4_A?/Run/",
    "high_accuracy_main_L5": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev5_A?/Run/",
}

L15_ode_fix_legend = {
    "high_accuracy_L1": "Ode Fix Level 1",
    "high_accuracy_L2": "Ode Fix Level 2",
    "high_accuracy_L3": "Ode Fix Level 3",
    "high_accuracy_L4": "Ode Fix Level 4",
    "high_accuracy_L5": "Ode Fix Level 5",
}
L15_ode_fix_runs = {
    "high_accuracy_L1": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev1_A?/Run/",
    "high_accuracy_L2": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev2_A?/Run/",
    "high_accuracy_L3": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev3_A?/Run/",
    "high_accuracy_L4": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev4_A?/Run/",
    "high_accuracy_L5": "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev5_A?/Run/",
}


L16_set1_legend = {
    "6_set1_L6s1": "Set1 Level 1",
    "6_set1_L6s2": "Set1 Level 2",
    "6_set1_L6s3": "Set1 Level 3",
    "6_set1_L6s4": "Set1 Level 4",
    "6_set1_L6s5": "Set1 Level 5",
    "6_set1_L6s6": "Set1 Level 6",
}
L16_set1_runs = {
    "6_set1_L6s1": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev1_A?/Run/",
    "6_set1_L6s2": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev2_A?/Run/",
    "6_set1_L6s3": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev3_A?/Run/",
    "6_set1_L6s4": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev4_A?/Run/",
    "6_set1_L6s5": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev5_A?/Run/",
    "6_set1_L6s6": "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev6_A?/Run/",
}


# ==============================================================================

SKIP_THIS = False

runs_to_plot_list = [L15_main_runs, L15_ode_fix_runs, L16_set1_runs]
legend_dict_list = [L15_main_legend, L15_ode_fix_legend, L16_set1_legend]
runs_set_name_list = ["L15_main", "L15_ode_fix", "L16_set1"]

for runs_to_plot, legend_dict, runs_set_name in zip(
    runs_to_plot_list, legend_dict_list, runs_set_name_list
):
    if SKIP_THIS:
        continue

    data_file_path = "ConstraintNorms/GhCe_Linf.dat"
    column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = True
    y_axis_list = None
    x_axis = "t(M)"

    plot_abs_diff = False

    minT = 1205
    maxT = 4000

    def plot_fun(x, y, label):
        return plt.semilogy(x, y, label=label)

    append_to_title = ""
    if "@" in data_file_path:
        append_to_title = " HorizonBH=" + data_file_path.split("@")[-1]

    with plt.style.context("ggplot"):
        plt.rcParams["figure.figsize"] = (6, 6)
        plt.rcParams["figure.autolayout"] = True

        y_axis = "Linf(GhCe) on SphereA0"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_SphereA0_Linf_GhCe.png"
        plt.savefig(save_name, dpi=300)
        plt.clf()
        print(f"Saved {save_name}!\n")

        y_axis = "Linf(GhCe) on SphereC6"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_SphereC6_Linf_GhCe.png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved {save_name}!\n")
        plt.clf()

    # ==============================================================================

    data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
    column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = True
    y_axis_list = None
    x_axis = "t(M)"

    plot_abs_diff = False

    minT = 1205
    maxT = 4000

    def plot_fun(x, y, label):
        return plt.semilogy(x, y, label=label)

    append_to_title = ""
    if "@" in data_file_path:
        append_to_title = " HorizonBH=" + data_file_path.split("@")[-1]

    with plt.style.context("ggplot"):
        plt.rcParams["figure.figsize"] = (6, 6)
        plt.rcParams["figure.autolayout"] = True

        y_axis = "Linf(NormalizedGhCe) on SphereA0"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = (
            save_folder_path / f"{runs_set_name}_SphereA0_Linf_NormalizedGhCe.png"
        )
        plt.savefig(save_name, dpi=300)
        plt.clf()
        print(f"Saved {save_name}!\n")

        y_axis = "Linf(NormalizedGhCe) on SphereC6"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = (
            save_folder_path / f"{runs_set_name}_SphereC6_Linf_NormalizedGhCe.png"
        )
        plt.savefig(save_name, dpi=300)
        print(f"Saved {save_name}!\n")
        plt.clf()

    # ==============================================================================

    data_file_path = "ConstraintNorms/GhCe_Norms.dat"
    column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = True
    y_axis_list = None
    x_axis = "t(M)"

    plot_abs_diff = False

    minT = 1205
    maxT = 4000

    def plot_fun(x, y, label):
        return plt.semilogy(x, y, label=label)

    append_to_title = ""
    if "@" in data_file_path:
        append_to_title = " HorizonBH=" + data_file_path.split("@")[-1]

    with plt.style.context("ggplot"):
        plt.rcParams["figure.figsize"] = (6, 6)
        plt.rcParams["figure.autolayout"] = True

        y_axis = "L2(GhCe)"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_L2(GhCe).png"
        plt.savefig(save_name, dpi=300)
        plt.clf()
        print(f"Saved {save_name}!\n")

        y_axis = "Linf(GhCe)"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_Linf(GhCe).png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved {save_name}!\n")
        plt.clf()

        y_axis = "VolLp(GhCe)"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_VolLp(GhCe).png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved {save_name}!\n")
        plt.clf()

    # ==============================================================================

    data_file_path = "ConstraintNorms/NormalizedGhCe_Norms.dat"
    column_names, runs_data_dict = load_data_from_levs(runs_to_plot, data_file_path)

    moving_avg_len = 0
    save_path = None
    diff_base = None
    constant_shift_val_time = None
    plot_abs_diff = True
    y_axis_list = None
    x_axis = "t(M)"

    plot_abs_diff = False

    minT = 1205
    maxT = 4000

    def plot_fun(x, y, label):
        return plt.semilogy(x, y, label=label)

    append_to_title = ""
    if "@" in data_file_path:
        append_to_title = " HorizonBH=" + data_file_path.split("@")[-1]

    with plt.style.context("ggplot"):
        plt.rcParams["figure.figsize"] = (6, 6)
        plt.rcParams["figure.autolayout"] = True

        y_axis = "L2(NormalizedGhCe)"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_L2(NormalizedGhCe).png"
        plt.savefig(save_name, dpi=300)
        plt.clf()
        print(f"Saved {save_name}!\n")

        y_axis = "Linf(NormalizedGhCe)"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_Linf(NormalizedGhCe).png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved {save_name}!\n")
        plt.clf()

        y_axis = "VolLp(NormalizedGhCe)"
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
            diff_base=diff_base,
            plot_abs_diff=plot_abs_diff,
            constant_shift_val_time=constant_shift_val_time,
            append_to_title=append_to_title,
        )

        plt.title("")
        plt.ylabel(y_axis)
        plt.xlabel("t(M)")
        plt.legend(loc="upper right")
        #   plt.ylim(1e-8, 1e-5)
        #   plt.ylim(1e-12, 1e-6)

        plt.tight_layout()
        save_name = save_folder_path / f"{runs_set_name}_VolLp(NormalizedGhCe).png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved {save_name}!\n")
        plt.clf()


# =================================================================================================
# Power spectrum
# =================================================================================================

L15_main_legend = {
    "high_accuracy_main_L1": "Old Level 1",
    "high_accuracy_main_L2": "Old Level 2",
    "high_accuracy_main_L3": "Old Level 3",
    "high_accuracy_main_L4": "Old Level 4",
    "high_accuracy_main_L5": "Old Level 5",
}

L15_main_h5_files = {
    "high_accuracy_main_L1": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/h5_files_Lev1"
    ),
    "high_accuracy_main_L2": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/h5_files_Lev2"
    ),
    "high_accuracy_main_L3": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/h5_files_Lev3"
    ),
    "high_accuracy_main_L4": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/h5_files_Lev4"
    ),
    "high_accuracy_main_L5": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/h5_files_Lev5"
    ),
}

L15_ode_fix_legend = {
    "high_accuracy_L1": "Ode Fix Level 1",
    "high_accuracy_L2": "Ode Fix Level 2",
    "high_accuracy_L3": "Ode Fix Level 3",
    "high_accuracy_L4": "Ode Fix Level 4",
    "high_accuracy_L5": "Ode Fix Level 5",
}

L15_ode_fix_h5_files = {
    "high_accuracy_L1": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/h5_files_Lev1"
    ),
    "high_accuracy_L2": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/h5_files_Lev2"
    ),
    "high_accuracy_L3": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/h5_files_Lev3"
    ),
    "high_accuracy_L4": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/h5_files_Lev4"
    ),
    "high_accuracy_L5": Path(
        "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/h5_files_Lev5"
    ),
}

L16_set1_legend = {
    "6_set1_L6s1": "Set1 Level 1",
    "6_set1_L6s2": "Set1 Level 2",
    "6_set1_L6s3": "Set1 Level 3",
    "6_set1_L6s4": "Set1 Level 4",
    "6_set1_L6s5": "Set1 Level 5",
    "6_set1_L6s6": "Set1 Level 6",
}

L16_set1_h5_files = {
    "6_set1_L6s1": Path(
        "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev1"
    ),
    "6_set1_L6s2": Path(
        "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev2"
    ),
    "6_set1_L6s3": Path(
        "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev3"
    ),
    "6_set1_L6s4": Path(
        "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev4"
    ),
    "6_set1_L6s5": Path(
        "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev5"
    ),
    "6_set1_L6s6": Path(
        "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev6"
    ),
}


# ==============================================================================

SKIP_THIS = False

levs_to_plot_list = [5, 6]
domains_to_plot_list = ["SphereA0", "SphereC6"]
vars_to_plot_list = ["psi", "kappa"]
topologies_to_plot_list = [0, 1]
runs_set_name_list = ["L15_main", "L15_ode_fix", "L16_set1"]

runs_to_plot_list = [L15_main_h5_files, L15_ode_fix_h5_files, L16_set1_h5_files]
runs_legend_list = [L15_main_legend, L15_ode_fix_legend, L16_set1_legend]

for runs_to_plot, runs_legend, runs_set_name in zip(
    # runs_to_plot_list, runs_legend_list, runs_set_name_list
    runs_to_plot_list,
    runs_legend_list,
    runs_set_name_list,
):
    if SKIP_THIS:
        continue

    with plt.style.context("ggplot"):
        plt.rcParams["figure.figsize"] = (6, 6)
        plt.rcParams["figure.autolayout"] = True

        for h5_path_key, domain, top_num, var in itertools.product(
            runs_to_plot,
            domains_to_plot_list,
            topologies_to_plot_list,
            vars_to_plot_list,
        ):
            h5_path = runs_to_plot[h5_path_key]
            current_lev = int(str(h5_path)[-1])
            if current_lev not in levs_to_plot_list:
                continue

            folder_paths = [
                Path(
                    f"{h5_path}/extracted-PowerDiagnostics/{domain}.dir/Power{var}.dir"
                )
            ]
            top_data_dict = {
                join_str_with_underscore(
                    str(folder_path).split("/")[-5:-3] + [domain]
                ): make_mode_dataframe(folder_path)
                for folder_path in folder_paths
            }

            top_name = get_top_name_from_number(top_num, domain)

            t_min = 1210
            t_max = 4000

            min_coef = -1
            max_coef = 100
            plot_slice = slice(0, None)

            title = ""
            has_more_than_one = len(list(top_data_dict.keys())) > 1
            style_list = ["-", ":", "--", "-."]
            num_legends = 0
            single_legend, max_col = True, -1
            for key_num, A, key in zip(
                range(100), string.ascii_uppercase, top_data_dict
            ):
                plt.gca().set_prop_cycle(None)
                data = top_data_dict[key][top_name]
                data = limit_by_col_val(t_min, t_max, "t", data)
                data = data.dropna(
                    axis=1, how="all"
                )  # Some columns will have just nans remove those
                column_names = data.columns[1:]
                visual_data = data[column_names]

                cols_to_use = [i for i in data.columns if "t" not in i]
                # df = np.log10(data[cols_to_use])
                df = data[cols_to_use]
                df["row_min"] = df.min(axis=1)
                df["row_max"] = df.max(axis=1)
                df["row_mean"] = df.mean(axis=1)
                df["row_std"] = df.std(axis=1)

                for i in cols_to_use[plot_slice]:
                    coef_num = int(i[4:])
                    if coef_num < min_coef or coef_num > max_coef:
                        continue
                    # plt.plot(data['t'], df[f'{i}'])
                    label = f"$a_{{{i[4:]}}}$"
                    if has_more_than_one:
                        if max_col < coef_num:
                            label = f"{A}: {i}"
                            num_legends = num_legends + 1
                        else:
                            label = None
                        max_col = max(coef_num, max_col)
                    plt.plot(
                        data["t"],
                        df[f"{i}"],
                        label=label,
                        linestyle=style_list[key_num % len(style_list)],
                    )

                if has_more_than_one:
                    title = title + f"{style_list[key_num % len(style_list)]}  {A}: "
                title = title + f"{domain} of {runs_legend[h5_path_key]} : {top_name}\n"

            if num_legends > 20:
                plt.legend(ncol=int(np.ceil(num_legends / 20)))
            else:
                plt.legend()

            plt.legend(loc="upper right")
            plt.title(title[:-1])
            plt.xlabel("t(M)")
            plt.ylabel(f"Power {var}")
            # plt.ylim(-16.5, 0.5)
            plt.yscale("log")
            plt.ylim(5e-17, 5)
            plt.grid(False)
            plt.tight_layout()

            save_name = (
                save_folder_path
                / f"{runs_set_name}_L{current_lev}_PS_{var}_{domain}_{top_num}.png"
            )
            plt.savefig(save_name, dpi=300)
            print(f"Saved {save_name}!\n")
            plt.clf()
