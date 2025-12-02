# %%
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d


def filter_by_regex(regex, col_list, exclude=False):
    filtered_set = set()
    if type(regex) is list:
        for reg in regex:
            for i in col_list:
                if re.search(reg, i):
                    filtered_set.add(i)
    else:
        for i in col_list:
            if re.search(regex, i):
                filtered_set.add(i)

    filtered_list = list(filtered_set)
    if exclude:
        col_list_copy = list(col_list.copy())
        for i in filtered_list:
            if i in col_list_copy:
                col_list_copy.remove(i)
        filtered_list = col_list_copy

    # Restore the original order
    filtered_original_ordered_list = []
    for i in list(col_list):
        if i in filtered_list:
            filtered_original_ordered_list.append(i)
    return filtered_original_ordered_list


def limit_by_col_val(min_val, max_val, col_name, df):
    filter = (df[col_name] >= min_val) & (df[col_name] <= max_val)
    return df[filter]


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


def moving_median_valid(array, avg_len):
    from scipy import ndimage

    # Apply median filter to full array, then trim to 'valid' size
    filtered = ndimage.median_filter(array, size=avg_len)
    # Trim edges to match 'valid' convolution behavior
    trim = (avg_len - 1) // 2
    if avg_len % 2 == 0:  # even window size
        return filtered[trim : -trim - 1]
    else:  # odd window size
        return filtered[trim:-trim] if trim > 0 else filtered


def moving_percentile_valid(array, avg_len, percentile=50):
    from scipy import ndimage

    filtered = ndimage.percentile_filter(array, percentile=percentile, size=avg_len)
    trim = (avg_len - 1) // 2
    if avg_len % 2 == 0:
        return filtered[trim : -trim - 1]
    else:
        return filtered[trim:-trim] if trim > 0 else filtered


def moving_trimmed_mean_valid(array, avg_len, trim_percent=0.2):
    import pandas as pd
    from scipy import stats

    # Use pandas rolling with trimmed mean
    result = (
        pd.Series(array)
        .rolling(avg_len)
        .apply(lambda x: stats.trim_mean(x, trim_percent))
        .dropna()
        .values
    )
    return result


def moving_robust_mean_valid(array, avg_len, mad_threshold=3):
    """Moving average that excludes outliers based on MAD"""
    import pandas as pd
    from scipy import stats

    def robust_mean(window):
        median = np.median(window)
        mad = stats.median_abs_deviation(window)
        mask = np.abs(window - median) <= mad_threshold * mad
        return np.mean(window[mask]) if np.any(mask) else median

    result = pd.Series(array).rolling(avg_len).apply(robust_mean).dropna().values
    return result


# def moving_average_valid(array, avg_len):
#     # return moving_median_valid(array, avg_len)
#     return moving_percentile_valid(array, avg_len, percentile=5)
#     # return moving_trimmed_mean_valid(array, avg_len, trim_percent=0.005)
#     # return moving_robust_mean_valid(array, avg_len, mad_threshold=3)


def path_to_folder_name(folder_name):
    return folder_name.replace("/", "_")


def find_file(pattern):
    return glob.glob(pattern, recursive=True)[0]


def is_the_current_run_going_on(run_folder):
    if len(find_file(run_folder + "/**/" + "TerminationReason.txt")) > 0:
        return False
    else:
        return True


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
            if "t(M)" in run_df.keys():
                sort_by = "t(M)"
            else:
                sort_by = run_df.keys()[0]
        runs_data_dict[run_name] = run_df.sort_values(by=sort_by)


def noise_function(x, y, df, y_axis, scipy_or_np="scipy", window=25):
    if scipy_or_np == "scipy":
        running_mean = uniform_filter1d(y, size=window, mode="nearest")
        noise_estimate = y - running_mean
    elif scipy_or_np == "np":
        running_mean = np.convolve(y, np.ones(window), mode="valid") / window
        noise_estimate = np.full(y.shape[0], np.nan, dtype=y.dtype)
        noise_estimate[window // 2 - 1 : -window // 2] = (
            y[window // 2 - 1 : -window // 2] - running_mean
        )
    else:
        raise Exception(f"Invalid scipy_or_np value: {scipy_or_np}")

    return x, noise_estimate, f"noise_{window}_{y_axis}"


def derivative_function(x, y, df, y_axis, order=1):
    if order == 1:
        grad = np.gradient(y, x)
    elif order == 2:
        grad = np.gradient(np.gradient(y, x), x)
    else:
        raise Exception(f"Invalid order value: {order}")

    return x, grad, f"deriv:{order}_{y_axis}"


def compute_center(x, y, df, y_axis):
    center = np.zeros_like(y)
    for i in [
        "CoordCenterInertial_0",
        "CoordCenterInertial_1",
        "CoordCenterInertial_2",
    ]:
        assert i in df.columns, f"Column {i} not found in DataFrame"
        center += df[i] ** 2

    return x, np.sqrt(center), "center"


def compute_dt_center(x, y, df, y_axis):
    center = np.zeros_like(y)
    for i in [
        "CoordCenterInertial_0",
        "CoordCenterInertial_1",
        "CoordCenterInertial_2",
    ]:
        assert i in df.columns, f"Column {i} not found in DataFrame"
        center += df[i] ** 2

    return x, np.gradient(np.sqrt(center), x), "dt_center"


def min_max_r_ratio(x, y, df, y_axis):
    ratio = np.zeros_like(y)
    for i in [
        "max(r)",
        "min(r)",
    ]:
        assert i in df.columns, f"Column {i} not found in DataFrame"
    ratio = df["min(r)"] / df["max(r)"]
    return x, ratio, "ratio"


def index_constraints_norm(x, y, df, y_axis, index_num=3, norm="Linf"):
    index_cols = [i for i in df.columns if f"{index_num}Con" in i]

    if norm == "Linf":
        index_norm = np.max(np.abs(df[index_cols]), axis=1)
    elif norm == "L2":
        index_norm = np.linalg.norm(df[index_cols], axis=1)
    elif norm == "RMS":
        index_norm = np.linalg.norm(df[index_cols], axis=1) / np.sqrt(len(index_cols))
    else:
        raise Exception(f"Invalid norm value: {norm}")

    label = f"{norm}({index_num}Con)"
    return x, index_norm, label


def get_num_points_subdomains(
    x, y, df, y_axis, regex_for_sd=r".*", print_sd_names=False
):
    # only consider col names that have _ in them
    filtered_cols = [col for col in df.columns if "_" in col]
    if len(filtered_cols) == 0:
        raise Exception("No subdomains with _ columns found in the dataframe")
    sd_names = set(
        [col.split("_")[0] for col in filtered_cols if re.match(regex_for_sd, col)]
    )
    if print_sd_names:
        print(f"All subdomains in the dataframe: {sd_names}")
    sd_names = sd_names - set(["t(M)", "TOfLastChange", "StartTime", "Version"])
    if len(sd_names) == 0:
        raise Exception(f"No subdomains found matching {regex_for_sd}")
    num_points = 0
    if print_sd_names:
        print(f"Subdomains matching {regex_for_sd}:")
    for sd in sd_names:
        points_in_current_sd = 0
        R_pts = df[f"{sd}_R"].fillna(0)
        L_pts = df[f"{sd}_L"].fillna(0)
        M_pts = df[f"{sd}_M"].fillna(0)
        points_in_current_sd = R_pts * L_pts * M_pts
        if "Sphere" in sd:
            points_in_current_sd = points_in_current_sd // 2
        if print_sd_names:
            print(f"{sd}: {points_in_current_sd} points")
        num_points += points_in_current_sd

    return x, num_points, f"NumPoints in {regex_for_sd}"


def compute_power_in_modes(x, y, df, y_axis, L_list=None):
    def get_l(key):
        return int(key.split(",")[0])

    if L_list is None:
        cols_to_use = []
        for col in df.columns:
            if col == "t(M)":
                continue
            if "diff_" in col:
                raise Exception(
                    "You have diff base enabled, this is not what you want!! Use the function compute_diff_power_in_modes"
                )
            cols_to_use.append(col)
        power = np.linalg.norm(df[cols_to_use], axis=1)
        return x, power, "power"
    else:
        cols_to_use = []
        for col in df.columns:
            if col == "t(M)":
                continue
            if "diff_" in col:
                raise Exception(
                    "You have diff base enabled, this is not what you want!! Use the function compute_diff_power_in_modes"
                )
            l_value = get_l(col)
            if l_value in L_list:
                cols_to_use.append(col)
        power = np.linalg.norm(df[cols_to_use], ord=2, axis=1)
        return x, power, f"power{L_list}"


def data_frame_diff_with_interpolation(df1, df2):
    # Will interpolate the df2 onto the time of df1 and then compute the difference
    # No interpolation is done if the time values are the same
    # Assumes that the time column is sorted and is named "t(M)"
    t1 = df1["t(M)"]
    # mint1 = t1.min()
    # maxt1 = t1.max()
    t2 = df2["t(M)"]
    if not np.array_equal(t1, t2):
        # interpolating_t_vals = t2[(t2 >= mint1) & (t2 <= maxt1)]
        diff = df1.copy()
        for col in df2.columns:
            # df1 cols will have diff_abs cols added by the main function. ignore those
            if col == "t(M)":
                continue
            diff[col] = df1[col] - sp.interpolate.CubicSpline(
                t2, df2[col], extrapolate=False
            )(t1)
        return diff
    else:
        diff = df1 - df2
        diff["t(M)"] = df1["t(M)"]
        return diff


def compute_diff_power_in_modes(x, y, df, y_axis, diff_base_df, L_list=None):
    def get_l(key):
        return int(key.split(",")[0])

    # def get_pandas

    if L_list is None:
        cols_to_use = []
        for col in df.columns:
            if col == "t(M)":
                continue
            if "diff_" in col:
                continue  # These keys will be added, just ignore
            cols_to_use.append(col)
        ylabel = "power"
    else:
        cols_to_use = []
        for col in df.columns:
            if col == "t(M)":
                continue
            if "diff_" in col:
                continue  # These keys will be added, just ignore
            l_value = get_l(col)
            if l_value in L_list:
                cols_to_use.append(col)
        ylabel = f"power{L_list}"

    power = np.linalg.norm(
        data_frame_diff_with_interpolation(df, diff_base_df)[cols_to_use], axis=1
    )
    return x, power, ylabel


def get_drift_from_center(x, y, df, y_axis):
    center = np.zeros_like(y)
    for i in [
        "COM_X",
        "COM_Y",
        "COM_Z",
    ]:
        assert i in df.columns, f"Column {i} not found in DataFrame"
        center += df[i] ** 2

    center = np.sqrt(center)
    return x, center, "COM_mag"


def add_norm_constraints(
    runs_data_dict,
    index_num=[1, 2, 3],
    norm=["Linf", "L2", "RMS"],
    subdomains_seperately=False,
    replace_runs_data_dict=False,
):
    for key, run_data in runs_data_dict.items():
        new_data = {}  # Add data later in one go otherwise index filtering will pickup added stuff
        for which_index in index_num:
            if not subdomains_seperately:
                index_cols = [
                    i
                    for i in run_data.columns
                    if re.search(rf"{which_index}Con.+\)", i)
                ]  # Prevent picking up new added cols on reruns
                for which_norm in norm:
                    label = f"{which_norm}({which_index}Con)"
                    if which_norm == "Linf":
                        new_data[label] = np.max(np.abs(run_data[index_cols]), axis=1)
                    elif which_norm == "L2":
                        new_data[label] = np.linalg.norm(run_data[index_cols], axis=1)
                    elif which_norm == "RMS":
                        new_data[label] = np.linalg.norm(
                            run_data[index_cols], axis=1
                        ) / np.sqrt(len(index_cols))
                    else:
                        raise Exception(f"Invalid norm value: {norm}")
            else:
                subdomains = [
                    col.split(" on ")[1] for col in run_data.columns if " on " in col
                ]
                subdomains = list(set(subdomains))
                for sd in subdomains:
                    index_cols = [
                        i
                        for i in run_data.columns
                        if (re.search(rf"{which_index}Con.+\)", i)) and (sd in i)
                    ]  # Prevent picking up new added cols on reruns
                    for which_norm in norm:
                        label = f"{which_norm}({which_index}Con) on {sd}"
                        if which_norm == "Linf":
                            new_data[label] = np.max(
                                np.abs(run_data[index_cols]), axis=1
                            )
                        elif which_norm == "L2":
                            new_data[label] = np.linalg.norm(
                                run_data[index_cols], axis=1
                            )
                        elif which_norm == "RMS":
                            new_data[label] = np.linalg.norm(
                                run_data[index_cols], axis=1
                            ) / np.sqrt(len(index_cols))
                        else:
                            raise Exception(f"Invalid norm value: {norm}")
                pass

        new_df = pd.DataFrame(new_data, index=run_data.index)
        if replace_runs_data_dict:
            new_df["t(M)"] = run_data["t(M)"]
            runs_data_dict[key] = new_df
        else:
            runs_data_dict[key] = pd.concat([run_data, new_df], axis=1)
            runs_data_dict[key] = runs_data_dict[key].loc[
                :, ~runs_data_dict[key].columns.duplicated(keep="last")
            ]
        new_indices = list(new_data.keys())  # Just of the last lev
    return new_indices, runs_data_dict


# %% [markdown]
# ### Add number of points per subdomain


# %%
def num_points_per_subdomain(runs_data_dict, replace_runs_data_dict=False):
    for key, run_data in runs_data_dict.items():
        new_data = {}  # Add data later in one go otherwise index filtering will pickup added stuff
        sd_names = set([col.split("_")[0] for col in run_data.columns])
        sd_names = sd_names - set(["t(M)", "TOfLastChange", "StartTime", "Version"])
        for sd in sd_names:
            points_in_current_sd = 0
            R_pts = run_data[f"{sd}_R"]
            L_pts = run_data[f"{sd}_L"]
            M_pts = run_data[f"{sd}_M"]
            points_in_current_sd = R_pts * L_pts * M_pts
            if "Sphere" in sd:
                points_in_current_sd = points_in_current_sd // 2
            new_data[f"NumPoints in {sd}"] = points_in_current_sd

        new_df = pd.DataFrame(new_data, index=run_data.index)
        if replace_runs_data_dict:
            new_df["t(M)"] = run_data["t(M)"]
            new_df.drop_duplicates(keep="last", inplace=True)
            runs_data_dict[key] = new_df
        else:
            runs_data_dict[key] = pd.concat([run_data, new_df], axis=1)
            runs_data_dict[key] = runs_data_dict[key].loc[
                :, ~runs_data_dict[key].columns.duplicated(keep="last")
            ]
        new_indices = list(new_data.keys())  # Just of the last lev
    return new_indices, runs_data_dict


# %% [markdown]
# ### Add power modes


# %%
def change_to_power_in_L_modes(runs_data_dict, LArr=None, RMS_Power=True, debug=False):
    for key in runs_data_dict:
        cols = list(runs_data_dict[key].columns)
        Lmax = int(cols[-1].split(",")[0])

        power_dict = {"t(M)": runs_data_dict[key]["t(M)"]}
        for L in range(0, Lmax + 1):
            num_m_modes_added = 0
            for m in range(-L, L + 1):
                mode_key = f"{L},{m}"
                if mode_key in cols:
                    if debug:
                        print(f"{L},{m}")
                    if str(L) not in power_dict:
                        # We need to square the magnitude of the mode, it is complex
                        power_dict[str(L)] = np.abs(runs_data_dict[key][mode_key]) ** 2
                    else:
                        power_dict[str(L)] += np.abs(runs_data_dict[key][mode_key]) ** 2
                    num_m_modes_added += 1

            if num_m_modes_added > 0:
                if RMS_Power:
                    if debug:
                        print(f"Normalizing power for L={L} by ({num_m_modes_added})")
                    # In theory num_m_modes_added should be 2*L+1, there are some var where negative m modes are not present.
                    power_dict[str(L)] = np.sqrt(power_dict[str(L)]) / num_m_modes_added
                else:
                    # Total power is the sum of squares
                    power_dict[str(L)] = power_dict[str(L)]

        runs_data_dict[key] = pd.DataFrame(power_dict)


# %% [markdown]
# ### Mismatch related functions


# %%
def interpolate_df_to_target(df1, df2, time_col="t(M)", t_min=None, t_max=None):
    # 1. Determine Source vs Target based on density (length)
    df1_is_source = len(df1) > len(df2)
    source_df = df1 if df1_is_source else df2
    target_df = df2 if df1_is_source else df1

    # 2. Filter Source FIRST based on User Constraints
    #    We handle None values for t_min/t_max here
    if t_min is not None:
        source_df = source_df[source_df[time_col] >= t_min]
    if t_max is not None:
        source_df = source_df[source_df[time_col] <= t_max]

    # Check if source became empty
    if source_df.empty:
        raise ValueError(
            f"Time constraints resulted in an empty source DataFrame: t_min={t_min}, t_max={t_max}"
        )

    source_df = source_df.reset_index(drop=True)
    source_times = source_df[time_col].values

    # 3. Define Valid Range based on ACTUAL filtered source data
    #    This prevents the "Discrete Gap" bug.
    valid_min = source_times.min()
    valid_max = source_times.max()

    # 4. Filter Target to be strictly within the Source's available range
    target_df = target_df[
        (target_df[time_col] >= valid_min) & (target_df[time_col] <= valid_max)
    ].reset_index(drop=True)

    target_times = target_df[time_col].values

    # Get column names (excluding time column)
    data_cols = [col for col in source_df.columns if col != time_col]

    # Initialize result dictionary
    result = {time_col: target_times}

    # Use scipy for better performance on large datasets
    for col in data_cols:
        # Create interpolator
        f = CubicSpline(source_times, source_df[col].values, extrapolate=False)

        # Interpolate
        result[col] = f(target_times)
        if np.any(np.isnan(result[col])):
            print(
                f"Warning: NaN values found in interpolated column {col}. This may be due to target times being outside the source time range."
            )
            return result[col], source_times, target_times
    # return df1 and df2 with both having same time steps now
    if df1_is_source:
        return pd.DataFrame(result), target_df
    else:
        return target_df, pd.DataFrame(result)


def waveform_diff_all_modes(
    df1, df2, time_col="t(M)", t_min=None, t_max=None, warnings=False
):
    if not np.array_equal(df1[time_col].values, df2[time_col].values):
        # print("Interpolating dataframes to have same time steps.")
        df1, df2 = interpolate_df_to_target(
            df1, df2, time_col=time_col, t_min=t_min, t_max=t_max
        )
    diff = df1 - df2
    diff[time_col] = df1[time_col]
    t = diff[time_col].values
    diff_dict = {}
    for cols in diff.columns:
        if cols == time_col:
            continue
        h1h2 = sp.integrate.simpson(np.abs(diff[cols]) ** 2, t)

        h1h1 = sp.integrate.simpson(np.abs(df1[cols]) ** 2, t)
        h2h2 = sp.integrate.simpson(np.abs(df2[cols]) ** 2, t)
        if (h1h1 - h2h2) > 1e-12:
            if warnings:
                print(
                    f"Warning {cols}: h1h1 ({h1h1}) and h2h2 ({h2h2}) differ significantly {abs(h1h1 - h2h2)}."
                )
        norm = np.sqrt(h1h1 * h2h2)
        diff_dict[cols] = (h1h2, norm)

    return diff_dict


def get_cumilative_waveform_diff(diff_dict, mode_list=None):
    total_h1h2 = 0.0
    total_norm = 0.0
    for key in diff_dict:
        if mode_list is not None and key not in mode_list:
            continue
        h1h2, norm = diff_dict[key]
        total_h1h2 += h1h2
        total_norm += norm
    total_waveform_diff = 0.5 * total_h1h2 / total_norm
    return total_waveform_diff


def getLM_from_key(key):
    L_str, M_str = key.split(",")
    return int(L_str), int(M_str)
