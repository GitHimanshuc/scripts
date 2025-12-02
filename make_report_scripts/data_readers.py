# %%
import re
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd


# %% [markdown]
# ### Functions to read horizon files


# %%
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


# %%
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


# %% [markdown]
# ### Functions to read dat and hist files


# %%
def get_top_name_from_number(top_number: int, subdomain_name: str) -> str:
    if re.match(r"Sphere", subdomain_name):
        return ["Bf0I1", "Bf1S2", "Bf1S2"][top_number]
    elif re.match(r"Cylinder", subdomain_name):
        return ["Bf0I1", "Bf1S1", "Bf2I1"][top_number]
    elif re.match(r"FilledCylinder", subdomain_name):
        return ["Bf0I1", "Bf1B2Radial", "Bf1B2"][top_number]
    else:
        raise Exception(f"{subdomain_name=} not recognized!")


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
    with open(file_name, "r") as f:
        lines = f.readlines()
        max_col_num = 0
        for line in lines:
            max_col_num = max(max_col_num, len(line.split()))
    maxL = np.sqrt(max_col_num - 4) - 1
    for l in range(int(maxL) + 1):
        for m in range(-l, l + 1):
            cols_names.append(f"{l},{m}")
    cols_names = ["t(M)", "Center-x", "Center-y", "Center-z"] + cols_names

    return pd.read_csv(file_name, comment="#", sep="\s+", names=cols_names)


def read_power_diagnostics_non_power_spectrum(
    file_path, dat_file_name, psi_or_kappa, top_num
):
    top_num = int(top_num)
    with h5py.File(file_path, "r") as f:
        data_dict = {}
        if psi_or_kappa == "psi":
            data_index = 1
        else:
            data_index = 2
        for sd_name in f.keys():
            top_name = get_top_name_from_number(top_num, sd_name)
            data_dict[
                f"{psi_or_kappa}_{dat_file_name.split('.')[0]}_{top_name} on {sd_name[:-4]}"
            ] = f[sd_name][f"{top_name}_{dat_file_name}"][:, data_index]

        # get time var
        any_subdomain = next(iter(f.keys()))
        top_name = get_top_name_from_number(top_num, any_subdomain)
        data_dict["t(M)"] = f[any_subdomain][f"{top_name}_{dat_file_name}"][:, 0]

        df = pd.DataFrame(data_dict)

    return df


def GetWTDataExtracRadii(folder_path: Path):
    extraction_radii_list = []
    for WT_data in folder_path.glob("BondiCceR*.h5"):
        extraction_radii_list.append(WT_data.name.split("BondiCceR")[1].split(".")[0])
    return sorted(extraction_radii_list)


def GetFiniteRadiiDataVars(folder_path: Path):
    finite_radii_files = []
    for file in folder_path.glob("*_CodeUnits.h5"):
        finite_radii_files.append(file.name.split("_")[0])
    return sorted(finite_radii_files)


def GetFiniteRadiusExtractionList(file_path: Path):
    with h5py.File(file_path, "r") as f:
        extraction_radii_list = []
        for radius_dir in f.keys():
            if "Version" in radius_dir:
                continue
            extraction_radii_list.append(radius_dir[1:5])
    return sorted(extraction_radii_list)


def FindMinMaxL(keys):
    "Finds the minimum and maximum L from the dat files in the finite radius H5 files."
    minL = 10000
    maxL = -10000
    for key in keys:
        if "Y_l" not in key:
            continue
        l = int(key.split("_")[1][1:])
        minL = min(minL, l)
        maxL = max(maxL, l)
    return minL, maxL


def read_finite_radius_quantaties(file_path, radius):
    with h5py.File(file_path, "r") as f:
        if f"R{radius}.dir" not in f.keys():
            # -1 at last drops the version history
            radii_list = [i.split(".")[0][1:] for i in f.keys()][:-1]
            raise Exception(f"R{radius}.dir not found in {file_path}\n{radii_list=}")
        rad_data = f[f"R{radius}.dir"]
        minL, maxL = FindMinMaxL(rad_data.keys())

        data_is_double_double = False
        data = {"t(M)": rad_data["Y_l2_m0.dat"][:, 0]}
        if data["t(M)"].dtype == np.dtype([("hi", "<f8"), ("lo", "<f8")]):
            data_is_double_double = True
            data["t(M)"] = data["t(M)"]["hi"]
        for l in range(minL, maxL + 1):
            for m in range(-l, l + 1):
                if data_is_double_double:
                    data[f"{l},{m}"] = (
                        rad_data[f"Y_l{l}_m{m}.dat"][:, 1]["hi"]
                        + 1j * rad_data[f"Y_l{l}_m{m}.dat"][:, 2]["hi"]
                    )
                else:
                    data[f"{l},{m}"] = (
                        rad_data[f"Y_l{l}_m{m}.dat"][:, 1]
                        + 1j * rad_data[f"Y_l{l}_m{m}.dat"][:, 2]
                    )

        return pd.DataFrame(data)


def read_WT_data(file_path: Path, var: str):
    with h5py.File(file_path, "r") as f:
        # all_m = ['DrJ.dat', 'H.dat', 'J.dat', 'Q.dat', 'U.dat', ]
        # some_m = ['Beta.dat', 'DuR.dat',  'R.dat', 'W.dat']

        col_names = list(f[var].attrs["Legend"])
        col_names[0] = "t(M)"
        name_to_index_dict = {name: i for i, name in enumerate(col_names)}

        data_is_double_double = False
        data = {"t(M)": f[var][:, name_to_index_dict["t(M)"]]}
        if data["t(M)"].dtype == np.dtype([("hi", "<f8"), ("lo", "<f8")]):
            data_is_double_double = True
            data["t(M)"] = data["t(M)"]["hi"]

        maxL = int(col_names[-1].split(",")[0][3:])
        for L in range(maxL + 1):
            for m in range(-L, L + 1):
                real_key = f"Re({L},{m})"
                img_key = f"Im({L},{m})"
                if data_is_double_double:
                    # Note tested because there is no Bondi data from float128 just yets
                    if real_key in name_to_index_dict:
                        data[f"{L},{m}"] = np.array(
                            f[var][:, name_to_index_dict[real_key]]["hi"],
                            dtype=np.complex128,
                        )
                    if img_key in name_to_index_dict:
                        data[f"{L},{m}"] += 1j * np.array(
                            f[var][:, name_to_index_dict[img_key]]["hi"],
                            dtype=np.complex128,
                        )
                else:
                    if real_key in name_to_index_dict:
                        data[f"{L},{m}"] = np.array(
                            f[var][:, name_to_index_dict[real_key]], dtype=np.complex128
                        )
                    if img_key in name_to_index_dict:
                        data[f"{L},{m}"] += 1j * np.array(
                            f[var][:, name_to_index_dict[img_key]], dtype=np.complex128
                        )

        # df1 = pd.DataFrame(f[var][:], columns=col_names)
        return pd.DataFrame(data)


def read_point_interpolation_file(file_path, loadALL):
    def get_col_name(file_path):
        # if the file is Int_kappa_ttt.dat which has
        # Points = (0,0,0), (0,0,2), (0,0,4), (0,0,6), (0,0,8)
        # returns a list of string ["kappa_ttt_(0,0,0)", "kappa_ttt_(0,0,2)", ...]
        with open(file_path, "r") as file:
            file_name = file_path.split("/")[-1][
                4:-4
            ]  # Get the file name without extension
            lines = file.readlines()
            for line in lines:
                if line.startswith("# Points"):
                    parts = line[11:].split(", ")
                    points = [f"{file_name}_{part.strip()}" for part in parts]
                    return points

    if loadALL:
        parent_folder, var = file_path.split("/Int_")
        parent_folder = Path(parent_folder)
        var = var.split("_")[0]
        all_dat_files = parent_folder.glob(f"Int_{var}_*.dat")
        dfs = []
        for dat_file in all_dat_files:
            print(dat_file)
            column_names = get_col_name(str(dat_file))
            column_names = ["t(M)"] + column_names
            dfs.append(
                pd.read_csv(dat_file, sep="\s+", comment="#", names=column_names)
            )

        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on="t(M)", how="outer")
        return result
    else:
        column_names = get_col_name(file_path)
        column_names = ["t(M)"] + column_names
        data = pd.read_csv(file_path, sep="\s+", comment="#", names=column_names)
        return data


def read_OrbitDiagnostics_file(file_name):
    with h5py.File(file_name, "r") as f:
        names = []
        f.visit(names.append)

        first_data_set = names[0]
        data = f[first_data_set]
        # This df now had t and some orbital quantities
        df = pd.DataFrame(data[:], columns=list(data.attrs["Legend"]))

        # Now read and append rest of the datasets
        for dataset_name in names[1:]:
            data = f[dataset_name]
            cols_ = list(data.attrs["Legend"])[1:]  # Skip time column
            for col in cols_:
                df[col] = data[:, list(data.attrs["Legend"]).index(col)]

    return df
