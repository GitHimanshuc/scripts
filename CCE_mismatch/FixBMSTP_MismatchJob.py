# %%
import itertools
import json
import pickle
import re
from pathlib import Path

import FindMismatch as FM
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scri
import spherical_functions as sf
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from spherical_functions import LM_index as lm

import time

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 10)


# %%
def moving_average_valid(array, avg_len):
    return np.convolve(array, np.ones(avg_len), "valid") / avg_len


def load_and_pickle(
    data_path: Path,
    reload_data: bool = False,
    data_type: str = "abd",
    options: dict = {},
):
    if not data_path.exists():
        raise Exception(f"{data_path} does not exist!")

    saved_data_path = data_path.parent / "saved.pkl"

    if saved_data_path.exists() and reload_data == False:
        with open(saved_data_path, "rb") as f:
            saved_data = pickle.load(f)
            print(f"Saved data loaded: {saved_data_path}")
    else:
        saved_data = {}
        if data_type == "abd":
            saved_data["abd"] = scri.create_abd_from_h5(
                file_name=str(data_path), file_format="spectrecce_v1", **options
            )
            with open(saved_data_path, "wb") as f:
                pickle.dump(saved_data, f)
            print(f"Data loaded and saved at : {saved_data_path}")

    return saved_data


def load_bondi_constraints(data_path: Path):
    if not data_path.exists():
        raise Exception(f"{data_path} does not exist!")
    saved_data_path = data_path.parent / "saved.pkl"
    if not saved_data_path.exists():
        raise Exception(f"{saved_data_path} does not exist")
    else:
        with open(saved_data_path, "rb") as f:
            saved_data = pickle.load(f)
            if "bondi_violation_norms" in saved_data:
                print(f"bondi_violation_norms loaded for {data_path}")
            else:
                print(f"Computing bondi_violation_norms for: {data_path}")
                saved_data["bondi_violation_norms"] = saved_data[
                    "abd"
                ].bondi_violation_norms
                with open(saved_data_path, "wb") as f:
                    pickle.dump(saved_data, f)

                print(f"Saved bondi_violation_norms for: {data_path}")
        return saved_data


def add_bondi_constraints(abd_data: dict):
    for key in abd_data:
        abd_data[key]["bondi_violation_norms"] = abd_data[key][
            "abd"
        ].bondi_violation_norms
        print(f"bondi_violation_norms computed for {key}")


def create_diff_dict_cce(
    WT_data_dict: dict, l: int, m: int, base_key: str, t_interpolate: np.ndarray
):
    h = WT_data_dict[base_key]["abd"].h.interpolate(t_interpolate)
    diff_dict = {"t": h.t}
    y_base = h.data[:, lm(l, m, h.ell_min)]
    y_norm = np.linalg.norm(y_base)
    for key in WT_data_dict:
        if key == base_key:
            continue
        h = WT_data_dict[key]["abd"].h.interpolate(t_interpolate)
        y_inter = h.data[:, lm(l, m, h.ell_min)]
        diff_dict[key + "_diff"] = y_inter - y_base
        diff_dict[key + "_absdiff"] = np.abs(y_inter - y_base)
        diff_dict[key + "_rel_diff"] = (y_inter - y_base) / y_norm
        diff_dict[key + "_rel_absdiff"] = np.abs(y_inter - y_base) / y_norm
    return diff_dict


def extract_radii(h5_file_path: Path):
    radii = set()
    with h5py.File(h5_file_path, "r") as f:
        names = []
        f.visit(names.append)
    for name in names:
        if "Version" in name:
            continue
        radii.add(name[1:5])
    radii = list(radii)
    radii.sort()
    return radii


def generate_columns(num_cols: int, beta_type=False):
    if beta_type:
        num_cols = num_cols * 2
    L_max = int(np.sqrt((num_cols - 1) / 2)) - 1
    # print(L_max,np.sqrt((num_cols-1)/2)-1)
    col_names = ["t(M)"]
    for l in range(0, L_max + 1):
        for m in range(-l, l + 1):
            if beta_type:
                if m == 0:
                    col_names.append(f"Re({l},{m})")
                elif m < 0:
                    continue
                else:
                    col_names.append(f"Re({l},{m})")
                    col_names.append(f"Im({l},{m})")
            else:
                col_names.append(f"Re({l},{m})")
                col_names.append(f"Im({l},{m})")
    return col_names


def WT_to_pandas(horizon_path: Path):
    assert horizon_path.exists()
    df_dict = {}
    beta_type_list = ["Beta.dat", "DuR.dat", "R.dat", "W.dat"]
    with h5py.File(horizon_path, "r") as hf:
        # Not all horizon files may have AhC
        for key in hf.keys():
            if key == "VersionHist.ver":
                continue
            if key in beta_type_list:
                df_dict[key] = pd.DataFrame(
                    hf[key], columns=generate_columns(hf[key].shape[1], beta_type=True)
                )
            else:
                df_dict[key] = pd.DataFrame(
                    hf[key], columns=generate_columns(hf[key].shape[1])
                )

    return df_dict


def create_diff_dict(WT_data_dict: dict, mode: str, variable: str, base_key: str):
    diff_dict = {"t(M)": WT_data_dict[base_key][variable]["t(M)"]}
    y_base = WT_data_dict[base_key][variable][mode]
    y_norm = np.linalg.norm(y_base)
    for key in WT_data_dict:
        if key == base_key:
            continue
        y = WT_data_dict[key][variable][mode]
        t = WT_data_dict[key][variable]["t(M)"]
        y_interpolator = interp1d(t, y, kind="cubic", fill_value="extrapolate")
        y_inter = y_interpolator(diff_dict["t(M)"])
        diff_dict[key + "_diff"] = y_inter - y_base
        diff_dict[key + "_absdiff"] = np.abs(y_inter - y_base)
        diff_dict[key + "_rel_diff"] = (y_inter - y_base) / y_norm
        diff_dict[key + "_rel_absdiff"] = np.abs(y_inter - y_base) / y_norm
    return diff_dict


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


def abs_mean_value_upto_l(pd_series, L_max: int):
    idx = pd_series.index
    abs_cum_sum = 0
    num = 0
    for i in idx:
        L = int(i.split(",")[0][3:])
        if L > L_max:
            continue
        else:
            abs_cum_sum = abs_cum_sum + abs(pd_series[i])
            num = num + 1
    return abs_cum_sum / num


def get_mode(name):
    return int(name.split("(")[-1].split(")")[0])


def get_radii(name):
    if name[-5] == "R":
        # R0257 -> 0257load_and_pickle
        return int(name.split("_")[-1][1:])
    else:
        return int(name.split("_")[-1])


def sort_by_power_modes(col_names):
    col_name_copy = list(col_names).copy()
    return sorted(col_name_copy, key=lambda x: int(get_mode(x)))


def add_L_mode_power(df: pd.DataFrame, L: int, ReOrIm: str):
    column_names = df.columns
    n = 0
    power = 0
    for m in range(-L, L + 1):
        col_name = f"{ReOrIm}({L},{m})"
        # print(col_name)
        if col_name in column_names:
            power = power + df[col_name] * df[col_name]
            n = n + 1
    if n != 0:
        power = power / n
        df[f"pow_{ReOrIm}({L})"] = power
    return power


def add_all_L_mode_power(df: pd.DataFrame, L_max: int):
    local_df = df.copy()
    total_power_Re = 0
    total_power_Im = 0
    for l in range(0, L_max + 1):
        total_power_Re = total_power_Re + add_L_mode_power(local_df, l, "Re")
        total_power_Im = total_power_Im + add_L_mode_power(local_df, l, "Im")
        local_df[f"pow_cum_Re({l})"] = total_power_Re
        local_df[f"pow_cum_Im({l})"] = total_power_Im
    return local_df


def create_power_diff_dict(
    power_dict: dict, pow_mode: str, variable: str, base_key: str
):
    diff_dict = {"t(M)": power_dict[base_key]["t(M)"]}
    y_base = power_dict[base_key][variable][pow_mode]
    y_norm = np.linalg.norm(y_base)
    for key in power_dict:
        if key == base_key:
            continue
        y = power_dict[key][variable][pow_mode]
        t = power_dict[key]["t(M)"]
        y_interpolator = interp1d(t, y, kind="cubic", fill_value="extrapolate")
        y_inter = y_interpolator(diff_dict["t(M)"])
        diff_dict[key + "_diff"] = y_inter - y_base
        diff_dict[key + "_absdiff"] = np.abs(y_inter - y_base)
        diff_dict[key + "_rel_diff"] = (y_inter - y_base) / y_norm
        diff_dict[key + "_rel_absdiff"] = np.abs(y_inter - y_base) / y_norm
    return diff_dict


# %%
def find_mismatch(abd_data: dict, key1, key2, t1, t2):
    W1 = FM.abd_to_WM(abd_data[key1]["abd"])
    W2 = FM.abd_to_WM(abd_data[key2]["abd"])
    return FM.SquaredError(W1, W2, t1, t2)


def find_mismatch_abd(abd1, abd2, t1, t2):
    W1 = FM.abd_to_WM(abd1)
    W2 = FM.abd_to_WM(abd2)
    return FM.SquaredError(W1, W2, t1, t2)


def include_radii(name, min, max):
    radius = int(name[-4:])
    if radius < min or radius > max:
        return False
    else:
        return True


def indices_of_a_mode(L, ell_min):
    indices = []
    if L < ell_min:
        return indices
    for m in range(-L, L + 1):
        indices.append(lm(L, m, ell_min))
    return indices


# %%
cce_data = {}
# cce_data["high_accuracy_Lev0_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/GW_data_lev0/BondiCceR0257/red_cce.h5")
# cce_data["high_accuracy_Lev1_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/GW_data_lev1/BondiCceR0257/red_cce.h5")
# cce_data["high_accuracy_Lev2_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/GW_data_lev2/BondiCceR0257/red_cce.h5")
cce_data["high_accuracy_Lev3_R0258"] = Path(
    "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/GW_data_lev3/BondiCceR0258/red_cce.h5"
)
cce_data["high_accuracy_Lev4_R0258"] = Path(
    "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/GW_data_lev4/BondiCceR0258/red_cce.h5"
)
cce_data["high_accuracy_Lev5_R0258"] = Path(
    "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/GW_data_lev5/BondiCceR0258/red_cce.h5"
)
# cce_data["high_accuracy_Lev3_R0472"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/GW_data_lev3/BondiCceR0472/red_cce.h5")
# cce_data["high_accuracy_Lev4_R0472"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/GW_data_lev4/BondiCceR0472/red_cce.h5")
# cce_data["high_accuracy_Lev5_R0472"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/GW_data_lev5/BondiCceR0472/red_cce.h5")
# cce_data["master_Lev0_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/GW_data/BondiCceR0257/red_cce.h5")
# cce_data["master_Lev1_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/GW_data_lev1/BondiCceR0257/red_cce.h5")
# cce_data["master_Lev2_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/GW_data_lev2/BondiCceR0257/red_cce.h5")
# cce_data["master_Lev3_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/GW_data_lev3/BondiCceR0257/red_cce.h5")
# cce_data["master_Lev4_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/GW_data_lev4/BondiCceR0257/red_cce.h5")
# cce_data["master_Lev5_R0257"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/GW_data_lev5/BondiCceR0257/red_cce.h5")
# cce_data["Lev5_bg_ah100_cd_01_uamr_full_R0258"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_cd_01_uamr_full/GW_data/BondiCceR0258/red_cce.h5")
# cce_data["Lev5_bg_ah100_cd_01_uamr_full_R0686"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_cd_01_uamr_full/GW_data/BondiCceR0686/red_cce.h5")
# cce_data["Lev5_bg_ah100_lapse_uamr_fullR0258"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_lapse_uamr_full/GW_data/BondiCceR0258/red_cce.h5")
# cce_data["Lev5_bg_ah100_lapse_uamr_full_R0100"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_lapse_uamr_full/GW_data/BondiCceR0100/red_cce.h5")
# cce_data["high_accuracy_Lev5_R0258_ZeroNonSmooth"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev5_variations/initial_data/ZeroNonSmooth/red_Lev5_R0258_VolumeData.h5")
# cce_data["high_accuracy_Lev5_R0258_NoIncomingRadiation"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev5_variations/initial_data/NoIncomingRadiation/red_Lev5_R0258_VolumeData.h5")
# cce_data["high_accuracy_Lev5_R0258_InverseCubic"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev5_variations/initial_data/InverseCubic/red_Lev5_R0258_VolumeData.h5")
# cce_data["high_accuracy_Lev5_R0258_ConformalFactor10"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev5_variations/initial_data/ConformalFactor10/red_Lev5_R0258_VolumeData.h5")
# cce_data["high_accuracy_Lev5_R0258_ConformalFactor7"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev5_variations/initial_data/ConformalFactor7/red_Lev5_R0258_VolumeData.h5")
# cce_data["high_accuracy_Lev5_R0258_ConformalFactor3"] = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev5_variations/initial_data/ConformalFactor3/red_Lev5_R0258_VolumeData.h5")

# cce_data["Lev01_test_ode_Lev2_257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/GW_data_lev2/BondiCceR0257/red_cce.h5")
# cce_data["Lev01_test_ode_Lev1_257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/GW_data_lev1/BondiCceR0257/red_cce.h5")

# cce_data["Lev01_test_Lev2_257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/GW_data_lev2/BondiCceR0257/red_cce.h5")

# cce_data["Lev01_test_Lev2_257"] = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/GW_data_lev2/BondiCceR0257/red_cce.h5")

# # levs,run_sets,radius= [],[],[]
# levs = [0,1,2,3,4,5,6]
# # levs = [2,3,4,5,6]
# # levs = [4,5,6]
# # levs = [1,3,6]
# # levs = [1,2,3]
# # levs = [3]
# # levs = [5,6]
# # levs = [6]
# run_sets = [1]
# radius = [250]
# radius = [350]
# # radius = [100,150,200,250,300,350,500,700,900]
# # radius = [150,200,250,300,350,500,700]
# # radius = [200,250,300,350,500]
# for l, s, r in itertools.product(levs, run_sets, radius):
#     if s == 2 and (l == 0 or l == 1):
#         continue
#     if l <= 3:
#         if s == 1:
#             cce_data[f"6_set{s}_L6s{l}_{r}"] = Path(
#                 f"/groups/sxs/hchaudha/spec_runs/6_segs/6_set{s}_L6/GW_data_lev{l}/BondiCceR0{r}/red_cce.h5"
#             )
#             cce_data[f"6_set{s}_L3s{l}_{r}"] = Path(
#                 f"/groups/sxs/hchaudha/spec_runs/6_segs/6_set{s}_L3/GW_data_lev{l}/BondiCceR0{r}/red_cce.h5"
#             )
#     else:
#         cce_data[f"6_set{s}_L6s{l}_{r}"] = Path(
#             f"/groups/sxs/hchaudha/spec_runs/6_segs/6_set{s}_L6/GW_data_lev{l}/BondiCceR0{r}/red_cce.h5"
#         )
#         pass


# cce_data[f"6_set1_L6s3_CAMR_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6_vars/L6s3_CAMR/GW_data_lev3/BondiCceR0{r}/red_cce.h5")
# cce_data[f"6_set1_L6s3_min_L_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6_vars/L6s3_min_L/GW_data_lev3/BondiCceR0{r}/red_cce.h5")

# cce_data[f"7_constAMR_set1_L6_base_{l}_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/GW_data_lev{l}/BondiCceR0{r}/red_cce.h5/")

# cce_data[f"10_4000M_CAMR_set1_L6_base_{l}_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/GW_data_lev{l}/BondiCceR0{r}/red_cce.h5/")
# cce_data[f"11_4000M_CAMR_set1_L6_maxExt_{l}_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/GW_data_lev{l}/BondiCceR0{r}/red_cce.h5/")

# cce_data[f"6_set1_L3s3_3"] = Path(f"/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/GW_data_lev2/BondiCceR0250/red_cce.h5")

# radius_list = []
# radius_list = ['0100','0150','0200','0250','0300','0350','0500','0700','0900','1100','1300','1400']
# radius_list = ['0250','0300','0350','0500','0700']
# for r in radius_list:
#   cce_data[f"12_set1_L3_1500_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/12_set1_L3_1500/GW_data_lev3/BondiCceR{r}/red_cce.h5/")

# radius_list = []
# radius_list = ['0100','0150','0200','0250','0300','0350','0500','0700','0900','1100','1300','1500','1700','1900']
# radius_list = ['0200','0250','0300','0350','0500','0700']
# for r in radius_list:
#   cce_data[f"12_set1_L3_2000_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/12_set1_L3_2000/GW_data_lev3/BondiCceR{r}/red_cce.h5/")

# radius_list = []
# radius_list = ['0100','0150','0200','0250','0300','0350','0500','0700','0900','1100','1300','1500','1700','1900','2100','2300']
# radius_list = ['0200','0250','0300','0350','0500','0700']
# for r in radius_list:
#   cce_data[f"12_set1_L3_2500_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/12_set1_L3_2500/GW_data_lev3/BondiCceR{r}/red_cce.h5/")

# radius_list = []
# radius_list = ['0100', '0150', '0200', '0250', '0300', '0350', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800', '2900']
# # radius_list = ['0250','0300','0350','0500','0700']
# for r in radius_list:
#   cce_data[f"13_set1_L3_3000_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/13_set1_L3_3000/GW_data_lev3/BondiCceR{r}/red_cce.h5/")

# radius_list = []
# radius_list = ['0100', '0150', '0200', '0250', '0300', '0350', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400']
# radius_list = ['0200','0250','0300','0350','0500','0700']
# for r in radius_list:
#   cce_data[f"13_set1_L4_1500_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/13_set1_L4_1500/GW_data_lev4/BondiCceR{r}/red_cce.h5/")

# radius_list = []
# radius_list = ['0100', '0150', '0200', '0250', '0300', '0350', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800', '2900']
# radius_list = ['0200','0250','0300','0350','0500','0700']
# for r in radius_list:
#   cce_data[f"13_set1_L4_3000_{r}"] =  Path(f"/groups/sxs/hchaudha/spec_runs/13_set1_L4_3000/GW_data_lev4/BondiCceR{r}/red_cce.h5/")

# radius_list = []
# radius_list = ['0100', '0150', '0200', '0250', '0300', '0350', '0400', '0500', '0600', '0700', '0800', '0900']
# for r in radius_list:
#   cce_data[f"16_set1_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/16_set1_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"16_set1_L3_HP32_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/16_set1_L3_HP32/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"16_set1_L3_HP28_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/16_set1_L3_HP28/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"16_set1_L3_HP32_AF_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/16_set1_L3_HP32_AF/GW_data_lev3/BondiCceR{r}/red_cce.h5")

# radius_list = []
# radius_list = ['0258', '0472', '0686', '0900']
# for r in radius_list:
#   cce_data[f"17_main_9_18_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/17_main_9_18_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")

# radius_list = []
# radius_list = ['0258', '0469', '0679', '0890']
# for r in radius_list:
#   cce_data[f"17_set_main_q3_18_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/17_set_main_q3_18_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")

# radius_list = []
# radius_list = ['0199', '0353', '0506', '0660']
# for r in radius_list:
#   cce_data[f"17_set_main_q3_15_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/17_set_main_q3_15_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")

# radius_list = []
# radius_list = ['0100', '0150', '0200', '0250', '0300', '0350', '0500', '0700', '0900']
# radius_list = [ '0300', '0250', '0350','0200']
# radius_list = [ '0350']
# for r in radius_list:
#   cce_data[f"17_set1_q3_18_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/17_set1_q3_18_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"17_set3_q3_18_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/17_set3_q3_18_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"17_set1_9_18_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/17_set1_9_18_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"17_set3_9_18_L3_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/17_set3_9_18_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"22_set1_L1_long_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_set1_L1_long/GW_data_lev1/BondiCceR{r}/red_cce.h5")
#   cce_data[f"22_set1_L3_long_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_set1_L3_long/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"22_L3_AC_L3_no_res_C_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_no_res_C/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"22_L3_AC_L3_res_10_C_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_res_10_C/GW_data_lev3/BondiCceR{r}/red_cce.h5")

#   cce_data[f"L1_AC_L2_long_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_segs/L1_AC_L2/GW_data_lev2/BondiCceR{r}/red_cce.h5")
#   cce_data[f"L1_AC_L3_long_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_segs/L1_AC_L3/GW_data_lev3/BondiCceR{r}/red_cce.h5")
#   cce_data[f"L3_AC_L1_long_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L1/GW_data_lev1/BondiCceR{r}/red_cce.h5")
#   cce_data[f"L3_AC_L2_long_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L2/GW_data_lev2/BondiCceR{r}/red_cce.h5")
#   cce_data[f"L3_AC_L4_long_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L4/GW_data_lev4/BondiCceR{r}/red_cce.h5")

# levs,radius,start = [],[],[]
# levs = [1,3]
# levs = [3]
# radius = ['0250', '0350']
# radius = ['0350']
# start = [3000,5000,7000,8000]
# start = [3000,7000]
# for l,r,s in itertools.product(levs,radius,start):
#   cce_data[f"L{l}_S{s}_r{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_cce_test/L{l}/start_{s}/BondiCceR{r}/red_cce.h5")

# levs,radius,start = [],[],[]
# levs = ['_IC','_NIR','_ZNS','']
# levs = ['_NIR','']
# radius = ['0250', '0350']
# radius = ['0350']
# start = [0,500,1000,3000,7000]
# start = [3000,7000]
# for l,r,s in itertools.product(levs,radius,start):
#   cce_data[f"L3{l}_S{s}_r{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_cce_test/L3/start_{s}{l}/BondiCceR{r}/red_cce.h5")

# radius = ['0020','0035','0050','0075','0100','0150','0200','0250','0300','0400','0500','0600','0800','1000','1500','2000','2500',]
# radius = ['0020','0050','0100','0200','0500','1000','1500','2000','2500',]
# for r in radius:
#   cce_data[f"14_NIR_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh/20_zero_spin_AMR_L5_10000M/GW_data/long_16_2565_10000_14_NIR/BondiCceR{r}/red_cce.h5")


# radius = ['0012', '0050', '0112', '0200', '0312', '0450', '0612', '0800', '1012', '1250', '1512', '1800', '2112', '2450', '2812', '3200', '3612', '4050', '4512', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0015', '0020', '0030', '0075']
# radius = sorted(['0050', '0112', '0200', '0612', '0800', '1012',  '1800',  '3200', '3612', '4050', '4512', '0006', '0020', '0075'])
# radius = sorted(['0050', '0112', '0200', '0612', '0800', '1012',  '0006', '0010', '0015', '0020', '0030', '0075','1800',  '3200','4512'])
# radius = ['0006']
# for r in radius:
#   cce_data[f"ex_rad_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh_CCE/runs/radius_dependence/ex_rad_{r}/red_cce.h5")

# radius = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000 ]
# radius = [0, 100, 200, 300, 400, 500, 600]
# for r in radius[::]:
#   cce_data[f"CF_350_start_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/22_cce_test/L3_new_executable/runs/ConformalFactor_start_{r}/red_cce.h5")

# for i in Path("/groups/sxs/hchaudha/spec_runs/22_cce_test/L3_merged_data/runs").glob("*/red_cce.h5"):
#     cce_data[i.parent.stem] = i
# for i in Path("/groups/sxs/hchaudha/spec_runs/22_cce_test/L3_merged_data/runs").glob("Delta*/red_cce.h5"):
#     cce_data[i.parent.stem] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/22_cce_test/runs/CF_0").glob(
#     "Delta*/red_cce.h5"
# ):
#     cce_data[i.parent.stem] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/22_set1_L3_long/GW_data_lev3_start_6000").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     cce_data["set1_L3_long_ST_6000M_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/22_set1_L3_long/GW_data_lev3_start_4000").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     cce_data["set1_L3_long_ST_4000M_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/22_set1_L3_long/GW_data_lev3_start_2000").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     cce_data["set1_L3_long_ST_2000M_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/32_RM_set1_L3/GW_data_lev3").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     cce_data["32_RM_set1_L3_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/").glob("29_set1_L3_ID_diff_?/GW_data_lev3/BondiCceR*/red_cce.h5"):
#     if not include_radii(i.parent.stem, 210, 260):
#         continue
#     cce_data[f"{str(i).split('/')[-4]}_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/31_segs/L1s3_cdg1_250/GW_data_lev3").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     if not include_radii(i.parent.stem, 200, 400):
#         continue
#     cce_data["32_RM_set1_L1s3_cdg1_250_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/31_segs/L1s3_cdg1_100/GW_data_lev3").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     if not include_radii(i.parent.stem, 200, 400):
#         continue
#     cce_data["32_RM_set1_L1s3_cdg1_100_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/31_segs/L1s3_cdg1_10/GW_data_lev3").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     if not include_radii(i.parent.stem, 200, 400):
#         continue
#     cce_data["32_RM_set1_L1s3_cdg1_10_"+i.parent.stem[-4:]] = i

# for i in Path("/resnick/groups/sxs/hchaudha/spec_runs/31_segs/L1s3/GW_data_lev3").glob(
#     "BondiCceR*/red_cce.h5"
# ):
#     if not include_radii(i.parent.stem, 200, 400):
#         continue
#     cce_data["32_RM_set1_L1s3_"+i.parent.stem[-4:]] = i
# cce_data['6_12_250'] = Path("/resnick/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/GW_data_lev6_12/BondiCceR0250/red_cce.h5")
# cce_data['6_10_250'] = Path("/resnick/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/GW_data_lev6_10/BondiCceR0250/red_cce.h5")
# cce_data['BondiCceR0334'] = Path("/resnick/groups/sxs/hchaudha/spec_runs/CCE_stuff/Lev4_061CCE/BondiCceR0334/red_cce.h5")
# cce_data['BondiCceR0586'] = Path("/resnick/groups/sxs/hchaudha/spec_runs/CCE_stuff/Lev4_061CCE/BondiCceR0586/red_cce.h5")
# cce_data['BondiCceR0838'] = Path("/resnick/groups/sxs/hchaudha/spec_runs/CCE_stuff/Lev4_061CCE/BondiCceR0838/red_cce.h5")
# cce_data['BondiCceR1090'] = Path("/resnick/groups/sxs/hchaudha/spec_runs/CCE_stuff/Lev4_061CCE/BondiCceR0838/red_cce.h5")


# for i in Path(
#     "/resnick/groups/sxs/hchaudha/spec_runs/33_const_inn_dom_runs/set1_L3_Lmin18/GW_data_lev3"
# ).glob("BondiCceR*/red_cce.h5"):
#     if not include_radii(i.parent.stem, 240, 260):
#         continue
#     cce_data["set1_L3_Lmin18_" + i.parent.stem[-4:]] = i

# for i in Path(
#     "/resnick/groups/sxs/hchaudha/spec_runs/33_const_inn_dom_runs/set1_L3_Lmin20_Rn2/GW_data_lev3"
# ).glob("BondiCceR*/red_cce.h5"):
#     if not include_radii(i.parent.stem, 240, 260):
#         continue
#     cce_data["set1_L3_Lmin20_Rn2_" + i.parent.stem[-4:]] = i

# for i in Path(
#     "/resnick/groups/sxs/hchaudha/spec_runs/33_const_inn_dom_runs/set1_L3_Rn1/GW_data_lev3"
# ).glob("BondiCceR*/red_cce.h5"):
#     if not include_radii(i.parent.stem, 240, 260):
#         continue
#     cce_data["set1_L3_Rn1_" + i.parent.stem[-4:]] = i

# for i in Path(
#     "/resnick/groups/sxs/hchaudha/spec_runs/33_const_inn_dom_runs/set1_L3_Rn2/GW_data_lev3"
# ).glob("BondiCceR*/red_cce.h5"):
#     if not include_radii(i.parent.stem, 240, 260):
#         continue
#     cce_data["set1_L3_Rn2_" + i.parent.stem[-4:]] = i


cce_data = dict(sorted(cce_data.items()))


fail_flag = False
for key in cce_data:
    if not cce_data[key].exists():
        fail_flag = True
        print(f"{cce_data[key]} does not exist!")
    if fail_flag:
        raise Exception("Some paths do not exist!")

print(cce_data.keys())


# %%
t_interpolate = np.linspace(-1000, 20000, num=2000)
# t_interpolate = np.linspace(-1000,4000,num=5000)

abd_data = {}
failed_keys = {}
for key in cce_data:
    try:
        abd_data[key] = load_and_pickle(
            cce_data[key], options={"t_interpolate": t_interpolate}
        )
        abd_data[key] = load_bondi_constraints(cce_data[key])
    except Exception as e:
        failed_keys[key] = str(e)
        print(f"Failed to load and pickle data for key {key}: {e}")
        continue

print(abd_data.keys())

# %%
WM_data = {key: FM.abd_to_WM(abd_data[key]["abd"]) for key in abd_data}
keys = list(WM_data.keys())

# %% [markdown]
# ### Do BMSPT fixing and compute the mismatch

# %%
t1 = 1200
t2 = 4000
modes = None
# Create BMSPT fixed WT_dict
WM_BMSPT = {}
base_key = keys[-1]
for i in keys[:-1]:
    start_time = time.time()
    WM_base, WM_Fixed = FM.fix_BMS_NRNR_t12(
        WM_data[base_key], WM_data[i], t1=2500, t2=3500
    )
    WM_BMSPT[i] = WM_Fixed
    WM_BMSPT[base_key] = WM_base
    elapsed = time.time() - start_time
    print(f"{i} fixed to {base_key}'s frame in {elapsed:.3f} seconds")

    mismatch = FM.SquaredError(WM_data[i], WM_data[base_key], t1=t1, t2=t2, modes=modes)
    print(f"{i}@{base_key}: {mismatch}")

pickle_save_path = (
    Path("/resnick/groups/sxs/hchaudha/spec_runs/CCE_mismatch_BMSPTfix") / ""
)
with open(pickle_save_path, "wb") as f:
    pickle.dump(WM_BMSPT, f)
    print(f"Saved WM_BMSPT to {pickle_save_path}")
