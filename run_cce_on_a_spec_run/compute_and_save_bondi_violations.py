# %%
import scri
import matplotlib.pyplot as plt
from pathlib import Path
from spherical_functions import LM_index as lm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import json
import h5py
import numpy as np
import pandas as pd
import pickle
import itertools
import pickle
import re

# %% [markdown]
# # Functions

# %%
def moving_average_valid(array,avg_len):
    return np.convolve(array,np.ones(avg_len),'valid')/avg_len

def load_and_pickle(data_path:Path, reload_data:bool=False, data_type:str='abd', options:dict={}):
  if not data_path.exists():
    raise Exception(f"{data_path} does not exist!")

  saved_data_path = data_path.parent/"saved.pkl"
  
  if saved_data_path.exists() and reload_data == False:
    with open(saved_data_path, 'rb') as f:
      saved_data = pickle.load(f)
      print(f"Saved data loaded: {saved_data_path}")
  else:
    saved_data = {}
    if data_type == 'abd':
      saved_data['abd']= scri.create_abd_from_h5(
          file_name=str(data_path),
          file_format="spectrecce_v1",
          **options
        )
      with open(saved_data_path, 'wb') as f:
        pickle.dump(saved_data,f)
      print(f"Data loaded and saved at : {saved_data_path}")

  return saved_data

def load_bondi_constraints(data_path:Path):
  if not data_path.exists():
    raise Exception(f"{data_path} does not exist!")
  saved_data_path = data_path.parent/"saved.pkl"
  if not saved_data_path.exists():
    raise Exception(f"{saved_data_path} does not exist")
  else:
    with open(saved_data_path, 'rb') as f:
      saved_data = pickle.load(f)
      if 'bondi_violation_norms' in saved_data:
        print(f"bondi_violation_norms loaded for {data_path}")
      else:
        print(f"Computing bondi_violation_norms for: {data_path}")
        saved_data['bondi_violation_norms'] = saved_data['abd'].bondi_violation_norms
        with open(saved_data_path, 'wb') as f:
          pickle.dump(saved_data,f)

        print(f"Saved bondi_violation_norms for: {data_path}")
    return saved_data

def add_bondi_constraints(abd_data:dict):
  for key in abd_data:
    abd_data[key]['bondi_violation_norms'] = abd_data[key]["abd"].bondi_violation_norms
    print(f"bondi_violation_norms computed for {key}")

def create_diff_dict_cce(WT_data_dict:dict, l:int, m:int, base_key:str, t_interpolate:np.ndarray):
  h = WT_data_dict[base_key]['abd'].h.interpolate(t_interpolate)
  diff_dict = {"t": h.t}
  y_base = h.data[:,  lm(l,m,h.ell_min)]
  y_norm = np.linalg.norm(y_base)
  for key in WT_data_dict:
    if key == base_key:
      continue
    h = WT_data_dict[key]['abd'].h.interpolate(t_interpolate)
    y_inter = h.data[:,  lm(l,m,h.ell_min)]
    diff_dict[key+"_diff"] = y_inter-y_base
    diff_dict[key+"_absdiff"] = np.abs(y_inter-y_base)
    diff_dict[key+"_rel_diff"] = (y_inter-y_base)/y_norm
    diff_dict[key+"_rel_absdiff"] = np.abs(y_inter-y_base)/y_norm
  return diff_dict

def extract_radii(h5_file_path:Path):
  radii = set()
  with h5py.File(h5_file_path,'r') as f:
    names = []
    f.visit(names.append)
  for name in names:
    if "Version" in name:
      continue
    radii.add(name[1:5])
  radii = list(radii)
  radii.sort()
  return radii


def generate_columns(num_cols:int,beta_type=False):
  if beta_type:
    num_cols = num_cols*2
  L_max = int(np.sqrt((num_cols-1)/2))-1
  # print(L_max,np.sqrt((num_cols-1)/2)-1)
  col_names = ['t(M)']
  for l in range(0,L_max+1):
    for m in range(-l,l+1):
      if beta_type:
        if m==0:
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


def WT_to_pandas(horizon_path:Path):
    assert(horizon_path.exists())
    df_dict = {}
    beta_type_list = ['Beta.dat', 'DuR.dat', 'R.dat', 'W.dat']
    with h5py.File(horizon_path,'r') as hf:
        # Not all horizon files may have AhC
        for key in hf.keys():
            if key == "VersionHist.ver":
              continue 
            if key in beta_type_list:
              df_dict[key] = pd.DataFrame(hf[key], columns=generate_columns(hf[key].shape[1],beta_type=True))
            else:
              df_dict[key] = pd.DataFrame(hf[key], columns=generate_columns(hf[key].shape[1]))


    return df_dict


def create_diff_dict(WT_data_dict:dict, mode:str, variable:str, base_key:str):
  diff_dict = {"t(M)":WT_data_dict[base_key][variable]['t(M)']}
  y_base = WT_data_dict[base_key][variable][mode]
  y_norm = np.linalg.norm(y_base)
  for key in WT_data_dict:
    if key == base_key:
      continue
    y = WT_data_dict[key][variable][mode]
    t = WT_data_dict[key][variable]['t(M)']
    y_interpolator = interp1d(t, y, kind='cubic',fill_value='extrapolate')
    y_inter = y_interpolator(diff_dict['t(M)'])
    diff_dict[key+"_diff"] = y_inter-y_base
    diff_dict[key+"_absdiff"] = np.abs(y_inter-y_base)
    diff_dict[key+"_rel_diff"] = (y_inter-y_base)/y_norm
    diff_dict[key+"_rel_absdiff"] = np.abs(y_inter-y_base)/y_norm
  return diff_dict

def filter_by_regex(regex,col_list,exclude=False):
  filtered_set = set()
  if type(regex) is list:
    for reg in regex:
      for i in col_list:
        if re.search(reg,i):
          filtered_set.add(i)
  else:
    for i in col_list:
      if re.search(regex,i):
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

def limit_by_col_val(min_val,max_val,col_name,df):
  filter = (df[col_name]>=min_val) &(df[col_name] <=max_val)
  return df[filter]

def abs_mean_value_upto_l(pd_series,L_max:int):
  idx = pd_series.index
  abs_cum_sum = 0
  num = 0
  for i in idx:
    L = int(i.split(",")[0][3:])
    if L > L_max:
      continue
    else:
      abs_cum_sum = abs_cum_sum+abs(pd_series[i])
      num = num +1
  return abs_cum_sum/num

def get_mode(name):
  return int(name.split("(")[-1].split(")")[0])
def get_radii(name):
  if name[-5]=='R':
    # R0257 -> 0257load_and_pickle
    return int(name.split('_')[-1][1:])
  else:
    return int(name.split('_')[-1])
def sort_by_power_modes(col_names):
  col_name_copy = list(col_names).copy()
  return sorted(col_name_copy, key=lambda x: int(get_mode(x)))

def add_L_mode_power(df:pd.DataFrame,L:int, ReOrIm:str):
  column_names = df.columns
  n = 0
  power = 0
  for m in range(-L,L+1):
    col_name = f'{ReOrIm}({L},{m})'
    # print(col_name)
    if col_name in column_names:
      power = power + df[col_name]*df[col_name]
      n = n + 1
  if n != 0:
    power = power/n
    df[f'pow_{ReOrIm}({L})'] = power
  return power

def add_all_L_mode_power(df:pd.DataFrame,L_max:int):
  local_df = df.copy()
  total_power_Re = 0
  total_power_Im = 0
  for l in range(0,L_max+1):
    total_power_Re = total_power_Re + add_L_mode_power(local_df,l,"Re")
    total_power_Im = total_power_Im + add_L_mode_power(local_df,l,"Im")
    local_df[f"pow_cum_Re({l})"] = total_power_Re
    local_df[f"pow_cum_Im({l})"] = total_power_Im
  return local_df


def create_power_diff_dict(power_dict:dict, pow_mode:str, variable:str, base_key:str):
  diff_dict = {"t(M)":power_dict[base_key]['t(M)']}
  y_base = power_dict[base_key][variable][pow_mode]
  y_norm = np.linalg.norm(y_base)
  for key in power_dict:
    if key == base_key:
      continue
    y = power_dict[key][variable][pow_mode]
    t = power_dict[key]['t(M)']
    y_interpolator = interp1d(t, y, kind='cubic',fill_value='extrapolate')
    y_inter = y_interpolator(diff_dict['t(M)'])
    diff_dict[key+"_diff"] = y_inter-y_base
    diff_dict[key+"_absdiff"] = np.abs(y_inter-y_base)
    diff_dict[key+"_rel_diff"] = (y_inter-y_base)/y_norm
    diff_dict[key+"_rel_absdiff"] = np.abs(y_inter-y_base)/y_norm
  return diff_dict

# %%
cce_data= {}
radius = [20,35,50,75,100,150,200,250,300,400,500,600,800,1000,1500,2000,2500,]
for r in radius:
  cce_data[f"20_zero_err_{r}"] = Path(f"/groups/sxs/hchaudha/spec_runs/single_bh/20_zero_spin_AMR_L5_10000M/GW_data/long_16_2565_10000_14/BondiCceR{r:04}/red_cce.h5")

# fail_flag = False
drop_unavailable_keys = True
unavailable_keys = []
for key in cce_data:
  if not cce_data[key].exists():
    fail_flag = True
    unavailable_keys.append(key)
    print(f"{cce_data[key]} does not exist!")
# if fail_flag:
#   raise Exception("Some paths do not exist!")

for key in unavailable_keys:
  cce_data.pop(key)

# %%
t_interpolate = None
# t_interpolate = np.linspace(-2000,8000,num=5000)

abd_data = {}
failed_keys = {}
for key in cce_data:
  try:
    if "temp" in key:
      abd_data[key] = load_and_pickle(cce_data[key],options = {'t_interpolate':t_interpolate}, reload_data=True)
      abd_data[key] = load_bondi_constraints(cce_data[key])
    elif t_interpolate is None:
      abd_data[key] = load_and_pickle(cce_data[key], reload_data=False)
    #   abd_data[key] = load_and_pickle(cce_data[key],reload_data=True)
      abd_data[key] = load_bondi_constraints(cce_data[key])
    else:
      abd_data[key] = load_and_pickle(cce_data[key],options = {'t_interpolate':t_interpolate}, reload_data=False)
    #   abd_data[key] = load_and_pickle(cce_data[key],reload_data=True)
      abd_data[key] = load_bondi_constraints(cce_data[key])
  except Exception as e:
    failed_keys[key] = str(e)
    print(f"Failed to load and pickle data for key {key}: {e}")
    continue

for key,val in failed_keys.items():
  abd_data.pop(key)
  print(f"{key}: {val}")

print(abd_data.keys())
