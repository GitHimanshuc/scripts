# %%
import numpy as np
import pandas as pd
import subprocess
import random
import re
import h5py
import copy
import sys
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from typing import List, Dict
import imageio.v3 as iio
import os
import glob


plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 10)
import json
import time
import matplotlib
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d

spec_home = "/home/himanshu/spec/my_spec"
matplotlib.matplotlib_fname()


# %% [markdown]
# # Various functions to read across levs
# ### Also functions to make reports

# %% [markdown]
# ### domain color 

# %%
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

def get_domain_name(col_name):
  def AMR_domains_to_decimal(subdoamin_name):
    # SphereC28.0.1
    a = subdoamin_name.split(".")
    # a = [SphereC28,0,1]
    decimal_rep = a[0]+"."
    # decimal_rep = SphereC28.
    for i in a[1:]:
      decimal_rep = decimal_rep + i
    # decimal_rep = SphereC28.01
    return decimal_rep

  if "on" in col_name:
    return AMR_domains_to_decimal(col_name.split(" ")[-1])
  elif "_" in col_name:
    return col_name.split("_")[0]
  elif "MinimumGridSpacing" in col_name:
    return col_name.split("[")[-1][:-1]
  else:
    raise Exception(f"{col_name} type not implemented in return_sorted_domain_names")

def return_sorted_domain_names(domain_names, repeated_symmetric=False, num_Excision=2):

  # def filtered_domain_names(domain_names, filter):
  #   return [i for i in domain_names if get_domain_name(i).startswith(filter)]

  def filtered_domain_names(domain_names, filter):
    return [i for i in domain_names if re.match(filter, get_domain_name(i))]

  def sort_spheres(sphere_list,reverse=False):
    if len(sphere_list) == 0:
      return []
    if "SphereA" in sphere_list[0]:
      return sorted(sphere_list, key=lambda x: float(get_domain_name(x).lstrip('SphereA')), reverse=reverse)
    elif "SphereB" in sphere_list[0]:
      return sorted(sphere_list, key=lambda x: float(get_domain_name(x).lstrip('SphereB')), reverse=reverse)
    elif "SphereC" in sphere_list[0]:
      return sorted(sphere_list, key=lambda x: float(get_domain_name(x).lstrip('SphereC')), reverse=reverse)
    elif "SphereD" in sphere_list[0]:
      return sorted(sphere_list, key=lambda x: float(get_domain_name(x).lstrip('SphereD')), reverse=reverse)
    elif "SphereE" in sphere_list[0]:
      return sorted(sphere_list, key=lambda x: float(get_domain_name(x).lstrip('SphereE')), reverse=reverse)

  FilledCylinderCA = filtered_domain_names(domain_names, r'FilledCylinder.{0,2}CA')
  CylinderCA = filtered_domain_names(domain_names, r'Cylinder.{0,2}CA')
  FilledCylinderEA = filtered_domain_names(domain_names, r'FilledCylinder.{0,2}EA')
  CylinderEA = filtered_domain_names(domain_names, r'Cylinder.{0,2}EA')
  SphereA = sort_spheres(filtered_domain_names(domain_names, 'SphereA'), reverse=True)
  CylinderSMA = filtered_domain_names(domain_names, r'CylinderS.{0,2}MA')
  FilledCylinderMA = filtered_domain_names(domain_names, r'FilledCylinder.{0,2}MA')

  FilledCylinderMB = filtered_domain_names(domain_names, r'FilledCylinder.{0,2}MB')
  CylinderSMB = filtered_domain_names(domain_names, r'CylinderS.{0,2}MB')
  SphereB = sort_spheres(filtered_domain_names(domain_names, 'SphereB'), reverse=True)
  CylinderEB = filtered_domain_names(domain_names, r'Cylinder.{0,2}EB')
  FilledCylinderEB = filtered_domain_names(domain_names, r'FilledCylinder.{0,2}EB')
  CylinderCB = filtered_domain_names(domain_names, r'Cylinder.{0,2}CB')
  FilledCylinderCB = filtered_domain_names(domain_names, r'FilledCylinder.{0,2}CB')

  SphereC = sort_spheres(filtered_domain_names(domain_names, 'SphereC'), reverse=False)
  SphereD = sort_spheres(filtered_domain_names(domain_names, 'SphereD'), reverse=False)
  SphereE = sort_spheres(filtered_domain_names(domain_names, 'SphereE'), reverse=False)
  
  NAN_cols = ['Excision']*num_Excision
  combined_columns = [FilledCylinderCA, CylinderCA, FilledCylinderEA, CylinderEA, SphereA, CylinderSMA, FilledCylinderMA, FilledCylinderMB, CylinderSMB, SphereB, CylinderEB, FilledCylinderEB, CylinderCB, FilledCylinderCB, SphereC, SphereD, SphereE]
  if repeated_symmetric:
    combined_columns = [SphereE[::-1], SphereD[::-1], SphereC[::-1],FilledCylinderCA[::-1], CylinderCA[::-1], FilledCylinderEA[::-1], CylinderEA[::-1], SphereA, NAN_cols, SphereA[::-1], CylinderSMA[::-1], FilledCylinderMA[::-1], FilledCylinderMB, CylinderSMB, SphereB,NAN_cols, SphereB[::-1], CylinderEB, FilledCylinderEB, CylinderCB, FilledCylinderCB, SphereC, SphereD, SphereE]
  combined_columns = [item for sublist in combined_columns for item in sublist]

  # Just append the domains not following any patterns in the front. Mostly domains surrounding sphereA for high spin and mass ratios
  combined_columns_set = set(combined_columns)
  domain_names_set = set()
  for i in domain_names:
    domain_names_set.add(i)
  subdomains_not_sorted = list(domain_names_set - combined_columns_set)
  return subdomains_not_sorted+combined_columns

class BBH_domain_sym_ploy:
  def __init__(self, center_xA, rA,RA,rC,RC,nA,nC,color_dict:dict=None):
    self.center_xA = center_xA
    self.color_dict = color_dict
    self.rA = rA # Largest SphereA radius
    self.RA = RA # Radius of FilledCylinderE
    self.rC = rC # Smallest SphereC radius
    self.RC = RC # Radius of the largest SphereC

    self.nA = nA # Number of SphereA
    self.nC = nC # Number of SphereC

    self.alpha_for_FilledCylinderE_from_Center_bh = np.radians(50)
    self.outer_angle_for_CylinderSM_from_Center_bh = np.arccos(self.center_xA/self.RA)
    self.inner_angle_for_CylinderSM_from_Center_bh = self.outer_angle_for_CylinderSM_from_Center_bh/3

    self.patches = []

    self.add_shpereCs()

    self.add_CylinderC(which_bh='A')
    self.add_FilledCylinderE(which_bh='A')
    self.add_CylinderE(which_bh='A')
    self.add_CylinderSM(which_bh='A')
    self.add_FilledCylinderM(which_bh='A')
    self.add_FilledCylinderC(which_bh='A')

    self.add_CylinderC(which_bh='B')
    self.add_FilledCylinderE(which_bh='B')
    self.add_CylinderE(which_bh='B')
    self.add_CylinderSM(which_bh='B')
    self.add_FilledCylinderM(which_bh='B')
    self.add_FilledCylinderC(which_bh='B')

    self.add_inner_shperes(which_bh='A')
    self.add_inner_shperes(which_bh='B')

    # print the unmatched domains
    print(self.color_dict)

  def get_matching_color(self, domain_name:str):
    if self.color_dict is None:
      return np.random.rand(3,)
    for key in self.color_dict.keys():
      if domain_name in key:
        # Remove the domain name from the key, this will allow us to see which domains were not matched
        return self.color_dict.pop(key)
    # No match found
    return 'pink'

  def add_inner_shperes(self,which_bh):
    center = self.center_xA
    if which_bh == 'B':
      center = -self.center_xA
  
    spheres_outer_radii = np.linspace(self.rA, 0, self.nA+2)
    i=nA-1
    for r in spheres_outer_radii[:-2]:
      domain_name = f'Sphere{which_bh}{i}'
      i = i-1
      color = self.get_matching_color(domain_name)
      self.patches.append(Circle((center, 0), r, facecolor=color, edgecolor='black'))

    domain_name = f'Sphere{which_bh}{i}'
    i = i-1
    color = self.get_matching_color(domain_name)
    self.patches.append(Circle((center, 0), spheres_outer_radii[-2], facecolor='black', edgecolor='black'))

  def add_shpereCs(self):
    spheres_outer_radii = np.linspace(self.RC, self.rC, self.nC+1)[:-1]
    i=nC-1
    for r in spheres_outer_radii:
      domain_name = f'SphereC{i}'
      i = i-1
      color = self.get_matching_color(domain_name)
      self.patches.append(Circle((0, 0), r, facecolor=color, edgecolor='black'))
    
  def add_FilledCylinderE(self,which_bh):
    alpha = self.alpha_for_FilledCylinderE_from_Center_bh
    
    x_inner = self.center_xA+self.rA*np.cos(alpha)
    y_inner = self.rA*np.sin(alpha)
    x_outer = self.center_xA+self.RA*np.cos(alpha)
    y_outer = self.RA*np.sin(alpha)

    if which_bh == 'B':
      x_inner = -x_inner
      x_outer = -x_outer
    vertices=[
      (x_inner,y_inner),
      (x_outer,y_outer),
      (x_outer,-y_outer),
      (x_inner,-y_inner),
    ]
    color = self.get_matching_color(f'FilledCylinderE{which_bh}')
    self.patches.append(Polygon(vertices, closed=True, facecolor=color, edgecolor='black'))

  def add_CylinderE(self,which_bh):
    alpha = self.alpha_for_FilledCylinderE_from_Center_bh
    beta = self.outer_angle_for_CylinderSM_from_Center_bh

    x_inner_away_from_center = self.center_xA+self.rA*np.cos(alpha)
    y_inner_away_from_center = self.rA*np.sin(alpha)
    x_outer_away_from_center = self.center_xA+self.RA*np.cos(alpha)
    y_outer_away_from_center = self.RA*np.sin(alpha)

    x_inner_closer_to_center = self.center_xA-self.rA*np.cos(beta)
    y_inner_closer_to_center = self.rA*np.sin(beta)
    x_outer_closer_to_center = 0
    y_outer_closer_to_center = self.RA*np.sin(beta)

    if which_bh == 'B':
      x_inner_away_from_center = -x_inner_away_from_center
      x_outer_away_from_center = -x_outer_away_from_center
      x_inner_closer_to_center = -x_inner_closer_to_center
      x_outer_closer_to_center = -x_outer_closer_to_center

    vertices=[
      (x_inner_away_from_center,y_inner_away_from_center),
      (x_outer_away_from_center,y_outer_away_from_center),
      (x_outer_closer_to_center,y_outer_closer_to_center),
      (x_inner_closer_to_center,y_inner_closer_to_center),
      (x_inner_closer_to_center,-y_inner_closer_to_center),
      (x_outer_closer_to_center,-y_outer_closer_to_center),
      (x_outer_away_from_center,-y_outer_away_from_center),
      (x_inner_away_from_center,-y_inner_away_from_center),
    ]
    color = self.get_matching_color(f'CylinderE{which_bh}')
    self.patches.append(Polygon(vertices, closed=True, facecolor=color, edgecolor='black'))

  def add_CylinderC(self,which_bh):
    alpha = self.alpha_for_FilledCylinderE_from_Center_bh
    beta = self.outer_angle_for_CylinderSM_from_Center_bh

    x_inner_away_from_center = self.center_xA+self.rA*np.cos(alpha)
    y_inner_away_from_center = self.rA*np.sin(alpha)
    x_outer_away_from_center = self.rC*np.cos(np.radians(30))
    y_outer_away_from_center = self.rC*np.sin(np.radians(30))

    x_inner_closer_to_center = 0
    y_inner_closer_to_center = self.RA*np.sin(beta)
    x_outer_closer_to_center = 0
    y_outer_closer_to_center = self.rC

    if which_bh == 'B':
      x_inner_away_from_center = -x_inner_away_from_center
      x_outer_away_from_center = -x_outer_away_from_center
      x_inner_closer_to_center = -x_inner_closer_to_center
      x_outer_closer_to_center = -x_outer_closer_to_center

    vertices=[
      (x_inner_closer_to_center,y_inner_closer_to_center),
      (x_outer_closer_to_center,y_outer_closer_to_center),
      (x_outer_away_from_center,y_outer_away_from_center),
      (x_inner_away_from_center,y_inner_away_from_center),
      (x_inner_away_from_center,-y_inner_away_from_center),
      (x_outer_away_from_center,-y_outer_away_from_center),
      (x_outer_closer_to_center,-y_outer_closer_to_center),
      (x_inner_closer_to_center,-y_inner_closer_to_center),
    ]
    color = self.get_matching_color(f'CylinderC{which_bh}')
    self.patches.append(Polygon(vertices, closed=True, facecolor=color, edgecolor='black'))

  def add_CylinderSM(self,which_bh):
    beta = self.outer_angle_for_CylinderSM_from_Center_bh
    gamma = self.inner_angle_for_CylinderSM_from_Center_bh

    x_inner_away_from_center = self.center_xA-self.rA*np.cos(beta)
    y_inner_away_from_center = self.rA*np.sin(beta)
    x_outer_away_from_center = 0
    y_outer_away_from_center = self.RA*np.sin(beta)

    x_inner_closer_to_center = self.center_xA-self.rA*np.cos(gamma)
    y_inner_closer_to_center = self.rA*np.sin(gamma)
    x_outer_closer_to_center = 0
    y_outer_closer_to_center = self.RA*np.sin(gamma)

    if which_bh == 'B':
      x_inner_away_from_center = -x_inner_away_from_center
      x_outer_away_from_center = -x_outer_away_from_center
      x_inner_closer_to_center = -x_inner_closer_to_center
      x_outer_closer_to_center = -x_outer_closer_to_center

    vertices=[
      (x_inner_away_from_center,y_inner_away_from_center),
      (x_outer_away_from_center,y_outer_away_from_center),
      (x_outer_closer_to_center,y_outer_closer_to_center),
      (x_inner_closer_to_center,y_inner_closer_to_center),
      (x_inner_closer_to_center,-y_inner_closer_to_center),
      (x_outer_closer_to_center,-y_outer_closer_to_center),
      (x_outer_away_from_center,-y_outer_away_from_center),
      (x_inner_away_from_center,-y_inner_away_from_center),
    ]
    color = self.get_matching_color(f'CylinderSM{which_bh}')
    self.patches.append(Polygon(vertices, closed=True, facecolor=color, edgecolor='black'))
    
  def add_FilledCylinderM(self,which_bh):
    gamma = self.inner_angle_for_CylinderSM_from_Center_bh

    x_inner = self.center_xA-self.rA*np.cos(gamma)
    y_inner = self.rA*np.sin(gamma)
    x_outer = 0
    y_outer = self.RA*np.sin(gamma)

    if which_bh == 'B':
      x_inner = -x_inner
      x_outer = -x_outer
    vertices=[
      (x_inner,y_inner),
      (x_outer,y_outer),
      (x_outer,-y_outer),
      (x_inner,-y_inner),
    ]
    color = self.get_matching_color(f'FilledCylinderM{which_bh}')
    self.patches.append(Polygon(vertices, closed=True, facecolor=color, edgecolor='black'))

  def add_FilledCylinderC(self,which_bh):
    alpha = self.alpha_for_FilledCylinderE_from_Center_bh

    x_inner = self.center_xA+self.RA*np.cos(alpha)
    y_inner = self.RA*np.sin(alpha)
    x_outer = self.rC*np.cos(np.radians(30))
    y_outer = self.rC*np.sin(np.radians(30))

    if which_bh == 'B':
      x_inner = -x_inner
      x_outer = -x_outer
    vertices=[
      (x_inner,y_inner),
      (x_outer,y_outer),
      (x_outer,-y_outer),
      (x_inner,-y_inner),
    ]
    color = self.get_matching_color(f'FilledCylinderC{which_bh}')
    self.patches.append(Polygon(vertices, closed=True, facecolor=color, edgecolor='black'))

def scalar_to_color(scalar_dict,min_max_tuple=None,color_map="viridis"):
  arr_keys,arr_vals = [], []
  for key,val in scalar_dict.items():
    if np.isnan(val):
      continue
    else:
      arr_keys.append(key)
      arr_vals.append(val)

  scalar_array = np.array(arr_vals, dtype=np.float64) 
  scalar_array = np.log10(scalar_array)
  min_val = np.min(scalar_array)
  max_val = np.max(scalar_array)
  print(min_val,max_val)
  if min_max_tuple is not None:
    min_val, max_val = min_max_tuple
  scalar_normalized = (scalar_array - min_val) / (max_val - min_val)

  colormap = plt.get_cmap(color_map)
  colors = {}
  for key,value in zip(arr_keys,scalar_normalized):
    colors[key] = colormap(value)

  # Get colorbar
  norm = Normalize(vmin=min_val, vmax=max_val)

  sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
  sm.set_array([])

  return colors,sm

# nA=4
# rA=nA*1.5
# center_xA=rA + 2
# RA=rA+5
# rC=RA*2
# nC=30
# RC=rC+nC

# fig, ax = plt.subplots(figsize=(12, 10))

# domain_color_local = domain_color.copy()
# patches_class = BBH_domain_sym_ploy(center_xA=center_xA, rA=rA, RA=RA, rC=rC, RC=RC, nA=nA, nC=nC, color_dict=domain_color_local) 
# for patch in patches_class.patches:
#   ax.add_patch(patch)

# ax.set_xlim(-RC, RC)
# ax.set_ylim(-RC, RC)
# ax.set_aspect('equal')

# %% [markdown]
# ### Functions to read h5 files

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
        if df.shape == (0,0):
            # data[:,0] is time and then we have the data
            data = h5_dir[var]
            
            # vars[:-4] to remove the .dat at the end
            col_names = make_col_names(var[:-4],data.shape[1]-1)
            col_names.append('t')
            # Reverse the list so that we get ["t","var_name"]
            col_names.reverse()            
            append_to_df(data[:],col_names,df)
            
        else:
            data = h5_dir[var]
            col_names = make_col_names(var[:-4],data.shape[1]-1)         
            append_to_df(data[:,1:],col_names,df)
            
    return df

def append_to_df(data,col_names,df):
    for i,col_name in enumerate(col_names):
        df[col_name] = data[:,i]
        
def make_col_names(val_name:str,val_size:int):
    col_names = []
    if val_size == 1:
        col_names.append(val_name)
    else:
        for i in range(val_size):
            col_names.append(val_name+f"_{i}")
    return col_names


def horizon_to_pandas(horizon_path:Path):
    assert(horizon_path.exists())
    df_dict = {}
    with h5py.File(horizon_path,'r') as hf:
        # Not all horizon files may have AhC
        for key in hf.keys():
            if key == 'VersionHist.ver':
                # Newer runs have this
                continue
            df_dict[key[:-4]] = make_Bh_pandas(hf[key])

    return df_dict

def read_horizon_across_Levs(path_list:List[Path]):
    df_listAB = []
    df_listC = []
    final_dict = {}
    for path in path_list:
        df_lev = horizon_to_pandas(path)
        # Either [AhA,AhB] or [AhA,AhB,AhC]
        if len(df_lev.keys()) > 1:
            df_listAB.append(df_lev)
        # Either [AhC] or [AhA,AhB,AhC]
        if (len(df_lev.keys()) == 1) or (len(df_lev.keys()) ==3):
            df_listC.append(df_lev)
    if len(df_listAB)==1:
        # There was only one lev
        final_dict = df_listAB[0]
    else:
        final_dict["AhA"] = pd.concat([df["AhA"] for df in df_listAB])
        final_dict["AhB"] = pd.concat([df["AhB"] for df in df_listAB])
        if len(df_listC) > 0:
            final_dict["AhC"] = pd.concat([df["AhC"] for df in df_listC])
    
    return final_dict

def load_horizon_data_from_levs(base_path:Path, runs_path:Dict[str,Path]):
  data_dict = {}
  for run_name in runs_path.keys():
    path_list = list(base_path.glob(runs_path[run_name]))
    print(path_list)
    data_dict[run_name] = read_horizon_across_Levs(path_list)
  return data_dict

def flatten_dict(horizon_data_dict:Dict[str,pd.DataFrame]) -> Dict[str,pd.DataFrame] :
  flattened_data = {}
  for run_name in horizon_data_dict.keys():
      for horizons in horizon_data_dict[run_name]:
          flattened_data[run_name+"_"+horizons] = horizon_data_dict[run_name][horizons]
          # print(run_name+"_"+horizons)
  return flattened_data

# %%
def read_profiler(file_name):
  with h5py.File(file_name,'r') as f:
    steps = set()
    procs = set()
    names = []
    f.visit(names.append)
    for name in names:
      step = name.split('.')[0][4:]
      steps.add(step)
      if 'Proc' in name:
        procs.add(name.split('/')[-1][4:-4])

    dict_list = []
    for step in steps:
      for proc in procs:
        data = f[f'Step{step}.dir/Proc{proc}.txt'][0].decode()

        lines = data.split("\n")
        time = float((lines[0].split("=")[-1])[:-1])

        curr_dict = {
            "t(M)": time,
            "step": step,
            "proc": proc
        }
        # Find where the columns end
        a = lines[4]
        event_end = a.find("Event")+5
        cum_end = a.find("cum(%)")+6
        exc_end = a.find("exc(%)")+6
        inc_end = a.find("inc(%)")+6

        for line in lines[6:-2]:
          Event = line[:event_end].strip()
          cum = float(line[event_end:cum_end].strip())
          exc = float(line[cum_end:exc_end].strip())
          inc = float(line[exc_end:inc_end].strip())
          N = int(line[inc_end:].strip())
          # print(a)
          # a = line.split("  ")
          # Event,cum,exc,inc,N = [i.strip() for i in a if i!= '']
          curr_dict[f'{Event}_cum'] = cum
          curr_dict[f'{Event}_exc'] = exc
          curr_dict[f'{Event}_inc'] = inc
          curr_dict[f'{Event}_N'] = N

        dict_list.append(curr_dict)
  return pd.DataFrame(dict_list)

def read_profiler_multiindex(folder_path:Path):
  dir_paths,dat_paths = list_all_dir_and_dat_files(folder_path)
  steps = set()
  # Get step names
  for dir in dir_paths:
    step = dir.name.split('.')[0][4:]
    steps.add(step)

  procs = set()
  # Get the proc names
  for txt in dir_paths[0].iterdir():
    if ".txt" in txt.name and "Summary" not in txt.name:
      procs.add(txt.name[4:-4])

  dict_list = []
  col_names = set()
  row_names = []
  for step in steps:
    for proc in procs:
      txt_file_path = folder_path/f'Step{step}.dir/Proc{proc}.txt'

      with txt_file_path.open("r") as f:
        lines = f.readlines()

      time = float((lines[0].split("=")[-1])[:-2])

      curr_dict = {
          "time": time,
          "step": step,
          "proc": proc
      }

      # Find where the columns end
      a = lines[4]
      event_end = a.find("Event")+5
      cum_end = a.find("cum(%)")+6
      exc_end = a.find("exc(%)")+6
      inc_end = a.find("inc(%)")+6

      row_names.append((str(proc),str(time)))

      for line in lines[6:-2]:
        Event = line[:event_end].strip()
        cum = float(line[event_end:cum_end].strip())
        exc = float(line[cum_end:exc_end].strip())
        inc = float(line[exc_end:inc_end].strip())
        N = int(line[inc_end:].strip())
        # print(a)
        # a = line.split("  ")
        # Event,cum,exc,inc,N = [i.strip() for i in a if i!= '']
        col_names.add(Event)
        curr_dict[("cum",Event)] = cum
        curr_dict[("exc",Event)] = exc
        curr_dict[("inc",Event)] = inc
        curr_dict[("N",Event)] = N

      dict_list.append(curr_dict)

  # Multi index rows
  index = pd.MultiIndex.from_tuples(row_names, names=["proc","t(M)"])
  df = pd.DataFrame(dict_list,index=index)
  
  # Multi index cols
  multi_index_columns = [(k if isinstance(k, tuple) else (k, '')) for k in df.columns]
  df.columns = pd.MultiIndex.from_tuples(multi_index_columns)
  df.columns.names = ['metric', 'process']

  # data.xs('24', level="proc")['N']
  # data.xs('0.511442', level="t(M)")['cum']
  # data.xs(('0','0.511442'),level=('proc','t(M)'))
  # data.xs('cum',level='metric',axis=1) = data['cum']
  # data.xs('MPI::MPreduceAdd(MV<double>)',level='process',axis=1)
  # data[data['time']<50]
  # data[data['time']<50]['cum'].xs('0',level='proc')['MPI::MPreduceAdd(MV<double>)']
  return df.sort_index()

# %% [markdown]
# ### Functions to read dat and hist files

# %%
def read_dat_file(file_name):
  cols_names = []
  # Read column names
  with open(file_name,'r') as f:
      lines = f.readlines()
      for line in lines:
        if "#" not in line:
          # From now onwards it will be all data
          break
        elif "=" in line:
          if ("[" not in line) and ("]" not in line):
             continue
          cols_names.append(line.split('=')[-1][1:-1].strip())
        else:
          continue

  return pd.read_csv(file_name,sep="\s+",comment="#",names=cols_names)

def hist_files_to_dataframe(file_path):
  # Function to parse a single line and return a dictionary of values
  def parse_line(line):
      data = {}
      # Find all variable=value pairs
      pairs = re.findall(r'([^;=\s]+)=\s*([^;]+)', line)
      for var, val in pairs:
          # Hist-GrDomain.txt should be parsed a little differently
          if 'ResizeTheseSubdomains' in var:
              items = val.split('),')
              items[-1] = items[-1][:-1]
              for item in items:
                name,_,vals = item.split("(")
                r,l,m=vals[:-1].split(',')
                data[f"{name}_R"] = int(r)
                data[f"{name}_L"] = int(l)
                data[f"{name}_M"] = int(m)
          else:
              data[var] = float(val) if re.match(r'^[\d.e+-]+$', val) else val
      return data
  
  with open(file_path, 'r') as file:
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
  with open(file_name,'r') as f:
    with open(temp_file,'w') as w:
      lines = f.readlines()
      for line in lines:
        if(line[0] != '#'): # This is data
          w.writelines(" ".join(line.split()[:col_length])+"\n")
        if(line[0:3] == '# [' or line[0:4] == '#  ['): # Some dat files have comments on the top
          cols_names.append(line.split('=')[-1][1:-1].strip())
          col_length = col_length+1


  return pd.read_csv(temp_file,delim_whitespace=True,names=cols_names)

def read_dat_file_across_AA(file_pattern):

  # ApparentHorizons/Horizons.h5@AhA
  if 'Horizons.h5@' in file_pattern:
    file_pattern,h5_key = file_pattern.split('@')

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
        returned_data = read_horizonh5(path,h5_key)
        if returned_data is not None:
            read_data_collection.append(returned_data)
    else:
        read_data_collection.append(read_dat_file(path))

  data = pd.concat(read_data_collection)
  rename_dict = {
     't':'t(M)',
     'time':'t(M)',
     'Time':'t(M)',
     'time after step':'t(M)',
  }
  data.rename(columns=rename_dict, inplace=True)
  # print(data.columns)
  return data

def read_horizonh5(horizonh5_path,h5_key):
  with h5py.File(horizonh5_path,'r') as hf:
    # h5_key = ['AhA','AhB','AhC']
    # Horizons.h5 has keys 'AhA.dir'
    key = h5_key+".dir"
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

  return dataA,dataB  

  
# Combines all the pvd files into a single file and save it in the base folder
def combine_pvd_files(base_folder:Path, file_pattern:str, output_path=None):
  pvd_start ="""<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n"""
  pvd_end ="  </Collection>\n</VTKFile>"

  vis_folder_name = file_pattern.split("/")[-1][:-4]
  Lev = file_pattern[0:4]

  if output_path is None:
    output_path = f"{base_folder}/{vis_folder_name}_{Lev}.pvd"

  pvd_files = list(base_folder.glob(file_pattern))
  pvd_folders = list(base_folder.glob(file_pattern[:-4]))


  with open(output_path,'w') as write_file:
    write_file.writelines(pvd_start)
    for files in pvd_files:
      print(files)
      with files.open("r") as f:
        for line in f.readlines():
          line = line.replace(vis_folder_name,str(files)[:-4])
          if "DataSet" in line:
            write_file.writelines(line)
    write_file.writelines(pvd_end)
  
  print(output_path)

def moving_average(array,avg_len):
    return np.convolve(array,np.ones(avg_len))/avg_len
    
def moving_average_valid(array,avg_len):
    return np.convolve(array,np.ones(avg_len),'valid')/avg_len


def path_to_folder_name(folder_name):
  return folder_name.replace("/","_")

# Give a dict of {"run_name" = runs_path} and data_file_path to get {"run_name" = dat_file_data}
def load_data_from_levs(runs_path, data_file_path):
  data_dict = {}
  column_list = ""
  for run_name in runs_path.keys():
    data_dict[run_name] = read_dat_file_across_AA(runs_path[run_name]+data_file_path)
    column_list = data_dict[run_name].columns
  return column_list, data_dict

def add_diff_columns(runs_data_dict, x_axis, y_axis, diff_base):
  if diff_base not in runs_data_dict.keys():
    raise Exception(f"{diff_base} not in {runs_data_dict.keys()}")

  unique_x_data, unique_indices = np.unique(runs_data_dict[diff_base][x_axis], return_index=True)
  # sorted_indices = np.sort(unique_indices)
  unique_y_data = runs_data_dict[diff_base][y_axis].iloc[unique_indices]
  interpolated_data = CubicSpline(unique_x_data,unique_y_data,extrapolate=False)

  for key in runs_data_dict:
    if key == diff_base:
      continue
    df = runs_data_dict[key]
    df['diff_abs_'+y_axis] = np.abs(df[y_axis] - interpolated_data(df[x_axis]))
    df['diff_'+y_axis] = df[y_axis] - interpolated_data(df[x_axis])

def plot_graph_for_runs_wrapper(runs_data_dict, x_axis, y_axis_list, minT, maxT, legend_dict=None, save_path=None, moving_avg_len=0, plot_fun = lambda x,y,label : plt.plot(x,y,label=label),sort_by=None, diff_base=None, title=None,append_to_title="",plot_abs_diff=False,constant_shift_val_time=None):

  # Do this better using columns of a pandas dataframe
  for y_axis in y_axis_list[:-1]:
    legend_dict = {}
    for key in runs_data_dict:
      legend_dict[key] = key+"_"+str(y_axis)
    plot_graph_for_runs(runs_data_dict, x_axis, y_axis, minT, maxT, legend_dict=legend_dict, save_path=None, moving_avg_len=moving_avg_len, plot_fun = plot_fun,sort_by=sort_by, diff_base=diff_base, title=title,append_to_title=append_to_title,plot_abs_diff=plot_abs_diff,constant_shift_val_time=constant_shift_val_time)

  # Save when plotting the last y_axis.
  y_axis = y_axis_list[-1]
  legend_dict = {}
  for key in runs_data_dict:
    legend_dict[key] = key+"_"+str(y_axis)
  plot_graph_for_runs(runs_data_dict, x_axis, y_axis, minT, maxT, legend_dict=legend_dict, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun = plot_fun,sort_by=sort_by, diff_base=diff_base, title=title,append_to_title=append_to_title,plot_abs_diff=plot_abs_diff,constant_shift_val_time=constant_shift_val_time)

  plt.ylabel("")
  plt.title(""+append_to_title)

  if save_path is not None:
    fig_x_label = ""
    fig_y_label = ""

    for y_axis in y_axis_list:
      fig_x_label = fig_x_label + x_axis.replace("/","_").replace(".","_")
      fig_y_label = fig_y_label + y_axis.replace("/","_").replace(".","_")
    save_file_name = f"{fig_y_label}_vs_{fig_x_label}_minT={minT}_maxT={maxT}".replace(".","_")
    if moving_avg_len > 0:
      save_file_name = save_file_name + f"_moving_avg_len={moving_avg_len}"
    if diff_base is not None:
      save_file_name = save_file_name + f"_diff_base={diff_base}"

    if len(save_file_name) >= 251: # <save_file_name>.png >=255
      save_file_name = save_file_name[:245]+str(random.randint(10000,99999))
      print(f"The filename was too long!! New filename is {save_file_name}")

    plt.savefig(save_path+save_file_name)

def plot_graph_for_runs(runs_data_dict, x_axis, y_axis, minT, maxT, legend_dict=None, save_path=None, moving_avg_len=0, plot_fun = lambda x,y,label : plt.plot(x,y,label=label),sort_by=None, diff_base=None, title=None,append_to_title="",plot_abs_diff=False,constant_shift_val_time=None):
  sort_run_data_dict(runs_data_dict,sort_by=sort_by)
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
      y_axis = "diff_"+y_axis
 
  # Find the indices corresponding to maxT and minT
  minT_indx_list={}
  maxT_indx_list={}

  if legend_dict is None:
    legend_dict = {}
    for run_name in current_runs_data_dict_keys:
      legend_dict[run_name] = None
  else:
    for run_name in current_runs_data_dict_keys:
      if run_name not in legend_dict:
        raise ValueError(f"{run_name} not in {legend_dict=}")

  
  for run_name in current_runs_data_dict_keys:
    minT_indx_list[run_name] = len(runs_data_dict[run_name][x_axis][runs_data_dict[run_name][x_axis] < minT])
    maxT_indx_list[run_name] = len(runs_data_dict[run_name][x_axis][runs_data_dict[run_name][x_axis] < maxT])

  if moving_avg_len == 0:

    for run_name in current_runs_data_dict_keys:
      x_data = runs_data_dict[run_name][x_axis][minT_indx_list[run_name]:maxT_indx_list[run_name]]
      y_data = runs_data_dict[run_name][y_axis][minT_indx_list[run_name]:maxT_indx_list[run_name]]

      if constant_shift_val_time is not None:
          shift_label_val = np.abs(x_data.iloc[-1] - x_data.iloc[0])/4
          unique_x_data, unique_indices = np.unique(x_data, return_index=True)
          # sorted_indices = np.sort(unique_indices)
          unique_y_data = y_data.iloc[unique_indices]
          try:
            interpolated_data = CubicSpline(unique_x_data,unique_y_data,extrapolate=False)
          except Exception as e:
            print(run_name,unique_y_data)
          y_data = y_data - interpolated_data(constant_shift_val_time)
      

    #   print(f"{len(x_data)=},{len(y_data)=},{len(np.argsort(x_data))=},{type(x_data)=}")

    #   sorted_indices = x_data.argsort()
    #   x_data = x_data.iloc[sorted_indices]
    #   y_data = y_data.iloc[sorted_indices]
      legend = legend_dict[run_name]
      if legend is None:
        legend = run_name
      plot_fun(x_data, y_data,legend)

      if constant_shift_val_time is not None:
        plt.axhline(y=y_data.iloc[-1], linestyle=':')
        plt.text(x=np.random.rand()*shift_label_val+x_data.iloc[0], y=y_data.iloc[-1], s=f'{y_data.iloc[-1]:.2e}', verticalalignment='bottom')

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if constant_shift_val_time is not None:
      plt.axvline(x=constant_shift_val_time, linestyle=':', color='red')
    if title is None:
      title = "\"" +  y_axis+"\" vs \""+x_axis+"\""
      if constant_shift_val_time is not None:
        title = title + f" constant_shift_val_time={constant_shift_val_time}"
      if diff_base is not None:
        title = title + f" diff_base={diff_base}"
    plt.title(title+append_to_title)
    plt.legend()

  else:
    for run_name in current_runs_data_dict_keys:
      x_data = np.array(runs_data_dict[run_name][x_axis][minT_indx_list[run_name] + moving_avg_len-1:maxT_indx_list[run_name]])
      y_data = np.array(moving_average_valid(runs_data_dict[run_name][y_axis][minT_indx_list[run_name]:maxT_indx_list[run_name]], moving_avg_len))

      if constant_shift_val_time is not None:
          shift_label_val = np.abs(x_data.iloc[-1] - x_data.iloc[0])/4
          unique_x_data, unique_indices = np.unique(x_data, return_index=True)
          # sorted_indices = np.sort(unique_indices)
          unique_y_data = y_data.iloc[unique_indices]
          
          interpolated_data = CubicSpline(unique_x_data,unique_y_data,extrapolate=False)
          y_data = y_data - interpolated_data(constant_shift_val_time)
      

    #   sorted_indices = np.argsort(x_data)
    #   x_data = x_data[sorted_indices]
    #   y_data = y_data[sorted_indices]
      legend = legend_dict[run_name]
      if legend is None:
        legend = run_name
      plot_fun(x_data, y_data,legend)

      if constant_shift_val_time is not None:
        plt.axhline(y=y_data.iloc[-1], linestyle=':')
        plt.text(x=np.random.rand()*shift_label_val+x_data.iloc[0], y=y_data.iloc[-1], s=f'{y_data.iloc[-1]:.1f}', verticalalignment='bottom')

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    if constant_shift_val_time is not None:
      plt.axvline(x=constant_shift_val_time, linestyle=':', color='red')
    if title is None:
      title = "\"" + y_axis+ "\" vs \"" + x_axis + "\"  " + f"avg_window_len={moving_avg_len}"
      if constant_shift_val_time is not None:
        title = title + f" constant_shift_val_time={constant_shift_val_time}"
      if diff_base is not None:
        title = title + f" diff_base={diff_base}"
    plt.title(title+append_to_title)
    plt.legend()

  
  if save_path is not None:
    fig_x_label = x_axis.replace("/","_").replace(".","_")
    fig_y_label = y_axis.replace("/","_").replace(".","_")
    save_file_name = f"{fig_y_label}_vs_{fig_x_label}_minT={minT}_maxT={maxT}".replace(".","_")
    if moving_avg_len > 0:
      save_file_name = save_file_name + f"_moving_avg_len={moving_avg_len}"
    if diff_base is not None:
      save_file_name = save_file_name + f"_diff_base={diff_base}"

    for run_name in current_runs_data_dict_keys:
      save_file_name = save_file_name + "__" + run_name.replace("/","_").replace(".","_")

    if len(save_file_name) >= 251: # <save_file_name>.png >=255
      save_file_name = save_file_name[:245]+str(random.randint(10000,99999))
      print(f"The filename was too long!! New filename is {save_file_name}")

    plt.savefig(save_path+save_file_name)


def find_file(pattern):
  return glob.glob(pattern, recursive=True)[0]

def plots_for_a_folder(things_to_plot,plot_folder_path,data_folder_path):
  for plot_info in things_to_plot:
    file_name = plot_info['file_name']
    y_arr = plot_info['columns'][1:]
    x_arr = [plot_info['columns'][0]]*len(y_arr)

    data = read_dat_file_across_AA(data_folder_path+"/**/"+file_name)
    plot_and_save(data,x_arr,y_arr,plot_folder_path,file_name)

def is_the_current_run_going_on(run_folder):
  if len(find_file(run_folder+"/**/"+"TerminationReason.txt")) > 0:
    return False
  else:
    return True

def plot_min_grid_spacing(runs_data_dict):
    '''
    runs_data_dict should have dataframes with MinimumGridSpacing.dat data.
    The function will compute the min grid spacing along all domains and plot it.
    '''
    keys = runs_data_dict.keys()
    if len(keys) == 0:
        print("There are no dataframes in the dict")

    for key in keys:
        t_step = runs_data_dict[key]['t']
        min_val = runs_data_dict[key].drop(columns=['t']).min(axis='columns')
        plt.plot(t_step,min_val,label=key)

    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Min Grid Spacing")
    plt.title("Min grid spacing in all domains")
    plt.show()

def plot_GrAdjustSubChunksToDampingTimes(runs_data_dict):
    keys = runs_data_dict.keys()
    if len(keys) > 1:
        print("To plot the Tdamp for various quantities only put one dataframe in the runs_data_dict")

    data:pd.DataFrame = runs_data_dict[list(keys)[0]]
    tdamp_keys = []
    for key in data.keys():
        if 'Tdamp' in key:
            tdamp_keys.append(key)

    # Get a colormap
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(tdamp_keys)))

    t_vals = data['time']
    for i, color, key in zip(range(len(tdamp_keys)),colors, tdamp_keys):
        if i%2==0:
            plt.plot(t_vals,data[key],label=key,color=color)
        else:
            plt.plot(t_vals,data[key],label=key,color=color,linestyle="--")


    min_tdamp = data[tdamp_keys].min(axis='columns')
    plt.plot(t_vals,min_tdamp,label="min_tdamp",linewidth=3,linestyle="dotted",color="red")

    plt.legend()
    plt.xlabel("time")
    plt.title(list(keys)[0])
    plt.show()

def add_max_and_min_val(runs_data_dict):
    # If we load a file with 5 columns with first being time, then find max and min values for all the other columns, at all times and add it to the dataframe.
    # Useful when you want to find like Linf across all domains at all times
    for run_name in runs_data_dict.keys():
        data_frame = runs_data_dict[run_name]
        t = data_frame.iloc[:,0]
        max_val = np.zeros_like(t)
        min_val = np.zeros_like(t)
        for i in range(len(t)):
            max_val[i] = data_frame.iloc[i,1:].max()
            min_val[i] = data_frame.iloc[i,1:].max()

        # Add the values to the dataframe
        data_frame['max_val'] = max_val
        data_frame['min_val'] = min_val

def sort_run_data_dict(runs_data_dict:dict,sort_by=None):
    for run_name in runs_data_dict.keys():
        run_df = runs_data_dict[run_name]
        if sort_by is None:
            sort_by = run_df.keys()[0]
        runs_data_dict[run_name] = run_df.sort_values(by=sort_by)

# %% [markdown]
# # Plot dat files

# %%
# Old runs
runs_to_plot = {}
# runs_to_plot["boost_ID_test_wrong"] =  "/groups/sxs/hchaudha/spec_runs/boost_ID_test_wrong/Ev/Lev3_A?/Run/"
# runs_to_plot["boost_ID_test_correct"] =  "/groups/sxs/hchaudha/spec_runs/boost_ID_test_correct/Ev/Lev3_A?/Run/"
# runs_to_plot["corrected_coord_spin1"] =  "/groups/sxs/hchaudha/spec_runs/corrected_coord_spin1/Ev/Lev3_A?/Run/"
# runs_to_plot["corrected_coord_spin2"] =  "/groups/sxs/hchaudha/spec_runs/corrected_coord_spin2/Ev/Lev3_A?/Run/"
# runs_to_plot["2_SpKS_q1_sA_0_0_0_sB_0_0_0_d15"] =  "/groups/sxs/hchaudha/spec_runs/2_SpKS_q1_sA_0_0_0_sB_0_0_0_d15/Ev/Lev3_A?/Run/"
# runs_to_plot["2_SpKS_q1_sA_0_0_0_sB_0_0_99_d15"] =  "/groups/sxs/hchaudha/spec_runs/2_SpKS_q1_sA_0_0_0_sB_0_0_99_d15/Ev/Lev3_A?/Run/"
# runs_to_plot["2_SpKS_q1_sA_0_0_0_sB_0_0_9_d15"] =  "/groups/sxs/hchaudha/spec_runs/2_SpKS_q1_sA_0_0_0_sB_0_0_9_d15/Ev/Lev3_A?/Run/"
# runs_to_plot["2_SpKS_q1_sA_0_0_9_sB_0_0_9_d15"] =  "/groups/sxs/hchaudha/spec_runs/2_SpKS_q1_sA_0_0_9_sB_0_0_9_d15/Ev/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["L3_tol8_eq"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol8_eq/Lev3_A?/Run/"
# runs_to_plot["L3_tol9"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol9/Lev3_A?/Run/"
# runs_to_plot["L3_tol10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol10/Lev3_A?/Run/"
# runs_to_plot["L3_tol10_hi"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol10_hi/Lev3_A?/Run/"
# runs_to_plot["L3_tol11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol11/Lev3_A?/Run/"
# runs_to_plot["L3_all_100_tol10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/L6_tol10/Lev3_A?/Run/"
# runs_to_plot["L3_all_1000_tol11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/L6_all_10_tol11/Lev3_A?/Run/"
# runs_to_plot["local_100_tol5_11"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/local_100_tol5_11/Lev3_A?/Run/"
# runs_to_plot["local_10_tol5_10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/local_10_tol5_10/Lev3_A?/Run/"
# runs_to_plot["L3_local_10_tol5_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/local_10_tol5_10/Lev3_A?/Run/"
# runs_to_plot["L3_local_100_tol5_11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/local_100_tol5_11/Lev3_A?/Run/"


# runs_to_plot["3_DH_q1_ns_d18_L3_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3/Ev/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_tol8_eq_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol8_eq/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_tol9_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol9/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_tol10_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol10/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_tol10_hi_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol10_hi/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_tol11_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/tol11/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_all_100_tol10_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/L6_tol10/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_all_1000_tol11_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/L6_all_10_tol11/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L3_rd"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3/Ev/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L6"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L6/Ev/Lev6_A?/Run/"
# runs_to_plot["L6_1.1"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L6_higher_acc/L6_1.1/Lev6_A?/Run/"
# runs_to_plot["L6_1.1_dp8_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L6_higher_acc/L6_1.1_dp8_tol_10/Lev6_A?/Run/"
# runs_to_plot["L6_1.1_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L6_higher_acc/L6_1.1_tol_10/Lev6_A?/Run/"
# runs_to_plot["L6_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L6_higher_acc/L6_tol_10/Lev6_A?/Run/"
# runs_to_plot["all_10"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_10/Lev3_A?/Run/"
# runs_to_plot["all_100"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100/Lev3_A?/Run/"
# runs_to_plot["near_bhs_10"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/near_bhs_10/Lev3_A?/Run/"
# runs_to_plot["near_bhs_100"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/near_bhs_100/Lev3_A?/Run/"
# runs_to_plot["same_obs"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/same_obs/Lev3_A?/Run/"
# runs_to_plot["all_10_obs"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_10_obs/Lev3_A?/Run/"
# runs_to_plot["all_100_obs"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_obs/Lev3_A?/Run/"
# runs_to_plot["all_10_obs_tol_10"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_10_obs_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_obs_tol_10"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_obs_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_1.1_b0"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_1.1_b0/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_1.1_b0_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_1.1_b0_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_1.1"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_1.1/Lev3_A?/Run/"
# runs_to_plot["all_100_1.1_dp8_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_1.1_dp8_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_1.1_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_1.1_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_2"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_2/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_2"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_2/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_3"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_3/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_dp8"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dp8/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_dp8_tol10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dp8_tol10/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_dp8_tol11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dp8_tol11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_dt"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.02"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.02/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.03"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.03/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.04"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.04/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.025"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.025/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.021"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.021/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.022"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.022/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.023"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.023/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.024"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.024/Lev3_A?/Run/"
runs_to_plot["all_100_t2690_obs_grid_dt_0.0225"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt_0.0225/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_dt005"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_dt0.005/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_tol_11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_tol_9"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_9/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_tol_1.128e-11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_tol_1.128e-11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_tol_1.692e-11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_tol_1.692e-11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_tol_3.383e-11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_tol_3.383e-11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690/Lev3_A?/Run/"
# runs_to_plot["near_bhs_10_obs"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/near_bhs_10_obs/Lev3_A?/Run/"
# runs_to_plot["near_bhs_100_obs"] =  "/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/near_bhs_100_obs/Lev3_A?/Run/"
# runs_to_plot["3_DH_q1_ns_d18_L6_AA"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L6_AA/Ev/Lev3_A?/Run/"
# runs_to_plot["4_SphKS_q1_15_SSphKS_ID"] =  "/groups/sxs/hchaudha/spec_runs/4_SphKS_q1_15_SSphKS_ID/Ev/Lev3_A?/Run/"
# runs_to_plot["4_SphKS_q1_15_SKS_ID"] =  "/groups/sxs/hchaudha/spec_runs/4_SphKS_q1_15_SKS_ID/Ev/Lev3_A?/Run/"
# runs_to_plot["5_gd_SphKS_gauge_ID"] =  "/groups/sxs/hchaudha/spec_runs/5_gd_SphKS_gauge_ID/Ev/Lev2_A[A-S]/Run/"
# runs_to_plot["5_ngd_SphKS_ID"] =  "/groups/sxs/hchaudha/spec_runs/5_ngd_SphKS_ID/Ev/Lev2_A?/Run/"
# runs_to_plot["5_ngd_KS_ID"] =  "/groups/sxs/hchaudha/spec_runs/5_ngd_KS_ID/Ev/Lev2_A?/Run/"

# runs_to_plot["all_100_t2690_eteq_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eteq_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eteq_tol_11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eteq_tol_11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eteq_tol_12"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eteq_tol_12/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eteq_tol_8"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eteq_tol_8/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eteq_tol_9"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eteq_tol_9/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eteq_tol_eq"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eteq_tol_eq/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eth_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eth_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eth_tol_11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eth_tol_11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eth_tol_12"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eth_tol_12/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eth_tol_8"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eth_tol_8/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eth_tol_9"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eth_tol_9/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_eth_tol_eq"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eth_tol_eq/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_etl_tol_10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_etl_tol_10/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_etl_tol_11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_etl_tol_11/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_etl_tol_8"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_etl_tol_8/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_etl_tol_9"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_etl_tol_9/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_etl_tol_eq"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_etl_tol_eq/Lev3_A?/Run/"

# runs_to_plot["all_100_t2690_obs_grid_linf"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_linf/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_tol_10_linf"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_10_linf/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_tol_11_linf"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_11_linf/Lev3_A?/Run/"
# runs_to_plot["all_100_t2690_obs_grid_tol_9_linf"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_9_linf/Lev3_A?/Run/"

# runs_to_plot["t6115_tol11"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol11/Lev3_A?/Run/"
# runs_to_plot["t6115_tol10"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol10/Lev3_A?/Run/"
# runs_to_plot["t6115_tol9"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol9/Lev3_A?/Run/"
# runs_to_plot["t6115_tol8"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol8/Lev3_A?/Run/"
# runs_to_plot["t6115_tol7"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol7/Lev3_A?/Run/"
# runs_to_plot["t6115_tol11_AMR"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol11_AMR/Lev3_A?/Run/"
# runs_to_plot["t6115_tol10_AMR"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol10_AMR/Lev3_A?/Run/"
# runs_to_plot["t6115_tol9_AMR"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol9_AMR/Lev3_A?/Run/"
# runs_to_plot["t6115_tol8_AMR"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol8_AMR/Lev3_A?/Run/"
# runs_to_plot["t6115_tol7_AMR"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol7_AMR/Lev3_A?/Run/"
# runs_to_plot["t6115_tol8_linf"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol8_linf/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.02"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.02/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.03"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.03/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.041"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.041/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.042"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.042/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.043"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.043/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.044"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.044/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.045"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.045/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.046"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.046/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.047"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.047/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.048"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.048/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.049"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.049/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.050"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.050/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.052"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.052/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.054"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.054/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_dt0.056"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_dt0.056/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_2.368e-07"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_2.368e-07/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_1.692e-07"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_1.692e-07/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_1.015e-07"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_1.015e-07/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_6.767e-08"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_6.767e-08/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_5.075e-08"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_5.075e-08/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_3.383e-08"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_3.383e-08/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_2.256e-08"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_2.256e-08/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_1.692e-08"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_1.692e-08/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_1.128e-08"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_1.128e-08/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_6.767e-09"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_6.767e-09/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_4.833e-09"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_4.833e-09/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_3.383e-09"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_3.383e-09/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_1.692e-09"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_1.692e-09/Lev3_A?/Run/"
# runs_to_plot["t6115_linf_tol_1.128e-09"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_linf_tol_1.128e-09/Lev3_A?/Run/"


# runs_to_plot["all_100_t2710_0.021_0.021"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_0.21/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_0.022"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_0.22/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_0.0225"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_0.225/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_0.023"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_0.23/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_0.024"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_0.24/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_max_tol8"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_max_tol8/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_max_tol9"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_max_tol9/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_max_tol10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_max_tol10/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_max_tol10.5"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_max_tol10.5/Lev3_AE/Run/"
# runs_to_plot["all_100_t2710_0.021_max_tol11"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/all_100_t2710_0.021_max_tol11/Lev3_AE/Run/"

# runs_to_plot["eq_t4000_tol10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/eq_t4000_tol10/Lev3_A?/Run/"
# runs_to_plot["eq_t4000_tol5_10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/eq_t4000_tol5_10/Lev3_A?/Run/"
# runs_to_plot["eq_t4000_tol5_11"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/eq_t4000_tol5_11/Lev3_A?/Run/"
# runs_to_plot["eq_t4000_tol9"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/eq_t4000_tol9/Lev3_A?/Run/"
# runs_to_plot["t4000_tol10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/t4000_tol10/Lev3_A?/Run/"
# runs_to_plot["t4000_tol5_10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/t4000_tol5_10/Lev3_A?/Run/"
# runs_to_plot["t4000_tol5_11"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/t4000_tol5_11/Lev3_A?/Run/"
# runs_to_plot["t4000_tol8"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/t4000_tol8/Lev3_A?/Run/"
# runs_to_plot["t4000_tol9"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_all_10/t4000_tol9/Lev3_A?/Run/"

# runs_to_plot["Lev3_AA_tol10_all_10"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol10_all_10/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol11_all_10"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol11_all_10/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol12_all_10"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol12_all_10/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol5_10_all_10"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol5_10_all_10/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol5_11_all_10"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol5_11_all_10/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol10"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol10/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol11"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol11/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol12"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol12/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol5_10"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol5_10/Lev3_A?_/Run/"
# runs_to_plot["Lev3_AA_tol5_11"]  = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/rd_all_10_tol11/Lev3_AA_tol5_11/Lev3_A?_/Run/"



# %%
runs_to_plot = {}

# runs_to_plot["ode_change_Run1"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev1/Run/"
# runs_to_plot["ode_change_Run2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev2/Run/"
# runs_to_plot["ode_change_Run3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev3/Run/"
# runs_to_plot["ode_change_Run4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev4/Run/"
# runs_to_plot["ode_change_Run5"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev5/Run/"
# runs_to_plot["ode_change_Run6"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev6/Run/"
# runs_to_plot["ode_change_Run7"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev7/Run/"
# runs_to_plot["ode_change_Run8"] = "/groups/sxs/hchaudha/spec_runs/single_bh/0/Lev8/Run/"

# runs_to_plot["1000M_Run1"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev1/Run/"
# runs_to_plot["1000M_Run2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev2/Run/"
# runs_to_plot["1000M_Run3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev3/Run/"
# runs_to_plot["1000M_Run4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev4/Run/"
# runs_to_plot["1000M_Run5"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev5/Run/"
# runs_to_plot["1000M_Run6"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev6/Run/"
# runs_to_plot["1000M_Run7"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev7/Run/"
# runs_to_plot["1000M_Run7_a"] = "/groups/sxs/hchaudha/spec_runs/single_bh/2_1000M/Lev7_a/Run/"

# runs_to_plot["Lin_Run1"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev1/Run/"
# runs_to_plot["Lin_Run2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev2/Run/"
# runs_to_plot["Lin_Run3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev3/Run/"
# runs_to_plot["Lin_Run4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev4/Run/"
# runs_to_plot["Lin_Run5"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev5/Run/"
# runs_to_plot["Lin_Run6"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev6/Run/"
# runs_to_plot["Lin_Run7"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev7/Run/"

# runs_to_plot["400M_base1"] = "/groups/sxs/hchaudha/spec_runs/single_bh/4_400M_base/Lev1/Run/"
# runs_to_plot["400M_base2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/4_400M_base/Lev2/Run/"
# runs_to_plot["400M_base3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/4_400M_base/Lev3/Run/"
# runs_to_plot["400M_base4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/4_400M_base/Lev4/Run/"
# runs_to_plot["400M_base5"] = "/groups/sxs/hchaudha/spec_runs/single_bh/4_400M_base/Lev5/Run/"
# runs_to_plot["Lin_Run7"] = "/groups/sxs/hchaudha/spec_runs/single_bh/3_Lin/Lev7/Run/"

# runs_to_plot["400M_phys_bc1"] = "/groups/sxs/hchaudha/spec_runs/single_bh/12_400M_phys_bc/Lev1/Run/"
# runs_to_plot["400M_phys_bc2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/12_400M_phys_bc/Lev2/Run/"
# runs_to_plot["400M_phys_bc3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/12_400M_phys_bc/Lev3/Run/"
# runs_to_plot["400M_phys_bc4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/12_400M_phys_bc/Lev4/Run/"
# runs_to_plot["400M_phys_bc5"] = "/groups/sxs/hchaudha/spec_runs/single_bh/12_400M_phys_bc/Lev5/Run/"

# runs_to_plot["400M_phys_bc5"] = "/groups/sxs/hchaudha/spec_runs/single_bh/12_400M_phys_bc/Lev5/Run/"

# runs_to_plot["13_Lev4_250"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_250/Run/"
# runs_to_plot["13_Lev4_300"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_300/Run/"
# runs_to_plot["13_Lev4_350"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_350/Run/"
# runs_to_plot["13_Lev4_400"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_400/Run/"
# runs_to_plot["13_Lev4_450"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_450/Run/"
# runs_to_plot["13_Lev4_500"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_500/Run/"
# runs_to_plot["13_Lev4_550"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_550/Run/"
# runs_to_plot["13_Lev4_600"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_600/Run/"
# runs_to_plot["13_Lev4_650"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_650/Run/"
# runs_to_plot["13_Lev4_700"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_700/Run/"
# runs_to_plot["13_Lev4_750"] = "/groups/sxs/hchaudha/spec_runs/single_bh/13_error_falloff/Lev4_750/Run/"

# runs_to_plot["14_Lev4_250"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_250/Run/"
# runs_to_plot["14_Lev4_300"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_300/Run/"
# runs_to_plot["14_Lev4_350"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_350/Run/"
# runs_to_plot["14_Lev4_400"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_400/Run/"
# runs_to_plot["14_Lev4_450"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_450/Run/"
# runs_to_plot["14_Lev4_500"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_500/Run/"
# runs_to_plot["14_Lev4_550"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_550/Run/"
# runs_to_plot["14_Lev4_600"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_600/Run/"
# runs_to_plot["14_Lev4_650"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_650/Run/"
# runs_to_plot["14_Lev4_700"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_700/Run/"
# runs_to_plot["14_Lev4_750"] = "/groups/sxs/hchaudha/spec_runs/single_bh/14_error_fo_2000M/Lev4_750/Run/"

# runs_to_plot["15_AMR_Lev0_255"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev0_255/Run/"
# runs_to_plot["15_AMR_Lev0_355"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev0_355/Run/"
# runs_to_plot["15_AMR_Lev0_455"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev0_455/Run/"
# runs_to_plot["15_AMR_Lev1_255"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev1_255/Run/"
# runs_to_plot["15_AMR_Lev1_355"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev1_355/Run/"
# runs_to_plot["15_AMR_Lev1_455"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev1_455/Run/"
# runs_to_plot["15_AMR_Lev2_255"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev2_255/Run/"
# runs_to_plot["15_AMR_Lev2_355"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev2_355/Run/"
# runs_to_plot["15_AMR_Lev2_455"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev2_455/Run/"
# runs_to_plot["15_AMR_Lev3_255"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev3_255/Run/"
# runs_to_plot["15_AMR_Lev3_355"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev3_355/Run/"
# runs_to_plot["15_AMR_Lev3_455"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev3_455/Run/"
# runs_to_plot["15_AMR_Lev4_255"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev4_255/Run/"
# runs_to_plot["15_AMR_Lev4_355"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev4_355/Run/"
# runs_to_plot["15_AMR_Lev4_455"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev4_455/Run/"
# runs_to_plot["15_AMR_Lev5_255"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev5_255/Run/"
# runs_to_plot["15_AMR_Lev5_355"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev5_355/Run/"
# runs_to_plot["15_AMR_Lev5_455"] = "/groups/sxs/hchaudha/spec_runs/single_bh/15_AMR_test/Lev5_455/Run/"

# runs_to_plot["15_Lev5_255"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255/Run/"
# runs_to_plot["15_Lev5_255_010"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_010/Run/"
# runs_to_plot["15_Lev5_255_0100"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_0100/Run/"
# runs_to_plot["15_Lev5_255_01000"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_01000/Run/"
# runs_to_plot["15_Lev5_255_010000"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_010000/Run/"
# runs_to_plot["15_Lev5_255_05"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_05/Run/"
# runs_to_plot["15_Lev5_255_050"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_050/Run/"
# runs_to_plot["15_Lev5_255_0500"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_0500/Run/"
# runs_to_plot["15_Lev5_255_05000"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_05000/Run/"
# runs_to_plot["15_Lev5_255_050000"] = "/groups/sxs/hchaudha/spec_runs/single_bh/16_AMR_ode_tol_test/Lev5_255_050000/Run/"

# runs_to_plot["400M_gamma1_2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/5_400M_gamma1/Lev2/Run/"
# runs_to_plot["400M_gamma1_3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/5_400M_gamma1/Lev3/Run/"
# runs_to_plot["400M_gamma1_4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/5_400M_gamma1/Lev4/Run/"

# runs_to_plot["400M_gamma1_001_2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/6_400M_gamma1_001/Lev2/Run/"
# runs_to_plot["400M_gamma1_001_3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/6_400M_gamma1_001/Lev3/Run/"
# runs_to_plot["400M_gamma1_001_4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/6_400M_gamma1_001/Lev4/Run/"

# runs_to_plot["400M_gamma1_01_2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/7_400M_gamma1_01/Lev2/Run/"
# runs_to_plot["400M_gamma1_01_3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/7_400M_gamma1_01/Lev3/Run/"
# runs_to_plot["400M_gamma1_01_4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/7_400M_gamma1_01/Lev4/Run/"

# runs_to_plot["400M_BDres_2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/8_400M_BDres/Lev2/Run/"
# runs_to_plot["400M_BDres_3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/8_400M_BDres/Lev3/Run/"
# runs_to_plot["400M_BDres_4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/8_400M_BDres/Lev4/Run/"

# runs_to_plot["9_400M_BDres_05_2"] = "/groups/sxs/hchaudha/spec_runs/single_bh/9_400M_BDres_05/Lev2/Run/"
# runs_to_plot["9_400M_BDres_05_3"] = "/groups/sxs/hchaudha/spec_runs/single_bh/9_400M_BDres_05/Lev3/Run/"
# runs_to_plot["9_400M_BDres_05_4"] = "/groups/sxs/hchaudha/spec_runs/single_bh/9_400M_BDres_05/Lev4/Run/"

# runs_to_plot["10_freezing"] = "/groups/sxs/hchaudha/spec_runs/single_bh/10_freezing/Lev4/Run/"
# runs_to_plot["11_physical_bc"] = "/groups/sxs/hchaudha/spec_runs/single_bh/11_physical_bc/Lev4/Run/"

# runs_to_plot["Lev0_265"] = "/groups/sxs/hchaudha/spec_runs/single_bh/17_zero_spin_AMR/Lev0_265/Run/"
# runs_to_plot["Lev1_265"] = "/groups/sxs/hchaudha/spec_runs/single_bh/17_zero_spin_AMR/Lev1_265/Run/"
# runs_to_plot["Lev2_265"] = "/groups/sxs/hchaudha/spec_runs/single_bh/17_zero_spin_AMR/Lev2_265/Run/"
# runs_to_plot["Lev3_265"] = "/groups/sxs/hchaudha/spec_runs/single_bh/17_zero_spin_AMR/Lev3_265/Run/"
# runs_to_plot["Lev4_265"] = "/groups/sxs/hchaudha/spec_runs/single_bh/17_zero_spin_AMR/Lev4_265/Run/"
# runs_to_plot["Lev5_265"] = "/groups/sxs/hchaudha/spec_runs/single_bh/17_zero_spin_AMR/Lev5_265/Run/"

runs_to_plot["Lev5_265"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265/Run/"
# runs_to_plot["Lev5_265_003"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_003/Run/"
# runs_to_plot["Lev5_265_03"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_03/Run/"
# runs_to_plot["Lev5_265_10"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_10/Run/"
# runs_to_plot["Lev5_265_100"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_100/Run/"
# runs_to_plot["Lev5_265_1000"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_1000/Run/"
# runs_to_plot["Lev5_265_10000"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_10000/Run/"
# runs_to_plot["Lev5_265_30"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_30/Run/"
# runs_to_plot["Lev5_265_9"] = "/groups/sxs/hchaudha/spec_runs/single_bh/18_zero_spin_const_damp/Lev5_265_9/Run/"
# runs_to_plot["23_nobounds_AMR"] = "/groups/sxs/hchaudha/spec_runs/23_nobounds_AMR/Ev/Lev3_??/Run/"
# runs_to_plot["23_allcd_gaussExc_400"] = "/groups/sxs/hchaudha/spec_runs/23_allcd_gaussExc_400/Ev/Lev3_??/Run/"
# runs_to_plot["24_allcd_gaussEx_5_800"] = "/groups/sxs/hchaudha/spec_runs/24_allcd_gaussEx_5_800/Ev/Lev3_??/Run/"
# runs_to_plot["24_allcd_gaussEx_10_800"] = "/groups/sxs/hchaudha/spec_runs/24_allcd_gaussEx_10_800/Ev/Lev3_??/Run/"

# runs_to_plot["L5_AD_L5_ps_10"] = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master_segs/L5_AD_L5_ps_10/Ev/Lev5_AD/Run/"
# runs_to_plot["L5_AD_L5_ps_5"] = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master_segs/L5_AD_L5_ps_5/Ev/Lev5_AD/Run/"
# runs_to_plot["L5_AD_L5_ps_2"] = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master_segs/L5_AD_L5_ps_2/Ev/Lev5_AD/Run/"
# runs_to_plot["L5_AD_L5_BCSC_8"] = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master_segs/L5_AD_L5_BCSC_8/Ev/Lev5_AD/Run/"

# runs_to_plot["Lev5_ode_controller_fixed_0.07"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_ode_controller_0.07/Lev5_AC/Run/"
# runs_to_plot["Lev5_ode_controller"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_ode_controller/Lev5_AC/Run/"

# runs_to_plot["eq_AMR_3_tier_const"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/eq_AMR_3_tier_const/Ev/Lev3_A?/Run/"
# runs_to_plot["eq_AMR_3_tier_const_gamma2"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/eq_AMR_3_tier_const_gamma2/Ev/Lev3_A?/Run/"
# runs_to_plot["three_tier_AMR_const_L1"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/three_tier_AMR_const/Ev/Lev1_A?/Run/"
# runs_to_plot["three_tier_AMR_const_L2"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/three_tier_AMR_const/Ev/Lev2_A?/Run/"
# runs_to_plot["three_tier_AMR_const_L3"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/three_tier_AMR_const/Ev/Lev3_A?/Run/"
# runs_to_plot["normal_constraints"]="/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/eq_AMR_3_tier_const_variations/normal_constraints/Lev3_A?/Run/"
# runs_to_plot["normal_constraints_12_AB"]="/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/eq_AMR_3_tier_const_variations/normal_constraints_12_AB/Lev3_A?/Run/"
# runs_to_plot["normal_constraints_const1"]="/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_contraints/eq_AMR_3_tier_const_variations/normal_constraints_const1/Lev3_A?/Run/"

# runs_to_plot["high_accuracy_L3_tol8"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev3_tol8_checkpoint/Lev3_A?/Run/"
# runs_to_plot["L4_tol8"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev4_tol8/Lev4_A?/Run/"
# runs_to_plot["Lev4_AD_uniform_5"]  = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev4_uniform_constraints/Lev4_AD_uniform_5/Run/"
# runs_to_plot["Lev4_AD_uniform_1"]  = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev4_uniform_constraints/Lev4_AD_uniform_1/Run/"
# runs_to_plot["Lev4_AD_uniform_0.1"]  = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev4_uniform_constraints/Lev4_AD_uniform_0.1/Run/"
# runs_to_plot["Lev4_AD_big_gaussian"]  = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev4_uniform_constraints/Lev4_AD_big_gaussian/Run/"
# runs_to_plot["Lev4_AD_uniform_1_gamma2_0999"]  = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev4_uniform_constraints/Lev4_AD_uniform_1_gamma2_0999/Run/"
# runs_to_plot["Lev4_AD_uniform_0.1_gamma2_0999 "]  = "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev4_uniform_constraints/Lev4_AD_uniform_0.1_gamma2_0999/Run/"

# runs_to_plot["3_100"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/3_100/Lev3_A?/Run/"
# runs_to_plot["3_10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/3_10/Lev3_A?/Run/"
# runs_to_plot["3_1"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/3_1/Lev3_A?/Run/"
# runs_to_plot["2_100"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/2_100/Lev3_A?/Run/"
# runs_to_plot["2_10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/2_10/Lev3_A?/Run/"
# runs_to_plot["2_1"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/2_1/Lev3_A?/Run/"
# runs_to_plot["1_100"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/1_100/Lev3_A?/Run/"
# runs_to_plot["1_10"] = "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_ringdown_tol/changing_spectral_grid/1_10/Lev3_A?/Run/"

# runs_to_plot["1686_1.0e-07_043"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_043/Run/"
# runs_to_plot["1686_1.0e-07_046"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_046/Run/"
# runs_to_plot["1686_1.0e-07_046_10"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_046_10/Run/"
# runs_to_plot["1686_1.0e-07_046_6"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_046_6/Run/"
# runs_to_plot["1686_1.0e-07_046x0.5"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_046x0.5/Run/"
# runs_to_plot["1686_1.0e-07_046x2"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_046x2/Run/"
# runs_to_plot["1686_1.0e-07_048"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_048/Run/"
# runs_to_plot["1686_1.0e-07_050"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_050/Run/"
# runs_to_plot["1686_1.0e-07_055"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/1686_1.0e-07_055/Run/"
# runs_to_plot["3555_1.0e-07_040"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_040/Run/"
# runs_to_plot["3555_1.0e-07_045"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_045/Run/"
# runs_to_plot["3555_1.0e-07_045_10"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_045_10/Run/"
# runs_to_plot["3555_1.0e-07_045_6"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_045_6/Run/"
# runs_to_plot["3555_1.0e-07_045x0.5"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_045x0.5/Run/"
# runs_to_plot["3555_1.0e-07_045x2"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_045x2/Run/"
# runs_to_plot["3555_1.0e-07_046"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_046/Run/"
# runs_to_plot["3555_1.0e-07_047"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_047/Run/"
# runs_to_plot["3555_1.0e-07_050"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_050/Run/"
# runs_to_plot["3555_1.0e-07_055"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_055/Run/"
# runs_to_plot["3555_1.0e-07_060"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_060/Run/"
# runs_to_plot["3555_1.0e-07_065"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_065/Run/"
# runs_to_plot["3555_1.0e-07_070"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_070/Run/"
# runs_to_plot["3555_1.0e-07_100"] = "/groups/sxs/hchaudha/spec_runs/6_lev3_step_size_check/3555_1.0e-07_100/Run/"

# runs_to_plot["5_ngd_SphKS_ID"] = "/groups/sxs/hchaudha/spec_runs/5_ngd_SphKS_ID/Ev/Lev2_??/Run/"
# runs_to_plot["5_ngd_KS_ID"] = "/groups/sxs/hchaudha/spec_runs/5_ngd_KS_ID/Ev/Lev2_??/Run/"
# runs_to_plot["5_gd_SphKS_gauge_ID"] = "/groups/sxs/hchaudha/spec_runs/5_gd_SphKS_gauge_ID/Ev/Lev2_??/Run/"

# runs_to_plot["6_set1_L3_fil_buff0"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff0/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_fil_buff3"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_del/6_set1_L3_template/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_fil_buff2"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff2/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_fil_buff4"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff4/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_fil_buff0_14012"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff0_14012/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_fil_buff2_14012"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff2_14012/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_fil_buff4_14012"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff4_14012/Ev/Lev3_AB/Run/"

# runs_to_plot["6_set1_L3_FK_9443_C"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_9443_C/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_FK_9443_All"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_9443_All/Ev/Lev3_AB/Run/"

# runs_to_plot["6_set1_L3_FK_14012_C"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_14012_C/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_FK_14012_All"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_14012_All/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_FK_14012_All_11"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_14012_All_11/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_FK_14012_C_9"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_14012_C_9/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_FK_14012_C_13"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_14012_C_13/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_FK_14012_C_copy"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_FK_14012_C_copy/Ev/Lev3_AB/Run/"

# runs_to_plot["6_set1_L3_EXP_FK_14012_5_6_30"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_5_6_30/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_EXP_FK_14012_10_6_30"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_10_6_30/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_EXP_FK_14012_1_6_30"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_1_6_30/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_EXP_FK_14012_5_4_30"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_5_4_30/Ev/Lev3_AB/Run/"
# runs_to_plot["6_set1_L3_EXP_FK_14012_5_2_30"] = "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_5_2_30/Ev/Lev3_AB/Run/"


# data_file_path = "FailedTStepperDiag.dat"
# data_file_path = "GhCe_Norms.dat"
# data_file_path = "GhCe.dat"
# data_file_path = "NormalizedGhCe.dat"
# data_file_path = "GhCe_Linf.dat"
# data_file_path = "GhCe.dat"
# data_file_path = "NormalizedGhCe_Linf.dat"
# data_file_path = "1Con.dat"
# data_file_path = "2Con.dat"
data_file_path = "3Con.dat"
# data_file_path = "kappaErr_Linf.dat"
# data_file_path = "psiErr_Linf.dat"
# data_file_path = "TStepperDiag.dat"

column_names, runs_data_dict = load_data_from_levs(runs_to_plot,data_file_path)
print(column_names)

# %%
runs_to_plot = {}

# runs_to_plot["73_gd_master_new_code"] =  "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/73_gd_master_new_code/Ev/Lev1_A?/Run/"
# runs_to_plot["119_gd_SUKS_3_20"] =  "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/119_gd_SUKS_3_20/Ev/Lev3_A?/Run/"
runs_to_plot["120W_gd_SUKS1_3_20"] =  "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/120W_gd_SUKS1_3_20/Ev/Lev3_A?/Run/"
# runs_to_plot["AccTest_q1ns_Lev9"] =  "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/AccTest_q1ns_Lev9/Ev/Lev9_A?/Run/"
# runs_to_plot["77_gd_Kerr_q3"] =  "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/77_gd_Kerr_q3/Ev_Kerr/Lev1_A?/Run/"
# runs_to_plot["77_gd_Kerr_q3"] =  "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/120W_gd_SUKS1_3_20/Ev/Lev3_A?/Run/"


# data_file_path = "ConstraintNorms/GhCe.dat"
# data_file_path = "ConstraintNorms/GhCeExt.dat"
# data_file_path = "ConstraintNorms/GhCeExt_L2.dat"
# data_file_path = "ConstraintNorms/GhCeExt_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_L2.dat"
data_file_path = "ConstraintNorms/GhCe_Linf.dat"
# data_file_path = "ConstraintNorms/Linf.dat"
# data_file_path = "ConstraintNorms/Constraints_Linf.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path = "ConstraintNorms/GhCe_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_VolL2.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Norms.dat"
# data_file_path = "CharSpeedNorms/CharSpeeds_Min_SliceLFF.SphereA0.dat"
# data_file_path = "MinimumGridSpacing.dat"
# data_file_path = "GrAdjustMaxTstepToDampingTimes.dat"
# data_file_path = "GrAdjustSubChunksToDampingTimes.dat"
# data_file_path = "DiagAhSpeedA.dat"
# data_file_path = "ApparentHorizons/AhA.dat"
# data_file_path = "ApparentHorizons/AhB.dat" 
# data_file_path = "ApparentHorizons/MinCharSpeedAhA.dat"
# data_file_path = "ApparentHorizons/RescaledRadAhA.dat"
# data_file_path = "ApparentHorizons/AhACoefs.dat"
# data_file_path = "ApparentHorizons/AhBCoefs.dat"
# data_file_path = "ApparentHorizons/Trajectory_AhB.dat"
# data_file_path = "ApparentHorizons/HorizonSepMeasures.dat"

# data_file_path = "ApparentHorizons/Horizons.h5@AhA"
# data_file_path = "TStepperDiag.dat"
# data_file_path = "TimeInfo.dat"
# data_file_path = "Hist-FuncSkewAngle.txt"
# data_file_path = "Hist-FuncCutX.txt"
# data_file_path = "Hist-FuncExpansionFactor.txt"
# data_file_path = "Hist-FuncLambdaFactorA0.txt"
# data_file_path = "Hist-FuncLambdaFactorA.txt"
# data_file_path = "Hist-FuncLambdaFactorB0.txt"
# data_file_path = "Hist-FuncLambdaFactorB.txt"
# data_file_path = "Hist-FuncQuatRotMatrix.txt"
# data_file_path = "Hist-FuncSkewAngle.txt"
# data_file_path = "Hist-FuncSmoothCoordSep.txt"
# data_file_path = "Hist-FuncSmoothMinDeltaRNoLam00AhA.txt"
# data_file_path = "Hist-FuncSmoothMinDeltaRNoLam00AhB.txt"
# data_file_path = "Hist-FuncSmoothRAhA.txt"
# data_file_path = "Hist-FuncSmoothRAhB.txt"
# data_file_path = "Hist-FuncTrans.txt"
# data_file_path = "Hist-GrDomain.txt"
# data_file_path = "Profiler.h5"
column_names, runs_data_dict = load_data_from_levs(runs_to_plot,data_file_path)
print(column_names)

# %%
runs_to_plot = {}

# runs_to_plot["high_accuracy_L3_tol8_wrong"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev3_tol8/Run/"
# runs_to_plot["high_accuracy_L3_rd"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev3_Ringdown/Lev3_A?/Run/"
# runs_to_plot["high_accuracy_L4_rd"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev4_Ringdown/Lev4_A?/Run/"
# runs_to_plot["high_accuracy_L5_rd"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev5_Ringdown/Lev5_A?/Run/"

# runs_to_plot["high_accuracy_L0"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev0_A?/Run/"
# runs_to_plot["high_accuracy_L1"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev1_A?/Run/"
# runs_to_plot["high_accuracy_L2"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev2_A?/Run/"
# runs_to_plot["high_accuracy_L3"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev3_A?/Run/"
# runs_to_plot["high_accuracy_L4"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev4_A?/Run/"
# runs_to_plot["high_accuracy_L5"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L45"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev45_A?/Run/"
# runs_to_plot["high_accuracy_L55"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev55_A?/Run/"
# runs_to_plot["high_accuracy_L6"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev6_A?/Run/"

# runs_to_plot["high_accuracy_L45n"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_seg_runs/new_L45_L55/Ev/Lev45_A?/Run/"
# runs_to_plot["high_accuracy_L55n"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_seg_runs/new_L45_L55/Ev/Lev55_A?/Run/"

# runs_to_plot["high_accuracy_L4n_no_tol_change"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_seg_runs/main_L4_to_L55/Ev/Lev4_A?/Run/"
# runs_to_plot["high_accuracy_L45n_no_tol_change"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_seg_runs/main_L4_to_L55/Ev/Lev45_A?/Run/"
# runs_to_plot["high_accuracy_L5n_no_tol_change"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_seg_runs/main_L4_to_L55/Ev/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L55n_no_tol_change"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_seg_runs/main_L4_to_L55/Ev/Lev55_A?/Run/"

# runs_to_plot["high_accuracy_L5_three_tier"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L5_three_tier_constra"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_constra/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L5_three_tier_constra200"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_constra_200/Lev5_A?/Run/"
# runs_to_plot["L3_step_bound_gauss_error"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_step_bound_gauss_error/Ev/Lev3_A?/Run/"
# runs_to_plot["L3_step_bound_gauss_error_rd"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_step_bound_gauss_error/Ev/Lev3_Ringdown/Lev3_A?/Run/"

# runs_to_plot["Lev5_big_gaussian_ah_tol10"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_ah_tol10/Lev5_A?/Run/"
# runs_to_plot["Lev5_big_gaussian_ah_tol100"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_ah_tol100/Lev5_A?/Run/"
# runs_to_plot["Lev5_bg_ah100_cd_01_uamr"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_cd_01_uamr_full/Lev5_A?/Run/"
# runs_to_plot["Lev5_bg_ah100_lapse"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_lapse_full/Lev5_A?/Run/"
# runs_to_plot["Lev5_bg_ah100_lapse_uamr"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_lapse_uamr_full/Lev5_A?/Run/"

# runs_to_plot["high_accuracy_L0_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev0_A?/Run/"
# runs_to_plot["high_accuracy_L1_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev1_A?/Run/"
# runs_to_plot["high_accuracy_L2_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev2_A?/Run/"
# runs_to_plot["high_accuracy_L3_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev3_A?/Run/"
# runs_to_plot["high_accuracy_L4_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev4_A?/Run/"
# runs_to_plot["high_accuracy_L5_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L45_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev45_A?/Run/"
# runs_to_plot["high_accuracy_L55_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev55_A?/Run/"
# runs_to_plot["high_accuracy_L6_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev6_A?/Run/"


# runs_to_plot["ode_impro_Lev0"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev0_A?/Run/'
# runs_to_plot["ode_impro_Lev1"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev1_A?/Run/'
# runs_to_plot["ode_impro_Lev2"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev2_A?/Run/'
# runs_to_plot["ode_impro_Lev3"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev3_A?/Run/'
# runs_to_plot["ode_impro_Lev4"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev4_A?/Run/'
# runs_to_plot["ode_impro_Lev5"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev5_A?/Run/'
# runs_to_plot["main_Lev0"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev0_A?/Run/'
# runs_to_plot["main_Lev2"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev2_A?/Run/'
# runs_to_plot["main_Lev1"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev1_A?/Run/'

# runs_to_plot["ode_impro_Lev0_rd"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev0_Ringdown/Lev0_A?/Run/'
# runs_to_plot["ode_impro_Lev2_rd"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev2_Ringdown/Lev2_A?/Run/'
# runs_to_plot["ode_impro_Lev1_rd"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev1_Ringdown/Lev1_A?/Run/'
# runs_to_plot["main_Lev0_rd"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev0_Ringdown/Lev0_A?/Run/'
# runs_to_plot["main_Lev2_rd"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev2_Ringdown/Lev2_A?/Run/'
# runs_to_plot["main_Lev1_rd"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev1_Ringdown/Lev1_A?/Run/'

# runs_to_plot["6_set1_L3s0"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/Ev/Lev0_A?/Run/"
# runs_to_plot["6_set1_L3s1"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/Ev/Lev1_A?/Run/"
# runs_to_plot["6_set1_L3s2"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/Ev/Lev2_A?/Run/"
# runs_to_plot["6_set1_L3s3"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["6_set1_L3s3_fil6"] =  "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_5517_6/Ev/Lev3_A[B-]/Run/"
# runs_to_plot["6_set1_L3s3_fil8"] =  "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_5517_8/Ev/Lev3_A[B-]/Run/"
# runs_to_plot["6_set1_L3s3_fil10"] =  "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_5517_10/Ev/Lev3_A[B-]/Run/"

# runs_to_plot["6_set1_L3_template_all"] =  "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_template_all/Ev/Lev3_A[B-]/Run/"
# runs_to_plot["6_set1_L3_template_1_29"] =  "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_template_1_29/Ev/Lev3_A[B-]/Run/"

# runs_to_plot["6_set1_L3s3_5517_CCopy"] =  "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_5517_CCopy/Ev/Lev3_A?/Run/"

# runs_to_plot["6_set2_L3s2"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L3/Ev/Lev2_A?/Run/"
# runs_to_plot["6_set2_L3s3"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["6_set3_L3s0"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L3/Ev/Lev0_A?/Run/"
# runs_to_plot["6_set3_L3s1"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L3/Ev/Lev1_A?/Run/"
# runs_to_plot["6_set3_L3s2"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L3/Ev/Lev2_A?/Run/"
# runs_to_plot["6_set3_L3s3"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["6_set1_L6s0"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev0_A?/Run/"
# runs_to_plot["6_set1_L6s1"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev1_A?/Run/"
# runs_to_plot["6_set1_L6s2"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev2_A?/Run/"
# runs_to_plot["6_set1_L6s3"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev3_A?/Run/"
# runs_to_plot["6_set1_L6s4"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev4_A?/Run/"
# runs_to_plot["6_set1_L6s5"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev5_A?/Run/"
# runs_to_plot["6_set1_L6s6"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/Ev/Lev6_A?/Run/"

# runs_to_plot["set1_L6s4_cd10"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/set1_L6s4_cd10/Ev/Lev4_A?/Run/"
# runs_to_plot["set1_L6s4_cd100"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/set1_L6s4_cd100/Ev/Lev4_A?/Run/"
# runs_to_plot["set1_L6s4_cd200"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/set1_L6s4_cd200/Ev/Lev4_A?/Run/"
# runs_to_plot["set1_L6s4_cd500"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/set1_L6s4_cd500/Ev/Lev4_A?/Run/"
# runs_to_plot["set1_L6s4_cd100_AMRL6"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/set1_L6s4_cd100_AMRL6/Ev/Lev4_A?/Run/"
# runs_to_plot["set1_L6s4_cd100_AMRL7"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/set1_L6s4_cd100_AMRL7/Ev/Lev4_A?/Run/"
# runs_to_plot["set1_L6s4_cd100_AMRL8"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/set1_L6s4_cd100_AMRL8/Ev/Lev4_A?/Run/"

# runs_to_plot["6_set2_L6s4"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L6/Ev/Lev4_A?/Run/"
# runs_to_plot["6_set2_L6s5"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L6/Ev/Lev5_A?/Run/"
# runs_to_plot["6_set2_L6s6"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L6/Ev/Lev6_A?/Run/"

# runs_to_plot["6_set3_L6s4"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L6/Ev/Lev4_A?/Run/"
# runs_to_plot["6_set3_L6s5"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L6/Ev/Lev5_A?/Run/"
# runs_to_plot["6_set3_L6s6"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L6/Ev/Lev6_A?/Run/"

# runs_to_plot["6_set1_L6s3_CAMR"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6_vars/L6s3_CAMR/Ev/Lev3_A?/Run/"
# runs_to_plot["6_set1_L6s3_min_L"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6_vars/L6s3_min_L/Ev/Lev3_A?/Run/"
# runs_to_plot["6_set1_L6s3_min_LR"] =  "/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6_vars/L6s3_min_LR/Ev/Lev3_A?/Run/"

# runs_to_plot["7_constAMR_set1_L6_base_0"] =  "/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/Ev/Lev0_A?/Run/"
# runs_to_plot["7_constAMR_set1_L6_base_1"] =  "/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/Ev/Lev1_A?/Run/"
# runs_to_plot["7_constAMR_set1_L6_base_2"] =  "/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/Ev/Lev2_A?/Run/"
# runs_to_plot["7_constAMR_set1_L6_base_3"] =  "/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/Ev/Lev3_A?/Run/"
# runs_to_plot["7_constAMR_set1_L6_base_4"] =  "/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/Ev/Lev4_A?/Run/"
# runs_to_plot["7_constAMR_set1_L6_base_5"] =  "/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/Ev/Lev5_A?/Run/"
# runs_to_plot["7_constAMR_set1_L6_base_6"] =  "/groups/sxs/hchaudha/spec_runs/7_constAMR_set1_L6_base/Ev/Lev6_A?/Run/"

# runs_to_plot["8_constAMR_set1_L6_base_0"] =  "/groups/sxs/hchaudha/spec_runs/8_constAMR_set1_L6_base/Ev/Lev0_A?/Run/"
# runs_to_plot["8_constAMR_set1_L6_base_1"] =  "/groups/sxs/hchaudha/spec_runs/8_constAMR_set1_L6_base/Ev/Lev1_A?/Run/"
# runs_to_plot["8_constAMR_set1_L6_base_2"] =  "/groups/sxs/hchaudha/spec_runs/8_constAMR_set1_L6_base/Ev/Lev2_A?/Run/"
# runs_to_plot["8_constAMR_set1_L6_base_3"] =  "/groups/sxs/hchaudha/spec_runs/8_constAMR_set1_L6_base/Ev/Lev3_A?/Run/"

# runs_to_plot["9_set1_L3s3_01"] =  "/groups/sxs/hchaudha/spec_runs/9_const_damp_var/set1_L3s3_01/Ev/Lev3_A?/Run/"
# runs_to_plot["9_set1_L3s3_001"] =  "/groups/sxs/hchaudha/spec_runs/9_const_damp_var/set1_L3s3_001/Ev/Lev3_A?/Run/"
# runs_to_plot["9_set1_L3s3_10"] =  "/groups/sxs/hchaudha/spec_runs/9_const_damp_var/set1_L3s3_10/Ev/Lev3_A?/Run/"
# runs_to_plot["9_set1_L3s3_100"] =  "/groups/sxs/hchaudha/spec_runs/9_const_damp_var/set1_L3s3_100/Ev/Lev3_A?/Run/"

# runs_to_plot["10_4000M_CAMR_set1_L6_base0"] =  "/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/Ev/Lev0_A?/Run/"
# runs_to_plot["10_4000M_CAMR_set1_L6_base1"] =  "/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/Ev/Lev1_A?/Run/"
# runs_to_plot["10_4000M_CAMR_set1_L6_base2"] =  "/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/Ev/Lev2_A?/Run/"
# runs_to_plot["10_4000M_CAMR_set1_L6_base3"] =  "/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/Ev/Lev3_A?/Run/"
# runs_to_plot["10_4000M_CAMR_set1_L6_base4"] =  "/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/Ev/Lev4_A?/Run/"
# runs_to_plot["10_4000M_CAMR_set1_L6_base5"] =  "/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/Ev/Lev5_A?/Run/"
# runs_to_plot["10_4000M_CAMR_set1_L6_base6"] =  "/groups/sxs/hchaudha/spec_runs/10_4000M_CAMR_set1_L6_base/Ev/Lev6_A?/Run/"

# runs_to_plot["11_4000M_CAMR_set1_L6_base0"] =  "/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/Ev/Lev0_A?/Run/"
# runs_to_plot["11_4000M_CAMR_set1_L6_base1"] =  "/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/Ev/Lev1_A?/Run/"
# runs_to_plot["11_4000M_CAMR_set1_L6_base2"] =  "/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/Ev/Lev2_A?/Run/"
# runs_to_plot["11_4000M_CAMR_set1_L6_base3"] =  "/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/Ev/Lev3_A?/Run/"
# runs_to_plot["11_4000M_CAMR_set1_L6_base4"] =  "/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/Ev/Lev4_A?/Run/"
# runs_to_plot["11_4000M_CAMR_set1_L6_base5"] =  "/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/Ev/Lev5_A?/Run/"
# runs_to_plot["11_4000M_CAMR_set1_L6_base6"] =  "/groups/sxs/hchaudha/spec_runs/11_4000M_CAMR_set1_L6_maxExt/Ev/Lev6_A?/Run/"

# runs_to_plot["12_set1_L3_1500"] =  "/groups/sxs/hchaudha/spec_runs/12_set1_L3_1500/Ev/Lev3_A?/Run/"
# runs_to_plot["12_set1_L3_2000"] =  "/groups/sxs/hchaudha/spec_runs/12_set1_L3_2000/Ev/Lev3_A?/Run/"
# runs_to_plot["12_set1_L3_2500"] =  "/groups/sxs/hchaudha/spec_runs/12_set1_L3_2500/Ev/Lev3_A?/Run/"

# runs_to_plot["13_set1_L3_3000"] =  "/groups/sxs/hchaudha/spec_runs/13_set1_L3_3000/Ev/Lev3_A?/Run/"
# runs_to_plot["13_set1_L4_1500"] =  "/groups/sxs/hchaudha/spec_runs/13_set1_L4_1500/Ev/Lev4_A?/Run/"
# runs_to_plot["13_set1_L4_3000"] =  "/groups/sxs/hchaudha/spec_runs/13_set1_L4_3000/Ev/Lev4_A?/Run/"

# runs_to_plot["14_set1_L4_1500_cd5"] =  "/groups/sxs/hchaudha/spec_runs/14_set1_L4_1500_cd5/Ev/Lev4_A?/Run/"
# runs_to_plot["14_set1_L4_1500_cd10"] =  "/groups/sxs/hchaudha/spec_runs/14_set1_L4_1500_cd10/Ev/Lev4_A?/Run/"
# runs_to_plot["14_set1_L4_1500_cd25"] =  "/groups/sxs/hchaudha/spec_runs/14_set1_L4_1500_cd25/Ev/Lev4_A?/Run/"
# runs_to_plot["14_set1_L4_1500_cd50"] =  "/groups/sxs/hchaudha/spec_runs/14_set1_L4_1500_cd50/Ev/Lev4_A?/Run/"

# runs_to_plot["15_set1_L4_1500_JY"] =  "/groups/sxs/hchaudha/spec_runs/15_set1_L4_1500_JY/Ev/Lev4_A?/Run/"

# runs_to_plot["16_set1_L3"] = "/groups/sxs/hchaudha/spec_runs/16_set1_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["16_set1_L3_HP32"] = "/groups/sxs/hchaudha/spec_runs/16_set1_L3_HP32/Ev/Lev3_A?/Run/"
# runs_to_plot["16_set1_L3_HP28"] = "/groups/sxs/hchaudha/spec_runs/16_set1_L3_HP28/Ev/Lev3_A?/Run/"
# runs_to_plot["16_set1_L3_HP32_AF"] = "/groups/sxs/hchaudha/spec_runs/16_set1_L3_HP32_AF/Ev/Lev3_A?/Run/"
# runs_to_plot["17_BDI_32_SAE_NONE"] = "/groups/sxs/hchaudha/spec_runs/17_BDI_32_SAE_NONE/Ev/Lev4_A?/Run/"
# runs_to_plot["17_BDI_32_SAE_32"] = "/groups/sxs/hchaudha/spec_runs/17_BDI_32_SAE_32/Ev/Lev4_A?/Run/"
# runs_to_plot["17_BDI_32_SAE_32_AF"] = "/groups/sxs/hchaudha/spec_runs/17_BDI_32_SAE_32_AF/Ev/Lev4_A?/Run/"

# runs_to_plot["17_set_main_q3_15_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set_main_q3_15_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set_main_99_15_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set_main_99_15_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["17_set_main_q3_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set_main_q3_18_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set1_q3_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set1_q3_18_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set3_q3_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set3_q3_18_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["17_set_main_99_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set_main_99_18_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set1_99_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set1_99_18_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set3_99_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set3_99_18_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["17_main_9_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_main_9_18_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set1_9_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set1_9_18_L3/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set3_9_18_L3"] = "/groups/sxs/hchaudha/spec_runs/17_set3_9_18_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["17_main_9_18_L3_correct"] = "/groups/sxs/hchaudha/spec_runs/17_main_9_18_L3_correct/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set1_9_18_L3_correct"] = "/groups/sxs/hchaudha/spec_runs/17_set1_9_18_L3_correct/Ev/Lev3_A?/Run/"
# runs_to_plot["17_set3_9_18_L3_correct"] = "/groups/sxs/hchaudha/spec_runs/17_set3_9_18_L3_correct/Ev/Lev3_A?/Run/"

# runs_to_plot["18_set1_L3_junk_resolved"] = "/groups/sxs/hchaudha/spec_runs/18_set1_L3_junk_resolved/Ev/Lev3_??/Run/"
# runs_to_plot["20_set1_L3_fine_cylinders"] = "/groups/sxs/hchaudha/spec_runs/20_set1_L3_fine_cylinders/Ev/Lev3_??/Run/"
# runs_to_plot["21_set1_L3_fine_cylinders_minExtent"] = "/groups/sxs/hchaudha/spec_runs/21_set1_L3_fine_cylinders_minExtent/Ev/Lev3_??/Run/"

# runs_to_plot["22_set1_L1_long"] = "/groups/sxs/hchaudha/spec_runs/22_set1_L1_long/Ev/Lev1_??/Run/"
# runs_to_plot["L1_AC_L3"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L1_AC_L3/Ev/Lev3_??/Run/"
# runs_to_plot["L1_AC_L2"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L1_AC_L2/Ev/Lev2_??/Run/"
# runs_to_plot["L1_AC_L1"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L1_AC_L1/Ev/Lev1_??/Run/"

# runs_to_plot["L3_AC_L3_cd_const_high"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_cd_const_high/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_cd_const_low"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_cd_const_low/Ev/Lev3_??/Run/"

# runs_to_plot["22_L3_AC_L3_no_res_C"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_no_res_C/Ev/Lev3_??/Run/"
# runs_to_plot["22_L3_AC_L3_res_10_C"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_res_10_C/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L1"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L1/Ev/Lev1_??/Run/"
# runs_to_plot["L3_AC_L2"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L2/Ev/Lev2_??/Run/"
# runs_to_plot["22_set1_L3_long"] = "/groups/sxs/hchaudha/spec_runs/22_set1_L3_long/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L4"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L4/Ev/Lev4_??/Run/"
# runs_to_plot["L3_AC_L5"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L5/Ev/Lev5_??/Run/"
# runs_to_plot["L3_AC_L6"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L6/Ev/Lev6_??/Run/"
# runs_to_plot["L3_AC_L7"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L7/Ev/Lev7_??/Run/"
# runs_to_plot["L3_AC_L8"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L8/Ev/Lev8_??/Run/"

# runs_to_plot["L3_AC_L3_3_01"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_3_01/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_3_02"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_3_02/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_5_04"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_5_04/Ev/Lev3_??/Run/"

# runs_to_plot["26_set1_L6_long"] = "/groups/sxs/hchaudha/spec_runs/26_set1_L6_long/Ev/Lev6_??/Run/"
# runs_to_plot["26_main_L6_long"] = "/groups/sxs/hchaudha/spec_runs/26_main_L6_long/Ev/Lev6_??/Run/"

# runs_to_plot["28_set1_cd_junk_5"] = "/groups/sxs/hchaudha/spec_runs/28_set1_cd_junk_5/Ev/Lev3_??/Run/"
# runs_to_plot["28_set1_cd_junk_1"] = "/groups/sxs/hchaudha/spec_runs/28_set1_cd_junk_1/Ev/Lev3_??/Run/"
# runs_to_plot["28_set1_cd_junk_01"] = "/groups/sxs/hchaudha/spec_runs/28_set1_cd_junk_01/Ev/Lev3_??/Run/"
# runs_to_plot["28_set1_cd_junk_001"] = "/groups/sxs/hchaudha/spec_runs/28_set1_cd_junk_001/Ev/Lev3_??/Run/"

# runs_to_plot["29_set1_L3_ID_diff_12"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_12/Ev/Lev3_??/Run/"
# runs_to_plot["29_set1_L3_ID_diff_8"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_8/Ev/Lev3_??/Run/"
# runs_to_plot["29_set1_L3_ID_diff_4"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_4/Ev/Lev3_??/Run/"

# runs_to_plot["29_set1_L3_ID_diff_4"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_4/Ev/Lev3_A?/Run/"
# runs_to_plot["29_set1_L3_ID_diff_5"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_5/Ev/Lev3_A?/Run/"
# runs_to_plot["29_set1_L3_ID_diff_6"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_6/Ev/Lev3_A?/Run/"
# runs_to_plot["29_set1_L3_ID_diff_7"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_7/Ev/Lev3_A?/Run/"
# runs_to_plot["29_set1_L3_ID_diff_8"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_8/Ev/Lev3_A?/Run/"
# runs_to_plot["29_set1_L3_ID_diff_12"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_12/Ev/Lev3_A?/Run/"
# runs_to_plot["29_set1_L3_ID_diff_0"] = "/resnick/groups/sxs/hchaudha/spec_runs/29_set1_L3_ID_diff_0/Ev/Lev3_A?/Run/"

# runs_to_plot["RM_1_Lev3"] = "/resnick/groups/sxs/hchaudha/spec_runs/RM_1_Lev3/Ev/Lev3_A?/Run/"
# runs_to_plot["RM_0_Lev6"] = "/resnick/groups/sxs/hchaudha/spec_runs/RM_0_test/Ev_456/Lev6_A?/Run/"

# runs_to_plot["30_RM_set1_L1"] = "/resnick/groups/sxs/hchaudha/spec_runs/30_RM_set1_L1/Ev/Lev1_A?/Run/"
# runs_to_plot["30_RM_set1_L3"] = "/resnick/groups/sxs/hchaudha/spec_runs/30_RM_set1_L3/Ev/Lev3_A?/Run/"

# runs_to_plot["RM_L3s3_k0"] = "/resnick/groups/sxs/hchaudha/spec_runs/30_segs_res/L3s3_k0/Ev/Lev3_A?/Run/"
# runs_to_plot["RM_L3s3_k0_cd10"] = "/resnick/groups/sxs/hchaudha/spec_runs/30_segs_res/L3s3_k0_cd10/Ev/Lev3_A?/Run/"
# runs_to_plot["RM_L3s3_k0_cd100"] = "/resnick/groups/sxs/hchaudha/spec_runs/30_segs_res/L3s3_k0_cd100/Ev/Lev3_A?/Run/"
# runs_to_plot["RM_L3s4_k0"] = "/resnick/groups/sxs/hchaudha/spec_runs/30_segs_res/L3s4_k0/Ev/Lev4_A?/Run/"
# runs_to_plot["RM_L3s5_k0"] = "/resnick/groups/sxs/hchaudha/spec_runs/30_segs_res/L3s5_k0/Ev/Lev5_A?/Run/"
# runs_to_plot["RM_0_Lev6"] = "/resnick/groups/sxs/hchaudha/spec_runs/RM_0_test/Ev_456/Lev6_A?/Run/"
# runs_to_plot["RM_1_Lev3"] = "/resnick/groups/sxs/hchaudha/spec_runs/RM_1_Lev3/Ev/Lev3_A?/Run/"
# runs_to_plot["RM_1_Lev6"] = "/resnick/groups/sxs/hchaudha/spec_runs/RM_1_Lev6/Ev/Lev6_A?/Run/"


# runs_to_plot["main_L6_AM_ode_MQOS"] = "/resnick/groups/sxs/hchaudha/spec_runs/26_segs_res/main_L6_AM_ode_MQOS/Ev/Lev6_A?/Run/"
# runs_to_plot["set1_L6_AK_ode_MQOS"] = "/resnick/groups/sxs/hchaudha/spec_runs/26_segs_res/set1_L6_AK_ode_MQOS/Ev/Lev6_A?/Run/"

# runs_to_plot["set1_L6_AG_cd_0100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AG_cd_0100/Ev/Lev6_A[G-Z]/Run/"
# runs_to_plot["set1_L6_AG_cd_100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AG_cd_100/Ev/Lev6_A[G-Z]/Run/"
# runs_to_plot["set1_L6_AK_cd_0100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_cd_0100/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_cd_100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_cd_100/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20_cd_10"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20_cd_10/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20_cd_100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20_cd_100/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20_cd_1000"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20_cd_1000/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20_cd_10000"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20_cd_10000/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20_no_Rmin"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20_no_Rmin/Ev/Lev6_A[K-Z]/Run/"

# runs_to_plot["set1_L6_AG_cd_0100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AG_cd_0100/Ev/Lev6_A[G-Z]/Run/"
# runs_to_plot["set1_L6_AG_cd_100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AG_cd_100/Ev/Lev6_A[G-Z]/Run/"
# runs_to_plot["set1_L6_AK_cd_0100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_cd_0100/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_cd_100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_cd_100/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20/Ev/Lev6_A[K-Z]/Run/"
# runs_to_plot["set1_L6_AK_S2_L20_no_Rmin"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20_no_Rmin/Ev/Lev6_A[K-Z]/Run/"

# runs_to_plot["set1_L6_AG_cd_0100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AG_cd_0100/Ev/Lev6_??/Run/"
# runs_to_plot["set1_L6_AG_cd_100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AG_cd_100/Ev/Lev6_??/Run/"
# runs_to_plot["set1_L6_AK_cd_0100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_cd_0100/Ev/Lev6_??/Run/"
# runs_to_plot["set1_L6_AK_cd_100"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_cd_100/Ev/Lev6_??/Run/"
# runs_to_plot["set1_L6_AK_S2_L20"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20/Ev/Lev6_??/Run/"
# runs_to_plot["set1_L6_AK_S2_L20_no_Rmin"] = "/groups/sxs/hchaudha/spec_runs/26_segs/set1_L6_AK_S2_L20_no_Rmin/Ev/Lev6_??/Run/"

# runs_to_plot["L3_AC_L3_minL17"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_minL17/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_minL19"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_minL19/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_minL21"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_minL21/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_01_cd_asymp"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_01_cd_asymp/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_single_exp"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_single_exp/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_single_Exp_large_sigma"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_single_Exp_large_sigma/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_cd_const_low"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_cd_const_low/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_cd_const_high"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_cd_const_high/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_sigma2"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_sigma2/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_sigma05"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_sigma05/Ev/Lev3_??/Run/"
# runs_to_plot["L3_AC_L3_sigma1_const"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_sigma1_const/Ev/Lev3_??/Run/"

# runs_to_plot["L3_AC_L3_AB_L8"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_AB_L8/Ev/Lev3_AC/Run/"
# runs_to_plot["L3_AC_L3_AB_R7"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_AB_R7/Ev/Lev3_AC/Run/"
# runs_to_plot["L3_AC_L3_AB0_L16"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_AB0_L16/Ev/Lev3_AC/Run/"
# runs_to_plot["L3_AC_L3_AB0_L15"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_AB0_L15/Ev/Lev3_AC/Run/"
# runs_to_plot["L3_AC_L3_ps_10"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_ps_10/Ev/Lev3_AC/Run/"
# runs_to_plot["L3_AC_L3_ps_01"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_ps_01/Ev/Lev3_AC/Run/"
# runs_to_plot["L3_AC_L3_BCSC_8"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_BCSC_8/Ev/Lev3_AC/Run/"
# runs_to_plot["L3_AC_L3_BCSC_12"] = "/groups/sxs/hchaudha/spec_runs/22_segs/L3_AC_L3_BCSC_12/Ev/Lev3_AC/Run/"

# runs_to_plot["119_gd_SUKS_3_20"] = "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/119_gd_SUKS_3_20/Ev/Lev3_A?/Run/"
# runs_to_plot["120W_gd_SUKS1_3_20"] = "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/120W_gd_SUKS1_3_20/Ev/Lev3_A?/Run/"
# runs_to_plot["120_gd_SUKS1_3_20"] = "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/120_gd_SUKS1_3_20/Ev/Lev3_A?/Run/"
# runs_to_plot["119_gd_SUKS_3_20"] = "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/119_gd_SUKS_3_20/Ev/Lev3_A?/Run/"
# runs_to_plot["67_master_mr3"] = "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/67_master_mr3/Ev/Lev1_A?/Run/"
# runs_to_plot["119_gd_SUKS_3_20"] = "/net/panfs/SXS/himanshu/gauge_stuff/gauge_driver_runs/runs/119_gd_SUKS_3_20/Ev/Lev3_A?/Run/"

# data_file_path = "ConstraintNorms/GhCe.dat"
# data_file_path = "ConstraintNorms/GhCeExt.dat"
# data_file_path = "ConstraintNorms/GhCeExt_L2.dat"
# data_file_path = "ConstraintNorms/GhCeExt_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_L2.dat"
data_file_path = "ConstraintNorms/GhCe_Linf.dat"
# data_file_path = "ConstraintNorms/Linf.dat"
# data_file_path = "ConstraintNorms/Constraints_Linf.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path = "ConstraintNorms/GhCe_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_VolL2.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Norms.dat"
# data_file_path = "CharSpeedNorms/CharSpeeds_Min_SliceLFF.SphereA0.dat"
# data_file_path = "MinimumGridSpacing.dat"
# data_file_path = "GrAdjustMaxTstepToDampingTimes.dat"
# data_file_path = "GrAdjustSubChunksToDampingTimes.dat"
# data_file_path = "DiagAhSpeedA.dat"
# data_file_path = "ApparentHorizons/AhA.dat"
# data_file_path = "ApparentHorizons/AhB.dat" 
# data_file_path = "ApparentHorizons/MinCharSpeedAhA.dat"
# data_file_path = "ApparentHorizons/RescaledRadAhA.dat"
# data_file_path = "ApparentHorizons/AhACoefs.dat"
# data_file_path = "ApparentHorizons/AhBCoefs.dat"
# data_file_path = "ApparentHorizons/Trajectory_AhB.dat"
# data_file_path = "ApparentHorizons/HorizonSepMeasures.dat"

# data_file_path = "ApparentHorizons/Horizons.h5@AhA"
# data_file_path = "ApparentHorizons/Horizons.h5@AhB"
# data_file_path = "TStepperDiag.dat"
# data_file_path = "TimeInfo.dat"
# data_file_path = "Hist-FuncSkewAngle.txt"
# data_file_path = "Hist-FuncCutX.txt"
# data_file_path = "Hist-FuncExpansionFactor.txt"
# data_file_path = "Hist-FuncLambdaFactorA0.txt"
# data_file_path = "Hist-FuncLambdaFactorA.txt"
# data_file_path = "Hist-FuncLambdaFactorB0.txt"
# data_file_path = "Hist-FuncLambdaFactorB.txt"
# data_file_path = "Hist-FuncQuatRotMatrix.txt"
# data_file_path = "Hist-FuncSkewAngle.txt"
# data_file_path = "Hist-FuncSmoothCoordSep.txt"
# data_file_path = "Hist-FuncSmoothMinDeltaRNoLam00AhA.txt"
# data_file_path = "Hist-FuncSmoothMinDeltaRNoLam00AhB.txt"
# data_file_path = "Hist-FuncSmoothRAhA.txt"
# data_file_path = "Hist-FuncSmoothRAhB.txt"
# data_file_path = "Hist-FuncTrans.txt"
# data_file_path = "Hist-GrDomain.txt"
# data_file_path = "Profiler.h5"
column_names, runs_data_dict = load_data_from_levs(runs_to_plot,data_file_path)
print(column_names)

# %%
moving_avg_len=0
save_path = None
diff_base = None
constant_shift_val_time = None
plot_abs_diff = False
y_axis_list = None
x_axis = 't(M)'

plot_abs_diff = True
# diff_base = '29_set1_L3_ID_diff_0'
# diff_base = '6_set1_L3s3'
# diff_base = '22_set1_L1_long'
# diff_base = 'L1_AC_L3'
# diff_base = 'high_accuracy_L5'
# diff_base = 'high_accuracy_L5_main'
# diff_base = 'Lev5_big_gaussian_ah_tol100'
# diff_base = '6_set1_L6s6'
# diff_base = '6_set1_L3s3'
# add_max_and_min_val(runs_data_dict)
# y_axis = 'max_val'
# y_axis = 'min_val'

# constant_shift_val_time = 7206
# constant_shift_val_time = 1200

y_axis = 'Linf(GhCe)'
# y_axis = 'L2(GhCe)'
# y_axis = 'VolLp(GhCe)'
# y_axis = 'VolLp(GhCeExt)'
# y_axis = 'Linf(GhCeExt)'
# y_axis = 'L2(NormalizedGhCe)'
# y_axis = 'Linf(NormalizedGhCe)'
# y_axis = 'VolLp(NormalizedGhCe)'
# y_axis = 'CharSpeed'
# y_axis = 'Linf(GhCeExt)'
# y_axis = 'Linf(NormalizedGhCe) on SphereA5'
# y_axis = 'Linf(NormalizedGhCe) on SphereB0'
# y_axis = 'Linf(NormalizedGhCe) on SphereA3'
# y_axis = 'Linf(NormalizedGhCe) on SphereC0'
# y_axis = 'Linf(NormalizedGhCe) on SphereC40'
# y_axis = 'Linf(NormalizedGhCe) on CylinderCB1.0.0'
# y_axis = 'Linf(NormalizedGhCe) on CylinderEB1.0.0'
# y_axis = 'Linf(NormalizedGhCe) on FilledCylinderCA0'
# y_axis = 'Linf(NormalizedGhCe) on FilledCylinderCA1'
# y_axis = 'Linf(GhCe) on SphereA6'
# y_axis = 'Linf(GhCe) on SphereA0'
# y_axis = 'Linf(GhCe) on SphereC0'
# y_axis = 'Linf(GhCe) on SphereC29' 
# y_axis = 'Linf(GhCe) on CylinderCA0.0.0'
# y_axis = 'Linf(GhCe) on CylinderSMB1.0'
# y_axis = 'Linf(GhCe) on FilledCylinderMB0'
# y_axis = 'Linf(GhCe) on FilledCylinderEA0'
# y_axis = 'Linf(GhCe) on FilledCylinderCA0'
# y_axis = 'Linf(GhCe) on FilledCylinderCA1'
# y_axis = 'Linf(1Conz) on SphereC0'
# y_axis = 'Linf(3Conyxx) on SphereC12'
# y_axis = 'Linf(2Conxx) on SphereC12'
# y_axis = 'MinimumGridSpacing[CylinderCB1.0.0]'
# y_axis = 'MinimumGridSpacing[SphereA0]'
# y_axis = 'MinimumGridSpacing[SphereC0]'
# y_axis = 'MinimumGridSpacing[SphereB0]'

# y_axis = 'Linf(sqrt(psiErr^2)) on SphereC8'
# y_axis = 'Linf(sqrt(psiErr^2)) on SphereC15'
# y_axis = 'Linf(sqrt(kappaErr^2)) on SphereC0'
# y_axis = 'Linf(sqrt(kappaErr^2)) on SphereC6'
# y_axis = 'Linf(GhCe) on SphereC15'
# y_axis = 'Linf(GhCe) on SphereD0'
# y_axis = 'Linf(NormalizedGhCe) on SphereE5'
# y_axis = 'Linf(GhCe) on SphereC8'
# y_axis = 'Linf(GhCe) on SphereE5'
# y_axis = 'Linf(sqrt(3Con^2)) on SphereD1'
# y_axis = 'dt'


# y_axis = 'Linf(3Conzzz) on CylinderSMB0.0'

# y_axis = 'MPI::MPwait_cum'
# x_axis = 't'
# y_axis = 'T [hours]'
# y_axis = 'dt/dT'

# x_axis = 't'
# y_axis = 'CoordSepHorizons'
# y_axis = 'ProperSepHorizons'
# y_axis = 'MinCharSpeedAhA[9]'
# y_axis = 'InertialCenter_x'
# y_axis = 'InertialCenter_z'
# y_axis = 'SmoothCoordSep'
# y_axis = 'SphereC5_L'
# y_axis = 'SphereA0_L'
# y_axis = 'SphereC29_R'
# y_axis = 'FilledCylinderMB0_R'
# y_axis = 'FilledCylinderMB0_M'
# y_axis = 'FilledCylinderMB0_L'
# y_axis = 'MinimumGridSpacing[SphereA0]'
# y_axis = 'MinimumGridSpacing[CylinderSMB0.0]'
# y_axis = 'dt/dT'

# x_axis = 'time'
# y_axis = 'NumIterations'
# y_axis = 'Residual'
# y_axis = 'ArealMass'
# y_axis = 'ChristodoulouMass'
# y_axis = 'CoordCenterInertial_1'
# y_axis = 'CoordSpinChiInertial_2'
# y_axis = 'CoordSpinChiMagInertial'
# y_axis = 'DimensionfulInertialCoordSpin_0'
# y_axis = 'DimensionfulInertialCoordSpinMag'
# y_axis = 'DimensionfulInertialSpin_0'
# y_axis = 'DimensionfulInertialSpinMag'
# y_axis = 'SpinFromShape_2'
# y_axis = 'CoordSpinChiMagInertial'
# y_axis = 'max(r)'
# y_axis = 'min(r)'


# y_axis = 'courant factor'
# y_axis = 'error/1e-08'
# y_axis = 'NfailedSteps'
# y_axis = 'NumRhsEvaluations in this segment'
# y_axis = 'dt'
# y_axis = 'FilledCylinderMA1_L'
# y_axis = 'SphereC6_L'
# y_axis = 'SphereA0_R'
# y_axis = 'SphereA0_L'

minT = 0
# minT = 85
# minT = 470
# minT = 1200
# minT = 1400
# minT = 3000
# minT = 3200
# minT = 4000
# minT = 6500
# minT = 7200
# minT = 9260
# minT = 5266
# minT = 9700

maxT = 40000
# maxT = 1700
# maxT = minT+30
# maxT = minT+10
# maxT = minT+0.5
# maxT = 100*3
# maxT = 700
# maxT = 1200
# maxT = 2000
maxT = 4000
# maxT = 8400
# maxT = 9300
# maxT = 10000
# moving_avg_len = 50
# moving_avg_len = 10


# y_axis_list = ["SphereC0_L","SphereC0_R"]
# y_axis_list = [f"SphereC{i}_R" for i in range(30)]
# y_axis_list = ["SphereC0_L","SphereC1_L","SphereC2_L","SphereC4_L","SphereC8_L","SphereC16_L","SphereC29_L"]
# y_axis_list = ["SphereC4_L","SphereC16_L","SphereC29_L"]
# y_axis_list = [f'Linf(GhCe) on SphereA{i}' for i in [0]]
# y_axis_list = ["SphereC0_R",'CylinderSMA0.0_R','FilledCylinderMA0_R','SphereA0_R']
# y_axis_list = ['SphereA0_L','SphereA1_L','SphereA2_L','SphereA3_L','SphereA4_L']
# y_axis_list = ['Linf(GhCe) on CylinderSMA0.0','Linf(GhCe) on FilledCylinderMA0','Linf(GhCe) on SphereA0']
# y_axis_list = [
#   'Linf(NormalizedGhCe) on SphereA0',
  # 'Linf(NormalizedGhCe) on SphereA1',
  # 'Linf(NormalizedGhCe) on SphereA2',
  # 'Linf(NormalizedGhCe) on SphereA3',
  # 'Linf(NormalizedGhCe) on SphereA4',
  # 'Linf(NormalizedGhCe) on CylinderSMA0.0',29_set1_L3_ID_diff_0
  # 'Linf(NormalizedGhCe) on FilledCylinderMA0',
  # 'Linf(NormalizedGhCe) on SphereC0',
  # 'Linf(NormalizedGhCe) on SphereC1',
  # 'Linf(NormalizedGhCe) on SphereC2',
  # 'Linf(NormalizedGhCe) on SphereC4',
  # 'Linf(NormalizedGhCe) on SphereC8',
  # 'Linf(NormalizedGhCe) on SphereC12',
  # 'Linf(NormalizedGhCe) on SphereC16',
  # 'Linf(NormalizedGhCe) on SphereC20',
  # 'Linf(NormalizedGhCe) on SphereC24',
  # 'Linf(NormalizedGhCe) on SphereC28',
  # ]
# y_axis_list = [f'Linf(1Con{v}) on SphereC0' for v in ['t','x','y','z']]
# y_axis_list = [f'Linf(sqrt(kappaErr^2)) on SphereC{i}' for i in range(0,12)]
# y_axis_list = [f'Linf(NormalizedGhCe) on SphereC{i}' for i in range(5,45,10)]
# y_axis_list = [f'Linf(NormalizedGhCe) on SphereA{i}' for i in range(0,6)]
# y_axis_list = [f'Linf(GhCe) on SphereC{i}' for i in range(55)]
# y_axis_list = ['MinimumGridSpacing[CylinderSMA0.0]','MinimumGridSpacing[FilledCylinderMA0]','MinimumGridSpacing[SphereA0]']
# y_axis_list = [i for i in column_names if ('SphereA' in i)]
# y_axis_inc_list = [f"SphereC{i}$" for i in range(0,45,5)]
# y_axis_list = []
# for col in column_names:
#   for i in y_axis_inc_list :
#     if re.search(i,col):
#       y_axis_list.append(col)
# print(y_axis_list)

plot_fun = lambda x,y,label : plt.plot(x,y,label=label)
# plot_fun = lambda x,y,label : plt.plot(x,y,label=label,marker='x')
plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label)

# plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label,marker='x')
# plot_fun = lambda x,y,label : plt.loglog(x,y,label=label) 
# plot_fun = lambda x,y,label : plt.scatter(x,y,label=label,s=10,marker="x",alpha=0.4)
# save_path = "/groups/sxs/hchaudha/rough/high_acc_plots/"
# save_path = "/groups/sxs/hchaudha/rough/plots/"
# save_path = "/home/hchaudha/notes/spec_accuracy/figures/"
# save_path = "/home/hchaudha/notes/spec_accuracy/L5_comparisons/"
# save_path = "/home/hchaudha/notes/spec_accuracy/L5_comparisons/L15_no_tol/"
legend_dict = {}
for key in runs_data_dict.keys():
  legend_dict[key] = None

# legend_dict = {
#     'high_accuracy_L1_main':"Old Level 1",
#     'high_accuracy_L2_main':"Old Level 2",
#     'high_accuracy_L3_main':"Old Level 3",
#     'high_accuracy_L4_main':"Old Level 4",
#     'high_accuracy_L5_main':"Old Level 5",
#     '6_set1_L6s1':'New Level 1',
#     '6_set1_L6s2':'New Level 2',
#     '6_set1_L6s3':'New Level 3',
#     '6_set1_L6s4':'New Level 4',
#     '6_set1_L6s5':'New Level 5',
#     'high_accuracy_L1':"New Level 1",
#     'high_accuracy_L2':"New Level 2",
#     'high_accuracy_L3':"New Level 3",
#     'high_accuracy_L4':"New Level 4",
#     'high_accuracy_L5':"New Level 5",
#  }

append_to_title = ""
if '@' in data_file_path:
  append_to_title = " HorizonBH="+data_file_path.split('@')[-1]

# with plt.style.context('default'):
with plt.style.context('ggplot'):
#   plt.rcParams["figure.figsize"] = (15,10)
#   plt.rcParams["figure.figsize"] = (4,4)
#   plt.rcParams["figure.figsize"] = (10,8)
  plt.rcParams["figure.figsize"] = (6,6)
  plt.rcParams["figure.autolayout"] = True
  # plt.ylim(1e-10,1e-4)
  if y_axis_list is None:
    plot_graph_for_runs(runs_data_dict, x_axis, y_axis, minT, maxT, legend_dict=legend_dict, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun=plot_fun, diff_base=diff_base, plot_abs_diff=plot_abs_diff,constant_shift_val_time=constant_shift_val_time,append_to_title=append_to_title)
  else:
    plot_graph_for_runs_wrapper(runs_data_dict, x_axis, y_axis_list, minT, maxT, legend_dict=legend_dict, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun=plot_fun, diff_base=diff_base, plot_abs_diff=plot_abs_diff,constant_shift_val_time=constant_shift_val_time,append_to_title=append_to_title)


#   plt.title("")
#   plt.ylabel("Constraint Violations near black holes")
#   plt.tight_layout()
#   plt.legend(loc='upper right')
#   plt.ylim(1e-8, 1e-5)
#   plt.ylim(1e-12, 1e-6)
#   save_name = "main_ode_impro_const_new_no_avg.png"

#   save_name = Path(f"/groups/sxs/hchaudha/scripts/report/figures/{save_name}")
#   if save_name.exists():
#     raise Exception("Change name")
#   plt.savefig(save_name,dpi=600)
  plt.tight_layout()
  plt.show()

# %% [markdown]
# #### Save all y axis

# %%
moving_avg_len=0
save_path = None
diff_base = None
constant_shift_val_time = None
plot_abs_diff = False
y_axis_list = None
x_axis = 't(M)'

plot_abs_diff = True

minT = 0
maxT = 40000
maxT = 4000

plot_fun = lambda x,y,label : plt.plot(x,y,label=label)
# plot_fun = lambda x,y,label : plt.plot(x,y,label=label,marker='x')
plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label)

legend_dict = {}
for key in runs_data_dict.keys():
    legend_dict[key] = None

append_to_title = ""
if '@' in data_file_path:
    append_to_title = " HorizonBH="+data_file_path.split('@')[-1]


# main_folder_path = Path("/resnick/groups/sxs/hchaudha/figures/spec_accuracy/high_accuracy_L1to5_main")
# for y_axis in column_names:
#     if y_axis == 't(M)':
#         continue
#     with plt.style.context('ggplot'):
#         plt.rcParams["figure.figsize"] = (6,6)
#         plt.rcParams["figure.autolayout"] = True
#         # plt.ylim(1e-10,1e-4)
#         if y_axis_list is None:
#             plot_graph_for_runs(runs_data_dict, x_axis, y_axis, minT, maxT, legend_dict=legend_dict, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun=plot_fun, diff_base=diff_base, plot_abs_diff=plot_abs_diff,constant_shift_val_time=constant_shift_val_time,append_to_title=append_to_title)
#         else:
#             plot_graph_for_runs_wrapper(runs_data_dict, x_axis, y_axis_list, minT, maxT, legend_dict=legend_dict, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun=plot_fun, diff_base=diff_base, plot_abs_diff=plot_abs_diff,constant_shift_val_time=constant_shift_val_time,append_to_title=append_to_title)


#         #   plt.title("")
#         #   plt.ylabel("Constraint Violations near black holes")
#         #   plt.tight_layout()
#         #   plt.legend(loc='upper right')
#         #   plt.ylim(1e-8, 1e-5)
#         #   plt.ylim(1e-12, 1e-6)
#         #   save_name = "main_ode_impro_const_new_no_avg.png"

#         save_name = main_folder_path/f"{y_axis}"
#         if save_name.exists():
#             raise Exception("Change name")
#         plt.tight_layout()
#         plt.savefig(save_name,dpi=600)
#         plt.close()
#         print(y_axis)



# %% [markdown]
# #### Noise in things

# %%
runs_data_dict.keys()

# %%
# data = runs_data_dict['26_set1_L6_long'].copy()
# data = runs_data_dict['26_main_L6_long'].copy()
data = runs_data_dict['6_set1_L6s6'].copy()
# data = runs_data_dict['6_set1_L3s3'].copy()
# data = runs_data_dict['high_accuracy_L3_main'].copy()

# %%
y_key = 'ArealMass'
scipy_or_np = 'scipy'
window = 6 # Choose appropriate window size
moving_avg_len = None
moving_avg_len = 50
# scipy_or_np = 'np'

for key in runs_data_dict:
    # if 'L3_AC' not in key:
    #     if '22_set1_L3_long' not in key:
    #         continue
    if 'L3_AC' not in key:
        if '22_set1_L3_long' not in key:
            continue
    data = runs_data_dict[key].copy()
    t = np.array(data['t(M)'])
    x = np.array(data[y_key])

    if scipy_or_np == 'scipy':
        running_mean = uniform_filter1d(x, size=window, mode='nearest')
        noise_estimate = x - running_mean
    elif scipy_or_np == 'np':
        running_mean = np.convolve(x,np.ones(window), mode='valid')/window
        t = t[window//2-1:-window//2]
        noise_estimate = x[window//2-1:-window//2] - running_mean
    else:
        raise Exception(f"Invalid scipy_or_np value: {scipy_or_np}")

    if moving_avg_len is None:
        plt.plot(t,np.abs(noise_estimate), label=key)
    else:
        t = t[moving_avg_len//2-1:-moving_avg_len//2]
        y = np.convolve(np.abs(noise_estimate),np.ones(moving_avg_len), mode='valid')/moving_avg_len
        plt.plot(t,y, label=key)

title = f"Noise estimate {y_key}, window={int(window*0.5)}M"
if moving_avg_len is not None:
    title += f", moving_avg_len={moving_avg_len//2}M"
plt.title(title)
plt.xlabel("t(M)")
plt.ylabel(f"Noise estimate: {y_key}")
plt.legend()
plt.yscale('log')
# plt.ylim(1e-13,1e-6)
# plt.tight_layout()
plt.show()

# %%
t = np.array(data['t(M)'])
x = np.array(data['ArealMass'])
# x is your time series
window = 25 # Choose appropriate window size
running_mean = uniform_filter1d(x, size=window, mode='nearest')
noise_estimate = x - running_mean

# running_mean = np.convolve(x,np.ones(window), mode='valid')/window
# t = t[window//2-1:-window//2]
# noise_estimate = x[window//2-1:-window//2] - running_mean


# %%
plt.plot(t,np.abs(noise_estimate))
plt.yscale('log')
plt.ylim(1e-13,1e-6)
plt.show()

# %%
cols = data.columns
cols  = [col for col in cols if 'SphereC13' in col]
# cols  = [col for col in cols if '2Con' in col]
fil_data = data[cols]
# fil_data

# %%
t = data['t(M)']

colors = ['r','b','g']
plt.plot(t,fil_data[[col for col in cols if '1Con' in col]].min(axis=1),color=colors[0])
plt.plot(t,fil_data[[col for col in cols if '1Con' in col]].max(axis=1),color=colors[0])
plt.plot(t,fil_data[[col for col in cols if '1Con' in col]].median(axis=1),label='1Con',color=colors[0])

plt.plot(t,fil_data[[col for col in cols if '2Con' in col]].min(axis=1),color=colors[1])
plt.plot(t,fil_data[[col for col in cols if '2Con' in col]].max(axis=1),color=colors[1])
plt.plot(t,fil_data[[col for col in cols if '2Con' in col]].median(axis=1),label='2Con',color=colors[1])

all_but_zero = list(set([col for col in cols if '3Con' in col])  - set([col for col in cols if '3Cont' in col] ))
plt.plot(t,fil_data[all_but_zero].min(axis=1),color=colors[2])
plt.plot(t,fil_data[all_but_zero].max(axis=1),color=colors[2])
plt.plot(t,fil_data[all_but_zero].median(axis=1),label='3Con',color=colors[2])

plt.yscale('log')
plt.legend()
plt.show()

# %%
t = data['t(M)']
# y = fil_data['Linf(3Conzyy) on SphereC10']
# y = fil_data['Linf(2Conzy) on SphereC10']
y = fil_data.min(axis=1)
# y = fil_data['Linf(1Cont) on SphereC10']
plt.plot(t,fil_data.min(axis=1),label='min')
plt.plot(t,fil_data.max(axis=1),label='max')
plt.plot(t,fil_data.median(axis=1),label='median')
# plt.plot(t,y)
plt.yscale('log')
plt.legend()
plt.show()

# %% [markdown]
# ### Domain vals vs time

# %%
runs_data_dict.keys()

# %% [markdown]
# ##### Linf

# %%
filtered_runs = {}
for key,val in runs_to_plot.items():
  # if "15_" not in key:
  #   filtered_runs[key] = val
  #   continue
  filtered_runs[key] = val
  
filtered_runs.keys()

# %%

# data_file_path = "ConstraintNorms/GhCe.dat"
# data_file_path = "ConstraintNorms/GhCeExt.dat"
# data_file_path = "ConstraintNorms/GhCeExt_L2.dat"
# data_file_path = "ConstraintNorms/GhCeExt_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_L2.dat"
# data_file_path = "ConstraintNorms/GhCe_Linf.dat"
# data_file_path = "ConstraintNorms/Linf.dat"
# data_file_path = "ConstraintNorms/Constraints_Linf.dat"
# data_file_path = "ConstraintNorms/GhCe_VolL2.dat"
data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"

# data_file_path = "GhCe.dat"
# data_file_path = "NormalizedGhCe.dat"
# data_file_path = "GhCe_Norms.dat"
# data_file_path = "kappaErr_Linf.dat"
# data_file_path = "psiErr_Linf.dat"

column_names, runs_data_dict_ghce = load_data_from_levs(filtered_runs,data_file_path)
# column_names, runs_data_dict_ghce = load_data_from_levs(runs_to_plot,data_file_path)
print(runs_data_dict_ghce.keys())

# %%
# key = '6_set1_L3_EXP_FK_14012_10_6_30'
# key = '16_set1_L3'
# key = '22_set1_L3_long'
key = '6_set1_L3s3'
# key = '67_master_mr3'
key = '119_gd_SUKS_3_20'
# key = '26_set1_L6_long'
# key = 'set1_L6_AK_S2_L20_cd_10000'
# key = 'L3_AC_L3_cd_const_high'
# key = 'L3_AC_L3_3_01'
# key = 'high_accuracy_L5_main'
# key = 'high_accuracy_L5'
# key = '28_set1_cd_junk_1'
# key = '22_set1_L3_long'
# key = 'set1_L6s4_cd10'
# key = 'set1_L6s4_cd100'
# key = 'set1_L6s4_cd200'
# key = 'set1_L6s4_cd500'
# key = 'set1_L6s4_cd100_AMRL6'
# key = 'set1_L6s4_cd100_AMRL7'
# key = 'set1_L6s4_cd100_AMRL8'
# key = '22_set1_L3_long'

repeated_symmetric = False
# repeated_symmetric = True

minT = 0
# minT = 10
# minT = 480
# minT = 1400
# minT = 3080
# minT = 9700
# minT = 10000

maxT = 40000
# maxT = 4000
# maxT = 590
maxT = 3500
maxT = 200
# maxT = 800
# maxT = 2

data = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_ghce[key])
data = data.iloc[::1].dropna(axis=1, how='all')

domain_col_list = filter_by_regex(regex=["Sphere","Cylinder"],col_list=data.columns)
domain_col_list = filter_by_regex(regex=["Linf"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC[4-9]"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC[2][0-9]"],col_list=data.columns)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list,exclude=True)
# domain_col_list = filter_by_regex(regex=["1Conx"],col_list=domain_col_list)
visual_data = data[domain_col_list]
visual_data = np.log10(visual_data)
# print(visual_data.columns)
# Plot using imshow

column_names = list(visual_data.columns)
# column_names = [i.split(" ")[-1] for i in visual_data.columns]
column_names = return_sorted_domain_names(column_names, repeated_symmetric=repeated_symmetric)
# print(column_names)

vmin_log,vmax_log = None,None
# vmin_log,vmax_log = -12.417834211445228 , -8.343109330686875
# vmin_log,vmax_log = -13.918609670001128 , -2.3373560237057256
# vmin_log,vmax_log = -9 , None
if vmin_log is None:
  vmin_log = visual_data.min().min()
if vmax_log is None:
  vmax_log = visual_data.max().max()
print(vmin_log,",",vmax_log)

print(len(domain_col_list))

if repeated_symmetric:
    visual_data["Excision"] = [np.nan for i in range(len(data['t(M)']))]

plt.figure(figsize=(18, 10))
imshow_plot = plt.imshow(
    visual_data[column_names], 
    aspect='auto', 
    cmap='RdYlGn_r', 
    origin='lower',interpolation='none',
    vmin=vmin_log, 
    vmax=vmax_log
)

if repeated_symmetric:
    # Set x-ticks and labels
    plt.xticks(
        ticks=np.arange(len(column_names)), 
        labels=[i.split(" ")[-1] for i in column_names], 
        rotation=90
    )
else:
    # Set x-ticks and labels
    plt.xticks(
        ticks=np.arange(len(visual_data.columns)), 
        labels=[i.split(" ")[-1] for i in column_names], 
        rotation=90
    )

ytick_step = 1
ytick_step = len(visual_data) // 10  # Show about 10 ticks
plt.yticks(
    ticks=np.arange(0, len(visual_data), ytick_step), 
    labels=data['t(M)'][::ytick_step].astype(int)
)

# Create colorbar
colorbar = plt.colorbar(imshow_plot, label=f'{column_names[0].split(" ")[0]}')

# Determine colorbar ticks that align with the fixed vmin and vmax
# tick_vals = np.linspace(vmin_log, vmax_log, num=5)

# Set these ticks on the colorbar
# colorbar.set_ticks(tick_vals)

# Convert ticks back to the original scale for labeling
# colorbar.set_ticklabels([f'{10**val:.2e}' for val in tick_vals])

plt.ylabel('t(M)')
plt.title(f'{key}')
plt.tight_layout() 

plt.grid(False)
plt.show()

# %% [markdown]
# ##### APS plot version

# %%
# key = '6_set1_L3_EXP_FK_14012_10_6_30'
# key = '16_set1_L3'
key = '22_set1_L3_long'
# key = 'L3_AC_L4'
key = '28_set1_cd_junk_1'
# key = 'high_accuracy_L5'
# key = '22_set1_L3_long'
# key = 'set1_L6_AK_cd_100'
# key = '6_set1_L6s6'

repeated_symmetric = False
# repeated_symmetric = True
num_Excision = 3


minT = 0
# minT = .2
# minT = 480
# minT = 1400
# minT = 3000
# minT = 3500
# minT = 7000

maxT = 40000
maxT = 4000
# maxT = 590
# maxT = 3500
# maxT = 340
# maxT = 800
# maxT = 2

data = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_ghce[key])
data = data.iloc[::1].dropna(axis=1, how='all')

domain_col_list = filter_by_regex(regex=["Sphere","Cylinder"],col_list=data.columns)
domain_col_list = filter_by_regex(regex=["Linf"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC[4-9]"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC[2][0-9]"],col_list=data.columns)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list,exclude=True)
# domain_col_list = filter_by_regex(regex=["1Conx"],col_list=domain_col_list)
visual_data = data[domain_col_list]
visual_data = np.log10(visual_data)
# print(visual_data.columns)
# Plot using imshow

column_names = list(visual_data.columns)
# column_names = [i.split(" ")[-1] for i in visual_data.columns]
column_names = return_sorted_domain_names(column_names, repeated_symmetric=repeated_symmetric, num_Excision=num_Excision)
# print(column_names)

vmin_log,vmax_log = None,None
# vmin_log,vmax_log = -12.417834211445228 , -8.343109330686875
# vmin_log,vmax_log = -10.183427321168152 , -2.540360921276033
vmin_log,vmax_log = -10.2 , -0.10
# vmin_log,vmax_log = -9 , None
if vmin_log is None:
  vmin_log = visual_data.min().min()
if vmax_log is None:
  vmax_log = visual_data.max().max()
print(vmin_log,",",vmax_log)

print(len(domain_col_list))

if repeated_symmetric:
    visual_data["Excision"] = [np.nan for i in range(len(data['t(M)']))]

fig, ax = plt.subplots(figsize=(7, 4))

imshow_plot = ax.imshow(
    visual_data[column_names], 
    aspect='auto', 
    cmap='RdYlGn_r', 
    origin='lower',
    interpolation='none',
    vmin=vmin_log, 
    vmax=vmax_log
)

# Set x-ticks and labels
# ax.set_xticks(np.arange(len(visual_data.columns)))
# ax.set_xticklabels([i.split(" ")[-1] for i in column_names], rotation=90)

ax.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)

# ytick_step = 1
# ytick_step = len(visual_data) // 10  # Show about 10 ticks
# ax.set_yticks(np.arange(0, len(visual_data), ytick_step))
# ax.set_yticklabels(data['t(M)'][::ytick_step].astype(int))

# Create colorbar
colorbar = fig.colorbar(imshow_plot, ax=ax, label=f'log(constraint violation)')

# Determine colorbar ticks that align with the fixed vmin and vmax
# tick_vals = np.linspace(vmin_log, vmax_log, num=5)

# Set these ticks on the colorbar
# colorbar.set_ticks(tick_vals)

# Convert ticks back to the original scale for labeling
# colorbar.set_ticklabels([f'{10**val:.2e}' for val in tick_vals])

ax.set_ylabel(r'time $\longrightarrow$')
# ax.set_xlabel(r'subdomain number')
ax.set_xticks(
    ticks=[0,44,65,len(column_names)], 
    labels=['Outer Boundary','bhA','bhB','Outer Boundary'], 
    # rotation=90
)
ax.set_xticks(
    ticks=[0,44,54,65,len(column_names)], 
    labels=['Outer Boundary','bhA','center','bhB','Outer Boundary'], 
    # rotation=90
)
# ax.set_title(f'{key}')
plt.tight_layout()
# annotation1 = ax.annotate('bh A', 
#             xy=(0.2,0.1),             # point to annotate
#             xytext=(0.1,0.5),  # text position
#             xycoords = 'subfigure fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05)
#             )
# annotation2 = ax.annotate('bh B', 
#             xy=(0.36,0.1),             # point to annotate
#             xytext=(0.25,0.5),  # text position
#             xycoords = 'subfigure fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05)
#             )
# annotation3 = ax.annotate('Outer boundary', 
#             xy=(0.82,0.1),             # point to annotate
#             xytext=(0.6,0.5),  # text position
#             xycoords = 'subfigure fraction',
#             arrowprops=dict(facecolor='black', shrink=0.05)
#             )


ax.grid(False)

# save_name = "Extra_L5_set1_sym.png"
# save_name = "Extra_L6_set1_sym_lim.png"
# save_name = "Extra_L5_main_sym.png"
# save_name = Path(f"/groups/sxs/hchaudha/scripts/report/figures/{save_name}")
# if save_name.exists():
#     raise Exception("Change name")
# plt.savefig(save_name,dpi=600,bbox_inches='tight', bbox_extra_artists=[annotation1, annotation2, annotation3])
# plt.savefig(save_name,dpi=600,bbox_inches='tight')


plt.show()

# %% [markdown]
# #### take diff between two different runs

# %%
print(runs_data_dict_ghce.keys())

# %%
key1 = '6_set1_L6s6'
key1 = '30_RM_set1_L3'
# key1 = '22_set1_L1_long'
# key2 = '30_RM_set1_L3'
key2 = '6_set1_L3s3'
# key2 = '28_set1_cd_junk_1'
# key2 = '28_set1_cd_junk_1'


minT = 0
# minT = 480
minT = 3080
# minT = 7360

maxT = 40000
# maxT = 150
# maxT = 590
# maxT = 1500
# maxT = 8500

data1 = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_ghce[key1])
data1 = data1.iloc[::1].dropna(axis=1, how='all')

data2 = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_ghce[key2])
data2 = data2.iloc[::1].dropna(axis=1, how='all')


# Set column 't(M)' as the index for both data1 and data2
data1.set_index('t(M)', inplace=True)
data2.set_index('t(M)', inplace=True)

# Now, 't(M)' is the index, and you can perform the subtraction safely
data = data1.copy()
for col in data.columns:
    # Negative(green) is good
    data[col] = (np.log10(data2[col]) - np.log10(data1[col]))
    # data[col] = (data2[col] - data1[col])*2/(data2[col] + data1[col])
    # Modify each value: set to 1 if positive, 0 if negative or zero
    # => red(1) if more error and green(0) is less
    # data[col] = np.where(data[col] > 0, 1, 0)

# Reset the index if you need 't(M)' back as a column
data.reset_index(inplace=True)

domain_col_list = filter_by_regex(regex=["Sphere","Cylinder"],col_list=data.columns)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list,exclude=True)
# domain_col_list = filter_by_regex(regex=["1Conx"],col_list=domain_col_list)

visual_data = data[domain_col_list]
# visual_data = np.log10(visual_data)
# Plot using imshow

column_names = list(visual_data.columns)
# column_names = [i.split(" ")[-1] for i in visual_data.columns]
column_names = return_sorted_domain_names(column_names)
print(column_names)
vmin_log,vmax_log = None,None
if vmin_log is None:
  # Get min non -inf value
  temp = visual_data.copy()
  temp.replace([np.inf, -np.inf], np.nan, inplace=True)
  vmin_log = temp.min().min()
if vmax_log is None:
  vmax_log = visual_data.max().max()
print(vmin_log,",",vmax_log)

# vmin_log, vmax_log = vmin_log, -vmin_log # This is to make it so white is 0, green is good and red is bad

# Emphasize the bad parts, i.e. more color resolution for the red part
# vmin_log, vmax_log = -vmax_log, vmax_log # This is to make it so white is 0, green is good and red is bad

# max_val = max(abs(vmin_log),abs(vmax_log))
# vmin_log, vmax_log = -max_val, max_val # This is to make it so white is 0, green is good and red is bad
# print(vmin_log,",",vmax_log)


# Example colormap centered around zero
plt.figure(figsize=(15, 10))
divnorm = mcolors.TwoSlopeNorm(vmin=vmin_log, vcenter=0, vmax=vmax_log)

imshow_plot = plt.imshow(
    visual_data[column_names], 
    aspect='auto', 
    cmap='RdYlGn_r',
    origin='lower',interpolation='none',
    norm=divnorm
)

# Set x-ticks and labels
plt.xticks(
    ticks=np.arange(len(visual_data.columns)), 
    labels=[i.split(" ")[-1] for i in column_names], 
    rotation=90
)

ytick_step = len(visual_data) // 10  # Show about 10 ticks
plt.yticks(
    ticks=np.arange(0, len(visual_data), ytick_step), 
    labels=data['t(M)'][::ytick_step].astype(int)
)

# Create colorbar
colorbar = plt.colorbar(imshow_plot, label=f'{column_names[0].split(" ")[0]}')

plt.ylabel('t(M)')
plt.title(f'{key2}(Green better) - {key1}')
plt.tight_layout()
plt.grid(False)
plt.show()

# %%

minT = 0
minT = 1200
# minT = 4000

maxT = 40000
maxT = 2300
# data = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_ghce[key1])
data = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_ghce[key2])
data = data.iloc[::1].dropna(axis=1, how='all')

domain_col_list = filter_by_regex(regex=["Sphere","Cylinder"],col_list=data.columns)
# domain_col_list = filter_by_regex(regex=["Cylinder"],col_list=data.columns)
# domain_col_list = filter_by_regex(regex=[r"SphereC14"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=[r"3Con"],col_list=domain_col_list)

data = data[domain_col_list+["t(M)"]]

max_info = {}
for col in data.columns:
    # Get the maximum value
    max_val = data[col].max()
    # Get all indices where the maximum value occurs
    idx_max_list = data[data[col] == max_val].index
    times = data.loc[idx_max_list, 't(M)']
    values = data.loc[idx_max_list, col]

    max_info[col] = {
        "max_idx_list": idx_max_list,
        "max_val_times": list(times),
        "max_vals": list(values),
    }
    for time, val in zip(max_info[col]['max_val_times'], max_info[col]['max_vals']):
        if val > 1e-9:
            print(col, time, val)

# %%
minT = 0
# minT = 1200
key = "6_set1_L6s2"
maxT = 40000
maxT = 4000
data = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_ghce[key])
data = data.iloc[::1].dropna(axis=1, how='all')

domain_col_list = filter_by_regex(regex=["Sphere","Cylinder"],col_list=data.columns)

# Calculate maximum error at each time point
data['max_error'] = data[domain_col_list].max(axis=1)

# Calculate L2 norm (Euclidean norm) of the errors at each time point
data['l2_norm'] = np.sqrt((data[domain_col_list] ** 2).sum(axis=1))

plt.plot(data['t(M)'], data["max_error"],label="max_error")
# plt.plot(data['t(M)'], data["l2_norm"],label="l2_norm")
plt.legend()
plt.yscale('log')
plt.title(key)
plt.show()

# %% [markdown]
# ### For domain resolutions

# %%
data_file_path = "MinimumGridSpacing.dat"
data_file_path = "Hist-GrDomain.txt"

# data_file_path = "Hist-Domain.txt"


column_names, runs_data_dict_domain = load_data_from_levs(runs_to_plot,data_file_path)
print(column_names)
print(runs_data_dict_domain.keys())

# %%
# key = '12_set1_L3_1500'
# key = '13_set1_L4_1500'
# key = '13_set1_L4_3000'
# key = '13_set1_L3_3000'
key = '17_set_main_q3_18_L3'
key = 'L3_AC_L3_minL17'
key = '23_nobounds_AMR'
key = '22_set1_L3_long'
key = 'L1_AC_L1'
# key = '26_main_L6_long'
# key = '26_set1_L6_long'
# data = runs_data_dict[key].iloc[1:8000:10].dropna(axis=1, how='all')

# key = '15_AMR_Lev0_455'

minT = 0
# minT = 1400
# minT = 3400
minT = 10000
maxT = 40000
# maxT = 1200
# maxT = 4000
data = limit_by_col_val(minT,maxT,'t(M)',runs_data_dict_domain[key])
data = data.iloc[::].dropna(axis=1, how='all')
domain_col_list = filter_by_regex(regex=["Sphere","Cylinder"],col_list=data.columns)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC[4-9]"],col_list=domain_col_list)
# domain_col_list = filter_by_regex(regex=["SphereC[2][0-9]"],col_list=data.columns)
# domain_col_list = filter_by_regex(regex=["SphereC"],col_list=domain_col_list,exclude=True)


visual_data = data[domain_col_list]
if "Grid" in data_file_path:
  visual_data = np.log10(visual_data)
# Plot using imshow

column_names = list(visual_data.columns)
if "Grid" not in data_file_path:
  column_names = [i for i in column_names if ("_R" in i)]
#   column_names = [i for i in column_names if ("_L" in i)]
  # column_names = [i for i in column_names if ("_M" in i)]
column_names = return_sorted_domain_names(column_names)
plt.figure(figsize=(15, 10))
plt.imshow(visual_data[column_names], aspect='auto', cmap='RdYlGn_r', origin='lower',interpolation='none')

# Set x-ticks and labels
plt.xticks(ticks=np.arange(len(column_names)), labels=[get_domain_name(i) for i in column_names], rotation=90)
# plt.xticks(ticks=np.arange(len(column_names)), labels=[i.split("_")[0] for i in column_names], rotation=90)

# ytick_step = len(visual_data)
ytick_step = len(visual_data) // 5  # show about 10 ticks
# ytick_step = len(visual_data)   # show about 10 ticks
# if ytick_step < 1:
#   ytick_step = 10
plt.yticks(ticks=np.arange(0, len(visual_data), ytick_step), 
           labels=data['t(M)'][::ytick_step].astype(int))
# plt.yticks(ticks=np.arange(len(visual_data)), labels=data['t(M)'].astype(int))

# plt.colorbar(label=f'{column_names[0].split(" ")[0]}')
plt.colorbar()
# plt.xlabel('Features')
plt.ylabel('t(M)')
plt.title(f'{key}')
plt.tight_layout()
plt.grid(False)
plt.show()

# %% [markdown]
# #### Plot L,M and R on the same graph

# %%
# Assuming return_sorted_domain_names and get_domain_name are predefined functions
# Here's how you can create three subplots with three different column_name filters:

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), constrained_layout=True)
column_name_filters = ["_R", "_L", "_M"]
column_name = ["Extent 0", "Extent 1", "Extent 2"]

for i, ax, filter_suffix in zip(range(len(column_name_filters)),axes, column_name_filters):
    filtered_column_names = [i for i in visual_data.columns if (filter_suffix in i)]
    filtered_column_names = return_sorted_domain_names(filtered_column_names)
    
    im = ax.imshow(visual_data[filtered_column_names], aspect='auto', cmap='RdYlGn_r', origin='lower',interpolation='none',)
    
    if filter_suffix == "_M":
      # Set x-ticks and labels
      ax.set_xticks(np.arange(len(filtered_column_names)))
      ax.set_xticklabels([get_domain_name(i) for i in filtered_column_names], rotation=90)
    
    # Set y-ticks and labels
    ytick_step = len(visual_data) // 5  # show about 10 ticks
    ax.set_yticks(np.arange(0, len(visual_data), ytick_step))
    ax.set_yticklabels(data['t(M)'][::ytick_step].astype(int))
    
    # Set labels and title
    ax.set_ylabel('t(M)')
    ax.set_title(f'{key} : {column_name[i]}, {filter_suffix[-1]}')
    
    # Add a colorbar to each subplot
    fig.colorbar(im, ax=ax)
    
    ax.grid(False)

plt.show()

# %% [markdown]
# ### Plot 2d grid

# %%
# data_file_path = "ConstraintNorms/GhCe.dat"
# data_file_path = "ConstraintNorms/GhCeExt.dat"
# data_file_path = "ConstraintNorms/GhCeExt_L2.dat"
# data_file_path = "ConstraintNorms/GhCeExt_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_L2.dat"
# data_file_path = "ConstraintNorms/GhCe_Linf.dat"
# data_file_path = "ConstraintNorms/Linf.dat"
# data_file_path = "ConstraintNorms/Constraints_Linf.dat"
# data_file_path = "ConstraintNorms/GhCe_Norms.dat"
# data_file_path = "ConstraintNorms/GhCe_VolL2.dat"
data_file_path = "ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Norms.dat"
column_names, runs_data_dict_ghce = load_data_from_levs(runs_to_plot,data_file_path)
print(runs_data_dict_ghce.keys())

# %%
# for key in runs_data_dict_ghce.keys():
for key in ["high_accuracy_L5"]:
  t_list = np.arange(0,4000,100)
  save_path = Path("/home/hchaudha/notes/spec_accuracy/del/domain_plots")
  save_path = save_path/key
  if not save_path.exists():
    save_path.mkdir()
  print(save_path)

  nA=5
  rA=nA*1.5
  center_xA=rA + 2
  RA=rA+5
  rC=RA*2
  nC=30
  RC=rC+nC

  saved_path_list_for_gif = []
  for t in t_list:
    fig, ax = plt.subplots(figsize=(12, 10))
    pandas_dict = runs_data_dict_ghce[key].iloc[2*t,:].to_dict()
    time = pandas_dict.pop("t(M)")
    # domain_color_local,sm = scalar_to_color(pandas_dict,(-11,-4),color_map='RdYlGn_r')
    domain_color_local,sm = scalar_to_color(pandas_dict,(-8,-4),color_map='RdYlGn_r')
    domain_color_local,sm = scalar_to_color(pandas_dict,color_map='RdYlGn_r')

    patches_class = BBH_domain_sym_ploy(center_xA=center_xA, rA=rA, RA=RA, rC=rC, RC=RC, nA=nA, nC=nC, color_dict=domain_color_local) 
    for patch in patches_class.patches:
      ax.add_patch(patch)

    ax.set_xlim(-RC, RC)
    ax.set_ylim(-RC, RC)
    ax.set_aspect('equal')

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f"{list(pandas_dict.keys())[0].split(' ')[0]}")
    plt.title(f"{key}  t(M) = {time}")
    plt.tight_layout()
    img_save_path = save_path/f"{t}.png"
    print(img_save_path)
    plt.savefig(img_save_path)
    saved_path_list_for_gif.append(img_save_path)
    plt.close()


  def create_gif(filenames, output_gif='output.gif'):
    images = []
    # Read each image file using iio.imread and append it to the list
    for filename in filenames:
        images.append(iio.imread(filename))
    # Write the images as an animated GIF using iio.imwrite
    iio.imwrite(output_gif, images, duration=500, loop=0)

  create_gif(saved_path_list_for_gif,output_gif=save_path/'output.gif')
  print(f"Gif created: {save_path/'output.gif'}")


# %%
t = np.linspace(0,4000,100)
data1 = runs_data_dict[key].iloc[2*t,:].to_dict()
data1['t(M)'].values()
for key in data1.keys():
  print(key.split(" ")[-1])

# %%
key = "high_accuracy_L5"
t_list = np.arange(0,4000,100)
t_list = [2000]

# save_path = Path("/home/hchaudha/notes/spec_accuracy/del/domain_plots")/key
# if not save_path.exists():
#   save_path.mkdir()
# print(save_path)

save_path = None

nA=4
rA=nA*1.5
center_xA=rA + 2
RA=rA+5
rC=RA*2
nC=30
RC=rC+nC


for t in t_list:
  fig, ax = plt.subplots(figsize=(12, 10))
  pandas_dict = runs_data_dict[key].iloc[2*t,:].to_dict()
  time = pandas_dict.pop("t(M)")
  domain_color_local,sm = scalar_to_color(pandas_dict,(-10,-2),color_map='RdYlGn_r')

  patches_class = BBH_domain_sym_ploy(center_xA=center_xA, rA=rA, RA=RA, rC=rC, RC=RC, nA=nA, nC=nC, color_dict=domain_color_local) 
  for patch in patches_class.patches:
    ax.add_patch(patch)

  ax.set_xlim(-RC, RC)
  ax.set_ylim(-RC, RC)
  ax.set_aspect('equal')

  cbar = plt.colorbar(sm, ax=ax)
  cbar.set_label(f"{list(pandas_dict.keys())[0].split(' ')[0]}")
  plt.title(f"{key}  t(M) = {time}")
  plt.tight_layout()
  if save_path is not None:
    print(save_path/f"{t}.png")
    plt.savefig(save_path/f"{t}.png")
  plt.show()

# %% [markdown]
# ### Plot and save all y_axis in this data

# %%
# moving_avg_len=25
plt.close()
save_path = Path("/home/hchaudha/notes/spec_accuracy/del/L35_all/")
for file in save_path.glob('*'):  # Use '*' to match all files
    if file.is_file():  # Check if it's a file
        file.unlink()  # Remove the file
save_path = str(save_path)+"/"

# save_path = "/home/hchaudha/notes/spec_accuracy/uniAMR_comparisons/all_set3/"
diff_base = None

plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label,marker='x')
plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label)
plot_fun = lambda x,y,label : plt.plot(x,y,label=label,marker='x')
# plot_fun = lambda x,y,label : plt.plot(x,y,label=label)
y_lower,y_upper=1e-8,1
y_lower,y_upper=1e-13,1e-4
y_lower,y_upper=1e-12,1e-3

legend_dict = {}
for key in runs_data_dict.keys():
  legend_dict[key] = None

minT = 0
# minT = 1800
maxT = 28000
# maxT = 1200

x_axis = 't(M)'

for y_axis in column_names:
  if y_axis == x_axis:
    continue
  try:
    with plt.style.context('default'):
      plt.rcParams["figure.figsize"] = (12,10)
      plt.rcParams["figure.autolayout"] = True
      # plt.ylim(y_lower,y_upper)
      plot_graph_for_runs(runs_data_dict, x_axis, y_axis, minT, maxT, legend_dict=legend_dict, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun=plot_fun, diff_base=diff_base)
      plt.close()
    print(f"{y_axis} done!")
  except Exception as e:
      print(f"Error plotting {y_axis}: {str(e)}")
      continue

# %%
df = runs_data_dict['all_100_t2690_obs']
minT = 2690
minT = 2700
maxT = 2710
maxT = minT+2
df = df[df['time']>minT]
df = df[df['time']<maxT]
from scipy.signal import find_peaks

# plt.plot(df['time'],df['Linf(GhCe) on SphereA0'],marker='x')
# plt.plot(df['time'],df['Linf(GhCe) on SphereA4'],marker='x')
plt.plot(df['time'],df['Linf(GhCe) on SphereA3'],marker='x')
# plt.yscale('log')
plt.show()

# %%
data = runs_data_dict['all_100_t2690_obs_grid_tol_10']
data = data[data['time after step']>2690]
dt_arr = np.array(data.dt)
averaged_dt = np.zeros_like(dt_arr)
averaged_dt[0] =  dt_arr.mean() 
N = 100
for i in range(len(dt_arr)-1):
  averaged_dt[i+1] = averaged_dt[i]*(N-1)/(N)+dt_arr[i+1]*1/N

# %%
plt.plot(data['time after step'],averaged_dt)
plt.plot(data['time after step'],dt_arr)
plt.show()

# %% [markdown]
# ## Plot max/min value for a run

# %%
# data_file_path="ConstraintNorms/GhCe_L2.dat"
data_file_path="ConstraintNorms/GhCe_Linf.dat"
# data_file_path="ConstraintNorms/NormalizedGhCe_Linf.dat"
# data_file_path="ConstraintNorms/Constraints_Linf.dat"
data_file_path="MinimumGridSpacing.dat"
# data_file_path="ConstraintNorms/GhCe_VolL2.dat"
# data_file_path = "ConstraintNorms/NormalizedGhCe_Norms.dat"
column_names_linf2, runs_data_dict_linf2 = load_data_from_levs(runs_to_plot,data_file_path)
# print(column_names_linf2)
print(runs_data_dict_linf2.keys())

# %%
# run_name = list(runs_data_dict_linf2.keys())[4]
# run_name = '3555_1.0e-07_060'
run_name = '120W_gd_SUKS1_3_20'
# run_name = 'high_accuracy_L3'
# run_name = '6_set2_L3s3'
# run_name = '6_set2_L6s6'
# run_name = '6_set1_L3s0'
# run_name = '6_set1_L3s3'
df = runs_data_dict_linf2[run_name].copy()
df = df.sort_values(by=df.columns[0])
save_path = None
# save_path = "/home/hchaudha/notes/spec_accuracy/del/normalized_norms/all/"

# max_or_min = "MAX"
max_or_min = "MIN"

tmin=0
# tmin= 1200
# tmin= 2050
# tmin= 3000
# tmin=9300
# tmin=9372
# tmin=2691
tmax=50000
tmax=4000
# tmax=tmin+4
# tmax=7000
# tmax=2800

df = df[df['t(M)']>=tmin]
df = df[df['t(M)']<tmax]
t_name = df.columns[0]
y_axis = df.columns[1].split(" ")[0]
all_cols_but_t = df.columns[1:]
all_cols_but_t = []

def find_sphere_num(string:str):
  match = re.search(r'Sphere([A-C])(\d{1,2})', string)

  if match:
      letter = match.group(1)  # Extract the letter (A, B, or C)
      number = match.group(2)  # Extract the number (one or two digits)
      return letter,int(number)
  else:
      print(f"No Sphere found in the {string}.")
      return None

only_include = None
# only_include = [
#   r"SphereA[0-9]",
#   r"SphereB[0-9]",
#   r"SphereC[0-9]$",
#   r"SphereC1[0-9]",
#   r"SphereC2[0-9]",
#   r"CylinderE[A,B]\d",
#   r"CylinderC[A,B]\d",
#   r"CylinderSM[A,B]\d",
#   r"FilledCylinderE[A,B]\d",
#   r"FilledCylinderC[A,B]\d",
#   r"FilledCylinderM[A,B]\d",
# ]
# only_include = [
#   r"SphereC\d$",
#   r"SphereC\d\d$",
# ]
exclude = None
# exclude = [
#    r"SphereC",
# ]
# exclude = [
#    r"1Con",
#    r"2Con",
# #    r"3Con",
# ]
def matches_any_pattern(label, patterns):
    for pattern in patterns:
        if re.search(pattern, label):
            return True
    return False

SphereCMin,SphereCMax = 6,6
# SphereCMin,SphereCMax = 30,50

SphereAMin,SphereAMax = 0,9
SphereBMin,SphereBMax = 0,9
# SphereAMin,SphereAMax = 5,9
# SphereBMin,SphereBMax = 5,9
for i in df.columns[1:]:
  # Things to include, this overrides things to exclude
  if only_include is None:
    pass
  elif matches_any_pattern(i,only_include):
    # We are allowed to include this domain
    pass
  else:
    # We are not supposed to include this domain
    continue
  
  if exclude is None:
    pass
  elif matches_any_pattern(i,exclude):
    continue
  else:
    pass
  # Things to exclude
  if 'SphereC' in i:
    _, sphereC_num = find_sphere_num(i)
    if (sphereC_num < SphereCMin) or (sphereC_num > SphereCMax):
      print(i)
      continue
  if 'SphereA' in i :
    _, sphere_num = find_sphere_num(i)
    if (sphere_num < SphereAMin) or (sphere_num > SphereAMax):
      print(i)
      continue
  if 'SphereB' in i:
    _, sphere_num = find_sphere_num(i)
    if (sphere_num < SphereBMin) or (sphere_num > SphereBMax):
      print(i)
      continue
  if 't(M)' in i:
    print(i)
    continue
  all_cols_but_t.append(i)

if max_or_min == "MAX":
  # Find the maximum value across columns B, C, D, and F for each row
  df['extreme_val'] = df[all_cols_but_t].max(axis=1)

  # Determine which column had the maximum value
  df['extreme_source'] = df[all_cols_but_t].idxmax(axis=1)

if max_or_min == "MIN":
  # Find the maximum value across columns B, C, D, and F for each row
  df['extreme_val'] = df[all_cols_but_t].min(axis=1)

  # Determine which column had the maximum value
  df['extreme_source'] = df[all_cols_but_t].idxmin(axis=1)

# List all columns that have at least one extreme value
columns_with_extreme = df['extreme_source'].unique()

# Generate a colormap for the columns with at least one extreme value
num_colors = len(columns_with_extreme)
colors = plt.get_cmap('tab20', num_colors)  # Using 'tab20' colormap
color_map = {column: colors(i) for i, column in enumerate(columns_with_extreme)}

# Plot max_BCD vs t with different colors for different sources
plt.figure(figsize=(18, 10))
for i,source in enumerate(columns_with_extreme):
    subset = df[df['extreme_source'] == source]
    if i%4 == 0:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker="^")
    if i%4 == 1:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker="v")
    if i%4 == 2:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker=">")
    if i%4 == 3:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker="<")

plt.xlabel(t_name)
plt.ylabel(y_axis)
# plt.yscale('log')
plt.title(f'{max_or_min}:{y_axis} vs {t_name} for {run_name} : A_{SphereAMin}_{SphereAMax}_B_{SphereBMin}_{SphereBMax}_C_{SphereCMin}_{SphereCMax}')
plt.legend()
plt.grid(True)  
plt.tight_layout()
if save_path is None:
   save_path = "/groups/sxs/hchaudha/rough/"
plt.savefig(f"{save_path}{run_name}_{max_or_min}:{y_axis}_vs_{t_name}_{tmin}_{tmax}_A_{SphereAMin}_{SphereAMax}_B_{SphereBMin}_{SphereBMax}_C_{SphereCMin}_{SphereCMax}.png", dpi=500)
plt.show()

# %% [markdown]
# ## Plots all ys for a single run

# %%
# data_file_path="ConstraintNorms/GhCe_L2.dat"
data_file_path="ConstraintNorms/GhCe_Linf.dat"
data_file_path="ConstraintNorms/NormalizedGhCe_Linf.dat"
data_file_path="MinimumGridSpacing.dat"
# data_file_path="ConstraintNorms/GhCe_VolL2.dat"
column_names_linf3, runs_data_dict_linf3 = load_data_from_levs(runs_to_plot,data_file_path)
# print(column_names_linf3)
print(runs_data_dict_linf3.keys())

# %%
# run_name = list(runs_data_dict_linf2.keys())[4]
run_name = '119_gd_SUKS_3_20'
# run_name = 'high_accuracy_L4'
# run_name = 'high_accuracy_L3'
# run_name = '6_set2_L3s3'
# run_name = '6_set2_L6s6'
# run_name = '6_set3_L6s6'
df = runs_data_dict_linf3[run_name].copy()
df = df.sort_values(by=df.columns[0])
save_path = None
# save_path = "/home/hchaudha/notes/spec_accuracy/del/normalized_norms/all/"

# tmin= 1200
tmin=0
# tmin= 2050
# tmin=9300
# tmin=9372
# tmin=2691
tmax=50000
# tmax=2000
# tmax=tmin+4
# tmax=7000
# tmax=2800

df = df[df['t(M)']>=tmin]
df = df[df['t(M)']<tmax]
t_name = df.columns[0]
y_axis = df.columns[1].split(" ")[0]
all_cols_but_t = df.columns[1:]
all_cols_but_t = []

def find_sphere_num(string:str):
  match = re.search(r'Sphere([A-C])(\d{1,2})', string)

  if match:
      letter = match.group(1)  # Extract the letter (A, B, or C)
      number = match.group(2)  # Extract the number (one or two digits)
      return letter,int(number)
  else:
      print(f"No Sphere found in the {string}.")
      return None

SphereCMin,SphereCMax = 0,30
# SphereCMin,SphereCMax = 30,50

SphereAMin,SphereAMax = 1,9
SphereBMin,SphereBMax = 1,9
# SphereAMin,SphereAMax = 5,9
# SphereBMin,SphereBMax = 5,9
for i in df.columns[1:]:
  if 'SphereC' in i:
    _, sphereC_num = find_sphere_num(i)
    if (sphereC_num < SphereCMin) or (sphereC_num > SphereCMax):
      print(i)
      continue
  if 'SphereA' in i :
    _, sphere_num = find_sphere_num(i)
    if (sphere_num < SphereAMin) or (sphere_num > SphereAMax):
      print(i)
      continue
  if 'SphereB' in i:
    _, sphere_num = find_sphere_num(i)
    if (sphere_num < SphereBMin) or (sphere_num > SphereBMax):
      print(i)
      continue
  if 't(M)' in i:
    print(i)
    continue
  all_cols_but_t.append(i)

if max_or_min == "MAX":
  # Find the maximum value across columns B, C, D, and F for each row
  df['extreme_val'] = df[all_cols_but_t].max(axis=1)

  # Determine which column had the maximum value
  df['extreme_source'] = df[all_cols_but_t].idxmax(axis=1)

if max_or_min == "MIN":
  # Find the maximum value across columns B, C, D, and F for each row
  df['extreme_val'] = df[all_cols_but_t].min(axis=1)

  # Determine which column had the maximum value
  df['extreme_source'] = df[all_cols_but_t].idxmin(axis=1)

# List all columns that have at least one extreme value
columns_with_extreme = df['extreme_source'].unique()

# Generate a colormap for the columns with at least one extreme value
num_colors = len(columns_with_extreme)
colors = plt.get_cmap('tab20', num_colors)  # Using 'tab20' colormap
color_map = {column: colors(i) for i, column in enumerate(columns_with_extreme)}

# Plot max_BCD vs t with different colors for different sources
plt.figure(figsize=(18, 10))
for i,source in enumerate(columns_with_extreme):
    subset = df[df['extreme_source'] == source]
    if i%4 == 0:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker="^")
    if i%4 == 1:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker="v")
    if i%4 == 2:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker=">")
    if i%4 == 3:
        plt.scatter(subset[t_name], subset['extreme_val'], color=color_map[source], label=source, s=10, marker="<")

plt.xlabel(t_name)
plt.ylabel(y_axis)
# plt.yscale('log')
plt.title(f'{max_or_min}:{y_axis} vs {t_name} for {run_name} : A_{SphereAMin}_{SphereAMax}_B_{SphereBMin}_{SphereBMax}_C_{SphereCMin}_{SphereCMax}')
plt.legend()
plt.grid(True)  
plt.tight_layout()
if save_path is None:
   save_path = "/groups/sxs/hchaudha/rough/"
plt.savefig(f"{save_path}{run_name}_{max_or_min}:{y_axis}_vs_{t_name}_{tmin}_{tmax}_A_{SphereAMin}_{SphereAMax}_B_{SphereBMin}_{SphereBMax}_C_{SphereCMin}_{SphereCMax}.png", dpi=500)
plt.show()

# %%


# %% [markdown]
# # plots for h5 files

# %%
runs_to_plot = {}
# runs_to_plot["high_accuracy_L0"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev0_A?/Run/"
# runs_to_plot["high_accuracy_L1"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev1_A?/Run/"
# runs_to_plot["high_accuracy_L2"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev2_A?/Run/"
# runs_to_plot["high_accuracy_L3"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev3_A?/Run/"
# runs_to_plot["high_accuracy_L4"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev4_A?/Run/"
# runs_to_plot["high_accuracy_L45"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev45_A?/Run/"
# runs_to_plot["high_accuracy_L5"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L55"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev55_A?/Run/"
# runs_to_plot["high_accuracy_L6"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev6_A?/Run/"

# runs_to_plot["high_accuracy_L5_three_tier"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L5_three_tier_constra"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_constra/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L5_three_tier_constra200"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_constra_200/Lev5_A?/Run/"
# runs_to_plot["L3_step_bound_gauss_error"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_step_bound_gauss_error/Ev/Lev3_A?/Run/"
# runs_to_plot["L3_step_bound_gauss_error_rd"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L3_step_bound_gauss_error/Ev/Lev3_Ringdown/Lev3_A?/Run/"

# runs_to_plot["Lev5_big_gaussian_ah_tol10"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_ah_tol10/Lev5_A?/Run/"
# runs_to_plot["Lev5_big_gaussian_ah_tol100"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_big_gaussian_ah_tol100/Lev5_A?/Run/"
# runs_to_plot["Lev5_bg_ah100_cd_01_uamr"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_cd_01_uamr_full/Lev5_A?/Run/"
# runs_to_plot["Lev5_bg_ah100_lapse"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_lapse_full/Lev5_A?/Run/"
# runs_to_plot["Lev5_bg_ah100_lapse_uamr"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_variations/Lev5_bg_ah100_lapse_uamr_full/Lev5_A?/Run/"

# runs_to_plot["high_accuracy_L0_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev0_A?/Run/"
# runs_to_plot["high_accuracy_L1_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev1_A?/Run/"
# runs_to_plot["high_accuracy_L2_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev2_A?/Run/"
# runs_to_plot["high_accuracy_L3_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev3_A?/Run/"
# runs_to_plot["high_accuracy_L4_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev4_A?/Run/"
# runs_to_plot["high_accuracy_L45_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev45_A?/Run/"
# runs_to_plot["high_accuracy_L5_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev5_A?/Run/"
# runs_to_plot["high_accuracy_L55_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev55_A?/Run/"
# runs_to_plot["high_accuracy_L6_main"] =  "/groups/sxs/hchaudha/spec_runs/high_accuracy_L35_master/Ev/Lev6_A?/Run/"

# runs_to_plot["ode_impro_Lev0"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev0_A?/Run/'
# runs_to_plot["ode_impro_Lev1"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev1_A?/Run/'
# runs_to_plot["ode_impro_Lev2"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev2_A?/Run/'
# runs_to_plot["ode_impro_Lev3"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev3_A?/Run/'
# runs_to_plot["ode_impro_Lev4"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev4_A?/Run/'
# runs_to_plot["ode_impro_Lev5"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev5_A?/Run/'
# runs_to_plot["main_Lev0"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev0_A?/Run/'
# runs_to_plot["main_Lev2"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev2_A?/Run/'
# runs_to_plot["main_Lev1"] = '/groups/sxs/hchaudha/spec_runs/Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev1_A?/Run/'

# runs_to_plot["high_accuracy_L5"] =  "/groups/sxs/hchaudha/spec_runs/truncation_error_diagnostics/high_accuracy_L35_Lev5/"
# runs_to_plot["high_accuracy_L5_main"] =  "/groups/sxs/hchaudha/spec_runs/truncation_error_diagnostics/high_accuracy_L35_master_Lev5/"

# runs_to_plot["6_set1_L3s0"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/h5_files_Lev0/'
# runs_to_plot["6_set1_L3s1"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/h5_files_Lev1/'
# runs_to_plot["6_set1_L3s2"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/h5_files_Lev2/'
# runs_to_plot["6_set1_L3s3"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L3/h5_files_Lev3/'

runs_to_plot["6_set1_L6s4"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev4/'
runs_to_plot["6_set1_L6s5"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev5/'
runs_to_plot["6_set1_L6s6"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set1_L6/h5_files_Lev6/'

# runs_to_plot["6_set2_L6s4"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L6/h5_files_Lev4/'
# runs_to_plot["6_set2_L6s5"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L6/h5_files_Lev5/'
# runs_to_plot["6_set2_L6s6"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set2_L6/h5_files_Lev6/'

# runs_to_plot["6_set3_L6s4"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L6/h5_files_Lev4/'
# runs_to_plot["6_set3_L6s5"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L6/h5_files_Lev5/'
# runs_to_plot["6_set3_L6s6"] = '/groups/sxs/hchaudha/spec_runs/6_segs/6_set3_L6/h5_files_Lev6/'

domain = "SphereC5"

# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_ConvergenceFactor.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_HighestThirdConvergenceFactor.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_NumberOfFilteredModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_NumberOfModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_NumberOfNonFilteredNonZeroModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_NumberOfPiledUpModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_PowerInFilteredModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_PowerInHighestUnfilteredModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_PredictedTruncationErrorForOneLessMode.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_RawConvergenceFactor.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_SpectrumIsDegenerate.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf0I1_TruncationError.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_ConvergenceFactor.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_HighestThirdConvergenceFactor.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_NumberOfFilteredModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_NumberOfModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_NumberOfNonFilteredNonZeroModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_NumberOfPiledUpModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_PowerInFilteredModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_PowerInHighestUnfilteredModes.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_PredictedTruncationErrorForOneLessMode.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_RawConvergenceFactor.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_SpectrumIsDegenerate.dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Bf1S2_TruncationError.dat"

psi_or_kappa = "psi"
psi_or_kappa = "kappa"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf0I1(10 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf0I1(11 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf0I1(12 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf0I1(13 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(20 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(21 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(22 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(23 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(24 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(25 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(26 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(27 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(28 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(29 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(30 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(31 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(32 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(33 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(34 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(35 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(36 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(37 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(38 modes).dat"
# data_file_path = f"extracted-PowerDiagnostics/{domain}.dir/Power{psi_or_kappa}.dir/Bf1S2(41 modes).dat"

# data_file_path = "extracted-FilterDiagnostics/BoundaryFilters.dir/ExpChebCoef.dir/SliceLFF.SphereA0.dat"
# data_file_path = "extracted-ProjectedCon/Subdomains.dir/SphereA0.dir/NormOf2Con.dat"
data_file_path = "extracted-ProjectedCon/Subdomains.dir/SphereC5.dir/NormOf2Con.dat"
# data_file_path = "extracted-PowerDiagnostics/SphereA0.dir/Bf0I1_NumberOfModes.dat"
# data_file_path = "extracted-PowerDiagnostics/FilledCylinderEA0.dir/Bf0I1_TruncationError.dat"
# data_file_path = "extracted-RhsExpense/CostPerProc.dir/Proc00.dat"
# data_file_path = "extracted-RhsExpense/CostPerSubdomain.dir/SphereA0.dat"
# data_file_path = "extracted-RhsExpense/CostPerSubdomain.dir/SphereB2.dat"
data_file_path = "extracted-OrbitDiagnostics/OrbitalPhase.dat"
# data_file_path = "extracted-AdjustGridExtents/SphereC5.dir/Bf0I1.dat"
# data_file_path = "extracted-AdjustGridExtents/SphereA0.dir/Bf1S2.dat"
# data_file_path = "extracted-AdjustGridExtents/SphereA0.dir/Extents.dat"
# data_file_path = "extracted-AdjustGridExtents/SphereA0.dir/Size.dat"
# data_file_path = "extracted-AdjustGridExtents/SphereC5.dir/Size.dat"
# data_file_path = "extracted-ControlNthDeriv/ExpansionFactor.dir/a.dat"
# data_file_path = "extracted-ControlNthDeriv/Trans.dir/Tx.dat"
# data_file_path = "extracted-FilterDiagnostics/SubdomainFilters.dir/ExpChebCoef.dir/SphereB0.dat"

column_names, runs_data_dict = load_data_from_levs(runs_to_plot,data_file_path)
print(column_names)

# %%
moving_avg_len=0
save_path = None
diff_base = None
x_axis = 't(M)'

diff_base = '6_set1_L6s6'
# diff_base = '6_set1_L6s5'
# diff_base = '6_set1_L3s3'
# diff_base = 'high_accuracy_L5'
# diff_base = 'ode_impro_Lev3'
# add_max_and_min_val(runs_data_dict)
# y_axis = 'max_val'
# y_axis = 'min_val'

# y_axis = 'coef0'
# y_axis = 'coef1'
# y_axis = 'MaxTruncationError'
# y_axis = 'ShiftedTruncErrorMax'
# y_axis = 'TruncationErrorExcess'
# y_axis = 'MaxNextTruncationError'
# y_axis = 'MinNumberOfPiledUpModes'
# y_axis = 'GridDiagPowerkappa'
# y_axis = 'GridDiagPowerpsi'
# y_axis = 'CostPerPoint'
# y_axis = 'Cost'
# y_axis = 'NumberOfPoints'
# y_axis = 'Q'
# y_axis = 'lambda'
# y_axis = 'Bf0I1'
y_axis = 'phi'
# y_axis = 'Extent[0]'
# y_axis = 'Size'

minT = 0
minT = 1100
# minT = 2700
# minT = 9100

maxT = 40000
# maxT = 2710
maxT = 4000
# maxT = 9300
# maxT = 9375
# moving_avg_len = 50
# moving_avg_len = 10

plot_fun = lambda x,y,label : plt.plot(x,y,label=label)
# plot_fun = lambda x,y,label : plt.plot(x,y,label=label,marker='x')
# plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label)
# plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label,marker='x')
# plot_fun = lambda x,y,label : plt.loglog(x,y,label=label) 
# plot_fun = lambda x,y,label : plt.scatter(x,y,label=label,s=10,marker="x")
# save_path = "/groups/sxs/hchaudha/rough/high_acc_plots/"
# save_path = "/groups/sxs/hchaudha/rough/plots/"
# save_path = "/home/hchaudha/notes/spec_accuracy/L5_comparisons/power_diagon/"
legend_dict = {}
for key in runs_data_dict.keys():
  legend_dict[key] = None

# legend_dict = { '3_DH_q1_ns_d18_L3': "Lev3",
#                 '3_DH_q1_ns_d18_L6': "Lev6",
#                 'all_10': "Lev3_all_tols_10",
#                 'all_100': "Lev3_all_tols_100",
#                 'near_bhs_10': "Lev3_sphere_AB_tols_10",
#                 'near_bhs_100': "Lev3_sphere_AB_tols_100"}
# legend_dict = {
#  '3_DH_q1_ns_d18_L3':"Lev3_ode_tol_1e-8",
#  '3_DH_q1_ns_d18_L3_tol9':"Lev3_ode_tol_1e-9",
#  '3_DH_q1_ns_d18_L3_tol10':"Lev3_ode_tol_1e-10",
#  '3_DH_q1_ns_d18_L3_tol11':"Lev3_ode_tol_1e-11",
#  '3_DH_q1_ns_d18_L3_all_100_tol10':"Lev3_AMR_tol_100_ode_tol_1e-11",
#  }
title = data_file_path
# for key in runs_data_dict:
#   title = title + "_" + 
with plt.style.context('default'):
  plt.rcParams["figure.figsize"] = (12,10)
  plt.rcParams["figure.autolayout"] = True
  plot_graph_for_runs(runs_data_dict, x_axis, y_axis, minT, maxT, legend_dict=legend_dict, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun=plot_fun, diff_base=diff_base, title=title)
  plt.show()

# %% [markdown]
# # Plots for one

# %%
main_path = Path("/groups/sxs/hchaudha/spec_runs/")
run_path = main_path/Path("3_DH_q1_ns_d18_L3/Ev/Lev3_A?/Run/")
# run_path = main_path/Path("3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs/Lev3_AD/Run/")
# run_path = main_path/Path("3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs/Lev3_AD/Run/")
# run_path = main_path/Path("3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_9/Lev3_AD/Run/")
# run_path = main_path/Path("3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_10/Lev3_AD/Run/")
# run_path = main_path/Path("3_DH_q1_ns_d18_L3_higher_acc/all_100_t2690_obs_grid_tol_11/Lev3_AD/Run/")

data_files = {}

# data_files["GrAdjustSubChunksToDampingTimes"] = {"path":run_path/"GrAdjustSubChunksToDampingTimes.dat", "prefix":None}
# data_files["MemoryInfo"] = {"path":run_path/"MemoryInfo.dat", "prefix":None}
# data_files["MinimumGridSpacing"] = {"path":run_path/"MinimumGridSpacing.dat", "prefix":None}
# data_files["DiagInclinationAngle"] = {"path":run_path/"DiagInclinationAngle.dat", "prefix":None}
# data_files["ApparentHorizons/Trajectory_AhA"] = {"path":run_path/"ApparentHorizons/Trajectory_AhA.dat", "prefix":"AhA"}
# data_files["ApparentHorizons/MinCharSpeedAhA"] = {"path":run_path/"ApparentHorizons/MinCharSpeedAhA.dat", "prefix":"AhA"}
# data_files["ApparentHorizons/Trajectory_AhB"] = {"path":run_path/"ApparentHorizons/Trajectory_AhB.dat", "prefix":"AhB"}
# data_files["ApparentHorizons/SmoothCoordSepHorizon"] = {"path":run_path/"ApparentHorizons/SmoothCoordSepHorizon.dat", "prefix":None}
# data_files["ApparentHorizons/MinCharSpeedAhB"] = {"path":run_path/"ApparentHorizons/MinCharSpeedAhB.dat", "prefix":"AhB"}
# data_files["ApparentHorizons/RescaledRadAhB"] = {"path":run_path/"ApparentHorizons/RescaledRadAhB.dat", "prefix":"AhB"}
# data_files["ApparentHorizons/AhACoefs"] = {"path":run_path/"ApparentHorizons/AhACoefs.dat", "prefix":"AhA"}
# data_files["ApparentHorizons/AhB"] = {"path":run_path/"ApparentHorizons/AhB.dat", "prefix":"AhB"}
# data_files["ApparentHorizons/HorizonSepMeasures"] = {"path":run_path/"ApparentHorizons/HorizonSepMeasures.dat", "prefix":None}
# data_files["ApparentHorizons/AhA"] = {"path":run_path/"ApparentHorizons/AhA.dat", "prefix":"AhA"}
# data_files["ApparentHorizons/RescaledRadAhA"] = {"path":run_path/"ApparentHorizons/RescaledRadAhA.dat", "prefix":"AhA"}
# data_files["ApparentHorizons/AhBCoefs"] = {"path":run_path/"ApparentHorizons/AhBCoefs.dat", "prefix":"AhB"}
# data_files["TimeInfo"] = {"path":run_path/"TimeInfo.dat", "prefix":None}
# data_files["GrAdjustMaxTstepToDampingTimes"] = {"path":run_path/"GrAdjustMaxTstepToDampingTimes.dat", "prefix":None}
# data_files["FailedTStepperDiag"] = {"path":run_path/"FailedTStepperDiag.dat", "prefix":None}
# data_files["DiagAhSpeedA"] = {"path":run_path/"DiagAhSpeedA.dat", "prefix":"AhA"}
# data_files["DiagAhSpeedB"] = {"path":run_path/"DiagAhSpeedB.dat", "prefix":"AhB"}
# data_files["CharSpeedNorms/CharSpeeds_Max_SliceLFF.SphereA0"] = {"path":run_path/"CharSpeedNorms/CharSpeeds_Max_SliceLFF.SphereA0.dat", "prefix":"Max_A0"}
# data_files["CharSpeedNorms/CharSpeeds_Min_SliceLFF.SphereA0"] = {"path":run_path/"CharSpeedNorms/CharSpeeds_Min_SliceLFF.SphereA0.dat", "prefix":"Min_A0"}
# data_files["CharSpeedNorms/CharSpeeds_Min_SliceLFF.SphereB0"] = {"path":run_path/"CharSpeedNorms/CharSpeeds_Min_SliceLFF.SphereB0.dat", "prefix":"Max_B0"}
# data_files["CharSpeedNorms/CharSpeeds_Max_SliceLFF.SphereB0"] = {"path":run_path/"CharSpeedNorms/CharSpeeds_Max_SliceLFF.SphereB0.dat", "prefix":"Min_B0"}
# data_files["CharSpeedNorms/CharSpeeds_Max_SliceUFF.SphereC29"] = {"path":run_path/"CharSpeedNorms/CharSpeeds_Max_SliceUFF.SphereC29.dat", "prefix":"Max_C29"}
# data_files["CharSpeedNorms/CharSpeeds_Min_SliceUFF.SphereC29"] = {"path":run_path/"CharSpeedNorms/CharSpeeds_Min_SliceUFF.SphereC29.dat", "prefix":"Min_C29"}
# data_files["ConstraintNorms/NormalizedGhCe_Norms"] = {"path":run_path/"ConstraintNorms/NormalizedGhCe_Norms.dat", "prefix":None}
# data_files["ConstraintNorms/GhCeExt_Norms"] = {"path":run_path/"ConstraintNorms/GhCeExt_Norms.dat", "prefix":None}
# data_files["ConstraintNorms/GhCe_L2"] = {"path":run_path/"ConstraintNorms/GhCe_L2.dat", "prefix":None}
# data_files["ConstraintNorms/GhCeExt_L2"] = {"path":run_path/"ConstraintNorms/GhCeExt_L2.dat", "prefix":None}
# data_files["ConstraintNorms/GhCeExt"] = {"path":run_path/"ConstraintNorms/GhCeExt.dat", "prefix":None}
# data_files["ConstraintNorms/GhCe"] = {"path":run_path/"ConstraintNorms/GhCe.dat", "prefix":None}
# data_files["ConstraintNorms/GhCe_VolL2"] = {"path":run_path/"ConstraintNorms/GhCe_VolL2.dat", "prefix":None}
data_files["ConstraintNorms/GhCe_Norms"] = {"path":run_path/"ConstraintNorms/GhCe_Norms.dat", "prefix":None}
# data_files["ConstraintNorms/NormalizedGhCe_Linf"] = {"path":run_path/"ConstraintNorms/NormalizedGhCe_Linf.dat", "prefix":None}
# data_files["ConstraintNorms/GhCe_Linf"] = {"path":run_path/"ConstraintNorms/GhCe_Linf.dat", "prefix":None}
data_files["TStepperDiag"] = {"path":run_path/"TStepperDiag.dat", "prefix":None}
# data_files["DiagCutXCorrection"] = {"path":run_path/"DiagCutXCorrection.dat", "prefix":None}


# data = read_dat_file_across_AA(str(data_files['TimeInfo']['path']))
for key in data_files:
  data_files[key]["dataframe"] = read_dat_file_across_AA(str(data_files[key]['path']))
  cols = list(data_files[key]["dataframe"].columns)
  # Make new cols names s.t. the first cols is 't' and add prefix as required
  new_cols = []
  new_cols.append('t')
  if data_files[key]['prefix'] is None:
    [new_cols.append(name) for name in cols[1:]]
  else:
    [new_cols.append(name+"_"+data_files[key]['prefix']) for name in cols[1:]]
  data_files[key]["dataframe"].columns = new_cols

  # Set 't' to be a index and copy it into a column called 'time'
  # data_files[key]["dataframe"]["time"] = data_files[key]["dataframe"]["t"]
  # data_files[key]["dataframe"].set_index('t', inplace=True)

combined = None
for key in data_files:
  if combined is None:
    combined = data_files[key]["dataframe"]
    continue
  else:
    combined  = pd.merge(combined,data_files[key]["dataframe"],on='t',how='outer')


# %%
combined = combined.sort_values(by='t')
for i in combined.columns:
  print(i)

# %%
# plt.scatter(combined['t'],combined['SuggestedDampingTime'])

x_val = 't'
plot_list=[
  # ('Linf(GhCe) on SphereA0','semilogy',None,None,'x'),
  ('Linf(GhCe)','semilogy',None,None,'.'),
  ('dt','plot',1e-3,None,None),
]

for i in plot_list:
  y_val, plot_type, mul_factor, add_factor, marker = i

  label = y_val
  if mul_factor is not None:
    label = f"{label}*{mul_factor}"
  else:
    mul_factor = 1
  if add_factor is not None:
    label = f"{label}+{add_factor}"
  else:
    add_factor = 0


  match plot_type:
    case 'semilogy':
      plt.semilogy(combined[x_val],combined[y_val]*mul_factor+add_factor,marker=marker,label=label)
    case 'plot':
      plt.plot(combined[x_val],combined[y_val]*mul_factor+add_factor,marker=marker,label=label)

title = str(run_path).split('/')[-4]
save_name = str(run_path).split('/')[-4]
for i in str(run_path).split('/')[-3:-1]:
  title = title +"/" +i
  save_name = save_name+"&"+i


plt.title(title)
plt.xlabel(x_val)
plt.legend()
plt.tight_layout()
plt.savefig(f"/groups/sxs/hchaudha/rough/plots/{save_name}.png",dpi=500)
plt.show()

# %%
import pandas as pd
def check_duplicate_rows(df, subset_column):
    """
    Function to find and print duplicate rows in a DataFrame based on a specified column.
    It checks if all duplicate rows are identical or if they differ in some columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to check for duplicates.
    subset_column (str): The column to check for duplicate values.

    Returns:
    None
    """
    # Get rows with duplicate values in the subset column
    duplicate_mask = df.duplicated(subset=[subset_column], keep=False)
    duplicate_rows = df[duplicate_mask]
    
    # Sort the duplicate rows by the subset column to group them together
    duplicate_rows_sorted = duplicate_rows.sort_values(by=subset_column)
    
    # Iterate through groups of rows with the same value in the subset column
    for _, group in duplicate_rows_sorted.groupby(subset_column):
        if len(group) > 1:
            print(f"\nRows with '{subset_column}' value: {group[subset_column].iloc[0]}")
            
            # Check if all rows in the group are identical
            identical = all(group.iloc[0].equals(row) for _, row in group.iterrows())
            if identical:
                print("All rows are identical:")
                print(group)
            else:
                print("Rows differ in some columns:")
                print(group)
                
                # Optionally, show which columns differ
                differing_columns = group.columns[group.nunique() > 1].tolist()
                print(f"Columns with differences: {differing_columns}")



# Usage example
# combined = pd.read_csv('your_data.csv')  # Load your data
# check_duplicate_rows(combined, 't')

# %%
check_duplicate_rows(combined,'t')

# %%
import pandas as pd

def check_duplicate_rows(df, subset_column):
    # Get rows with duplicate values in the subset column
    duplicate_mask = df.duplicated(subset=[subset_column], keep=False)
    duplicate_rows = df[duplicate_mask]
    
    # Sort the duplicate rows by the subset column to group them together
    duplicate_rows_sorted = duplicate_rows.sort_values(by=subset_column)
    
    # Iterate through groups of rows with the same value in the subset column
    for _, group in duplicate_rows_sorted.groupby(subset_column):
        if len(group) > 1:
            print(f"\nRows with '{subset_column}' value: {group[subset_column].iloc[0]}")
            
            # Check if all rows in the group are identical
            if group.drop_duplicates().shape[0] == 1:
                print("All rows are identical:")
                print(group)
            else:
                print("Rows differ in some columns:")
                print(group)
                
                # Show which columns differ
                differing_columns = group.columns[group.nunique() > 1].tolist()
                print(f"Columns with differences: {differing_columns}")
                
                # Show the differences
                for col in differing_columns:
                    if col != subset_column:
                        print(f"\nDifferences in column '{col}':")
                        print(group[['t', col]])

# Usage example
# combined = pd.read_csv('your_data.csv')  # Load your data
check_duplicate_rows(combined, 't')

# %%
combined = combined.sort_values(by='t')

# %%
combined['MinCharSpeedAhA[7]_AhA']

# %%


# %%
col_names = []
for i in data:
  print(i.columns.is_unique)
  col_names = col_names+list(i.columns)

# %%
len(col_names), len(set(col_names))

# %%
pd.concat(data,axis=1)

# %% [markdown]
# # Linf plots

# %%
runs_to_plot = {}
runs_to_plot["t6115_tol8_linf"] =  "/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_high_tol/t6115_tol8_linf/Lev3_A?/Run/"

data_file_path="ConstraintNorms/Linf.dat"


column_names_linf, runs_data_dict_linf = load_data_from_levs(runs_to_plot,data_file_path)
print(column_names_linf)
print(runs_data_dict_linf.keys())

# %%
data = runs_data_dict_linf['t6115_tol8_linf']

# %%


# %%
domain_list = []
constraint_list = []
component_list = []

# domain_list.append("SphereB0")
domain_list.append("erMA0")

# constraint_list.append('1Con')
constraint_list.append('2Con')
# constraint_list.append('3Con')

component_list.append('t')

temp_list = copy.copy(column_names_linf)
col_domains = []
for col in temp_list:
  for domain in domain_list:
    if domain in col:
      col_domains.append(col)

temp_list = col_domains
col_domains = []
for col in temp_list:
  for constraint in constraint_list:
    if constraint in col:
      col_domains.append(col)

if len(component_list) > 0:
  temp_list = col_domains
  col_domains = []
  for col in temp_list:
    for component in component_list:
      if component in col:
        col_domains.append(col)

col_domains

# %%
moving_avg_len = None

x = 'time'
# moving_avg_len = 50*3


for col in col_domains:
  if moving_avg_len is not None:
    plt.semilogy(np.array(data[x])[moving_avg_len-1:],moving_average_valid(data[col],moving_avg_len),label=col)
  else:
    plt.semilogy(data[x],data[col],label=col)
  # plt.plot(data[x],data[col],label=col)
plt.xlabel(x)
plt.legend()
plt.show()

# %%
len(moving_average_valid(data[col],average_over))

# %% [markdown]
# # Plot horizons.h5

# %%
base_path = Path("/groups/sxs/hchaudha/spec_runs")
runs_to_plot = {}

# runs_to_plot["high_accuracy_L0"] =  "high_accuracy_L35/Ev/Lev0_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L1"] =  "high_accuracy_L35/Ev/Lev1_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L2"] =  "high_accuracy_L35/Ev/Lev2_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L3"] =  "high_accuracy_L35/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L4"] =  "high_accuracy_L35/Ev/Lev4_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L45"] =  "high_accuracy_L35/Ev/Lev45_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L5"] =  "high_accuracy_L35/Ev/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L55"] =  "high_accuracy_L35/Ev/Lev55_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L6"] =  "high_accuracy_L35/Ev/Lev6_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L3_master"] =  "high_accuracy_L35_master/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L4_master"] =  "high_accuracy_L35_master/Ev/Lev4_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L5_master"] =  "high_accuracy_L35_master/Ev/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L5_big_gauss"] =  "high_accuracy_L35_variations/Lev5_big_gaussian/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["Lev5_big_gaussian_constra"] =  "high_accuracy_L35_variations/Lev5_big_gaussian_constra/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["Lev5_big_gaussian_constra_200"] =  "high_accuracy_L35_variations/Lev5_big_gaussian_constra_200/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L5_a10"] =  "high_accuracy_L35_variations/Lev5_big_gaussian_ah_tol10/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["high_accuracy_L5_a100"] =  "high_accuracy_L35_variations/Lev5_big_gaussian_ah_tol100/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["L3_step_bound_gauss_error"] =  "high_accuracy_L3_step_bound_gauss_error/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5"

# runs_to_plot["eq_AMR_3_tier_const"] =  "high_accuracy_L3_contraints/eq_AMR_3_tier_const/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["eq_AMR_3_tier_const_gamma2"] =  "high_accuracy_L3_contraints/eq_AMR_3_tier_const_gamma2/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["three_tier_AMR_const_L3"] =  "high_accuracy_L3_contraints/three_tier_AMR_const/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["normal_constraints"]="high_accuracy_L3_contraints/eq_AMR_3_tier_const_variations/normal_constraints/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["normal_constraints_12_AB"]="high_accuracy_L3_contraints/eq_AMR_3_tier_const_variations/normal_constraints_12_AB/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
# runs_to_plot["normal_constraints_const1"]="high_accuracy_L3_contraints/eq_AMR_3_tier_const_variations/normal_constraints_const1/Lev3_A?/Run/ApparentHorizons/Horizons.h5"

# runs_to_plot["ode_impro_Lev0"] = 'Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev0_A?/Run/ApparentHorizons/Horizons.h5'
# runs_to_plot["ode_impro_Lev1"] = 'Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev1_A?/Run/ApparentHorizons/Horizons.h5'
# runs_to_plot["ode_impro_Lev2"] = 'Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev2_A?/Run/ApparentHorizons/Horizons.h5'
# runs_to_plot["ode_impro_Lev3"] = 'Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5'
# runs_to_plot["ode_impro_Lev4"] = 'Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev4_A?/Run/ApparentHorizons/Horizons.h5'
# runs_to_plot["ode_impro_Lev5"] = 'Lev01_test/new_ode_tol/high_accuracy_L35/Ev/Lev5_A?/Run/ApparentHorizons/Horizons.h5'

# runs_to_plot["main_Lev0"] = 'Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev0_A?/Run/ApparentHorizons/Horizons.h5'
# runs_to_plot["main_Lev2"] = 'Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev2_A?/Run/ApparentHorizons/Horizons.h5'
# runs_to_plot["main_Lev1"] = 'Lev01_test/old_ode_tol/high_accuracy_L35_master/Ev/Lev1_A?/Run/ApparentHorizons/Horizons.h5'


runs_to_plot["6_set1_L3s0"] =  "6_segs/6_set1_L3/Ev/Lev0_A?/Run/ApparentHorizons/Horizons.h5"
runs_to_plot["6_set1_L3s1"] =  "6_segs/6_set1_L3/Ev/Lev1_A?/Run/ApparentHorizons/Horizons.h5"
runs_to_plot["6_set1_L3s2"] =  "6_segs/6_set1_L3/Ev/Lev2_A?/Run/ApparentHorizons/Horizons.h5"
runs_to_plot["6_set1_L3s3"] =  "6_segs/6_set1_L3/Ev/Lev3_A?/Run/ApparentHorizons/Horizons.h5"

runs_to_plot["6_set1_L6s4"] =  "6_segs/6_set1_L6/Ev/Lev4_A?/Run/ApparentHorizons/Horizons.h5"
runs_to_plot["6_set1_L6s5"] =  "6_segs/6_set1_L6/Ev/Lev5_A?/Run/ApparentHorizons/Horizons.h5"
runs_to_plot["6_set1_L6s6"] =  "6_segs/6_set1_L6/Ev/Lev6_A?/Run/ApparentHorizons/Horizons.h5"

# runs_to_plot["all_100"] =  "3_DH_q1_ns_d18_L3_higher_acc/all_100/Lev3_A?/Run/ApparentHorizons/Horizons.h5"
data_dict = load_horizon_data_from_levs(base_path, runs_to_plot)
data_dict = flatten_dict(data_dict)
data_dict[list(data_dict.keys())[0]].columns

# %%
moving_avg_len = 0
save_path = None

x_axis = 't'
y_axis = 'ArealMass'
y_axis = 'ChristodoulouMass'
# y_axis = 'CoordCenterInertial_0'
# y_axis = 'CoordCenterInertial_1'
# y_axis = 'CoordCenterInertial_2'
# y_axis = 'DimensionfulInertialSpin_0'
# y_axis = 'DimensionfulInertialSpin_1'
# y_axis = 'DimensionfulInertialSpin_2'
# y_axis = 'DimensionfulInertialCoordSpin_0'
# y_axis = 'DimensionfulInertialCoordSpin_1'
# y_axis = 'DimensionfulInertialCoordSpin_2'
# y_axis = 'DimensionfulInertialSpinMag'
# y_axis = 'SpinFromShape_0'
# y_axis = 'SpinFromShape_1'
# y_axis = 'SpinFromShape_2'
# y_axis = 'SpinFromShape_3'
# y_axis = 'chiInertial_0'
# y_axis = 'chiInertial_1'
# y_axis = 'chiInertial_2'
# y_axis = 'chiMagInertial'



# moving_avg_len=25
minT = 0
minT = 1200
maxT = 25000
# maxT = 7000
maxT = 7500
maxT = 4000

plot_fun = lambda x,y,label : plt.plot(x,y,label=label)
# plot_fun = lambda x,y,label : plt.semilogy(x,y,label=label)
# plot_fun = lambda x,y,label : plt.loglog(x,y,label=label)
# plot_fun = lambda x,y,label : plt.scatter(x,y,label=label)
# save_path = "/panfs/ds09/sxs/himanshu/scripts/report/not_tracked/temp2/"

filtered_dict = {}
allowed_horizons = ["AhA"]
for horizons in allowed_horizons:
  for runs_keys in data_dict.keys():
    if horizons in runs_keys:
      filtered_dict[runs_keys] = data_dict[runs_keys]
 
with plt.style.context('default'):
  plt.rcParams["figure.figsize"] = (12,10)
  plt.rcParams["figure.figsize"] = (8,6)
  plt.rcParams["figure.autolayout"] = True
  plot_graph_for_runs(filtered_dict, x_axis, y_axis, minT, maxT, save_path=save_path, moving_avg_len=moving_avg_len, plot_fun=plot_fun)

plt.show()

# %%
bh = 'corrected_coord_spin2_AhB'
y_axis1 = 'chiInertial_0'
y_axis2 = 'CoordSpinChiInertial_0'

X = data_dict[bh][x_axis]
Y1 = data_dict[bh][y_axis1]
Y2 = data_dict[bh][y_axis2]
plt.plot(X,Y1,label=y_axis1)
plt.plot(X,Y2,label=y_axis2)
plt.xlabel(x_axis)
# plt.ylabel(y_axis1+" - "+y_axis2)
plt.legend()
# plt.title()
plt.show()

# %%
filtered_dict.keys()

# %%
x_axis = 't'
y_axis = 'ChristodoulouMass'
minT = 500
maxT = 800
run1 = filtered_dict['AccTest_q1ns_Lev5_AhA']
# run1 = filtered_dict['AccTest_q1ns_Lev6p_AhA']
run2 = filtered_dict['AccTest_q1ns_Lev6p_AhA']
interp_grid_pts = run1[x_axis].size

# %%
interp_run1 = CubicSpline(run1[x_axis].to_numpy(),run1[y_axis].to_numpy())
interp_run2 = CubicSpline(run2[x_axis].to_numpy(),run2[y_axis].to_numpy())
interp_grid = np.arange(minT,maxT,(maxT-minT)/interp_grid_pts)

plt.plot(interp_grid, interp_run2(interp_grid) - interp_run1(interp_grid))
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()

# %%
plt.plot(interp_grid, interp_run2(interp_grid) - interp_run1(interp_grid))

# %%
def inertial_dist(run_name:str, data_dict):
    ct = data_dict[f"{run_name}_AhA"].t
    dx = data_dict[f"{run_name}_AhA"].CoordCenterInertial_0 - data_dict[f"{run_name}_AhB"].CoordCenterInertial_0
    dy = data_dict[f"{run_name}_AhA"].CoordCenterInertial_1 - data_dict[f"{run_name}_AhB"].CoordCenterInertial_1
    dz = data_dict[f"{run_name}_AhA"].CoordCenterInertial_2 - data_dict[f"{run_name}_AhB"].CoordCenterInertial_2

    dx = np.sqrt(dx**2 + dy**2 + dz**2)

    return ct,dx


# %%
for run_name in runs_to_plot.keys():
    ct,dx = inertial_dist(run_name,data_dict)
    plt.plot(ct,dx,label=run_name)
    plt.legend()

# %%
print(data_dict.keys())
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(data_dict['76_ngd_master_mr1_50_3000_AhA'].describe())

# %% [markdown]
# # Combine all paraview files into a single file

# %%
base_folder = Path("/central/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_L3_higher_acc/near_bhs_100_obs/")

Lev = 3
file_pattern =f"Lev{Lev}_A[A-Z]/Run/GaugeVis.pvd"
file_patternGrid =f"Lev{Lev}_A[A-Z]/Run/GaugeVisGrid.pvd"
file_patternAll =f"Lev{Lev}_A[A-Z]/Run/GuageVisAll.pvd"

combine_pvd_files(base_folder,file_pattern)
combine_pvd_files(base_folder,file_patternGrid)
combine_pvd_files(base_folder,file_patternAll)

# %% [markdown]
# ```Python
# # Create GaugeVis
# command = f"cd {base_folder} && mkdir ./GaugeVis"
# status = subprocess.run(command, capture_output=True, shell=True, text=True)
# if status.returncode == 0:
#   print(f"Succesfully created GaugeVis in {base_folder}")
# else:
#   sys.exit(
#       f"GaugeVis creation failed in {base_folder} with error: \n {status.stderr}")
# 
# # Create GaugeVis subfolder
# vtu_folder_path = base_folder+"/GaugeVis/GaugeVis"
# command = f"mkdir {vtu_folder_path}"
# status = subprocess.run(command, capture_output=True, shell=True, text=True)
# if status.returncode == 0:
#   print(f"Succesfully created {vtu_folder_path}")
# else:
#   sys.exit(
#       f"GaugeVis creation failed as {vtu_folder_path} with error: \n {status.stderr}")
# 
# 
# # Copy vtu files
# GaugeVisFolder=[]
# 
# for paths in path_collection:
#   GaugeVisFolder.append(paths[:-4])
# 
# for paths in GaugeVisFolder:
#   command = f"cp {paths}/*.vtu {vtu_folder_path}/"
#   status = subprocess.run(command, capture_output=True, shell=True, text=True)
#   if status.returncode == 0:
#     print(f"Succesfully copied vtu files from {paths}")
#   else:
#     sys.exit(
#         f"Copying vtu files from {paths} failed with error: \n {status.stderr}")
# 
# ```

# %% [markdown]
# # Profiler results

# %% [markdown]
# # Make report (do not run randomly)

# %%
with open("./report_new_gauge.json") as report_data:
  data = json.load(report_data)

os.mkdir(data['report_folder'])

subfolders = []
for folders in data['runs_to_track']:
  subfolders_path = data['report_folder'] + "/" + path_to_folder_name(folders) + "/"
  print(subfolders_path)
  os.mkdir(subfolders_path)
  subfolders.append(subfolders_path)

# %%
runs_still_going_on = True
while runs_still_going_on:
  # time.sleep(data['report_generation_frequency'])

  for i,run_folder_path in enumerate(data['runs_to_track']):
    # if is_the_current_run_going_on(run_folder_path) or True:
    if True:
      plots_for_a_folder(data['things_to_plot'],subfolders[i],run_folder_path)
    print(run_folder_path)


  runs_still_going_on = False
  print("all done")


# %% [markdown]
# ### Save all columns and data files paths

# %%
# Write all the cols in the dat files for reference
lev_golb="/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/gauge_driver_kerr_target_50_50_0_16_16_01/Ev/Lev1_AA"
dat_files_glob=lev_golb+"/Run/**/**.dat"
path_pattern = dat_files_glob

path_collection = []
for folder_name in glob.iglob(path_pattern, recursive=True):
    if os.path.isdir(folder_name) or os.path.isfile(folder_name):
        path_collection.append(folder_name)
        print(folder_name.split("/")[-1])


column_data_for_dat_files = {
  'columns_of_dat_files' : [
  ] 
}

for file_path in path_collection:
  file_name = file_path.split("/")[-1]
  columns_list =  list(read_dat_file(file_path).columns)
  column_data_for_dat_files['columns_of_dat_files'].append({
    'file_name': file_name,
    'file_path': file_path,
    'columns': columns_list
  })


with open('./column_data_for_dat_files.json', 'w') as outfile:
  json.dump(column_data_for_dat_files, outfile, indent=2)

# %%
def JoinH5(h5_file_list, output_path, output_file_name):

  file_list_to_str = ""
  for h5file in h5_file_list:
    file_list_to_str += h5file + " "

  command = f"cd {output_path} && {spec_home}/Support/bin/JoinH5 -o {output_file_name} {file_list_to_str}"
  status = subprocess.run(command, capture_output=True, shell=True, text=True)
  if status.returncode == 0:
    print(f"Succesfully ran JoinH5 in {output_path}")
  else:
    sys.exit(
        f"JoinH5 failed in {output_path} with error: \n {status.stderr}")


def ExtractFromH5(h5_file, output_path):

  command = f"cd {output_path} && {spec_home}/Support/bin/ExtractFromH5 {h5_file}"
  status = subprocess.run(command, capture_output=True, shell=True, text=True)
  if status.returncode == 0:
    print(f"Succesfully ran ExtractFromH5 in {output_path}")
  else:
    sys.exit(
        f"ExtractFromH5 failed in {output_path} with error: \n {status.stderr}")



# %%
output_base_path= "/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/profiler_results"


base_folder = "/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/49_ngd_weird_gauge_mr1"
file_pattern = base_folder+"/Ev/Lev1_A?/Run/Profiler.h5"

path_pattern = file_pattern
path_collection = []

# make a folder in the output directory
save_folder = output_base_path+"/"+base_folder.split("/")[-1]
os.mkdir(save_folder)


# Find all the files that match the required pattern of the file
for folder_name in glob.iglob(path_pattern, recursive=True):
    if os.path.isdir(folder_name) or os.path.isfile(folder_name):
        path_collection.append(folder_name)
        print(folder_name)

JoinH5(path_collection,save_folder,"Profiler.h5")
ExtractFromH5("Profiler.h5",save_folder)

# Save path of all the summary files in extracted data

file_pattern = base_folder+"/Ev/Lev1_A?/Run/Profiler.h5"

path_pattern = file_pattern
path_collection = []

# Find all the files that match the required pattern of the file
for folder_name in glob.iglob(path_pattern, recursive=True):
    if os.path.isdir(folder_name) or os.path.isfile(folder_name):
        path_collection.append(folder_name)
        print(folder_name)

# %%
# Find all the Summary files 
summary_file_pattern = save_folder+"/**/Summary.txt"
summary_file_collection = []

for file_path in glob.iglob(summary_file_pattern, recursive=True):
    if os.path.isdir(file_path) or os.path.isfile(file_path):
        summary_file_collection.append(file_path)
        print(file_path)

summary_file_collection.sort()

# %%
file_path = "/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/profiler_results/49_ngd_weird_gauge_mr1/extracted-Profiler/Step10522.dir/Summary.txt"



# %% [markdown]
# ## AmrTolerances.input

# %%
Lev=3
TruncationErrorMax = 0.000216536 * 4**(-Lev)
ProjectedConstraintsMax = 0.216536 * 4**(-Lev)
TruncationErrorMaxA = TruncationErrorMax*1.e-4
TruncationErrorMaxB = TruncationErrorMax*1.e-4

AhMaxRes  = TruncationErrorMax
AhMinRes  = AhMaxRes / 10.0

AhMaxTrunc=TruncationErrorMax
AhMinTrunc=AhMaxTrunc / 100.0

print(f"AhMinRes={AhMinRes};")
print(f"AhMaxRes={AhMaxRes};")
print(f"AhMinTrunc={AhMinTrunc};")
print(f"AhMaxTrunc={AhMaxTrunc};")
print(f"TruncationErrorMax={TruncationErrorMax};")
print(f"TruncationErrorMaxA={TruncationErrorMaxA};")
print(f"TruncationErrorMaxB={TruncationErrorMaxB};")
print(f"ProjectedConstraintsMax={ProjectedConstraintsMax};")


# %%
def ode_tol_val(Lev):
  TruncationErrorMax = 0.000216536 * 4**(-Lev)
  ode_tol = TruncationErrorMax/2000
  return ode_tol

for i in [0,1,2,3,4,4.5,5,5.5,6]:
  print(i,"->",ode_tol_val(i))

# %%
#SBATCH --reservation=sxs_standing

# %% [markdown]
# # Simulating time steps

# %%
class time_step_simulator():
  def __init__(self,average=0.05,variation_frac=0.1,running_avg_bound_N=50, step_growth_frac=0.07,chunk_time_diff=1.5, chunk_step_modify=3,chunk_start=1.5) -> None:
    self.average = average
    self.variation_frac = variation_frac

    self.ode_error_estimate = 1e-8
    self.ode_error_estimate_variation_fraction = 0.1 #How much can the error change from step to step
    self.ode_tolerance = 1e-8
    self.ode_order = 4

    # Time step  tracking
    self.time_till_now = 0
    self.time_steps = []
    self.time_after_step = []

    # Chunking stuff
    self.chunk_time_diff = chunk_time_diff
    self.chunk_step_modify = chunk_step_modify
    self.next_chunk_time = chunk_start
    
    # Running average stuff
    self.running_average_step_size = average
    self.step_growth_frac = step_growth_frac
    self.running_avg_bound_N = running_avg_bound_N
    self.running_average_step_size_arr = []


  def take_step(self):
    curr_step = self.average + (2*np.random.rand()-1)*self.variation_frac*self.average

    # Bound the step size growth
    if curr_step > self.running_average_step_size*(1+self.step_growth_frac):
      curr_step = self.running_average_step_size*(1+self.step_growth_frac)

    steps_left = (self.next_chunk_time-self.time_till_now)/curr_step
    if steps_left <=0:
      raise ValueError(f"{steps_left=} is an invalid value.")

    if steps_left < self.chunk_step_modify:
      if (steps_left < 1+1e-14):
        # Take exact time step to hit the chunk
        curr_step = (self.next_chunk_time-self.time_till_now)
        # set the next chunk time
        self.next_chunk_time = self.next_chunk_time + self.chunk_time_diff
      else:
        # We need to decrease the time steps to hit the chunk
        curr_step = (self.next_chunk_time-self.time_till_now)*1.05/(np.floor(steps_left)+1.0)

    avg_frac = (self.running_avg_bound_N-1)/self.running_avg_bound_N
    self.running_average_step_size = self.running_average_step_size*avg_frac + (1-avg_frac)*curr_step
    self.running_average_step_size_arr.append(self.running_average_step_size)

    self.time_till_now = self.time_till_now + curr_step
    self.time_after_step.append(self.time_till_now)
    self.time_steps.append(curr_step)

  def take_steps(self,num_steps=100):
    for i in range(num_steps):
      self.take_step()




# %%
stepper = time_step_simulator(variation_frac=0.1,chunk_step_modify=3)
stepper.take_steps(100)

# %%
plt.plot(stepper.time_after_step, stepper.time_steps,marker="x")
plt.plot(stepper.time_after_step, stepper.running_average_step_size_arr,marker="x")
# plt.plot(stepper.time_after_step, stepper.time_steps)
# plt.plot(stepper.time_after_step, stepper.running_average_step_size_arr)
plt.show()

# %% [markdown]
# # Extract h5 files

# %%
Ev_path = Path("/groups/sxs/hchaudha/spec_runs/Lev01_test/new_ode_tol/high_accuracy_L35/Ev")
save_run_path = Path("/groups/sxs/hchaudha/spec_runs/del.txt")

# %%
with save_run_path.open('w') as f:
  path_list = list(Ev_path.glob("Lev*_A?/Run"))
  path_list.sort()
  ringdown_path_list = list(Ev_path.glob("Lev*_Ringdown/Lev*_A?/Run"))
  ringdown_path_list.sort()
  for i in (path_list+ringdown_path_list):
    if not os.path.islink(i.parent): # If Lev5_AA is symlink then do not write
      f.write(f"{i}\n")
  # for i in Ev_path.glob("Lev?_A?/Run/extrac*"):
    # f.write(f"rm -r {i}\n")

# %% [markdown]
# ```bash
# #!/bin/bash
# 
# folder=$1
# 
# cd $1
# echo "Running in the folder: $folder"
# /home/hchaudha/spec/Support/bin/ExtractFromH5 -o . -strippath ./*.h5
# # cat del.txt | xargs -P 8 -I {} ./del.sh {}
# ```

# %%
os.path.islink(Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev6_AA")),os.path.islink(Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/Ev/Lev6_AA/Run"))

# %% [markdown]
# # Hist-GrDomain

# %%
def parse_ResizeTheseSubdomains(input_str):
    # Remove the initial part before '=' and trailing semicolon
    input_str = input_str.split('=', 1)[1].strip(';')
    
    # Use regex to find all subdomain entries
    pattern = r'(\w+(?:\.\d+)*)\(Extents=\((\d+,\d+,\d+)\)\)'
    matches = re.findall(pattern, input_str)
    
    # Initialize the dictionary
    subdomains_dict = {}

    for name, extents in matches:
        # Convert the extents string into a tuple of integers
        extents = tuple(map(int, extents.split(',')))
        
        # Add to the dictionary
        subdomains_dict[name] = extents
    
    return subdomains_dict

def find_max_extents_for_domain_type(parsed_dict, domain_type):
  def max_vals_in_a_tuple(t1, t2):
    return tuple(max(a, b) for a, b in zip(t1, t2))
  max_extents = (0, 0, 0)
  for domain_name in parsed_dict:
    if domain_type in domain_name:
      max_extents = max_vals_in_a_tuple(max_extents, parsed_dict[domain_name])
  return max_extents


def AMR_input_content_spheres(sphere_name, MinExtent0, MinL, DoNotChangeBeforeTime,TruncationErrorMax=None):
  if TruncationErrorMax is None:
    amr_str = f"""{sphere_name}(MinExtent0={MinExtent0};MinL={MinL};
          DoNotChangeExtent0BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeLBeforeRadiusPlusTime = {DoNotChangeBeforeTime};
        ),
"""
  else:
    amr_str = f"""{sphere_name}(MinExtent0={MinExtent0};MinL={MinL};
          DoNotChangeExtent0BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeLBeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          TruncationErrorMax = {TruncationErrorMax};
          Center = 0,0,0;
        ),
"""
  return amr_str

def AMR_input_content_cylinder(cylinder_name, MinExtent0, MinExtent1, MinExtent2, DoNotChangeBeforeTime,TruncationErrorMax=None):
  if TruncationErrorMax is None:
    amr_str = f"""{cylinder_name}*(MinExtent0={MinExtent0};MinExtent1={MinExtent1};MinExtent2={MinExtent2};
          DoNotChangeExtent0BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeExtent1BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeExtent2BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
        ),
"""
  else:
    amr_str = f"""{cylinder_name}*(MinExtent0={MinExtent0};MinExtent1={MinExtent1};MinExtent2={MinExtent2};
          DoNotChangeExtent0BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeExtent1BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeExtent2BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          TruncationErrorMax = {TruncationErrorMax};
          Center = 0,0,0;
        ),
"""
  return amr_str

def make_odd(n):
  if n%2==0:
    return n+1
  else:
    return n

def AMR_input_content_spheres_min_max(sphere_name, MinExtent0, MinL, DoNotChangeBeforeTime):
  amr_str = f"""{sphere_name}(MinExtent0={MinExtent0};MinL={MinL};
        MaxExtent0={MinExtent0};MaxL={MinL};
        SplitAfterMaxExtent0IsReached=False;
        SplitAfterMaxLIsReached=False;
        DoNotChangeExtent0BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
        DoNotChangeLBeforeRadiusPlusTime = {DoNotChangeBeforeTime};
      ),
"""
  return amr_str

def AMR_input_content_cylinder_min_max(cylinder_name, MinExtent0, MinExtent1, MinExtent2, DoNotChangeBeforeTime):
  MinExtent0, MinExtent1, MinExtent2 = make_odd(MinExtent0), make_odd(MinExtent1), make_odd(MinExtent2)
  amr_str = f"""{cylinder_name}*(MinExtent0={MinExtent0};MinExtent1={MinExtent1};MinExtent2={MinExtent2};
          MaxExtent0={MinExtent0};MaxExtent1={MinExtent1};MaxExtent2={MinExtent2};
          SplitAfterMaxExtent0IsReached=False;
          SplitAfterMaxExtent1IsReached=False;
          SplitAfterMaxExtent2IsReached=False;
          DoNotChangeExtent0BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeExtent1BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
          DoNotChangeExtent2BeforeRadiusPlusTime = {DoNotChangeBeforeTime};
        ),
"""
  return amr_str

def adjust_parsed_dict_for_lev(parsed_dict, lev):
  adjustment = int(lev - 3)
  for domain_name in parsed_dict:
    parsed_dict[domain_name] = tuple([int(i+adjustment) for i in parsed_dict[domain_name]])
    if "Sphere" in domain_name:
      extents = parsed_dict[domain_name]
      # For spheres M is always 2*L
      parsed_dict[domain_name] = (extents[0], extents[1], 2*extents[1])
  return parsed_dict

input_str_base_set3_lev3 = "ResizeTheseSubdomains=SphereA0(Extents=(8,18,36)),SphereA1(Extents=(8,19,38)),SphereA2(Extents=(8,20,40)),SphereA3(Extents=(8,21,42)),SphereA4(Extents=(8,22,44)),SphereB0(Extents=(8,18,36)),SphereB1(Extents=(8,19,38)),SphereB2(Extents=(8,20,40)),SphereB3(Extents=(8,21,42)),SphereB4(Extents=(8,22,44)),SphereC0(Extents=(15,15,30)),SphereC1(Extents=(15,15,30)),SphereC2(Extents=(15,15,30)),SphereC3(Extents=(15,15,30)),SphereC4(Extents=(15,15,30)),SphereC5(Extents=(15,15,30)),SphereC6(Extents=(15,15,30)),SphereC7(Extents=(15,15,30)),SphereC8(Extents=(15,15,30)),SphereC9(Extents=(15,15,30)),SphereC10(Extents=(15,15,30)),SphereC11(Extents=(15,16,32)),SphereC12(Extents=(15,16,32)),SphereC13(Extents=(15,16,32)),SphereC14(Extents=(15,16,32)),SphereC15(Extents=(15,16,32)),SphereC16(Extents=(15,15,30)),SphereC17(Extents=(15,15,30)),SphereC18(Extents=(15,15,30)),SphereC19(Extents=(15,15,30)),SphereC20(Extents=(15,15,30)),SphereC21(Extents=(15,15,30)),SphereC22(Extents=(15,15,30)),SphereC23(Extents=(15,16,32)),SphereC24(Extents=(15,16,32)),SphereC25(Extents=(15,16,32)),SphereC26(Extents=(15,16,32)),SphereC27(Extents=(15,16,32)),SphereC28(Extents=(15,16,32)),SphereC29(Extents=(15,16,32)),CylinderEB0.0.0(Extents=(10,23,15)),CylinderEB1.0.0(Extents=(12,23,15)),CylinderCB0.0.0(Extents=(13,23,16)),CylinderCB1.0.0(Extents=(13,23,15)),CylinderEA0.0.0(Extents=(10,23,15)),CylinderEA1.0.0(Extents=(12,23,15)),CylinderCA0.0.0(Extents=(13,23,16)),CylinderCA1.0.0(Extents=(13,23,15)),FilledCylinderEB0(Extents=(10,9,21)),FilledCylinderEB1(Extents=(10,9,21)),FilledCylinderCB0(Extents=(11,8,21)),FilledCylinderCB1(Extents=(12,8,21)),FilledCylinderMB0(Extents=(10,9,19)),FilledCylinderMB1(Extents=(12,9,19)),CylinderSMB0.0(Extents=(10,21,11)),CylinderSMB1.0(Extents=(12,21,12)),FilledCylinderEA0(Extents=(10,9,21)),FilledCylinderEA1(Extents=(10,9,21)),FilledCylinderCA0(Extents=(11,8,21)),FilledCylinderCA1(Extents=(12,8,21)),FilledCylinderMA0(Extents=(10,9,19)),FilledCylinderMA1(Extents=(12,9,19)),CylinderSMA0.0(Extents=(10,21,11)),CylinderSMA1.0(Extents=(12,21,12));"
parsed_dict_set3_lev3 = parse_ResizeTheseSubdomains(input_str_base_set3_lev3)

input_str_base_set1_lev3 = "ResizeTheseSubdomains=SphereA0(Extents=(8,18,36)),SphereA1(Extents=(8,19,38)),SphereA2(Extents=(8,20,40)),SphereA3(Extents=(8,21,42)),SphereA4(Extents=(8,22,44)),SphereB0(Extents=(8,18,36)),SphereB1(Extents=(8,19,38)),SphereB2(Extents=(8,20,40)),SphereB3(Extents=(8,21,42)),SphereB4(Extents=(8,22,44)),SphereC0(Extents=(15,15,30)),SphereC1(Extents=(15,15,30)),SphereC2(Extents=(15,15,30)),SphereC3(Extents=(15,15,30)),SphereC4(Extents=(15,15,30)),SphereC5(Extents=(15,15,30)),SphereC6(Extents=(15,15,30)),SphereC7(Extents=(15,15,30)),SphereC8(Extents=(15,15,30)),SphereC9(Extents=(15,15,30)),SphereC10(Extents=(15,16,32)),SphereC11(Extents=(15,16,32)),SphereC12(Extents=(15,16,32)),SphereC13(Extents=(15,16,32)),SphereC14(Extents=(15,16,32)),SphereC15(Extents=(15,15,30)),SphereC16(Extents=(15,15,30)),SphereC17(Extents=(15,15,30)),SphereC18(Extents=(15,15,30)),SphereC19(Extents=(15,15,30)),SphereC20(Extents=(15,15,30)),SphereC21(Extents=(15,15,30)),SphereC22(Extents=(15,16,32)),SphereC23(Extents=(15,16,32)),SphereC24(Extents=(15,16,32)),SphereC25(Extents=(15,16,32)),SphereC26(Extents=(15,16,32)),SphereC27(Extents=(15,16,32)),SphereC28(Extents=(15,16,32)),SphereC29(Extents=(15,16,32)),CylinderEB0.0.0(Extents=(10,23,15)),CylinderEB1.0.0(Extents=(12,23,15)),CylinderCB0.0.0(Extents=(13,23,16)),CylinderCB1.0.0(Extents=(13,23,15)),CylinderEA0.0.0(Extents=(10,23,15)),CylinderEA1.0.0(Extents=(12,23,15)),CylinderCA0.0.0(Extents=(13,23,16)),CylinderCA1.0.0(Extents=(13,23,15)),FilledCylinderEB0(Extents=(10,9,21)),FilledCylinderEB1(Extents=(10,9,21)),FilledCylinderCB0(Extents=(11,8,21)),FilledCylinderCB1(Extents=(12,8,21)),FilledCylinderMB0(Extents=(10,9,19)),FilledCylinderMB1(Extents=(12,9,19)),CylinderSMB0.0(Extents=(10,21,11)),CylinderSMB1.0(Extents=(12,21,12)),FilledCylinderEA0(Extents=(10,9,21)),FilledCylinderEA1(Extents=(10,9,21)),FilledCylinderCA0(Extents=(11,8,21)),FilledCylinderCA1(Extents=(12,8,21)),FilledCylinderMA0(Extents=(10,9,19)),FilledCylinderMA1(Extents=(12,9,19)),CylinderSMA0.0(Extents=(10,21,11)),CylinderSMA1.0(Extents=(12,21,12));"
parsed_dict_set1_lev3 = parse_ResizeTheseSubdomains(input_str_base_set1_lev3)

input_str_base_lev6 = "ResizeTheseSubdomains=SphereA0(Extents=(11,23,46)),SphereA1(Extents=(11,24,48)),SphereA2(Extents=(11,25,50)),SphereA3(Extents=(11,26,52)),SphereA4(Extents=(13,27,54)),SphereB0(Extents=(11,22,44)),SphereB1(Extents=(11,23,46)),SphereB2(Extents=(11,24,48)),SphereB3(Extents=(11,25,50)),SphereB4(Extents=(13,26,52)),SphereC0(Extents=(18,18,36)),SphereC1(Extents=(18,17,34)),SphereC2(Extents=(18,18,36)),SphereC3(Extents=(18,18,36)),SphereC4(Extents=(18,18,36)),SphereC5(Extents=(18,19,38)),SphereC6(Extents=(18,19,38)),SphereC7(Extents=(18,19,38)),SphereC8(Extents=(18,19,38)),SphereC9(Extents=(18,19,38)),SphereC10(Extents=(18,19,38)),SphereC11(Extents=(18,19,38)),SphereC12(Extents=(18,19,38)),SphereC13(Extents=(18,19,38)),SphereC14(Extents=(18,19,38)),SphereC15(Extents=(18,19,38)),SphereC16(Extents=(18,20,40)),SphereC17(Extents=(18,20,40)),SphereC18(Extents=(18,20,40)),SphereC19(Extents=(18,20,40)),SphereC20(Extents=(18,20,40)),SphereC21(Extents=(18,20,40)),SphereC22(Extents=(18,21,42)),SphereC23(Extents=(18,21,42)),SphereC24(Extents=(18,21,42)),SphereC25(Extents=(18,22,44)),SphereC26(Extents=(18,22,44)),SphereC27(Extents=(18,22,44)),SphereC28(Extents=(18,22,44)),SphereC29(Extents=(18,21,42)),CylinderEB0.0.0(Extents=(14,29,19)),CylinderEB1.0.0(Extents=(16,29,21)),CylinderCB0.0.0(Extents=(17,29,20)),CylinderCB1.0.0(Extents=(17,27,19)),CylinderEA0.0.0(Extents=(14,29,19)),CylinderEA1.0.0(Extents=(16,29,21)),CylinderCA0.0.0(Extents=(17,29,20)),CylinderCA1.0.0(Extents=(17,27,19)),FilledCylinderEB0(Extents=(13,11,29)),FilledCylinderEB1(Extents=(14,11,29)),FilledCylinderCB0(Extents=(15,11,29)),FilledCylinderCB1(Extents=(16,10,27)),FilledCylinderMB0(Extents=(14,11,27)),FilledCylinderMB1(Extents=(16,11,23)),CylinderSMB0.0(Extents=(15,27,15)),CylinderSMB1.0(Extents=(16,25,15)),FilledCylinderEA0(Extents=(13,11,29)),FilledCylinderEA1(Extents=(14,11,29)),FilledCylinderCA0(Extents=(15,11,29)),FilledCylinderCA1(Extents=(16,10,27)),FilledCylinderMA0(Extents=(14,11,27)),FilledCylinderMA1(Extents=(16,11,23)),CylinderSMA0.0(Extents=(15,27,15)),CylinderSMA1.0(Extents=(16,25,15));"
parsed_dict_lev6 = parse_ResizeTheseSubdomains(input_str_base_lev6)

# %%
input_str = "ResizeTheseSubdomains=SphereA0(Extents=(8,18,36)),SphereA1(Extents=(8,19,38)),SphereA2(Extents=(8,20,40)),SphereA3(Extents=(8,21,42)),SphereA4(Extents=(8,22,44)),SphereB0(Extents=(8,18,36)),SphereB1(Extents=(8,19,38)),SphereB2(Extents=(8,20,40)),SphereB3(Extents=(8,21,42)),SphereB4(Extents=(8,22,44)),SphereC0(Extents=(15,15,30)),SphereC1(Extents=(15,15,30)),SphereC2(Extents=(15,15,30)),SphereC3(Extents=(15,15,30)),SphereC4(Extents=(15,15,30)),SphereC5(Extents=(15,15,30)),SphereC6(Extents=(15,15,30)),SphereC7(Extents=(15,15,30)),SphereC8(Extents=(15,15,30)),SphereC9(Extents=(15,15,30)),SphereC10(Extents=(15,16,32)),SphereC11(Extents=(15,16,32)),SphereC12(Extents=(15,16,32)),SphereC13(Extents=(15,16,32)),SphereC14(Extents=(15,16,32)),SphereC15(Extents=(15,15,30)),SphereC16(Extents=(15,15,30)),SphereC17(Extents=(15,15,30)),SphereC18(Extents=(15,15,30)),SphereC19(Extents=(15,15,30)),SphereC20(Extents=(15,15,30)),SphereC21(Extents=(15,15,30)),SphereC22(Extents=(15,16,32)),SphereC23(Extents=(15,16,32)),SphereC24(Extents=(15,16,32)),SphereC25(Extents=(15,16,32)),SphereC26(Extents=(15,16,32)),SphereC27(Extents=(15,16,32)),SphereC28(Extents=(15,16,32)),SphereC29(Extents=(15,16,32)),CylinderEB0.0.0(Extents=(10,23,15)),CylinderEB1.0.0(Extents=(12,23,15)),CylinderCB0.0.0(Extents=(13,23,16)),CylinderCB1.0.0(Extents=(13,23,15)),CylinderEA0.0.0(Extents=(10,23,15)),CylinderEA1.0.0(Extents=(12,23,15)),CylinderCA0.0.0(Extents=(13,23,16)),CylinderCA1.0.0(Extents=(13,23,15)),FilledCylinderEB0(Extents=(10,9,21)),FilledCylinderEB1(Extents=(10,9,21)),FilledCylinderCB0(Extents=(11,8,21)),FilledCylinderCB1(Extents=(12,8,21)),FilledCylinderMB0(Extents=(10,9,19)),FilledCylinderMB1(Extents=(12,9,19)),CylinderSMB0.0(Extents=(10,21,11)),CylinderSMB1.0(Extents=(12,21,12)),FilledCylinderEA0(Extents=(10,9,21)),FilledCylinderEA1(Extents=(10,9,21)),FilledCylinderCA0(Extents=(11,8,21)),FilledCylinderCA1(Extents=(12,8,21)),FilledCylinderMA0(Extents=(10,9,19)),FilledCylinderMA1(Extents=(12,9,19)),CylinderSMA0.0(Extents=(10,21,11)),CylinderSMA1.0(Extents=(12,21,12));"

DoNotChangeBeforeTime = 40000
Lev = 3

parsed_dict = parse_ResizeTheseSubdomains(input_str)
adjust_parsed_dict_for_lev(parsed_dict, Lev) 

max_extent_dict = {
  # Domains that are not spheres will all have the maximum extents
  "FilledCylinderCA" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderCA"),
  "CylinderCA" : find_max_extents_for_domain_type(parsed_dict,"CylinderCA"),
  "FilledCylinderEA" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderEA"),
  "CylinderEA" : find_max_extents_for_domain_type(parsed_dict,"CylinderEA"),
  "CylinderSMA" : find_max_extents_for_domain_type(parsed_dict,"CylinderSMA"),
  "FilledCylinderMA" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderMA"),
  "FilledCylinderMB" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderMB"),
  "CylinderSMB" : find_max_extents_for_domain_type(parsed_dict,"CylinderSMB"),
  "CylinderEB" : find_max_extents_for_domain_type(parsed_dict,"CylinderEB"),
  "FilledCylinderEB" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderEB"),
  "CylinderCB" : find_max_extents_for_domain_type(parsed_dict,"CylinderCB"),
  "FilledCylinderCB" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderCB"),
}

for key,extent in parsed_dict.items():
  if 'Sphere' in key:
    max_extent_dict[key] = extent

for key in max_extent_dict:
  if "Sphere" in key:
    if "SphereC" in key:
      if "SphereC0" in key:
        extents = max_extent_dict["SphereC0"]
        print(AMR_input_content_spheres("SphereC*", extents[0], extents[1], DoNotChangeBeforeTime=DoNotChangeBeforeTime))
      continue
    print(AMR_input_content_spheres(key, max_extent_dict[key][0], max_extent_dict[key][1], DoNotChangeBeforeTime=DoNotChangeBeforeTime))
  elif "Cylinder" in key:
    print(AMR_input_content_cylinder(key, max_extent_dict[key][0], max_extent_dict[key][1], max_extent_dict[key][2], DoNotChangeBeforeTime=DoNotChangeBeforeTime))


# %%
input_str = "ResizeTheseSubdomains=SphereA0(Extents=(8,18,36)),SphereA1(Extents=(8,19,38)),SphereA2(Extents=(8,20,40)),SphereA3(Extents=(8,21,42)),SphereA4(Extents=(8,22,44)),SphereB0(Extents=(8,18,36)),SphereB1(Extents=(8,19,38)),SphereB2(Extents=(8,20,40)),SphereB3(Extents=(8,21,42)),SphereB4(Extents=(8,22,44)),SphereC0(Extents=(15,15,30)),SphereC1(Extents=(15,15,30)),SphereC2(Extents=(15,15,30)),SphereC3(Extents=(15,15,30)),SphereC4(Extents=(15,15,30)),SphereC5(Extents=(15,15,30)),SphereC6(Extents=(15,15,30)),SphereC7(Extents=(15,15,30)),SphereC8(Extents=(15,15,30)),SphereC9(Extents=(15,15,30)),SphereC10(Extents=(15,16,32)),SphereC11(Extents=(15,16,32)),SphereC12(Extents=(15,16,32)),SphereC13(Extents=(15,16,32)),SphereC14(Extents=(15,16,32)),SphereC15(Extents=(15,15,30)),SphereC16(Extents=(15,15,30)),SphereC17(Extents=(15,15,30)),SphereC18(Extents=(15,15,30)),SphereC19(Extents=(15,15,30)),SphereC20(Extents=(15,15,30)),SphereC21(Extents=(15,15,30)),SphereC22(Extents=(15,16,32)),SphereC23(Extents=(15,16,32)),SphereC24(Extents=(15,16,32)),SphereC25(Extents=(15,16,32)),SphereC26(Extents=(15,16,32)),SphereC27(Extents=(15,16,32)),SphereC28(Extents=(15,16,32)),SphereC29(Extents=(15,16,32)),CylinderEB0.0.0(Extents=(10,23,15)),CylinderEB1.0.0(Extents=(12,23,15)),CylinderCB0.0.0(Extents=(13,23,16)),CylinderCB1.0.0(Extents=(13,23,15)),CylinderEA0.0.0(Extents=(10,23,15)),CylinderEA1.0.0(Extents=(12,23,15)),CylinderCA0.0.0(Extents=(13,23,16)),CylinderCA1.0.0(Extents=(13,23,15)),FilledCylinderEB0(Extents=(10,9,21)),FilledCylinderEB1(Extents=(10,9,21)),FilledCylinderCB0(Extents=(11,8,21)),FilledCylinderCB1(Extents=(12,8,21)),FilledCylinderMB0(Extents=(10,9,19)),FilledCylinderMB1(Extents=(12,9,19)),CylinderSMB0.0(Extents=(10,21,11)),CylinderSMB1.0(Extents=(12,21,12)),FilledCylinderEA0(Extents=(10,9,21)),FilledCylinderEA1(Extents=(10,9,21)),FilledCylinderCA0(Extents=(11,8,21)),FilledCylinderCA1(Extents=(12,8,21)),FilledCylinderMA0(Extents=(10,9,19)),FilledCylinderMA1(Extents=(12,9,19)),CylinderSMA0.0(Extents=(10,21,11)),CylinderSMA1.0(Extents=(12,21,12));"

DoNotChangeBeforeTime = 40000
Lev = 6

parsed_dict = parse_ResizeTheseSubdomains(input_str)
adjust_parsed_dict_for_lev(parsed_dict, Lev) 

max_extent_dict = {
  # Domains that are not spheres will all have the maximum extents
  "FilledCylinderCA" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderCA"),
  "CylinderCA" : find_max_extents_for_domain_type(parsed_dict,"CylinderCA"),
  "FilledCylinderEA" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderEA"),
  "CylinderEA" : find_max_extents_for_domain_type(parsed_dict,"CylinderEA"),
  "CylinderSMA" : find_max_extents_for_domain_type(parsed_dict,"CylinderSMA"),
  "FilledCylinderMA" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderMA"),
  "FilledCylinderMB" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderMB"),
  "CylinderSMB" : find_max_extents_for_domain_type(parsed_dict,"CylinderSMB"),
  "CylinderEB" : find_max_extents_for_domain_type(parsed_dict,"CylinderEB"),
  "FilledCylinderEB" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderEB"),
  "CylinderCB" : find_max_extents_for_domain_type(parsed_dict,"CylinderCB"),
  "FilledCylinderCB" : find_max_extents_for_domain_type(parsed_dict,"FilledCylinderCB"),
}

for key,extent in parsed_dict.items():
  if 'Sphere' in key:
    max_extent_dict[key] = extent

for key in max_extent_dict:
  if "Sphere" in key:
    if "SphereC" in key:
      if "SphereC0" in key:
        extents = max_extent_dict["SphereC0"]
        print(AMR_input_content_spheres_min_max("SphereC*", extents[0], extents[1], DoNotChangeBeforeTime=DoNotChangeBeforeTime))
      continue
    print(AMR_input_content_spheres_min_max(key, max_extent_dict[key][0], max_extent_dict[key][1], DoNotChangeBeforeTime=DoNotChangeBeforeTime))
  elif "Cylinder" in key:
    print(AMR_input_content_cylinder_min_max(key, max_extent_dict[key][0], max_extent_dict[key][1], max_extent_dict[key][2], DoNotChangeBeforeTime=DoNotChangeBeforeTime))


# %% [markdown]
# # Rough

# %%
runs_data_dict.keys()

# %%
data = runs_data_dict['high_accuracy_L5']

# filtered_cols = [i.split(" ")[-1] for i in data.columns if 'SphereA' in i]
filtered_cols = ['t(M)']+[i for i in data.columns if 'e' in i]
filtered_cols

# %%


# %%


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data creation (same as in the previous example)
np.random.seed(0)
data = {
    'time': np.arange(0, 10, 1),
    'velocity': np.random.uniform(50, 100, 10),
    'acceleration': np.random.uniform(-5, 5, 10),
    'fuel_left': np.random.uniform(100, 200, 10),
    'burning_rate': np.random.uniform(0, 10, 10)
}
df = pd.DataFrame(data)

# Normalize the data to [0, 1]
normalized_df = (df - df.min()) / (df.max() - df.min())

# Remove time for visualization
visual_data = normalized_df.drop('time', axis=1)

# Plot using imshow
plt.figure(figsize=(12, 6))
plt.imshow(visual_data, aspect='auto', cmap='viridis', origin='lower')

# Set x-ticks and labels
xtick_positions = np.arange(len(visual_data.columns))
plt.xticks(ticks=xtick_positions, labels=visual_data.columns, rotation=90)
# plt.xticks(ticks=np.arange(len(visual_data.columns)), labels=visual_data.columns, rotation=90)
plt.yticks(ticks=np.arange(len(visual_data)), labels=normalized_df['time'].astype(int))

plt.colorbar(label='Normalized Value')
# plt.xlabel('Features')
plt.ylabel('Time')
plt.title('Feature Intensity Over Time')
plt.tight_layout()
plt.show()

# %%
x = np.arange(100)/100
y1 = np.sin(x)
y2 = np.cos(x)

styles =  plt.style.available

for style in styles:
    print(style)
    plt.style.use(style)
    plt.plot(x,y1,label="y1asfasd")
    plt.plot(x,y2,label="y3asfasd")
    plt.title("asdf")
    plt.legend()
    plt.savefig(f"/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/make_report_del/{style}.png")
    plt.close()

# %%
w = 4
print(np.convolve(np.arange(1,10),np.ones(w),'valid')/w)
print(np.arange(1,10))

# %%
def moving_average(array,avg_len):
    return np.convolve(array,np.ones(avg_len))/avg_len

# %% [markdown]
# 

# %%
def gauss(x,cen,scale):
  return np.exp(-(x-cen)**2/scale**2)

# %%
d0 = 18
q = 1
d1 = 1/(1+q)*d0
d2 = q/(1+q)*d0
cen1 = d1
cen2 = -d2
scale1 = d2
scale2 = d1

tol1 = 3.38337e-10
tol2 = 3.38337e-10
tol_base = 3.38337e-6

d = 100
x_min,x_max = -d/2,d/2
X = np.linspace(x_min,x_max,1000)

fac_base = np.ones_like(X)
fac1 = gauss(X,cen1,scale1)
fac2 = gauss(X,cen2,scale2)

fac_total = fac_base+fac1+fac2
val = fac_base*np.log10(tol_base) + fac1*np.log10(tol1)+fac2*np.log10(tol2)
val = 10**(val/fac_total)
print(val.min(),val.max())

# %%
d0 = 18
q = 1
cen1 = 0
cen2 = 0
scale1 = 100
scale2 = 1000

tol1 = 3.38337e-10
tol2 = 3.38337e-9
tol_base = 3.38337e-6

d = 1000
x_min,x_max = -d/2,d/2
X = np.linspace(x_min,x_max,1000)

fac_base = np.ones_like(X)
fac1 = gauss(X,cen1,scale1)
fac2 = gauss(X,cen2,scale2)

fac_total = fac_base+fac1+fac2
val = fac_base*np.log10(tol_base) + fac1*np.log10(tol1)+fac2*np.log10(tol2)
val = 10**(val/fac_total)
print(val.min(),val.max())

# %%
0.00000216536 * 4**(-3)/2

# %%
f = plt.semilogy
f(X,val,label='tot')
# f(X,fac1*tol1,label='tol1')
# f(X,fac2*tol2,label='tol2')
# f(X,fac_base*tol_base,label='tol_base')
hor_lines = [val.min(),val[int(len(val)/2)],val.max()]
plt.hlines(y=hor_lines, xmin=-d/2, xmax=d/2, colors=['r', 'g', 'b'], linestyles='--', linewidth=2)
for y_value in hor_lines:
    plt.text(0, y_value*1.1, f'{y_value:.3e}', va='center', ha='left')


plt.ylabel('AMR tolerance')
plt.xlabel('x_axis')
plt.legend()
plt.show()

# %%
plt.semilogy(X,gauss(X,0,50))
plt.show()

# %%
np.exp(-4)

# %%
def hist_files_to_dataframe(file_path):
  # Function to parse a single line and return a dictionary of values
  def parse_line(line):
      data = {}
      # Find all variable=value pairs
      pairs = re.findall(r'([^;=\s]+)=\s*([^;]+)', line)
      for var, val in pairs:
          # Hist-GrDomain.txt should be parsed a little differently
          if 'ResizeTheseSubdomains' in var:
              items = val.split('),')
              items[-1] = items[-1][:-1]
              for item in items:
                name,_,vals = item.split("(")
                r,l,m=vals[:-1].split(',')
                data[f"{name}_R"] = int(r)
                data[f"{name}_L"] = int(l)
                data[f"{name}_M"] = int(m)
          else:
              data[var] = float(val) if re.match(r'^[\d.e+-]+$', val) else val
      return data
  
  with open(file_path, 'r') as file:
    # Parse the lines
    data = []
    for line in file.readlines():
        data.append(parse_line(line.strip()))

    # Create a DataFrame
    df = pd.DataFrame(data)

  return df

hist_files_to_dataframe('/groups/sxs/hchaudha/spec_runs/2_SpKS_q1_sA_0_0_9_sB_0_0_9_d15/Ev/Lev3_AB/Run/Hist-GrDomain.txt')


# %%
hist_files_to_dataframe('/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_etl_tol_10/Lev3_AD/Run_old/Hist-GrDomain.txt')


# %%
hist_files_to_dataframe('/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_etl_tol_10/Lev3_AD/Run_old/Hist-GrDomain.txt')
# hist_files_to_dataframe('/groups/sxs/hchaudha/spec_runs/2_SpKS_q1_sA_0_0_9_sB_0_0_9_d15/Ev/Lev3_AB/Run/Hist-FuncLambdaFactorB.txt')

# %%
import pandas as pd
import re

# Function to parse a single line and return a dictionary of values
def parse_line(line):
    data = {}
    # Find all variable=value pairs
    pairs = re.findall(r'([^;=]+)=\s*([\d.e+-]+)', line)
    for var, val in pairs:
        data[var] = float(val) if re.match(r'[\d.e+-]+', val) else val
    return data

# Read the file
with open('/groups/sxs/hchaudha/spec_runs/2_SpKS_q1_sA_0_0_9_sB_0_0_9_d15/Ev/Lev3_AB/Run/Hist-FuncLambdaFactorB.txt', 'r') as file:
    lines = file.readlines()

# Parse the lines
data = []
for line in lines:
    data.append(parse_line(line.strip()))

# Create a DataFrame
df = pd.DataFrame(data)

print(df)


# %%
import pandas as pd
import re

# The input string
data = """SphereA0(Extents=(12,22,44)),SphereA1(Extents=(10,23,46)),SphereA2(Extents=(10,24,48)),SphereA3(Extents=(10,25,50)),SphereA4(Extents=(13,26,52)),SphereB0(Extents=(12,22,44)),SphereB1(Extents=(10,23,46)),SphereB2(Extents=(10,24,48)),SphereB3(Extents=(10,25,50)),SphereB4(Extents=(13,26,52)),SphereC0(Extents=(15,15,30)),SphereC1(Extents=(15,15,30)),SphereC2(Extents=(15,14,28)),SphereC3(Extents=(15,15,30)),SphereC4(Extents=(15,15,30)),SphereC5(Extents=(15,15,30)),SphereC6(Extents=(15,15,30)),SphereC7(Extents=(15,16,32)),SphereC8(Extents=(15,15,30)),SphereC9(Extents=(15,16,32)),SphereC10(Extents=(15,16,32)),SphereC11(Extents=(15,15,30)),SphereC12(Extents=(15,15,30)),SphereC13(Extents=(15,15,30)),SphereC14(Extents=(15,15,30)),SphereC15(Extents=(15,15,30)),SphereC16(Extents=(16,15,30)),SphereC17(Extents=(16,16,32)),SphereC18(Extents=(16,16,32)),SphereC19(Extents=(16,16,32)),SphereC20(Extents=(15,16,32)),SphereC21(Extents=(15,16,32)),SphereC22(Extents=(15,16,32)),SphereC23(Extents=(15,16,32)),SphereC24(Extents=(15,15,30)),SphereC25(Extents=(15,16,32)),SphereC26(Extents=(15,16,32)),SphereC27(Extents=(15,16,32)),SphereC28(Extents=(15,16,32)),SphereC29(Extents=(15,16,32)),CylinderEB0.0.0(Extents=(13,31,19)),CylinderEB1.0.0(Extents=(17,25,18)),CylinderCB0.0.0(Extents=(17,23,17)),CylinderCB1.0.0(Extents=(14,21,15)),CylinderEA0.0.0(Extents=(13,31,19)),CylinderEA1.0.0(Extents=(14,25,18)),CylinderCA0.0.0(Extents=(17,23,18)),CylinderCA1.0.0(Extents=(14,21,15)),FilledCylinderEB0(Extents=(12,11,25)),FilledCylinderEB1(Extents=(12,10,25)),FilledCylinderCB0(Extents=(12,9,21)),FilledCylinderCB1(Extents=(12,8,19)),FilledCylinderMB0(Extents=(14,11,25)),FilledCylinderMB1(Extents=(16,10,21)),CylinderSMB0.0(Extents=(14,27,15)),CylinderSMB1.0(Extents=(18,25,15)),FilledCylinderEA0(Extents=(12,11,25)),FilledCylinderEA1(Extents=(12,10,25)),FilledCylinderCA0(Extents=(12,9,21)),FilledCylinderCA1(Extents=(12,8,19)),FilledCylinderMA0(Extents=(14,11,25)),FilledCylinderMA1(Extents=(14,10,21)),CylinderSMA0.0(Extents=(14,27,15)),CylinderSMA1.0(Extents=(15,25,15))"""

# Split the string into individual items
items = data.split('),')

# # Function to parse each item
# def parse_item(item):
#     name, values = re.match(r'(\w+)\(Extents=\((.*?)\)', item).groups()
#     r, l, m = map(int, values.split(','))
#     return {'Name': name, 'R': r, 'L': l, 'M': m}

# # Parse all items
# parsed_data = [parse_item(item) for item in items]

# # Create DataFrame
# df = pd.DataFrame(parsed_data)

# # Set 'Name' as index
# df.set_index('Name', inplace=True)

# # Create the specific variables for SphereA2
# SphereA2_R = df.loc['SphereA2', 'R']
# SphereA2_L = df.loc['SphereA2', 'L']
# SphereA2_M = df.loc['SphereA2', 'M']

# print(df)
# print(f"\nSphereA2_R = {SphereA2_R}")
# print(f"SphereA2_L = {SphereA2_L}")
# print(f"SphereA2_M = {SphereA2_M}")

# %%
items = data.split('),')
name,_,vals = items[0].split("(")
r,l,m=vals[:-1].split(',')
{
  name+"_R":r,
  name+"_L":l,
  name+"_M":m
}

# %%
items[0].split("(")

# %%
vals[:-1].split(',')

# %%
folder_path = Path("/groups/sxs/hchaudha/spec_runs")
del_path = Path("/groups/sxs/hchaudha/spec_runs/del.sh")

with del_path.open('w') as f:
  for i in folder_path.iterdir():
    if i.is_dir():
      if "ID" in str(i):
        continue
      if "del" in str(i):
        continue
      f.writelines(str(i)+"\n")

# %%
def write_dir(folder_path:Path,del_path_opened):
  for i in folder_path.iterdir():
    if i.is_dir():
      print(i)
      if "ID" in str(i):
        continue
      if "del" in str(i):
        continue
      if "Lev" in str(i) and i.is_symlink():
        continue
      if "Run" == i.name:
        del_path_opened.writelines(str(i)+"\n")
        return
      write_dir(i,del_path_opened)

folder_path = Path("/groups/sxs/hchaudha/spec_runs")
# folder_path = Path("/groups/sxs/hchaudha/spec_runs/3_DH_q1_ns_d18_all_100_t2690/all_100_t2690_eteq_tol_8")
del_path = Path("/groups/sxs/hchaudha/spec_runs/del.sh")
with del_path.open('w') as f:
  write_dir(folder_path, f)

# %%
folder_path.name == "asd"

# %%
a = np.linspace(0.02,0.07,100)
a = np.array([0.03,0.05])
b = np.exp(-a*10)
plt.plot(a,b)
plt.plot(a,b**3)
plt.show()

# %%
np.exp(-0.03*10)**3,np.exp(-0.05*10)**3

# %%
import pickle

file = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/path_dict.pkl")
with file.open('rb') as f:
  data = pickle.load(f)

# %%
asd = data['Lev3_R0050'].sort()

# %%
data['Lev3_R0050']

# %%
for i in range(8):
  amr_tol = 0.000216536 * 4**(-i)
  ode_tol = amr_tol/2000


# %%
file_path = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev3_R0200/Lev3_R0200.h5")
file_path = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev4_R0200/Lev4_R0200.h5")
file_path = Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35/cce_bondi/Lev5_R0050/Lev5_R0050.h5")

# %%
with h5py.File(file_path,'r') as f:
    names = []
    f.visit(names.append)
    f.visit(print)
    data = np.array(f['Beta.dat'])
    # print(np.array(data),np.array(data).shape)

print(names)

# %%
plt.plot(data[:,0])
plt.show()

# %%
a = 6.76675e-09
for i in range(7):
  print(i-2,a*4**(4-i))


