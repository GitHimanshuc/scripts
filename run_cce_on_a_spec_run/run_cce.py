# %%
import numpy as np
import pandas as pd
import subprocess
import sys
from numba import njit
import matplotlib.pyplot as plt
import os
import glob
# plt.style.use('seaborn-talk')
plt.rcParams["figure.figsize"] = (12,10)
import json
import re
import time
import pickle
from pathlib import Path
import scri
import h5py
import sxs
import scipy
import scipy.integrate as integrate
spec_home="/home/hchaudha/spec"

# %% [markdown]
# ## Functions to convert the data to scri format

# %%
def make_config_file(BoundaryDataPath: Path,
                     InputSavePath: Path = None,
                     VolumeFilePostFix: str = None) -> Path :
        
    if InputSavePath is None:
        InputSavePath = BoundaryDataPath.parent/(str(BoundaryDataPath.stem)+".yaml")
    assert(InputSavePath.parent.exists())

    if VolumeFilePostFix is None:
        VolumeFilePostFix = "_VolumeData"

    config_file=\
f"""
# Distributed under the MIT License.
# See LICENSE.txt for details.

# This block is used by testing and the SpECTRE command line interface.
Executable: CharacteristicExtract
Testing:
  Check: parse
  Priority: High

---
Evolution:
  InitialTimeStep: 0.25
  InitialSlabSize: 10.0

ResourceInfo:
  AvoidGlobalProc0: false
  Singletons: Auto

Observers:
  VolumeFileName: "vol_{str(InputSavePath.stem)+VolumeFilePostFix}"
  ReductionFileName: "red_{str(InputSavePath.stem)+VolumeFilePostFix}"

EventsAndTriggers:
  # Write the CCE time step every Slab. A Slab is a fixed length of simulation
  # time and is not influenced by the dynamically adjusted step size.
  - Trigger:
      Slabs:
        EvenlySpaced:
          Offset: 0
          Interval: 1
    Events:
      - ObserveTimeStep:
          # The output is written into the "ReductionFileName" HDF5 file under
          # "/SubfileName.dat"
          SubfileName: CceTimeStep
          PrintTimeToTerminal: true

Cce:
  Evolution:
    TimeStepper:
      AdamsBashforth:
        Order: 3 # Going to higher order doesn't seem necessary for CCE
    StepChoosers:
      - Constant: 0.1 # Don't take steps bigger than 0.1M
      - Increase:
          Factor: 2
      - ErrorControl(SwshVars):
          AbsoluteTolerance: 1e-8
          RelativeTolerance: 1e-6
          # These factors control how much the time step is changed at once.
          MaxFactor: 2
          MinFactor: 0.25
          # How close to the "perfect" time step we take. Since the "perfect"
          # value assumes a linear system, we need some safety factor since our
          # system is nonlinear, and also so that we reduce how often we retake
          # time steps.
          SafetyFactor: 0.9
      - ErrorControl(CoordVars):
          AbsoluteTolerance: 1e-8
          RelativeTolerance: 1e-7
          # These factors control how much the time step is changed at once.
          MaxFactor: 2
          MinFactor: 0.25
          # How close to the "perfect" time step we take. Since the "perfect"
          # value assumes a linear system, we need some safety factor since our
          # system is nonlinear, and also so that we reduce how often we retake
          # time steps.
          SafetyFactor: 0.9

  # The number of angular modes used by the CCE evolution. This must be larger
  # than ObservationLMax. We always use all of the m modes for the LMax since
  # using fewer m modes causes aliasing-driven instabilities.
  LMax: 20
  # Probably don't need more than 15 radial grid points, but could increase
  # up to ~20
  NumberOfRadialPoints: 15
  # The maximum ell we use for writing waveform output. While CCE can dump
  # more, you should be cautious with higher modes since mode mixing, truncation
  # error, and systematic numerical effects can have significant contamination
  # in these modes.
  ObservationLMax: 8

  InitializeJ:
    # To see what other J-initialization procedures are available, comment
    # out this group of options and do, e.g. "Blah:" The code will print
    # an error message with the available options and a help string.
    # More details can be found at spectre-code.org.
    ConformalFactor:
      AngularCoordTolerance: 1e-13
      MaxIterations: 1000 # Do extra iterations in case we improve.
      RequireConvergence: False # Often don't converge to 1e-13, but that's fine
      OptimizeL0Mode: True
      UseBetaIntegralEstimate: False
      ConformalFactorIterationHeuristic: SpinWeight1CoordPerturbation
      UseInputModes: False
      InputModes: []

  StartTime: Auto
  EndTime: Auto
  ExtractionRadius: Auto

  BoundaryDataFilename: {BoundaryDataPath.name}
  H5IsBondiData: True
  H5Interpolator:
    BarycentricRationalSpanInterpolator:
      MinOrder: 10
      MaxOrder: 10
  FixSpecNormalization: False

  H5LookaheadTimes: 10000

  Filtering:
    RadialFilterHalfPower: 64
    RadialFilterAlpha: 35.0
    FilterLMax: 18

  ScriInterpOrder: 5
  ScriOutputDensity: 1

"""

    with InputSavePath.open('w') as f:
        f.writelines(config_file)

    return InputSavePath

def make_submit_file(path_dict:dict):
    path_dict['submit_script_paths'] = []
    base_path = path_dict['base_path']
    for cce_folder_name in path_dict['cce_paths_keys']:
        run_name = f"{cce_folder_name}_{base_path.stem}"
        run_path = base_path/"cce_bondi"/cce_folder_name
        input_file_name = cce_folder_name+".yaml"
        

        submit_script=\
f"""#!/bin/bash -
#SBATCH -J CCE_{run_name}              # Job Name
#SBATCH -o CCE.stdout                # Output file name
#SBATCH -e CCE.stderr                # Error file name
#SBATCH -n 2                          # Number of cores
#SBATCH -p expansion                  # Queue name
#SBATCH --ntasks-per-node 2        # number of MPI ranks per node
#SBATCH -t 24:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue
#SBATCH --reservation=sxs_standing
#SBATCH --constraint=skylake

# Go to the correct folder with the boundary data
cd {run_path}

# run CCE
{path_dict["CCE_Executable"]} --input-file ./{input_file_name}
"""
        submit_script_path = base_path/"cce_bondi"/cce_folder_name/"submit.sh"
        path_dict['submit_script_paths'].append(submit_script_path)

        # with submit_script_path.open('w') as f:
        #     f.writelines(submit_script)
        submit_script_path.write_text(submit_script)
    
def submit_all_jobs(path_dict:dict):
    for submit_script_path in path_dict['submit_script_paths']:
        command = f"cd {submit_script_path.parent} && qsub {submit_script_path}"
        status = subprocess.run(command, capture_output=True, shell=True, text=True)
        if status.returncode == 0:
          print(f"Succesfully submitted {submit_script_path}\n{status.stdout}")
        else:
          sys.exit(
              f"Job submission failed for {submit_script_path} with error: \n{status.stdout} \n{status.stderr}")

# %% [markdown]
# ## Functions to deal with combining the cce data

# %%
# 'Lev_list': ['1','3']
def add_levs(path_dict):
  base_path = path_dict["base_path"]
  path_dict["Lev_list"] = []
  for lev in range(10):
    if (base_path/f"Ev/Lev{lev}_AA").exists():
      path_dict["Lev_list"].append(f"{lev}")

# 'cce_radius': ['0112', '0540', '0397', '0255']
def add_cce_radius(path_dict):
  some_lev = path_dict["Lev_list"][0]
  cce_list = list(path_dict["base_path"].glob(f"Ev/Lev{some_lev}_AA/Run/GW2/BondiCceR????.h5"))
  path_dict["cce_radius"] = [file.name[9:-3] for file in cce_list]


# 'cce_paths_keys': ['Lev_1_radius_0112', 'Lev_1_radius_0255']
# 'Lev1_R0112': [path_list],
# 'Lev1_R0255': [path_list],
def add_cce_data_paths(path_dict):
  path_dict["cce_paths_keys"]=[]
  for lev in path_dict["Lev_list"]:
    for radius in path_dict["cce_radius"]:
      key_name = f"Lev{lev}_R{radius}"
      path_dict["cce_paths_keys"].append(key_name)
      path_dict[key_name] = list(path_dict["base_path"].glob(f"Ev/Lev{lev}_??/Run/GW2/BondiCceR{radius}.h5"))


# create directories to save cce waveforms
def create_folders_to_save_cce_data(path_dict):
  for cce_lev_radius in path_dict["cce_paths_keys"]:
    folder_to_create = path_dict["base_path"]/f"cce_bondi/{cce_lev_radius}"
    folder_to_create.mkdir(parents=True,exist_ok=True)


def run_JoinH5(save_folder,h5_file_list,output_file_name):
  file_list_str = ""
  for file_path in h5_file_list:
    file_list_str += f" {file_path}"

  command = f"cd {save_folder} && {spec_home}/Support/bin/JoinH5 -o {output_file_name} {file_list_str}"
  status = subprocess.run(command, capture_output=True, shell=True, text=True)
  if status.returncode == 0:
    print(f"Succesfully saved joined h5 file {output_file_name} in {save_folder}")
  else:
    sys.exit(
        f"JoinH5 failed in {save_folder} with error: \n {status.stderr}")

# Combines h5 files for different levs and radius
def save_joined_cce_h5_files(path_dict,cce_paths_keys_list=None):
  if cce_paths_keys_list is None:
    cce_paths_keys_list = path_dict["cce_paths_keys"]
    
  for cce_lev_radius in cce_paths_keys_list:
    save_folder = str(path_dict["base_path"]/f"cce_bondi/{cce_lev_radius}")
    h5_file_list = path_dict[cce_lev_radius]
    output_file_name = cce_lev_radius+".h5"

    # check that the outputfile is not already present
    output_file_path = path_dict["base_path"]/f"cce_bondi/{cce_lev_radius}/{output_file_name}"
    if output_file_path.exists():
      print(f"File {output_file_path} already exisits. Doing nothing!!!")
    else:
      run_JoinH5(save_folder,h5_file_list,output_file_name)
      
# Makes input files for CCE for each radius of CCE
def make_config_files_in_all_folders(path_dict:dict):
    path_dict['config_file_paths'] = []
    for bd_path in path_dict['boundary_data_paths']:
        path_dict['config_file_paths'].append(make_config_file(bd_path))

# Saves the path of the combined boundary data files into path_dict
def save_boundary_data_paths(path_dict):
  path_dict['boundary_data_paths'] = list(path_dict['base_path'].glob("cce_bondi/*/*.h5"))

def pickle_path_dict(path_dict):
  with open(path_dict['base_path']/"cce_bondi/path_dict.pkl",'wb') as f:
    pickle.dump(path_dict,f)

# %% [markdown]
# ## Function to do it all

# %%
def do_CCE(run_path_list: list, CCE_executable: Path, submit_jobs=True):
    for base_path in run_path_list:
        path_dict = {"base_path":Path(base_path)}
        path_dict['CCE_Executable'] = CCE_executable
        
        add_levs(path_dict)
        add_cce_radius(path_dict)
        add_cce_data_paths(path_dict)
        create_folders_to_save_cce_data(path_dict)
        save_joined_cce_h5_files(path_dict)
        save_boundary_data_paths(path_dict)
        pickle_path_dict(path_dict)

        # make cce input files and submit the job
        make_config_files_in_all_folders(path_dict)          
        make_submit_file(path_dict)
        if submit_jobs:
          submit_all_jobs(path_dict)

runs_paths = [
    Path("/groups/sxs/hchaudha/spec_runs/high_accuracy_L35")
]

CCE_executable = Path("/groups/sxs/hchaudha/spec_runs/CCE_stuff/CceExecutables/CharacteristicExtract")
do_CCE(runs_paths,CCE_executable)