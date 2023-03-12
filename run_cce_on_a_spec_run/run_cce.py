import numpy as np
import pandas as pd
import subprocess
import sys
import os
import glob
import json
import time
import pickle
from pathlib import Path
spec_home="/home/himanshu/spec/my_spec"

def RunCCE(CCE_executable: Path, BoundaryDataPath: Path, VolumeFilePostFix: str = None)-> Path:
    assert (CCE_executable.exists())
    assert (BoundaryDataPath.exists())

    if VolumeFilePostFix is None:
        VolumeFilePostFix = "_Data"

    # Input file that will be created
    InputSavePath = BoundaryDataPath.parent/(str(BoundaryDataPath.stem)+".yaml")

    # Create input file
    make_config_file(BoundaryDataPath,InputSavePath,VolumeFilePostFix)

    command = f"cd {InputSavePath.parent} && {CCE_executable} +p8 --input-file {InputSavePath}"
    status = subprocess.run(command,
                            capture_output=True,
                            shell=True,
                            text=True)
    if status.returncode == 0:  
        print(f"Succesfully ran CCE for file {BoundaryDataPath.name}")
    else:
        sys.exit(
            f"CCE failed for file {BoundaryDataPath.name} with error: \n {status.stderr}"
        )
    # Return the path of the output
    return list(BoundaryDataPath.parent.glob(f"*{BoundaryDataPath.stem}*{VolumeFilePostFix}*.h5"))

def make_config_file_inverse_cube(BoundaryDataPath: Path,
                     InputSavePath: Path = None,
                     VolumeFilePostFix: str = None) -> Path :
        
    if InputSavePath is None:
        InputSavePath = BoundaryDataPath.parent/(str(BoundaryDataPath.stem)+".yaml")
    assert(InputSavePath.parent.exists())

    if VolumeFilePostFix is None:
        VolumeFilePostFix = "_Data"

    config_file=\
f"""
# Distributed under the MIT License.
# See LICENSE.txt for details.

# Executable: CharacteristicExtract
# Check: parse

Evolution:
  InitialTimeStep: 0.25
  InitialSlabSize: 10.0

ResourceInfo:
  AvoidGlobalProc0: false
  Singletons:
    CharacteristicEvolution:
      Proc: Auto
      Exclusive: False
    H5WorldtubeBoundary:
      Proc: Auto
      Exclusive: False

Observers:
  VolumeFileName: {str(InputSavePath.stem)+VolumeFilePostFix}
  ReductionFileName: {str(InputSavePath.stem)+VolumeFilePostFix}

Cce:
  Evolution:
    TimeStepper:
      AdamsBashforth:
        Order: 3
    StepChoosers:
      - Constant: 0.5
      - Increase:
          Factor: 2
      - ErrorControl(SwshVars):
          AbsoluteTolerance: 1e-8
          RelativeTolerance: 1e-6
          MaxFactor: 2
          MinFactor: 0.25
          SafetyFactor: 0.9
      - ErrorControl(CoordVars):
          AbsoluteTolerance: 1e-8
          RelativeTolerance: 1e-7
          MaxFactor: 2
          MinFactor: 0.25
          SafetyFactor: 0.9

  LMax: 20
  NumberOfRadialPoints: 12
  ObservationLMax: 8

  InitializeJ:
    InverseCubic

  StartTime: Auto
  EndTime: Auto
  BoundaryDataFilename: {BoundaryDataPath.name}
  H5IsBondiData: False
  H5Interpolator:
    BarycentricRationalSpanInterpolator:
      MinOrder: 10
      MaxOrder: 10
  ExtractionRadius: 257.0
  FixSpecNormalization: False

  H5LookaheadTimes: 10000

  Filtering:
    RadialFilterHalfPower: 24
    RadialFilterAlpha: 35.0
    FilterLMax: 18

  ScriInterpOrder: 5
  ScriOutputDensity: 5

"""

    with InputSavePath.open('w') as f:
        f.writelines(config_file)

    return InputSavePath        

def make_config_file(BoundaryDataPath: Path,
                     InputSavePath: Path = None,
                     VolumeFilePostFix: str = None) -> Path :
        
    if InputSavePath is None:
        InputSavePath = BoundaryDataPath.parent/(str(BoundaryDataPath.stem)+".yaml")
    assert(InputSavePath.parent.exists())

    if VolumeFilePostFix is None:
        VolumeFilePostFix = "_Data"

    config_file=\
f"""
# Distributed under the MIT License.
# See LICENSE.txt for details.

# Executable: CharacteristicExtract
# Check: parse

Evolution:
  InitialTimeStep: 0.25
  InitialSlabSize: 10.0

ResourceInfo:
  AvoidGlobalProc0: false
  Singletons:
    CharacteristicEvolution:
      Proc: Auto
      Exclusive: False
    H5WorldtubeBoundary:
      Proc: Auto
      Exclusive: False

Observers:
  VolumeFileName: {str(InputSavePath.stem)+VolumeFilePostFix}
  ReductionFileName: "CharacteristicExtractUnusedReduction"

Cce:
  Evolution:
    TimeStepper:
      AdamsBashforth:
        Order: 3
    StepChoosers:
      - Constant: 0.5
      - Increase:
          Factor: 2
      - ErrorControl(SwshVars):
          AbsoluteTolerance: 1e-8
          RelativeTolerance: 1e-6
          MaxFactor: 2
          MinFactor: 0.25
          SafetyFactor: 0.9
      - ErrorControl(CoordVars):
          AbsoluteTolerance: 1e-8
          RelativeTolerance: 1e-7
          MaxFactor: 2
          MinFactor: 0.25
          SafetyFactor: 0.9

  LMax: 20
  NumberOfRadialPoints: 12
  ObservationLMax: 8

  InitializeJ:
    ConformalFactor:
      AngularCoordTolerance: 1e-13
      MaxIterations: 1000
      RequireConvergence: False
      OptimizeL0Mode: True
      UseBetaIntegralEstimate: False
      ConformalFactorIterationHeuristic: SpinWeight1CoordPerturbation
      UseInputModes: False
      InputModes: []

  StartTime: Auto
  EndTime: Auto
  BoundaryDataFilename: {BoundaryDataPath.name}
  H5IsBondiData: False
  H5Interpolator:
    BarycentricRationalSpanInterpolator:
      MinOrder: 10
      MaxOrder: 10
  ExtractionRadius: 257.0
  FixSpecNormalization: False

  H5LookaheadTimes: 10000

  Filtering:
    RadialFilterHalfPower: 24
    RadialFilterAlpha: 35.0
    FilterLMax: 18

  ScriInterpOrder: 5
  ScriOutputDensity: 5

"""

    with InputSavePath.open('w') as f:
        f.writelines(config_file)

    return InputSavePath


def make_submit_file(path_dict:dict):
    path_dict['submit_script_paths'] = []
    base_path = path_dict['base_path']
    for cce_folder_name in path_dict['cce_paths_keys']:
        run_name = f"{cce_folder_name}_{base_path.stem}"
        run_path = base_path/"cce"/cce_folder_name
        input_file_name = cce_folder_name+".yaml"
        

        submit_script=\
f"""#!/bin/bash -
#SBATCH -J {run_name}              # Job Name
#SBATCH -o SpEC.stdout                # Output file name
#SBATCH -e SpEC.stderr                # Error file name
#SBATCH -n 4                  # Number of cores
#SBATCH --ntasks-per-node 4        # number of MPI ranks per node
#SBATCH -t 24:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue

# Go to the correct folder with the boundary data
cd {run_path}

export SPECTRE_HOME=/panfs/ds09/sxs/himanshu/spectre
export SPECTRE_DEPS=/panfs/ds09/sxs/himanshu/SPECTRE_DEPS

# Setup spectre environment
. $SPECTRE_HOME/support/Environments/wheeler_gcc.sh && spectre_setup_modules $SPECTRE_DEPS && echo \"Modules build\" && spectre_load_modules && echo \"Modules loaded\"

# run CCE
/panfs/ds09/sxs/himanshu/spectre/build/bin/CharacteristicExtract +p4 --input-file ./{input_file_name}
"""
        submit_script_path = base_path/"cce"/cce_folder_name/"submit.sh"
        path_dict['submit_script_paths'].append(submit_script_path)
        with submit_script_path.open('w') as f:
            f.writelines(submit_script)
    
def submit_all_jobs(path_dict:dict):
    for submit_script_path in path_dict['submit_script_paths']:
        os.chdir(str(submit_script_path.parent))
        command = f"qsub {submit_script_path}"
        status = subprocess.run(command, capture_output=True, shell=True, text=True)
        if status.returncode == 0:
          print(f"Succesfully submitted {submit_script_path}\n{status.stdout}")
        else:
          sys.exit(
              f"Job submission failed for {submit_script_path} with error: \n{status.stdout} \n{status.stderr}")
          

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
  cce_list = list(path_dict["base_path"].glob(f"Ev/Lev{some_lev}_AA/Run/GW2/CceR????.h5"))
  path_dict["cce_radius"] = [file.name[4:-3] for file in cce_list]


# 'cce_paths_keys': ['Lev_1_radius_0112', 'Lev_1_radius_0255']
# 'Lev1_R0112': [path_list],
# 'Lev1_R0255': [path_list],
def add_cce_data_paths(path_dict):
  path_dict["cce_paths_keys"]=[]
  for lev in path_dict["Lev_list"]:
    for radius in path_dict["cce_radius"]:
      key_name = f"Lev{lev}_R{radius}"
      path_dict["cce_paths_keys"].append(key_name)
      path_dict[key_name] = list(path_dict["base_path"].glob(f"Ev/Lev{lev}_??/Run/GW2/CceR{radius}.h5"))


# create directories to save cce waveforms
def create_folders_to_save_cce_data(path_dict):
  for cce_lev_radius in path_dict["cce_paths_keys"]:
    folder_to_create = path_dict["base_path"]/f"cce/{cce_lev_radius}"
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
    save_folder = str(path_dict["base_path"]/f"cce/{cce_lev_radius}")
    h5_file_list = path_dict[cce_lev_radius]
    output_file_name = cce_lev_radius+".h5"

    # check that the outputfile is not already present
    output_file_path = path_dict["base_path"]/f"cce/{cce_lev_radius}/{output_file_name}"
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
  path_dict['boundary_data_paths'] = list(path_dict['base_path'].glob("cce/*/*.h5"))

def pickle_path_dict(path_dict):
  with open(path_dict['base_path']/"cce/path_dict.pkl",'wb') as f:
    pickle.dump(path_dict,f)



def do_CCE(run_path_list:list, CCE_executable:Path = None):
    if CCE_executable is None:
        CCE_executable = Path("/panfs/ds09/sxs/himanshu/spectre/build/bin/CharacteristicExtract")

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
        submit_all_jobs(path_dict)



# runs_paths = [
#     Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_harmonic_mr1_50_400/"),
#     Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_harmonic_mr1_200_400/"),
#     Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_mr1_50_400/"),
#     Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_mr1_200_400/")
# ]
# runs_paths = [
#     Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_mr1_200_400/")
# ]
runs_paths = [
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/71_ngd_master_mr1_50_400_no_roll_on'),
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/71_ngd_master_mr1_200_400_no_roll_on'),
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/72_ngd_master_mr1_50_400_no_roll_on_pow2'),
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/72_ngd_master_mr1_200_400_no_roll_on_pow2')
]

CCE_executable = Path("/panfs/ds09/sxs/himanshu/spectre/build/bin/CharacteristicExtract")
do_CCE(runs_paths,CCE_executable)