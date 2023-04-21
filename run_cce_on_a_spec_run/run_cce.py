# %%
import numpy as np
import pandas as pd
import subprocess
import sys
from numba import njit
import matplotlib.pyplot as plt
import os
import glob
plt.style.use('seaborn-talk')
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
spec_home="/home/himanshu/spec/my_spec"

# %% [markdown]
# ## Functions to convert the data to scri format

# %%

def block_mode_data_to_ylm_timeseries(volume_file:Path, output_dir_name:Path, output_prefix):
    def LM_index(L, M):
        return 2 * (L**2 + L + M) + 1

    with h5py.File(volume_file, 'r') as input_h5:
        for i, dataset in enumerate(input_h5):
            if "Version" in dataset or "tar.gz" in dataset:
                continue
            
            for variable in input_h5[dataset]:
                if i == 0:
                    min_time = input_h5[dataset][variable][0, 0]
                    max_time = input_h5[dataset][variable][-1, 0]
                else:
                    if min_time < input_h5[dataset][variable][0, 0]:
                        min_time = input_h5[dataset][variable][0, 0]
                    if max_time > input_h5[dataset][variable][-1, 0]:
                        max_time = input_h5[dataset][variable][-1, 0]
                        
        for dataset in input_h5:
            if "Version" in dataset or "tar.gz" in dataset:
                continue
            
            for variable in input_h5[dataset]:
                idx1 = np.argmin(abs(input_h5[dataset][variable][:, 0] - min_time))
                idx2 = np.argmin(abs(input_h5[dataset][variable][:, 0] - max_time))
                data_array = input_h5[dataset][variable][idx1:idx2]
                
                # sort the array according to the time
                data_array = data_array[data_array[:, 0].argsort()]
                number_of_columns = data_array.shape[1]
                L_max = int(np.sqrt((number_of_columns - 1) / 2 - 1))
                with h5py.File(output_dir_name + variable[:-4] + output_prefix + ".h5", 'w') as output_h5:
                    for L in range(L_max + 1):
                        for M in range(-L, L + 1):
                            output_h5.create_dataset(
                                "/Y_l" + str(L) + "_m" + str(M) + ".dat",
                                data=np.append(
                                    data_array[:, 0:1],
                                    data_array[:, LM_index(L, M):LM_index(L, M) + 2],
                                    axis=1))

def make_variables_dimensionless(WM, ChMass=None, metadata_filename=None):
        if WM.m_is_scaled_out:
            raise ValueError("Data is already dimensionless!")
        if (ChMass is None and metadata_filename is None):
            raise ValueError("Either ChMass OR metadata_filename must be supplied.")
        elif (ChMass is not None and metadata_filename is not None):
            raise ValueError("Either ChMass OR metadata_filename must be supplied, but not both.")

        if ChMass is None:
            metadata = sxs.metadata.Metadata.from_file(metadata_filename)
            mass1 = metadata['reference-mass1']
            mass2 = metadata['reference-mass2']
            ChMass = float(mass1) + float(mass2)

        if WM.dataType in [scri.psi4, scri.psi3, scri.psi2, scri.psi1, scri.psi0]:
            unit_scale_factor = (ChMass)**(WM.dataType-4)
        elif WM.dataType == scri.h:
            unit_scale_factor = 1/ChMass
        elif WM.dataType == scri.hdot:
            unit_scale_factor = 1.0
        else:
            raise ValueError("DataType not determined.")

        WM.t = WM.t / ChMass
        WM.data = WM.data * unit_scale_factor
        WM.m_is_scaled_out = True


def plot_and_save_bianchi_violations(violation_dict:dict,save_dir:Path):
    plt.semilogy(violation_dict['t'],violation_dict['5'],label='5')
    plt.semilogy(violation_dict['t'],violation_dict['4'],label='4')
    plt.semilogy(violation_dict['t'],violation_dict['3'],label='3')
    plt.semilogy(violation_dict['t'],violation_dict['2'],label='2')
    plt.semilogy(violation_dict['t'],violation_dict['1'],label='1')
    plt.semilogy(violation_dict['t'],violation_dict['0'],label='0')
    plt.xlabel('t')
    plt.ylabel("violations")
    plt.legend()
    plt.savefig(save_dir/"violations.png")

        
def post_process_jobs(path_dict , output_dir_name="extracted_data"):
    cce_scri_data_names = {
        'Strain' : (scri.h, 'rhOverM'),
        'News'   : (scri.hdot, 'r2News'),
        'Psi4'   : (scri.psi4, 'rMPsi4'),
        'Psi3'   : (scri.psi3, 'r2Psi3'),
        'Psi2'   : (scri.psi2, 'r3Psi2OverM'),
        'Psi1'   : (scri.psi1, 'r4Psi1OverM2'),
        'Psi0'   : (scri.psi0, 'r5Psi0OverM3')
    }

    bianchi_violations={}
    for bd_data_path in path_dict['boundary_data_paths']:
        bd_folder_path = bd_data_path.parent



        directory = str(bd_folder_path)+"/"
        radius = bd_folder_path.stem[-4:]
        

        os.system(f'rm -r {directory}/{output_dir_name} 2> /dev/null')
        os.system(f'mkdir {directory}/{output_dir_name}')

        block_mode_data_to_ylm_timeseries(bd_data_path,
                                          f'{directory}{output_dir_name}/',
                                          f'_BondiCce_R{radius}_unprocessed')

        variables = {}
        for input_h5_file in list(np.sort(glob.glob(f'{directory}/{output_dir_name}/*R{radius}_unprocessed.h5'))):
            input_data_name = input_h5_file.split('/')[-1].split('_')[0]
            if input_data_name in cce_scri_data_names:
                input_data_type = cce_scri_data_names[input_data_name][0]
                
                WM = scri.SpEC.read_from_h5(
                    input_h5_file,
                    frameType = scri.Inertial,
                    dataType = input_data_type,
                    r_is_scaled_out = True,
                    m_is_scaled_out = False,
                )
                if os.path.exists(f'{directory}metadata.txt'):
                    metadata_filename = f'{directory}metadata.txt'
                else:
                    metadata_filename = f'{directory}metadata.json'
                make_variables_dimensionless(WM, metadata_filename=metadata_filename)
                WM.t = WM.t - float(radius)
                variables[input_data_name] = WM
                                
        min_time = variables['Strain'].t[0]; max_time = variables['Strain'].t[-1]; idx = 0
        for i, WM_name in enumerate(variables):
            WM = variables[WM_name]
            if WM.t[0] > min_time and WM.t[-1] < max_time:
                min_time = WM.t[0]
                max_time = WM.t[-1]
                idx = i

        t_common = variables[list(variables.keys())[idx]].t
        for WM_name in variables:
            WM = variables[WM_name]
            variables[WM_name] = WM.interpolate(t_common)

        for WM_name in variables:
            scri.SpEC.file_io.write_to_h5(variables[WM_name],
                                          f'{directory}{output_dir_name}/BondiCce_R{radius}.h5')

        # remove the unprocessed parts
        os.system(f'rm {directory}/{output_dir_name}/*unprocessed.h5')


        # compute bianchi violations and save the pickel

        abd = scri.SpEC.file_io.create_abd_from_h5(h=f'{directory}{output_dir_name}/rhOverM_BondiCce_R{radius}.h5',
                                                  Psi4=f'{directory}{output_dir_name}/rMPsi4_BondiCce_R{radius}.h5',
                                                  Psi3=f'{directory}{output_dir_name}/r2Psi3_BondiCce_R{radius}.h5',
                                                  Psi2=f'{directory}{output_dir_name}/r3Psi2OverM_BondiCce_R{radius}.h5',
                                                  Psi1=f'{directory}{output_dir_name}/r4Psi1OverM2_BondiCce_R{radius}.h5',
                                                  Psi0=f'{directory}{output_dir_name}/r5Psi0OverM3_BondiCce_R{radius}.h5',
                                                  file_format='SXS')
      
        violations = abd.bondi_violation_norms

        # dump the dict as a pickel
        violations_dict = {
          't': abd.t,
          '0': violations[0],
          '1': violations[1],
          '2': violations[2],
          '3': violations[3],
          '4': violations[4],
          '5': violations[5]
        }
        with open(f'{directory}bondi_violation_dict.pkl','wb') as f:
            pickle.dump(violations_dict,f)
        plot_and_save_bianchi_violations(violations_dict,bd_folder_path)

        total_violations = []
        for violation in violations:
            total_violations.append(scipy.integrate.trapezoid(violation, abd.t))
            bianchi_violations[str(bd_folder_path.stem)] = total_violations


    # Save the bianchi violation dict
    with open(f'{directory}bianchi_violations.json', 'w') as f:
        json.dump(bianchi_violations, f, indent=2, separators=(",", ": "), ensure_ascii=True)


# %% [markdown]
# ## Functions to deal with cce extraction

# %%
def RunCCE(CCE_executable: Path, BoundaryDataPath: Path, VolumeFilePostFix: str = None)-> Path:
    assert (CCE_executable.exists())
    assert (BoundaryDataPath.exists())

    if VolumeFilePostFix is None:
        VolumeFilePostFix = "_VolumeData"

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
        VolumeFilePostFix = "_VolumeData"

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
    InverseCubic

  StartTime: Auto
  EndTime: Auto
  BoundaryDataFilename: {BoundaryDataPath.name}
  H5IsBondiData: False
  H5Interpolator:
    BarycentricRationalSpanInterpolator:
      MinOrder: 10
      MaxOrder: 10
  ExtractionRadius: Auto
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
        VolumeFilePostFix = "_VolumeData"

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
  VolumeFileName: vol_{str(InputSavePath.stem)+VolumeFilePostFix}
  ReductionFileName: redu_{str(InputSavePath.stem)+VolumeFilePostFix}

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
  ExtractionRadius: Auto
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

def create_metadata_with_masses(path_dict:dict):
    file_path = path_dict['base_path']/"ID/SpEC.out"

    # Read the masses from the ID SpEC.out file
    with file_path.open('r') as f:
        file_contents = f.read() 
        MA = float((re.findall("MA=0.\d*",file_contents))[-1].split('=')[-1])
        MB = float((re.findall("MB=0.\d*",file_contents))[-1].split('=')[-1])

    metadata_file_contents=f"""{{ "reference_mass1": {MA},  "reference_mass2": {MB}}}"""

    for bd_path in path_dict['boundary_data_paths']:
        metadata_file = bd_path.parent/"metadata.json"

        with metadata_file.open('w') as f:
            f.write(metadata_file_contents)

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
#SBATCH -n 4                          # Number of cores
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

# %% [markdown]
# ## Function to do it all

# %%
def do_CCE(run_path_list: list, CCE_executable: Path = None, submit_jobs=True):
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
        create_metadata_with_masses(path_dict)
        make_submit_file(path_dict)
        if submit_jobs:
          submit_all_jobs(path_dict)

runs_paths = [
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/67_master_mr1/"),
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/66_master_harmonic_mr1/"),
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/67_master_mr3/"),
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/66_master_harmonic_mr3/"),
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_harmonic_mr1_50_400/"),
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_harmonic_mr1_200_400/"),
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_mr1_50_400/"),
    Path("/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/70_ngd_master_mr1_200_400/"),
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/71_ngd_master_mr1_50_400_no_roll_on'),
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/71_ngd_master_mr1_200_400_no_roll_on'),
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/72_ngd_master_mr1_50_400_no_roll_on_pow2'),
    Path('/panfs/ds09/sxs/himanshu/gauge_stuff/gauge_driver_runs/runs/72_ngd_master_mr1_200_400_no_roll_on_pow2')
]

CCE_executable = Path("/panfs/ds09/sxs/himanshu/spectre/build/bin/CharacteristicExtract")
do_CCE(runs_paths,CCE_executable)