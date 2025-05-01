import numpy as np
import re
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import shutil
import h5py
import numpy as np
from typing import Dict, Any
from pathlib import Path
import subprocess
import sys


def read_h5_file_dump_tensors(file_path: str) -> Dict[str, Any]:
    def read_group(group) -> Dict[str, Any]:
        result = {}
        # Read all datasets in current group
        for name, item in group.items():
            if isinstance(item, h5py.Dataset):
                # Convert dataset to numpy array
                result[name] = item[()]
            elif isinstance(item, h5py.Group):
                # Recursively read nested group
                result[name] = read_group(item)
            else:
                print(name, item)
        for name, item in group.attrs.items():
            result[name] = item
        return result

    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file {file_path} does not exist")
    try:
        with h5py.File(file_path, "r") as f:
            # Read all contents
            data = {}
            # Read main groups
            for name, item in f.items():
                data[name] = read_group(f[name])
            # # Read main groups
            # for group_name in ["InitGridHi", "InitHhatt", "kappa", "psi"]:
            #     if group_name in f:
            #         data[group_name] = read_group(f[group_name])
        return data

    except OSError as e:
        raise OSError(f"Error reading HDF5 file: {str(e)}")


def create_modification_dict(filter_modes_output_path: Path, filtered_vars_folders_path:Path):

    filtered_sub_domains = set()
    with filter_modes_output_path.open() as f:
        for l in f.readlines():
            if re.match(r"Filtered=", l):
                domain_name = l.split("=")[-1].strip()
                filtered_sub_domains.add(domain_name)

    filtered_dict = {}
    vars_h5_files = list(filtered_vars_folders_path.glob("Vars_*.h5"))
    # load data for subdomains that were filtered
    for fp in vars_h5_files:
        domain_name = fp.stem.split("_")[-1]
        if domain_name in filtered_sub_domains:
            filtered_dict[domain_name] = read_h5_file_dump_tensors(fp)

    return filtered_dict

def copy_and_modify_h5file(input_file, output_file, modification_data_dict):
    shutil.copy(input_file, output_file)
    with h5py.File(output_file, "r+") as outfile:
        # Level 1: Root level items
        for key1, item1 in outfile.items():
            if isinstance(item1, h5py.Group):
                # Level 2: First nested level
                for key2, item2 in item1.items():
                    if isinstance(item2, h5py.Group):
                        # Level 3: Second nested level
                        for key3, item3 in item2.items():
                            if isinstance(item3, h5py.Dataset):
                                # print(f"{key1}/{key2}/{key3}")
                                if key1 in modification_data_dict:
                                    print(f"----Modifying {key1}/{key3}")
                                    outfile[key1][key2][key3][()] = modification_data_dict[key1][key2][key3]
                            else:
                                raise ValueError(f"Unexpected item type: {type(item3)}")
                    else:
                        raise ValueError(f"Unexpected item type: {type(item2)}")
            else:
                raise ValueError(f"Unexpected item type: {type(item1)}")


def make_filtered_checkpoints(
    checkpoint_folder: Path, new_checkpoint_folder: Path, data_dict: dict
):
    # Copy the original checkpoint file
    for fp in checkpoint_folder.glob("*.txt"):
        shutil.copy(fp, new_checkpoint_folder)

    for fp in checkpoint_folder.glob("*.h5"):
        file_name = fp.stem
        # get domain name for the h5 files
        if file_name == "SerialCheckpoint":
            # Copy this without change
            shutil.copy(fp, new_checkpoint_folder)
            continue

        domain_name = file_name.split("_")[-1]

        if domain_name in data_dict:
            print(f"Modifying the domain {domain_name}")
            modified_data = data_dict[domain_name]
            modified_fp = new_checkpoint_folder / fp.name
            copy_and_modify_h5file(fp, modified_fp, modified_data)
        else:
            print(f"Copying the domain {domain_name}")
            shutil.copy(fp, new_checkpoint_folder)

def make_input_file_content(half_power_R, half_power_L):
    input_file_text = f"""DataBoxItems =
        ReadFromFile(File=./SpatialCoordMap.input),
        ReadFromFile(File=./GaugeItems.input),
        Domain(Items=
            AddGeneralizedHarmonicInfo(MatterSourceName=;)
            ),
        Subdomain(Items =
                Add3Plus1ItemsFromGhPsiKappa(psi=psi;kappa=kappa;OutputPrefix=),
                AddSpacetimeJacobianAndHessianItems(MapPrefix=GridToInertial;),
                GlobalDifferentiator
                (GlobalDifferentiator=
                    MatrixMultiply(MultiDim_by_BasisFunction=yes;
                    TopologicalDifferentiator
                    =Spectral(SetBasisFunctionsFromTimingInfo=yes;
                            # BasisFunctions= (ChebyshevGaussLobatto=ChebyshevGaussLobattoMatrix);
                            )
                    );
                ),"""
    for R,L,i in zip(half_power_R,half_power_L,np.arange(len(half_power_R))):
        input_file_text = input_file_text+f"""
    # ==============================================================================
                FilterVar(
                    Input = psi;
                    Output = fil_psi{i};
                    WhichFilter = Exp;
                    RadialHalfPower = {R};
                    AngularHalfPower = {L};
                    DomainRegex = SphereC{i};
                ),
                FilterVar(
                    Input = kappa;
                    Output = fil_kappa{i};
                    WhichFilter = Exp;
                    RadialHalfPower = {R};
                    AngularHalfPower = {L};
                    DomainRegex = SphereC{i};
                ),
                FirstDeriv(
                    Input = fil_psi{i};
                    Output = fil_dpsi_TT{i};
                    # SetDerivDimFromTensorDim=True;
                    MapPrefix = GridToInertial;
                ),
                FlattenDeriv(
                    Input = fil_dpsi_TT{i};
                    Output = fil_dpsi{i};
                    DerivPosition = First;
                    ZeroFillOffset=1;
                ),
                FakeKappa(
                    FildPsi = fil_dpsi{i};
                    FilKappa = fil_kappa{i};
                    Output = fake_kappa{i};
                ),
"""

    input_file_text = input_file_text + """


    ;);#Subdomains
    Observers ="""

    for R,L,i in zip(half_power_R,half_power_L,np.arange(len(half_power_R))):
        input_file_text = input_file_text+f"""        DumpTensors(
        Input = fil_psi{i},fake_kappa{i};
        FileNames = psi,kappa;
        OnlyTheseSubdomains = SphereC{i};
        ),        
        ObserveInSubdir
        (Subdir=SpecCoeffs;
          Observers =
          DumpSpectralCoefficients(
            Input = fake_kappa{i}, fil_psi{i}, fil_kappa{i}, fil_dpsi{i};
            TakeLog=no;
            Subdomains =SphereC{i};
            # Eps = 1e-16;
          ),
        ),
"""

    input_file_text = input_file_text +"""
        ObserveInSubdir
        (Subdir=SpecCoeffs;
          Observers =
          DumpSpectralCoefficients(
            Input = psi, kappa;
            TakeLog=no;
            Subdomains = *;
            # Eps = 1e-16;
          ),
        ),
        ;
    """
    return input_file_text

def make_filtered_checkpoint_from_another(
    ApplyObserversPath: Path,
    InputAndHistFolderPath: Path,
    CheckpointFolderPath: Path,
    work_dir: Path,
    HalfPowerArr:np.ndarray
    ):

    if not CheckpointFolderPath.exists():
      raise Exception(f"{CheckpointFolderPath} does not exist!!")

    filtered_checkpoint_path = work_dir / "filtered_checkpoint"
    filtered_text_files_path = work_dir / "data"

    work_dir.mkdir(parents=True, exist_ok=True)
    filtered_checkpoint_path.mkdir(parents=False, exist_ok=True)
    filtered_text_files_path.mkdir(parents=False, exist_ok=True)

    for f in CheckpointFolderPath.glob("*"):
        shutil.copy(f, work_dir)
    for f in InputAndHistFolderPath.glob("Hist*.txt"):
        shutil.copy(f, work_dir)
    for f in InputAndHistFolderPath.glob("*.input"):
        shutil.copy(f, work_dir)

    apply_observer = f"""#!/bin/bash
. /home/hchaudha/spec/MakefileRules/this_machine.env
cd {work_dir}

{ApplyObserversPath} -t psi,kappa,InitHhatt,InitGridHi -r 11,122,,1 -d 4,4,1,3 -domaininput ./GrDomain.input -h5prefix Cp-VarsGr ./input_file.input
    """
    with open(work_dir / "input_file.input", "w") as f:
        f.write(make_input_file_content(
                HalfPowerArr,
                HalfPowerArr
            ))

    with open(work_dir / "apply_observer.sh", "w") as f:
        f.write(apply_observer)

    command = f"cd {work_dir} && bash ./apply_observer.sh > ./apply_observer.out"
    apply_observer_output_path =work_dir/"apply_observer.out"

    status = subprocess.run(command, capture_output=True, shell=True, text=True)
    if status.returncode == 0:
        print(f"Ran FilterModes observer in {work_dir}.\n {status.stdout}")
    else:
        sys.exit(
            f"Failed to run FilterModes observer in {work_dir}. \n {status.stderr}"
        )

    # load the filtered data in txt format into a dictionary
    filtered_data_dict = create_modification_dict(
        filter_modes_output_path=apply_observer_output_path,
        filtered_vars_folders_path=work_dir
    )
    print(f"Loaded filtered data for the following domains: {filtered_data_dict.keys()}")

    # create new checkpoint files with the filtered data
    make_filtered_checkpoints(
        CheckpointFolderPath, filtered_checkpoint_path, filtered_data_dict
    )

    # Rename the original checkpoint folder
    new_original_checkpoint_folder_name = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_original"
    shutil.move(CheckpointFolderPath,new_original_checkpoint_folder_name)

    # Copy the filtered checkpoint data into the place of the original checkpoint data
    shutil.copytree(work_dir / "filtered_checkpoint",CheckpointFolderPath)

    # Copy the main file for reproducibility
    shutil.copy(Path(__file__).absolute(), work_dir,follow_symlinks=True)

ApplyObserversPath = Path("/groups/sxs/hchaudha/spec_runs/filtered_checkpoints/binaries_EXP_fake_kappa/ApplyObservers")


# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_5_6_30_del/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# InputAndHistFolderPath = CheckpointFolderPath.parent.parent
# work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

# N,mean,std = 30,12,5
# minHalf,maxHalf = 6,30
# x = np.arange(N)
# guass = (1-np.exp(-((x - mean) ** 2) / (2 * std ** 2)))*(maxHalf-minHalf)+minHalf
# HalfPowerArr = np.array([np.ceil(i) for i in guass], dtype=int)

# make_filtered_checkpoint_from_another(
#     ApplyObserversPath,
#     InputAndHistFolderPath,
#     CheckpointFolderPath,
#     work_dir,
#     HalfPowerArr
# )


# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_5_6_30/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# InputAndHistFolderPath = CheckpointFolderPath.parent.parent
# work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

# N,mean,std = 30,12,5
# minHalf,maxHalf = 6,30
# x = np.arange(N)
# guass = (1-np.exp(-((x - mean) ** 2) / (2 * std ** 2)))*(maxHalf-minHalf)+minHalf
# HalfPowerArr = np.array([np.ceil(i) for i in guass], dtype=int)

# make_filtered_checkpoint_from_another(
#     ApplyObserversPath,
#     InputAndHistFolderPath,
#     CheckpointFolderPath,
#     work_dir,
#     HalfPowerArr
# )


# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_10_6_30/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# InputAndHistFolderPath = CheckpointFolderPath.parent.parent
# work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

# N,mean,std = 30,12,10
# minHalf,maxHalf = 6,30
# x = np.arange(N)
# guass = (1-np.exp(-((x - mean) ** 2) / (2 * std ** 2)))*(maxHalf-minHalf)+minHalf
# HalfPowerArr = np.array([np.ceil(i) for i in guass], dtype=int)

# make_filtered_checkpoint_from_another(
#     ApplyObserversPath,
#     InputAndHistFolderPath,
#     CheckpointFolderPath,
#     work_dir,
#     HalfPowerArr
# )


# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_1_6_30/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# InputAndHistFolderPath = CheckpointFolderPath.parent.parent
# work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

# N,mean,std = 30,12,1
# minHalf,maxHalf = 6,30
# x = np.arange(N)
# guass = (1-np.exp(-((x - mean) ** 2) / (2 * std ** 2)))*(maxHalf-minHalf)+minHalf
# HalfPowerArr = np.array([np.ceil(i) for i in guass], dtype=int)

# make_filtered_checkpoint_from_another(
#     ApplyObserversPath,
#     InputAndHistFolderPath,
#     CheckpointFolderPath,
#     work_dir,
#     HalfPowerArr
# )


# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_5_4_30/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# InputAndHistFolderPath = CheckpointFolderPath.parent.parent
# work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

# N,mean,std = 30,12,5
# minHalf,maxHalf = 4,30
# x = np.arange(N)
# guass = (1-np.exp(-((x - mean) ** 2) / (2 * std ** 2)))*(maxHalf-minHalf)+minHalf
# HalfPowerArr = np.array([np.ceil(i) for i in guass], dtype=int)

# make_filtered_checkpoint_from_another(
#     ApplyObserversPath,
#     InputAndHistFolderPath,
#     CheckpointFolderPath,
#     work_dir,
#     HalfPowerArr
# )


# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_EXP_FK_14012_5_2_30/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# InputAndHistFolderPath = CheckpointFolderPath.parent.parent
# work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

# N,mean,std = 30,12,5
# minHalf,maxHalf = 2,30
# x = np.arange(N)
# guass = (1-np.exp(-((x - mean) ** 2) / (2 * std ** 2)))*(maxHalf-minHalf)+minHalf
# HalfPowerArr = np.array([np.ceil(i) for i in guass], dtype=int)

# make_filtered_checkpoint_from_another(
#     ApplyObserversPath,
#     InputAndHistFolderPath,
#     CheckpointFolderPath,
#     work_dir,
#     HalfPowerArr
# )