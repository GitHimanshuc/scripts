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


def read_spec_into_pd(file_path: Path):

    with file_path.open("r") as f:
        file_content = f.read()

    # Extract metadata
    time_value = float(re.search(r"Time\[0\]\s*=\s*([\d.]+)", file_content).group(1))
    extents = tuple(
        map(int, re.search(r"Extents\s*=\s*\((\d+),(\d+)\)", file_content).groups())
    )

    # Extract data rows
    data_rows = re.findall(r"\(:,(\d+)\):\s*([\d\s\.,eE+-]+)", file_content)

    # Convert data into a list of lists
    data = {}
    for row in data_rows:
        index = int(row[0])
        values = list(map(float, row[1].split(",")))
        data[index] = values

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Add metadata as attributes (optional)
    df.attrs["Time"] = time_value
    df.attrs["Extents"] = extents

    return df


def read_phy_into_pd(file_path: Path):

    with file_path.open("r") as f:
        file_content = f.read()

    # Extract metadata
    time_value = float(re.search(r"Time\[0\]\s*=\s*([\d.]+)", file_content).group(1))
    extents = tuple(
        map(
            int,
            re.search(r"Extents\s*=\s*\((\d+),(\d+),(\d+)\)", file_content).groups(),
        )
    )

    # Extract data rows
    data_rows = re.findall(r"\(:,(\d+),(\d+)\):\s*(.*)", file_content)

    # Convert data into a list of lists
    data = {}
    for row in data_rows:
        l = int(row[0])
        m = int(row[1])
        values = list(map(float, row[2].split(",")))
        data[f"{l},{m}"] = values

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Add metadata as attributes (optional)
    df.attrs["Time"] = time_value
    df.attrs["Extents"] = extents

    return df.T


def read_h5_file(file_path: str) -> Dict[str, Any]:
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
        return result

    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file {file_path} does not exist")
    try:
        with h5py.File(file_path, "r") as f:
            # Read all contents
            data = {}
            # Read main groups
            for group_name in ["InitGridHi", "InitHhatt", "kappa", "psi"]:
                if group_name in f:
                    data[group_name] = read_group(f[group_name])
        return data

    except OSError as e:
        raise OSError(f"Error reading HDF5 file: {str(e)}")


def load_filtered_data_into_dict(filtered_txt_folder_path: Path):

    print(f"Loading data from {filtered_txt_folder_path}")
    file_list = list(filtered_txt_folder_path.glob("*.txt")) 
    if len(file_list) == 0:
        raise FileNotFoundError(f"No txt files found in {filtered_txt_folder_path}")
    data_dict = {}

    for fp in file_list:
        if "fil_phy" not in fp.stem:
            continue
        domain_name = fp.stem.split("_")[-1]
        if domain_name not in data_dict:
            data_dict[domain_name] = {
                "psi": {},
                "kappa": {},
            }
        if "psi" in fp.stem:
            index = fp.stem.split("_")[-2][-2:]
            data_dict[domain_name]["psi"][index] = (
                read_phy_into_pd(fp).to_numpy().flatten()
            )
        if "kappa" in fp.stem:
            index = fp.stem.split("_")[-2][-3:]
            data_dict[domain_name]["kappa"][index] = (
                read_phy_into_pd(fp).to_numpy().flatten()
            )
    return data_dict

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
                                    print(f"Modifying {key1}/{key3}")
                                    outfile[key1][key2][key3][()] = modification_data_dict[key1][key3]
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

def make_input_file_content(DomainRegex:str, Threshold:float):
  content=f"""
DataBoxItems =
    ReadFromFile(File=./SpatialCoordMap.input),
    ReadFromFile(File=./GaugeItems.input),
    Domain(Items=
           AddGeneralizedHarmonicInfo(MatterSourceName=;)
           ),
    Subdomain(Items =
              Add3Plus1ItemsFromGhPsiKappa(psi=psi;kappa=kappa;OutputPrefix=),
              GlobalDifferentiator
              (GlobalDifferentiator=
                MatrixMultiply(MultiDim_by_BasisFunction=yes;
                TopologicalDifferentiator
                =Spectral(SetBasisFunctionsFromTimingInfo=yes;)
                );
              );
    );

Observers =
    FilterModes(
      Variables = psi, kappa;
      DomainRegex = {DomainRegex};
      Threshold = {Threshold};
    ),
    ;
"""
  return content

def make_filtered_checkpoint_from_another(
    ApplyObserversPath: Path,
    ConvertDumpToTextPath: Path,
    InputAndHistFolderPath: Path,
    CheckpointFolderPath: Path,
    work_dir: Path,
    FilterThreshold: float,
    DomainRegex: str):

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

# For now the FilterMode observer is hard coded to output things in the data folder
cd ./data

DIRECTORY="{filtered_text_files_path}"
# Loop through all .dump files in the directory
for file in "$DIRECTORY"/*.dump; do
    if [ -f "$file" ]; then
        new_file="${{file%.dump}}.txt"
        # if [ -f "$new_file" ]; then
        #     continue
        # fi
        # Run the command on the file
        {ConvertDumpToTextPath} < $file > $new_file
        # Replace 'echo' with your desired command
        echo "$file"
    fi
done

    """
    with open(work_dir / "input_file.input", "w") as f:
        f.write(make_input_file_content(DomainRegex, FilterThreshold))

    with open(work_dir / "apply_observer.sh", "w") as f:
        f.write(apply_observer)

    command = f"cd {work_dir} && bash ./apply_observer.sh > ./apply_observer.out"
    status = subprocess.run(command, capture_output=True, shell=True, text=True)
    if status.returncode == 0:
        print(f"Ran FilterModes observer in {work_dir}.\n {status.stdout}")
    else:
        sys.exit(
            f"Failed to run FilterModes observer in {work_dir}. \n {status.stderr}"
        )

    # load the filtered data in txt format into a dictionary
    filtered_data_dict = load_filtered_data_into_dict(filtered_text_files_path)
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

ApplyObserversPath = Path("/groups/sxs/hchaudha/spec_runs/filtered_checkpoints/binaries/ApplyObservers")
ConvertDumpToTextPath = Path("/groups/sxs/hchaudha/spec_runs/filtered_checkpoints/binaries/ConvertDumpToText")

CheckpointFolderPath = Path(
    "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_5517_6/Ev/Lev3_AA/Run/Checkpoints/5517"
)

InputAndHistFolderPath = CheckpointFolderPath.parent.parent
work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

make_filtered_checkpoint_from_another(
    ApplyObserversPath,
    ConvertDumpToTextPath,
    InputAndHistFolderPath,
    CheckpointFolderPath,
    work_dir,
    1e-6,
    "SphereC*"
)

CheckpointFolderPath = Path(
    "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_5517_8/Ev/Lev3_AA/Run/Checkpoints/5517"
)

InputAndHistFolderPath = CheckpointFolderPath.parent.parent
work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

make_filtered_checkpoint_from_another(
    ApplyObserversPath,
    ConvertDumpToTextPath,
    InputAndHistFolderPath,
    CheckpointFolderPath,
    work_dir,
    1e-8,
    "SphereC*"
)

CheckpointFolderPath = Path(
    "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_5517_10/Ev/Lev3_AA/Run/Checkpoints/5517"
)

InputAndHistFolderPath = CheckpointFolderPath.parent.parent
work_dir = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_workdir"

make_filtered_checkpoint_from_another(
    ApplyObserversPath,
    ConvertDumpToTextPath,
    InputAndHistFolderPath,
    CheckpointFolderPath,
    work_dir,
    1e-10,
    "SphereC*"
)
