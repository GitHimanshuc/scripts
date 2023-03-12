import numpy as np
import pandas as pd
import subprocess
import sys
import json
import time
import pickle
import argparse
from pathlib import Path
spec_home="/home/himanshu/spec/my_spec"


# 'Lev_list': ['1','3']
def add_levs(path_dict):
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

# Saves the path of the combined boundary data files into path_dict
def save_boundary_data_paths(path_dict):
  path_dict['boundary_data_paths'] = list(path_dict['base_path'].glob("cce/*/*.h5"))

def pickle_path_dict(path_dict):
  with open(path_dict['base_path']/"cce/path_dict.pkl",'wb') as f:
    pickle.dump(path_dict,f)

    

parser = argparse.ArgumentParser(description='For a given folder save all combine all its boundary data files.')
parser.add_argument('base_path', metavar='base_path', type=str,
                    help='Base path of the folder')

args = parser.parse_args()
base_path = Path(args.base_path).absolute()
path_dict = {"base_path":base_path}
add_levs(path_dict)
add_cce_radius(path_dict)
add_cce_data_paths(path_dict)
create_folders_to_save_cce_data(path_dict)
save_joined_cce_h5_files(path_dict)
save_boundary_data_paths(path_dict)
pickle_path_dict(path_dict)