import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import copy
from typing import List, Dict
import pandas as pd
import os
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (10,8)

def get_info(h5path:Path):
    with h5py.File(h5path,'r') as f:
        names = []
        f.visit(names.append)
    var_names = set()
    num_comp_rad = 0
    for name in names:
        if 'VolumeData/' in name and 'CompactifiedRadius' not in name:
            var_names.add(name.split('/')[-1])
        if 'VolumeData/' in name and 'CompactifiedRadius' in name:
            num_comp_rad = max(num_comp_rad,int(name.split("CompactifiedRadius_")[-1].split('.')[0]))

    return var_names,num_comp_rad


def get_data_all_comp_rad(h5_datapath:Path, var_name:str, comp_rad_list:list, time_slice:slice , red_func=np.linalg.norm):
    with h5py.File(h5_datapath,'r') as f:
        data = {}
        for comp_rad in comp_rad_list:
            curr_data = f[f'Cce/VolumeData/{var_name}/CompactifiedRadius_{comp_rad}.dat'][time_slice,1:]
            data[comp_rad] = red_func(curr_data,axis=1)
        t = f[f'Cce/VolumeData/{var_name}/CompactifiedRadius_0.dat'][time_slice,0]
    return t,data


data_path = Path("./red_cce.h5").resolve()
if not data_path.exists():
    raise FileNotFoundError(f"File {data_path} does not exist.")

# Create dir to save the vol data for all but the last compactified radius
current_dir = data_path.parent
save_dir = current_dir/'plots_all_but_last'
save_dir.mkdir(exist_ok=False)

var_names,num_comp_rad = get_info(data_path)
print(var_names)

for var_name in var_names:
    try:
        t,var_data = get_data_all_comp_rad(data_path, var_name, range(num_comp_rad), slice(0,-1,100))
        for comp_rad in var_data:
            plt.plot(t,var_data[comp_rad],label=f'CompactifiedRadius_{comp_rad}')
        plt.xlabel('t')
        plt.ylabel(f'L2({var_name})')
        plt.title("Extraction radius: " + str(data_path).split("/")[-2][4:] + "M")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir/f"{var_name}.png")
        plt.close()
    except:
        print(f"Error processing variable: {var_name}")


# Create dir to save CCE vol data
save_dir = current_dir/'plots'
save_dir.mkdir(exist_ok=False)

var_names,num_comp_rad = get_info(data_path)


for var_name in var_names:
    try:
        t,var_data = get_data_all_comp_rad(data_path, var_name, range(num_comp_rad+1), slice(0,-1,100))
        for comp_rad in var_data:
            plt.plot(t,var_data[comp_rad],label=f'CompactifiedRadius_{comp_rad}')
        plt.xlabel('t')
        plt.ylabel(f'L2({var_name})')
        plt.title("Extraction radius: " + str(data_path).split("/")[-2][4:] + "M")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir/f"{var_name}.png")
        plt.close()
    except:
        print(f"Error processing variable: {var_name}")
