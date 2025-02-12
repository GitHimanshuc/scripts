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
import os
import pickle
from functools import wraps
from pathlib import Path
import itertools
from typing import Callable
import numpy as np
from typing import Tuple, Optional, Union
from numpy.typing import ArrayLike
import h5py
import numpy as np
from typing import Dict, Any
from pathlib import Path

from typing import Optional, Tuple


class FilterCheckpoints:
    def __init__(
        self,
        chkpt_path: Path,
        filter_coeffs_buffer:int = 3,
        which_filter_func: str = "simple_truncation",
    ):
        self._chkpt_path = chkpt_path
        self._which_filter_func = which_filter_func
        self._fil_spec_data = {}
        self._fil_phys_data = {}
        self.filter_coeffs_buffer = filter_coeffs_buffer

        self._chkpt_data = self.load_all_checkpointed_domains(self._chkpt_path)
        self._data_was_filtered = False

        self._vars_and_comps = [
            ("psi", ["tt", "xx", "yy", "zz", "tx", "ty", "tz", "xy", "xz", "yz"]),
            (
                "kappa",
                [
                    "ttt",
                    "ttx",
                    "tty",
                    "ttz",
                    "txx",
                    "txy",
                    "txz",
                    "tyy",
                    "tyz",
                    "tzz",
                    "xtt",
                    "xtx",
                    "xty",
                    "xtz",
                    "xxx",
                    "xxy",
                    "xxz",
                    "xyy",
                    "xyz",
                    "xzz",
                    "ytt",
                    "ytx",
                    "yty",
                    "ytz",
                    "yxx",
                    "yxy",
                    "yxz",
                    "yyy",
                    "yyz",
                    "yzz",
                    "ztt",
                    "ztx",
                    "zty",
                    "ztz",
                    "zxx",
                    "zxy",
                    "zxz",
                    "zyy",
                    "zyz",
                    "zzz",
                ],
            ),
        ]

    def load_all_checkpointed_domains(
        self, checkpoints_files_folder: Path, checkpoint_prefix: str = "Cp-VarsGr_"
    ):
        chkpt_dict = {}
        for f in checkpoints_files_folder.iterdir():
            if f.is_file():
                if checkpoint_prefix in f.stem and f.suffix == ".h5":
                    domain_name = f.stem[len(checkpoint_prefix) :]
                    chkpt_dict[domain_name] = self.read_h5_file(f)
                    # print(domain_name)
        print(f"Files loaded from : {checkpoints_files_folder}")
        return chkpt_dict

    def read_h5_file(self, file_path: str) -> Dict[str, Any]:
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
                for group_name in ["InitGridHi", "InitHhatt", "kappa", "psi"]:
                    if group_name in f:
                        data[group_name] = read_group(f[group_name])
            return data

        except OSError as e:
            raise OSError(f"Error reading HDF5 file: {str(e)}")

    def truncation_error_spec(self, logabs_arr, filtered_coeff_start=-1):
        # Ignoring the filtered coeffs
        N = len(logabs_arr[:filtered_coeff_start])
        max_coef_val = np.max(logabs_arr[:2])
        scale = np.exp(-((np.arange(N) - N + 0.5) ** 2))
        return max_coef_val - np.sum(
            logabs_arr[:filtered_coeff_start] * scale
        ) / np.sum(scale)

    def truncation_error_simple(self, logabs_arr, filtered_coeff_start=-1):
        # Ignoring the filtered coeffs
        max_coef_val = np.max(logabs_arr[:3])
        min_coef_val = np.min(
            logabs_arr[-3 + filtered_coeff_start : filtered_coeff_start]
        )
        return max_coef_val - min_coef_val

    def min_coef_val(self, logabs_arr, filtered_coeff_start=-1):
        # Ignoring the filtered coeffs find the smallest coeff in the last five coeffs
        return np.min(logabs_arr[-5 + filtered_coeff_start : filtered_coeff_start])

    def min_coef_val(self, logabs_arr):
        # maximum coeff in the first three coeffs
        return np.max(logabs_arr[:3])

    def this_domain_has_saturated_has_noise(self, logabs_arr, filtered_coeff_start=-1):
        # Give log_coeffs = np.log10(spectral_coeffs + 1e-16) as the input
        # This is only for chebyshev coeffs(note that we are ignoring the last coeff),
        # for ylms one needs to account for the filtered coeffs.
        has_saturated = False
        has_noise = False
        mean_tail = np.mean(logabs_arr[-5:filtered_coeff_start])
        median_tail = np.std(logabs_arr[-5:filtered_coeff_start])
        if mean_tail < -13:
            has_saturated = True
        if median_tail < 0.5:
            has_noise = True
        return has_saturated, has_noise, mean_tail, median_tail

    def dct2d_gcl_pts(self, arr, chebN):
        list2d = arr.reshape(-1, chebN)
        # dct_list = scipy.fftpack.dct(list2d, type=1, axis=1) / (chebN - 1)
        dct_list = scipy.fft.dct(list2d, type=1, axis=1) / (chebN - 1)
        dct_list[:, 0] /= 2
        dct_list[:, -1] /= 2
        dct_list[:, 1::2] *= -1

        return dct_list

    def idct2d_gcl_pts(self, arr, chebN):
        arr[:, 0] *= 2
        arr[:, -1] *= 2
        arr[:, 1::2] *= -1
        arr *= chebN - 1
        idct_list = scipy.fft.idct(arr, type=1, axis=1).reshape(-1)
        return idct_list

    def abslog10(self, arr):
        return np.log10(np.abs(arr) + 1e-16)

    def filter_data(self):
        self._data_was_filtered = True
        self._fil_spec_data = {}
        self._fil_phys_data = {}
        for sd in self._chkpt_data:
            self._fil_spec_data[sd] = {"subdomain_was_filtered": False}
            self._fil_phys_data[sd] = {"subdomain_was_filtered": False}
            subdomain_was_filtered = False
            for var, comp_list in self._vars_and_comps:
                self._fil_spec_data[sd][var] = {}
                self._fil_phys_data[sd][var] = {}
                for comp in comp_list:
                    chebN = self._chkpt_data[sd][var]["Step000000"]["Extents"][0]
                    comp_data = self._chkpt_data[sd][var]["Step000000"][comp]

                    self._fil_spec_data[sd][var][comp] = self.dct2d_gcl_pts(
                        comp_data, chebN
                    )

                    has_saturated, has_noise, mean_tail, median_tail = (
                        self.this_domain_has_saturated_has_noise(
                            np.mean(
                                self.abslog10(self._fil_spec_data[sd][var][comp]),
                                axis=0,
                            )
                        )
                    )

                    if has_saturated and not has_noise:
                        # Leave the SphereA/B alone
                        print(
                            f"{sd=} : {var=}{comp}, {has_saturated=} : {mean_tail=:.5}; {has_noise=} : {median_tail=:.5}"
                        )
                    elif has_saturated and has_noise:
                        # Mostly spheres C before junk reaches them. Leave for now.
                        # If filtering is done early then this prevents changing the boundary sphereCs
                        print(
                            f"{sd=} : {var=}{comp}, {has_saturated=} : {mean_tail=:.5}; {has_noise=} : {median_tail=:.5}"
                        )
                    elif re.search("SphereC", sd):
                        # Technically I should filter the sphereCs and the domains connected to it atleast. But for now just filter the sphereCs
                        subdomain_was_filtered = True
                        print(
                            f"Filtered: {sd=} : {var=}{comp}, {has_saturated=} : {mean_tail=:.5}; {has_noise=} : {median_tail=:.5}"
                        )
                        leave_out_N_coeffs = 6
                        maxL = len(self._fil_spec_data[sd][var][comp][0, :])
                        for i in range(self._fil_spec_data[sd][var][comp].shape[0]):
                            match self._which_filter_func:
                                case "simple_truncation":
                                    start_idx, noise_level, noise_indices = (
                                        self.estimate_noise_level(
                                            self._fil_spec_data[sd][var][comp][i, :],
                                            window_size=5,
                                            derivative_threshold=0.2,
                                        )
                                    )
                                    mask = np.arange(maxL) > start_idx + self.filter_coeffs_buffer
                                    mask[:leave_out_N_coeffs] = False

                                    self._fil_spec_data[sd][var][comp][i, mask] = 0.0
                                case "None":
                                    pass
                                case _:
                                    raise Exception(
                                        f"which_filter_func = {self._which_filter_func} not defined"
                                    )
                    else:
                        print(
                            f"{sd=} : {var=}{comp}, {has_saturated=} : {mean_tail=:.5}; {has_noise=} : {median_tail=:.5}"
                        )

                    self._fil_phys_data[sd][var][comp] = self.idct2d_gcl_pts(
                        self._fil_spec_data[sd][var][comp], chebN
                    )
            # Mark that some part of this subdomain was filtered
            self._fil_spec_data[sd]["subdomain_was_filtered"] = subdomain_was_filtered
            self._fil_phys_data[sd]["subdomain_was_filtered"] = subdomain_was_filtered

    def simple_truncation(self, arr, threshold, leave_out_N_coeffs=5):
        mask = np.abs(arr) < threshold
        mask[:leave_out_N_coeffs] = False  # Preserve first N coefficients
        arr[mask] = 0
        return arr

    def estimate_noise_level(
        self,
        coefficients: np.ndarray,
        window_size: int = 5,
        derivative_threshold: float = 0.2,
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:

        log_coeffs = self.abslog10(coefficients)
        moving_avg = np.convolve(
            log_coeffs, np.ones(window_size) / window_size, mode="valid"
        )
        derivatives = np.diff(moving_avg)
        plateau_idx = np.where(np.abs(derivatives) < derivative_threshold)[0]
        if len(plateau_idx) == 0:
            # print("No plateau found")
            start_idx = -1
            noise_level = 10 ** np.array(log_coeffs)[-1]
            noise_indices = []
        else:
            start_idx = plateau_idx[0]
            noise_level = 10 ** log_coeffs[start_idx]
            noise_indices = np.arange(start_idx, len(coefficients))
        return start_idx, noise_level, noise_indices

    def copy_and_modify_h5file(self, input_file, output_file, modification_data_dict):
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
                                    if key1 not in ['psi','kappa']:
                                        continue
                                    if key1 in modification_data_dict:
                                        print(f"-------Modifying {key1}/{key3}")
                                        outfile[key1][key2][key3][()] = (
                                            modification_data_dict[key1][key3]
                                        )
                                    else:
                                        raise Exception(
                                            f"[{key1}][{key3}] not found in modification_data_dict"
                                        )
                                else:
                                    raise ValueError(
                                        f"Unexpected item type: {type(item3)}"
                                    )
                        else:
                            raise ValueError(f"Unexpected item type: {type(item2)}")
                else:
                    raise ValueError(f"Unexpected item type: {type(item1)}")

    def make_filtered_checkpoints(self, new_checkpoint_folder: Path):
        if not self._data_was_filtered:
            print(f"Filtering the data!")
            self.filter_data()

        new_checkpoint_folder.mkdir(parents=False, exist_ok=True)
        print(f"Putting the filtered data in the folder: {new_checkpoint_folder}")

        # Copy the original checkpoint file
        for fp in self._chkpt_path.glob("*.txt"):
            shutil.copy(fp, new_checkpoint_folder)

        for fp in self._chkpt_path.glob("*.h5"):
            file_name = fp.stem
            # get domain name for the h5 files
            if file_name == "SerialCheckpoint":
                # Copy this without change
                shutil.copy(fp, new_checkpoint_folder)
                continue

            domain_name = file_name.split("_")[-1]

            if self._fil_phys_data[domain_name]["subdomain_was_filtered"]:
                print(f"Modifying:".ljust(15, " "), f"{domain_name}")
                modified_data = self._fil_phys_data[domain_name]
                modified_fp = new_checkpoint_folder / fp.name
                self.copy_and_modify_h5file(fp, modified_fp, modified_data)
            else:
                print(f"Copying:".ljust(15, " "), f"{domain_name}")
                shutil.copy(fp, new_checkpoint_folder)

def make_filtered_checkpoint_from_another(CheckpointFolderPath: Path,filter_coeffs_buffer=3):

    if not CheckpointFolderPath.exists():
      raise Exception(f"{CheckpointFolderPath} does not exist!!")

    work_dir = CheckpointFolderPath.parent
    filtered_checkpoint_path = work_dir / "filtered_checkpoint"
    filtered_checkpoint_path.mkdir(parents=False, exist_ok=False)

    FilCheck = FilterCheckpoints(
        CheckpointFolderPath,
        filter_coeffs_buffer = filter_coeffs_buffer,
        which_filter_func = "simple_truncation"
    )

    FilCheck.make_filtered_checkpoints(filtered_checkpoint_path)

    # Rename the original checkpoint folder
    new_original_checkpoint_folder_name = CheckpointFolderPath.parent/f"{CheckpointFolderPath.stem}_original"
    shutil.move(CheckpointFolderPath,new_original_checkpoint_folder_name)

    # Copy the filtered checkpoint data into the place of the original checkpoint data
    shutil.copytree(filtered_checkpoint_path, CheckpointFolderPath)

    # Copy the main file for reproducibility
    shutil.copy(Path(__file__).absolute(), work_dir,follow_symlinks=True)

# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_del/6_set1_L3_template/Ev/Lev3_AA/Run/Checkpoints/5517"
# )
# make_filtered_checkpoint_from_another(
#     CheckpointFolderPath,
# )

# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff0/Ev/Lev3_AA/Run/Checkpoints/5517"
# )
# make_filtered_checkpoint_from_another(
#     CheckpointFolderPath,
#     filter_coeffs_buffer=0
# )

# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff2/Ev/Lev3_AA/Run/Checkpoints/5517"
# )
# make_filtered_checkpoint_from_another(
#     CheckpointFolderPath,
#     filter_coeffs_buffer=2
# )

# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff4/Ev/Lev3_AA/Run/Checkpoints/5517"
# )
# make_filtered_checkpoint_from_another(
#     CheckpointFolderPath,
#     filter_coeffs_buffer=4
# )

# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff0_14012/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# make_filtered_checkpoint_from_another(
#     CheckpointFolderPath,
#     filter_coeffs_buffer=0
# )

# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff2_14012/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# make_filtered_checkpoint_from_another(
#     CheckpointFolderPath,
#     filter_coeffs_buffer=2
# )

# CheckpointFolderPath = Path(
#     "/groups/sxs/hchaudha/spec_runs/19_filtered_checkpoint_runs/6_set1_L3_fil_buff4_14012/Ev/Lev3_AA/Run/Checkpoints/14012"
# )
# make_filtered_checkpoint_from_another(
#     CheckpointFolderPath,
#     filter_coeffs_buffer=4
# )
