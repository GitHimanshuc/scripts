# Call from the cce folder and this will load the path dict from the pickel and start the bianchi stuff.

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
plt.style.use('seaborn-talk')
plt.rcParams["figure.figsize"] = (12,10)
import json
import pickle
from pathlib import Path
import scri
import h5py
import sxs
import scipy
import scipy.integrate as integrate
spec_home="/home/himanshu/spec/my_spec"


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
    plt.savefig(save_dir/"violations.png",bbox_inches='tight')
    plt.close()

        
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
        bd_data_path = list(bd_data_path.parent.glob("redu_*.h5"))[0]
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

path_dict = {}
with open("./path_dict.pkl",'rb') as f:
    path_dict = pickle.load(f)

post_process_jobs(path_dict)