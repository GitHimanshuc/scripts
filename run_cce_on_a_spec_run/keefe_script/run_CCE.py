#!/usr/local/python/anaconda3-2019.10/bin/pythonbondi_cce

# python /home/kmitman/AutomateCce/run_CCE.py -i CharacteristicExtract.yaml -n 2 -o bondi_cce -d "[...,...,...]"

"""
This will run SpECTRE's CCE on the worldtube files in a provided directory.
The metadata file (either a .txt or .json) must be in the same directory.
"""

import os
import sys
import sxs
import scri
import h5py
import glob
import json
import scipy
import argparse
import numpy as np

cwd = os.getcwd() + '/'
input_command = 'python ' + ' '.join(sys.argv)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawTextHelpFormatter)

p.add_argument(
    '--worldtube_prefix',
    '-w',
    dest='worldtube_prefix',
    type=str,
    default='BondiCce',
    help='Prefix for worldtube file.')
p.add_argument(
    '--dirs',
    '-d',
    dest='dirs',
    type=str,
    default='.',
    help='Directories containing CCE worldtubes, i.e., must contain:\n'\
    '- BondiCceR{XXXX}.h5 for XXXX some number, e.g., 0100, where'\
    'BondiCce is the worldtube prefix,\n'\
    'metadata.txt or metadata.json')
p.add_argument(
    '--number_of_simulations_per_node',
    '-n',
    dest='number_of_simulations_per_node',
    type=int,
    default=3,
    help='Number of simulations per node to run (1 <= n <= 3).')
p.add_argument(
    '--input_filename',
    '-i',
    dest='input_filename',
    type=str,
    default='CharacteristicExtract.yaml',
    help='Directory where CharacteristicExtract has been built.')
p.add_argument(
    '--spectre_build_dir',
    '-s',
    dest='spectre_build_dir',
    type=str,
    default='/home/kmitman/spectre_new/build/',
    help='Directory where CharacteristicExtract has been built.')
p.add_argument(
    '--automate_dir',
    '-a',
    dest='automate_dir',
    type=str,
    default='/home/kmitman/AutomateCce/',
    help='Directory where the AutomateCce files are stored.')
p.add_argument(
    '--post_process',
    '-p',
    dest='post_process',
    default=False,
    action='store_true',
    help='Whether or not to only post process the volume files.')
p.add_argument(
    '--output_dir_name',
    '-o',
    dest='output_dir_name',
    type=str,
    default='bondi_cce',
    help='Directory where the waveform files will be stored.')
p.add_argument(
    '--compute_bianchi_violations',
    '-c',
    dest='compute_bianchi_violations',
    default=False,
    action='store_true',
    help='Whether or not to compute the violation of the Bianchi identities.')
p.add_argument(
    '--keep_volume_files',
    '-k',
    dest='keep_volume_files',
    default=False,
    action='store_true',
    help='Whether or not to keep the volume files produced by CCE.')

args = p.parse_args()

def process_inputs(args):
    directories_and_files = {}
    for i, directory in enumerate(args.dirs.strip('][').split(',')):
        if directory[-1] != '/':
            directory += '/'
            
        if not os.path.exists(directory):
            print(f"The directory {directory} does not exist. Skipping this input.")
            continue
        
        files = [x for x in os.listdir(directory) if args.worldtube_prefix in x and '.h5' in x]
        if len(files) == 0:
            print(f"No worldtube files found. Skipping this input.")
            continue

        if not (os.path.exists(f'{directory}metadata.txt') or os.path.exists(f'{directory}metadata.json')):
            print(f"No metadata file found. Skipping this input.")
            continue
        
        radii = [x.split('R')[1][:4] for x in files]
        directories_and_files[i] = {
            'directory': directory,
            'radii': radii
        }

    if args.number_of_simulations_per_node < 1 or args.number_of_simulations_per_node > 3:
        raise ValueError("Number of simulations per node must be between 1 and 3.")

    if not os.path.exists(args.spectre_build_dir):
        raise ValueError("The SpECTRE build directory does not exist.")
    
    if not os.path.exists(args.automate_dir):
        raise ValueError("The AutomateCce direcotry does not exist.")
    
    if not os.path.exists(f'{args.automate_dir}yaml_files/{args.input_filename}'):
        raise ValueError("The input file does not exist.")
    
    return directories_and_files

def copy_and_replace_info_in_input_file(directory, automate_dir, input_filename, worldtube_prefix, radius):
    input_file = f'{automate_dir}yaml_files/{input_filename}'
    initialization_filename = f'CharacteristicExtract_R{radius}.yaml'
    initialization_filename_temp = (initialization_filename + '.')[:-1]
    initialization_filename_temp = initialization_filename_temp.replace('.', f'_template.')

    os.system(f'cp {input_file} {directory}CharacteristicExtract.yaml')
    os.system(f'cp {input_file} {directory}{initialization_filename_temp}')
    
    with open(f'{directory}{initialization_filename_temp}', 'rt') as f_input:
        with open(f'{directory}{initialization_filename}', 'wt') as f_output:
            for line in f_input:
                f_output.write(line
                               .replace('RADIUS', radius)
                               .replace('WORLDTUBE_PREFIX', worldtube_prefix))

    os.system(f'rm {directory}{initialization_filename_temp}')

def create_submission_file(directories_and_files, automate_dir, spectre_build_dir, worldtube_prefix, submission_idx):
    os.system(f'cp {automate_dir}/submission_script_base.sh {cwd}/submission_script_template.sh')
    
    with open(f'{cwd}submission_script_template.sh', 'rt') as f_input:
        with open(f'{cwd}submission_script_{submission_idx}.sh', 'wt') as f_output:
            for line in f_input:
                if '# GIT GET' in line:
                    for i, idx in enumerate(directories_and_files):
                        directory = directories_and_files[idx]['directory']
                    
                        f_output.write(f'cd {directory}\n')
                        f_output.write('#\n')
                        f_output.write(f'git annex get {worldtube_prefix}*\n')
                        f_output.write('#\n')
                if '# CCE JOBS' in line:
                    for i, idx in enumerate(directories_and_files):
                        directory = directories_and_files[idx]['directory']
                        radii = directories_and_files[idx]['radii']
                        
                        f_output.write(f'cd {directory}\n')
                        f_output.write('#\n')
                        for j, radius in enumerate(radii):
                            initialization_filename = f'CharacteristicExtract_R{radius}.yaml'

                            f_output.write(f'{spectre_build_dir}bin/CharacteristicExtract ++ppn 1 +setcpuaffinity +pemap '\
                                           + f'{2*len(radii)*i+2*j} +commap {2*len(radii)*i+2*j+1} --input-file ./{initialization_filename} 2>&1 &\n')
                            f_output.write('last_pid=$!\n')
                            f_output.write('cce_pids+=("$last_pid")\n')
                            f_output.write('#\n')
                elif '# GIT DROP' in line:
                    for i, idx in enumerate(directories_and_files):
                        directory = directories_and_files[idx]['directory']

                        f_output.write(f'cd {directory}\n')
                        f_output.write('#\n')
                        f_output.write(f'git annex drop {worldtube_prefix}*\n')
                        f_output.write('#\n')
                elif '# POST PROCESS' in line:
                    f_output.write(f'cd {cwd}\n')
                    
                    first_part = input_command.split('[')[0]
                    new_part = ','.join([directories_and_files[idx]['directory'] for idx in directories_and_files])
                    later_part = input_command.split(']')[1]
                    f_output.write(first_part + '[' + new_part + ']' + later_part + ' -p')
                else:
                    f_output.write(line.replace('SPECTRE_BUILD_DIR', spectre_build_dir))

    os.system(f'rm {cwd}/submission_script_template.sh')
                        
    return
    
def submit_jobs(directories_and_files, input_filename, worldtube_prefix,
                spectre_build_dir, automate_dir, keep_volume_files, submission_idx):
    print(f"Submitting N = {len(directories_and_files)} jobs.")
    
    # For each directory, submit the corresponding runs
    for i in directories_and_files:
        directory = directories_and_files[i]['directory']
        radii = directories_and_files[i]['radii']

        for radius in radii:
            copy_and_replace_info_in_input_file(directory, automate_dir, input_filename, worldtube_prefix, radius)

    create_submission_file(directories_and_files, automate_dir, spectre_build_dir, worldtube_prefix, submission_idx)

    os.chdir(cwd)

    os.system(f'sbatch submission_script_{submission_idx}.sh')

def block_mode_data_to_ylm_timeseries(volume_file, output_dir_name, output_prefix):
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
        
def post_process_jobs(directories_and_files, output_dir_name):
    cce_scri_data_names = {
        'Strain' : (scri.h, 'rhOverM'),
        'News'   : (scri.hdot, 'r2News'),
        'Psi4'   : (scri.psi4, 'rMPsi4'),
        'Psi3'   : (scri.psi3, 'r2Psi3'),
        'Psi2'   : (scri.psi2, 'r3Psi2OverM'),
        'Psi1'   : (scri.psi1, 'r4Psi1OverM2'),
        'Psi0'   : (scri.psi0, 'r5Psi0OverM3')
    }
    
    for i in directories_and_files:
        directory = directories_and_files[i]['directory']
        radii = directories_and_files[i]['radii']

        os.system(f'rm -r {directory}/{output_dir_name} 2> /dev/null')
        os.system(f'mkdir {directory}/{output_dir_name}')

        for radius in radii:
            block_mode_data_to_ylm_timeseries(f'{directory}CharacteristicExtractVolume_R{radius}.h5',
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

def clean_up_directories(directories_and_files, output_dir_name, keep_volume_files, automate_dir):
    for idx in directories_and_files:
        directory = directories_and_files[idx]['directory']

        if not keep_volume_files:
            os.system(f'rm {directory}/CharacteristicExtractVolume_R*.h5')
        os.system(f'rm {directory}/CharacteristicExtract_*.yaml')
        os.system(f'rm {directory}/{output_dir_name}/*unprocessed.h5')

    os.system(f'rm {cwd}submission_script_*.sh')
        
def compute_bianchi_violations(directories_and_files, output_dir_name):
    for i in directories_and_files:
        directory = directories_and_files[i]['directory']
        radii = directories_and_files[i]['radii']
        
        bianchi_violations = {}
        min_bianchi_violation = np.inf
        for radius in radii:
            abd = scri.SpEC.file_io.create_abd_from_h5(h=f'{directory}{output_dir_name}/rhOverM_BondiCce_R{radius}.h5',
                                                       Psi4=f'{directory}{output_dir_name}/rMPsi4_BondiCce_R{radius}.h5',
                                                       Psi3=f'{directory}{output_dir_name}/r2Psi3_BondiCce_R{radius}.h5',
                                                       Psi2=f'{directory}{output_dir_name}/r3Psi2OverM_BondiCce_R{radius}.h5',
                                                       Psi1=f'{directory}{output_dir_name}/r4Psi1OverM2_BondiCce_R{radius}.h5',
                                                       Psi0=f'{directory}{output_dir_name}/r5Psi0OverM3_BondiCce_R{radius}.h5',
                                                       file_format='SXS')
            
            violations = abd.bondi_violation_norms
            total_violations = []
            for violation in violations:
                total_violations.append(scipy.integrate.trapezoid(violation, abd.t))
                bianchi_violations[radius] = total_violations
                
            # rank based on Psi2 violation
            violation = total_violations[2]
            if violation < min_bianchi_violation:
                best_radius = radius
                min_bianchi_violation = violation

        bianchi_violations['best Psi2 violation'] = best_radius

        with open(f'{directory}bianchi_violations.json', 'w') as f:
            json.dump(bianchi_violations, f, indent=2, separators=(",", ": "), ensure_ascii=True)

if __name__ == '__main__':
    # Get directories that contain the files that we will run CCE on 
    directories_and_files = process_inputs(args)
    number_of_simulations_per_node = args.number_of_simulations_per_node 

    # Coordinate submissions
    number_of_submissions = int(np.ceil(len(directories_and_files) / number_of_simulations_per_node))

    if not args.post_process:
        for n in range(number_of_submissions):
            submit_jobs(dict([(key, value) for key,value in directories_and_files.items()
                              if key in range(n*number_of_simulations_per_node, (n + 1)*number_of_simulations_per_node)]),
                        args.input_filename,
                        args.worldtube_prefix,
                        args.spectre_build_dir,
                        args.automate_dir,
                        args.keep_volume_files,
                        n)
    else:
        print("post processing")
        post_process_jobs(directories_and_files, args.output_dir_name)
        
        if args.compute_bianchi_violations:
            compute_bianchi_violations(directories_and_files, args.output_dir_name)
            
        clean_up_directories(directories_and_files, args.output_dir_name, args.keep_volume_files, args.automate_dir)
            
