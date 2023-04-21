#!/usr/local/python/anaconda3-2019.10/bin/python
"""
This will fix the BMS frame of CCE waveforms
to be either the PN BMS frame or the superrest frame.
"""

import os
import sys
import scri
import scipy
import argparse
import numpy as np

cwd = os.getcwd() + '/'
input_command = 'python ' + ' '.join(sys.argv)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawTextHelpFormatter)

p.add_argument(
    '--dirs',
    '-d',
    dest='dirs',
    type=str,
    default='.',
    help='Directories containing a CCE directory and its metadata file.')
p.add_argument(
    '--cce_prefix',
    '-p',
    dest='cce_prefix',
    type=str,
    default='bondi_cce',
    help='CCE directory.')
p.add_argument(
    '--number_of_simulations_per_node',
    '-n',
    dest='number_of_simulations_per_node',
    type=int,
    default=24,
    help='Number of simulations per node to run (1 <= n <= 24).')
p.add_argument(
    '--automate_dir',
    '-a',
    dest='automate_dir',
    type=str,
    default='/home/kmitman/AutomateCce/',
    help='Directory where the AutomateCce files are stored.')
p.add_argument(
    '--target_frame',
    '-f',
    dest='target_frame',
    type=str,
    default='superrest',
    help='Target BMS frame (superrest or PN BMS).')
p.add_argument(
    '--time',
    '-t',
    dest='time',
    type=float,
    default=0.0,
    help='Time at which to map to the superrest frame, or the start of the matching window for the PN BMS frame.')
p.add_argument(
    '--n_orbits',
    '-c',
    dest='n_orbits',
    type=float,
    default=3,
    help='The size of the matching window for the PN BMS frame in orbits.')
p.add_argument(
    '--PN_strain',
    '-i',
    dest='PN_strain_name',
    type=str,
    default='rhOverM_Inertial_PN_Dict_T4.h5',
    help='PN strain filename.')
p.add_argument(
    '--PN_supermomentum',
    '-j',
    dest='PN_supermomentum_name',
    type=str,
    default='supermomentum_PN_Dict_T4.h5',
    help='PN supermomentum filename.')
p.add_argument(
    '--EXT_strain',
    '-k',
    dest='EXT_strain_name',
    type=str,
    default='rhOverM_Asymptotic_GeometricUnits_CoM.h5/Extrapolated_N2.dir',
    help='EXT strain filename.')
p.add_argument(
    '--save_suffix',
    '-s',
    dest='save_suffix',
    type=str,
    default='',
    help='A suffix to append to the saved files.')
p.add_argument(
    '--json_name',
    '-o',
    dest='json_name',
    type=str,
    default='PN_BMS_errors.json',
    help='JSON file to write to.')

args = p.parse_args()

def process_inputs(args):
    directories_and_files = {}
    for i, directory in enumerate(args.dirs.strip('][').split(',')):
        if directory[-1] != '/':
            directory += '/'
            
        if not os.path.exists(directory):
            print(f"The directory {directory} does not exist. Skipping this input.")
            continue
        
        if not os.path.exists(f'{directory}{args.cce_prefix}'):
            print(f"No CCE directory found. Skipping this input.")
            continue

        if not (os.path.exists(f'{directory}metadata.txt') or os.path.exists(f'{directory}metadata.json')):
            print(f"No metadata file found. Skipping this input.")
            continue
            
        directories_and_files[i] = {
            'directory': directory,
            'radius': None
        }

    if args.number_of_simulations_per_node < 1 or args.number_of_simulations_per_node > 24:
        raise ValueError("Number of simulations per node must be between 1 and 24.")
    
    if not os.path.exists(args.automate_dir):
        raise ValueError("The AutomateCce direcotry does not exist.")
    
    return directories_and_files

def create_submission_file(directories_and_files, cce_prefix, automate_dir, target_frame, time, n_orbits, PN_strain_name, PN_supermomentum_name, EXT_strain_name, save_suffix, json_name, submission_idx):
    os.system(f'cp {automate_dir}/submission_script_frame_fixing_base.sh {cwd}/submission_script_frame_fixing_template.sh')
    
    with open(f'{cwd}submission_script_frame_fixing_template.sh', 'rt') as f_input:
        with open(f'{cwd}submission_script_frame_fixing_{submission_idx}.sh', 'wt') as f_output:
            for line in f_input:
                if '# FRAME FIXING JOBS' in line:
                    for i, idx in enumerate(directories_and_files):
                        directory = directories_and_files[idx]['directory']
                        radius = directories_and_files[idx]['radius']
                        
                        f_output.write(f'cd {directory}\n')
                        f_output.write('#\n')
                        
                        if target_frame == 'superrest':
                            f_output.write(f'python {automate_dir}fix_BMS_frame.py --cce_prefix {cce_prefix} --dir {directory} --target_frame {target_frame} --time {time}\n')
                        elif target_frame == 'PN_BMS':
                            f_output.write(f'python {automate_dir}fix_BMS_frame.py --cce_prefix {cce_prefix} --dir {directory} --target_frame {target_frame} --time {time} --n_orbits {n_orbits} ' +
                                           f'--PN_strain {PN_strain_name} --PN_supermomentum {PN_supermomentum_name} --EXT_strain {EXT_strain_name} --save_suffix {save_suffix} --json_name {json_name}\n')
                        f_output.write('#\n')
                        
                        f_output.write('last_pid=$!\n')
                        f_output.write('# Pin the job to a certain core.\n')
                        f_output.write(f'taskset -pc {i} $last_pid\n')
                        f_output.write('#\n')
                else:
                    f_output.write(line)

    os.system(f'rm {cwd}/submission_script_frame_fixing_template.sh')
                        
    return
    
def submit_jobs(directories_and_files, cce_prefix, automate_dir, target_frame, time, n_orbits, PN_strain_name, PN_supermomentum_name, EXT_strain_name, save_suffix, json_name, submission_idx):
    print(f"Submitting N = {len(directories_and_files)} jobs.")
    
    create_submission_file(directories_and_files, cce_prefix, automate_dir, target_frame, time, n_orbits, PN_strain_name, PN_supermomentum_name, EXT_strain_name, save_suffix, json_name, submission_idx)

    os.chdir(cwd)

    os.system(f'sbatch submission_script_frame_fixing_{submission_idx}.sh')

if __name__ == '__main__':
    # Get directories that contain the files that we will run CCE on 
    directories_and_files = process_inputs(args)
    number_of_simulations_per_node = args.number_of_simulations_per_node 

    # Coordinate submissions
    number_of_submissions = int(np.ceil(len(directories_and_files) / number_of_simulations_per_node))

    for n in range(number_of_submissions):
        submit_jobs(dict([(key, value) for key,value in directories_and_files.items()
                          if key in range(n*number_of_simulations_per_node, (n + 1)*number_of_simulations_per_node)]),
                    args.cce_prefix,
                    args.automate_dir,
                    args.target_frame,
                    args.time,
                    args.n_orbits,
                    args.PN_strain_name,
                    args.PN_supermomentum_name,
                    args.EXT_strain_name,
                    args.save_suffix,
                    args.json_name,
                    n)

    os.system(f'rm {cwd}/submission_script_frame_fixing_*')
