import subprocess
import shutil
import sys
import matplotlib.pyplot as plt
# plt.style.use('seaborn-talk')
plt.rcParams["figure.figsize"] = (12,10)
import re
import time
from pathlib import Path
spec_home="/home/hchaudha/spec"


def submit_bianchi_violation_job(cce_folder_path: Path, binachi_violation_script: Path, conda_env='sxs',submit_job=True):
    
    if not binachi_violation_script.exists():
        raise Exception(f"{binachi_violation_script} does not exists.")

    if not cce_folder_path.exists():
        raise Exception(f"{cce_folder_path} does not exists.")

    run_name = f"bianchi_{cce_folder_path.parent.stem}"
    run_path = cce_folder_path

    shutil.copy(binachi_violation_script,run_path)
        

    submit_script=\
f"""#!/bin/bash -
#SBATCH -J {run_name}              # Job Name
#SBATCH -o SpEC.stdout                # Output file name
#SBATCH -e SpEC.stderr                # Error file name
#SBATCH -n 1                          # Number of cores
#SBATCH --ntasks-per-node 1        # number of MPI ranks per node
#SBATCH -t 1:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue
#SBATCH --reservation=sxs_standing
#SBATCH --constraint=skylake

# Go to the correct cce folder 
cd {run_path}

# activate the environment and run python script
source ~/.zshrc
conda activate {conda_env}
conda env list
which python
python {binachi_violation_script.name}

"""
    submit_script_path = cce_folder_path/"bianchi_submit.sh"
    with submit_script_path.open('w') as f:
        f.writelines(submit_script)
    if submit_job:
        command = f"cd {submit_script_path.parent} && qsub {submit_script_path}"
        status = subprocess.run(command, capture_output=True, shell=True, text=True)
        if status.returncode == 0:
          print(f"Succesfully submitted {submit_script_path}\n{status.stdout}")
        else:
          sys.exit(
              f"Job submission failed for {submit_script_path} with error: \n{status.stdout} \n{status.stderr}")

def are_runs_going_on(re_text=r"Lev\d_R\d\d\d\d",user='himanshu'):
    command = f"qstat -u {user}"
    status = subprocess.run(command, capture_output=True, shell=True, text=True)

    qstat_output = status.stdout.split("\n")

    JOBS_STILL_PENDING = False

    for line in qstat_output:
        matches = list(re.finditer(re_text, line, re.MULTILINE))
        if len(matches)>0:
            JOBS_STILL_PENDING = True

    return JOBS_STILL_PENDING
    

runs_paths = [
    Path('/groups/sxs/hchaudha/spec_runs/high_accuracy_L35')
]

while True:
    if are_runs_going_on(re_text=r"Lev\d_R\d\d\d\d",user='himanshu'):
        time.sleep(10)
    else:
        for path in runs_paths:
            cce_folder_path = path/"cce_bondi"
            binachi_violation_script = Path("/groups/sxs/hchaudha/scripts/run_cce_on_a_spec_run/bianchi_violation.py")
            submit_bianchi_violation_job(cce_folder_path=cce_folder_path,binachi_violation_script=binachi_violation_script, submit_job=True)
        break
            
  