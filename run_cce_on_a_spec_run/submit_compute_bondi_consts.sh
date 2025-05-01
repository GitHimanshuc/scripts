#!/bin/zsh -
#SBATCH -J casbc
#SBATCH -o casbc.stdout                 # Output file name
#SBATCH -e casbc.stderr                 # Error file name
#SBATCH -n 8                          # Number of cores
#SBATCH -p expansion                  # Queue name
#SBATCH --ntasks-per-node 8           # number of MPI ranks per node
#SBATCH -t 8:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue
#SBATCH --reservation=sxs_standing

set -x
# sleep(5)
cd /groups/sxs/hchaudha/scripts/run_cce_on_a_spec_run
ls -l

which python
ls -l

PYTHONUNBUFFERED=1 python ./compute_and_save_bondi_violations.py >> ./compute_and_save_bondi_violations.out