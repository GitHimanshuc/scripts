#!/bin/bash -
#SBATCH -J casbc_2
#SBATCH -n 16                          # Number of cores
#SBATCH -p expansion                  # Queue name
#SBATCH --ntasks-per-node 16           # number of MPI ranks per node
#SBATCH -t 24:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue
#SBATCH --reservation=sxs_standing
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /resnick/groups/sxs/hchaudha/scripts/run_cce_on_a_spec_run
source /groups/sxs/hchaudha/softwares/anaconda3/etc/profile.d/conda.sh

conda activate sxs
conda info --envs
which python

PYTHONUNBUFFERED=1 python ./compute_and_save_bondi_violations_del.py > ./compute_and_save_bondi_violations_del.out 2>&1
