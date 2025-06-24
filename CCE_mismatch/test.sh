#!/bin/bash -l
#SBATCH -J NRNR                   # Job Name
#SBATCH -o SpEC.stdout                # Output file name
#SBATCH -e SpEC.stderr                # Error file name
#SBATCH -n 5                  # Number of cores
#SBATCH --ntasks-per-node 5        # number of MPI ranks per node
#SBATCH -p expansion                  # Queue name
#SBATCH -t 10:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue
#SBATCH --reservation=sxs_standing

echo $0

set -x

source /groups/sxs/hchaudha/softwares/anaconda3/etc/profile.d/conda.sh

conda activate sxs
which python

cd /resnick/groups/sxs/hchaudha/spec_runs/CCE_mismatch/code
python ./NR_NR_fix_job.py > ./NR_NR_fix_job.out 2>&1