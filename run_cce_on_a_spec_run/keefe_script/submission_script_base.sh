#!/bin/bash -
#SBATCH -J BondiCce              # Job Name
#SBATCH -o BondiCce-%j.stdout       # Output file name
#SBATCH -e BondiCce-%j.stderr       # Error file name
#SBATCH -n 1                     # Number of cores
#SBATCH --ntasks-per-node 1      # Number of MPI ranks per node
#SBATCH -c 24                    # Get 24 cores on the node
#SBATCH -p productionQ
#SBATCH -t 24:00:00              # Run time
#SBATCH -A sxs                   # Account name
#SBATCH --no-requeue
#SBATCH --exclude=wheeler061,wheeler063,wheeler085,wheeler086,wheeler099,wheeler101,wheeler105,wheeler110,wheeler126

#
# GIT GET
source SPECTRE_BUILD_DIR../support/Environments/wheeler_clang.sh
module purge
spectre_load_modules
export OPENBLAS_NUM_THREADS=1
#
# cce_pids is an array of proccess IDs (PIDs) that we wait on for completion.
cce_pids=()
#
# CCE JOBS
wait ${cce_pids[@]}
#
source /home/kmitman/SpEC/MakefileRules/this_machine.env
module unload python/anaconda2-4.1.1
module load python/anaconda3-2019.10
source deactivate
source activate scri
#
# GIT DROP
#
# POST PROCESS
