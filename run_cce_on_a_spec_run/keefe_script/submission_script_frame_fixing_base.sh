#!/bin/bash -
#SBATCH -J FrameFixing              # Job Name
#SBATCH -o FrameFixing-%j.stdout       # Output file name
#SBATCH -e FrameFixing-%j.stderr       # Error file name
#SBATCH -n 1                     # Number of cores
#SBATCH --ntasks-per-node 1      # Number of MPI ranks per node
#SBATCH -c 24                    # Get 24 cores on the node
#SBATCH -p productionQ
#SBATCH -t 24:00:00              # Run time
#SBATCH -A sxs                   # Account name
#SBATCH --no-requeue
#SBATCH --exclude=wheeler061,wheeler063,wheeler099,wheeler105,wheeler110,wheeler126

# FRAME FIXING JOBS
