#!/bin/zsh -
#SBATCH -J gcc12                   # Job Name
#SBATCH -n 4                  # Number of cores
#SBATCH --ntasks-per-node 4        # number of MPI ranks per node
#SBATCH -p expansion                  # Queue name
#SBATCH -t 10:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --reservation=sxs_standing

spack install gcc@12