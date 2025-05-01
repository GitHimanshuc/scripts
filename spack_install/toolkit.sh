#!/bin/zsh -
#SBATCH -J oneapi                   # Job Name
#SBATCH -n 4                  # Number of cores
#SBATCH --ntasks-per-node 4        # number of MPI ranks per node
#SBATCH -p expansion                  # Queue name
#SBATCH -t 10:0:00   # Run time
#SBATCH -A sxs                # Account name
#SBATCH --reservation=sxs_standing

spack graph intel-oneapi-advisor intel-oneapi-compilers intel-oneapi-inspector intel-oneapi-vtune intel-oneapi-itac intel-oneapi-mpi intel-oneapi-tbb