#!/bin/bash -
#SBATCH -J pvpython                  # Job Name
#SBATCH -o SpEC.stdout                # Output file name
#SBATCH -e SpEC.stderr                # Error file name
#SBATCH -n 1                  # Number of cores
#SBATCH --ntasks-per-node 1        # number of MPI ranks per node
#SBATCH -t 1:0:00             # Run time
#SBATCH -A sxs                # Account name
#SBATCH --no-requeue


/panfs/ds09/sxs/himanshu/softwares/ParaView-5.10.0-osmesa-MPI-Linux-Python3.9-x86_64/bin/pvpython /panfs/ds09/sxs/himanshu/scripts/plotting/paraview/load_and_take_slice.py