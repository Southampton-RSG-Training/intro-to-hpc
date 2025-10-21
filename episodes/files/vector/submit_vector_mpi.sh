#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:01:00

module load openmpi
mpicc vector_mpi.exe -o vector_mpi.exe
srun ./vector_mpi.exe
