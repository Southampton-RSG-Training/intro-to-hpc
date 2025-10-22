#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:01:00

module load openmpi/5.0.3
mpicc vector_mpi.c -o vector_mpi.exe
srun ./vector_mpi.exe
