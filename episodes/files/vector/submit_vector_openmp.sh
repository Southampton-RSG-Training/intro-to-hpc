#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:01:00

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gcc
gcc -fopenmp vector_openmp.exe -o vector_openmp.exe
./vector_openmp.exe
