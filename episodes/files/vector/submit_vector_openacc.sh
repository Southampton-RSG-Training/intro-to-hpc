#!/bin/bash

#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

module load nvhpc
nvc -acc vector_openacc.c -o vector_openacc.exe
nvidia-smi
./vector_openacc.exe
