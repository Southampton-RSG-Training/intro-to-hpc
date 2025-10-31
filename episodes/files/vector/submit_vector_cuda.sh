#!/bin/bash

#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

module load nvhpc
nvcc vector_cuda.cu -o vector_cuda.exe
nvidia-smi
./vector_cuda.exe
