#!/bin/bash

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00

module load gcc
gcc vector_serial.c -o vector_serial.exe
./vector_serial.exe
