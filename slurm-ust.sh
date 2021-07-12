#!/bin/bash
#SBATCH -e logs/slurm-error.log
#SBATCH -o logs/slurm.log
#SBATCH -N 1
#SBATCH --cpus-per-task 16
#SBATCH -p gpu-share
#SBATCH -w hhnode-ib-144
source ~/.bashrc
srun ./horovod_mpi_ust.sh

