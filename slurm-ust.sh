#!/bin/bash
#SBATCH -e logs/slurm-error.log
#SBATCH -o logs/slurm.log
#SBATCH -N 1
#SBATCH -p gpu-share
#SBATCH -w hhnode-ib-104 
source ~/.bashrc
srun ./horovod_mpi_ust.sh

