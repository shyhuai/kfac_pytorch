#!/bin/bash
#SBATCH -e logs/slurm-error.log
#SBATCH -o logs/slurm.log
#SBATCH -N 2
#SBATCH -w hkbugpusrv04,hkbugpusrv05
#srun -n 2 /home/comp/20481896/shshi/pytorch1.4/bin/python /home/comp/20481896/shshi/helloworld.py
source ~/.bashrc
srun ./horovod_mpi_daai.sh

