#!/bin/bash
MPI_HOME=/home/esetstore/.local/openmpi-4.0.1
# 100GbIB
$MPI_HOME/bin/mpirun --prefix $MPI_HOME --oversubscribe -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot -mca pml ob1 -mca btl openib -mca btl_openib_allow_ib 1 \
-x LD_LIBRARY_PATH \
-x NCCL_DEBUG=INFO \
-x CUDA_VISIBLE_DEVICES=0,1,2,3 \
-x NCCL_SOCKET_IFNAME=ib0 \
python tests/test_comm.py
