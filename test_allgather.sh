#!/bin/bash
nworkers="${nworkers:-4}"
rdma="${rdma:-1}"
gpuids="${gpuids:-0,1,2,3}"

MPIPATH=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/pytorch1.4/bin/python

if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=VERSION \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0 \
    -x CUDA_VISIBLE_DEVICES=${gpuids}"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=VERSION \
    -x HOROVOD_CACHE_CAPACITY=0"
fi
    #-x HOROVOD_FUSION_THRESHOLD=0 \

#HOROVOD_TIMELINE=./logs/profile-timeline-${dnn}-kfac-${kfac}-json.log 
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY scripts/bench_communication.py
    #$PY scripts/test_allgather.py
