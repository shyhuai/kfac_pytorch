#!/bin/bash
dnn="${dnn:-resnet32}"
#dnn="${dnn:-resnet50}"
nworkers="${nworkers:-4}"
batch_size="${batch_size:-32}"
rdma="${rdma:-0}"
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=/home/comp/20481896/shshi/pytorch1.4/bin/python

if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include bond0 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=bond0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=INFO \
    -x HOROVOD_CACHE_CAPACITY=0"
fi

if [ "$dnn" = "resnet32" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile daai-cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_cifar10_resnet.py \
        --base-lr 0.1 --epochs 100 --kfac-update-freq 0 --model $dnn --lr-decay 35 75 90 --batch-size $batch_size --dir /home/datasets/cifar10
else
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile daai-cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_imagenet_resnet.py \
          --base-lr 0.0125 --epochs 55 --kfac-update-freq 1 --model $dnn  --batch-size $batch_size --lr-decay 25 35 40 45 50 \
          --train-dir /home/datasets/ILSVRC2012_dataset/train --val-dir /home/datasets/ILSVRC2012_dataset/val
fi
