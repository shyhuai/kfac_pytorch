#!/bin/bash
#dnn="${dnn:-resnet32}"
dnn="${dnn:-resnet50}"
nworkers="${nworkers:-4}"
MPIPATH=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/pytorch1.4/bin/python

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0 \
    $PY examples/pytorch_imagenet_resnet.py \
          --base-lr 0.0125 --epochs 55 --kfac-update-freq 0 --model $dnn  --lr-decay 25 35 40 45 50 \
          --train-dir /localdata/ILSVRC2012_dataset/train \
          --val-dir /localdata/ILSVRC2012_dataset/val

    #$PY examples/pytorch_cifar10_resnet.py \
    #--epochs 100 --kfac-update-freq 10 --model $dnn --lr-decay 35 75 90 --base-lr 0.1 \
    #--dir /home/esetstore/repos/p2p/data

