#!/bin/bash
#dnn="${dnn:-resnet32}"
dnn="${dnn:-resnet50}"
nworkers="${nworkers:-4}"
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=/home/comp/15485625/pytorch1.4/bin/python

params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include em1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=em1 \
    -x NCCL_TREE_THRESHOLD=0"

if [ "$dnn" = "resnet50" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile sci-cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_imagenet_resnet.py \
          --base-lr 0.0125 --epochs 55 --kfac-update-freq 0 --model $dnn  --lr-decay 25 35 40 45 50 --batch-size 128 \
          --train-dir /home/datasets/imagenet/ILSVRC2012_dataset/train \
          --val-dir /home/datasets/imagenet/ILSVRC2012_dataset/val
else
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_cifar10_resnet.py \
        --base-lr 0.1 --epochs 100 --kfac-update-freq 1 --model $dnn --lr-decay 35 75 90 --dir ./data
fi
