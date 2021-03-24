#!/bin/bash
dnn="${dnn:-resnet50}"
nworkers="${nworkers:-4}"
batch_size="${batch_size:-32}"
rdma="${rdma:-1}"
kfac="${kfac:-1}"
epochs="${epochs:-55}"
kfac_name="${kfac_name:-inverse}"
exclude_parts="${exclude_parts:-''}"

MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=/home/comp/15485625/pytorch1.4/bin/python

params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include em1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=em1 \
    -x NCCL_TREE_THRESHOLD=0"

if [ "$dnn" = "resnet32" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_cifar10_resnet.py \
        --base-lr 0.1 --epochs 100 --kfac-update-freq $kfac --model $dnn --lr-decay 35 75 90 --dir ./data
else
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile sci-cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_imagenet_resnet.py \
          --base-lr 0.0125 --epochs $epochs --kfac-update-freq $kfac --kfac-cov-update-freq $kfac --model $dnn --kfac-name $kfac_name --exclude-parts ${exclude_parts} --batch-size $batch_size --lr-decay 25 35 40 45 50 \
          --train-dir /home/datasets/imagenet/ILSVRC2012_dataset/train \
          --val-dir /home/datasets/imagenet/ILSVRC2012_dataset/val

fi
