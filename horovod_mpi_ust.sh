#!/bin/bash
#dnn="${dnn:-resnet32}"
dnn="${dnn:-resnet50}"
nworkers="${nworkers:-8}"
batch_size="${batch_size:-32}"
rdma="${rdma:-1}"
kfac="${kfac:-1}"
lr="${lr:-0.1}"
sparse_ratio="${sparse_ratio:-1}"
epochs="${epochs:-55}"
kfac_name="${kfac_name:-inverse}"
damping="${damping:-0.003}"
exclude_parts="${exclude_parts:-''}"
gpuids="${gpuids:-0,1,2,3}"

MPIPATH=/home/shaohuais/.local/openmpi-4.0.3
PY=/home/shaohuais/pytorch1.4/bin/python

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
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ust-cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_cifar10_resnet.py \
        --base-lr $lr --epochs 100 --kfac-update-freq $kfac --model $dnn --lr-decay 35 75 90 --batch-size $batch_size --sparse-ratio $sparse_ratio --kfac-name $kfac_name --damping $damping --warmup-epochs 0 --dir /scratch/PI/shaohuais/cifar10
else
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile ust-cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_imagenet_resnet.py \
          --base-lr $lr --epochs $epochs --kfac-update-freq $kfac --kfac-cov-update-freq $kfac --model $dnn --kfac-name $kfac_name --sparse-ratio $sparse_ratio --exclude-parts ${exclude_parts} --batch-size $batch_size --lr-decay 25 35 40 45 50 \
          --train-dir /scratch/PI/shaohuais/imagenet/train --val-dir /scratch/PI/shaohuais/imagenet/val
fi
