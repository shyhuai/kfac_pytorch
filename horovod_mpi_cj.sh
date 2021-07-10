#!/bin/bash
#dnn="${dnn:-resnet32}"
dnn="${dnn:-resnet50}"
nworkers="${nworkers:-4}"
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

MPIPATH=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/pytorch1.4/bin/python
#PY=/home/esetstore/pytorch1.8/bin/python

if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -mca mpi_warn_on_fork 0 \
    -x RDMA=$rdma \
    -x NCCL_DEBUG=VERSION  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0 \
    -x CUDA_VISIBLE_DEVICES=${gpuids}"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -mca mpi_warn_on_fork 0 \
    -x RDMA=$rdma \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=VERSION \
    -x HOROVOD_CACHE_CAPACITY=0"
fi
    #-x HOROVOD_FUSION_THRESHOLD=0 \

if [ "$dnn" = "resnet20" ] || [ "$dnn" = "resnet32" ] || [ "$dnn" = "resnet56" ] || [ "$dnn" = "resnet110" ]; then
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_cifar10_resnet.py \
        --base-lr $lr --epochs 100 --kfac-update-freq $kfac --model $dnn --lr-decay 35 75 90 --batch-size $batch_size --sparse-ratio $sparse_ratio --kfac-name $kfac_name --damping $damping --warmup-epochs 5 
else
#HOROVOD_TIMELINE=./logs/profile-timeline-${dnn}-kfac-${kfac}-json.log 
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY examples/pytorch_imagenet_resnet.py \
          --base-lr $lr --epochs $epochs --kfac-update-freq $kfac --kfac-cov-update-freq $kfac --model $dnn --kfac-name $kfac_name --sparse-ratio $sparse_ratio --exclude-parts ${exclude_parts} --batch-size $batch_size --lr-decay 25 35 40 45 50 \
          --train-dir /localdata/ILSVRC2012_dataset/train \
          --val-dir /localdata/ILSVRC2012_dataset/val
          #--base-lr 0.0125 --epochs 20 --kfac-update-freq $kfac --kfac-cov-update-freq $kfac --model $dnn  --batch-size $batch_size --lr-decay 8 14 16 18 --damping 0.0015 \
          #--base-lr 0.0125 --epochs $epochs --kfac-update-freq $kfac --kfac-cov-update-freq $kfac --model $dnn --kfac-name $kfac_name --exclude-parts ${exclude_parts} --batch-size $batch_size --lr-decay 25 35 40 45 50 \ 
fi
