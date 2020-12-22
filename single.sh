batch_size="${batch_size:-32}"
dnn="${dnn:-resnet50}"
kfac="${kfac:-1}"
epochs="${epochs:-55}"
PY=/home/esetstore/pytorch1.4/bin/python
$PY examples/pytorch_imagenet_resnet.py --base-lr 0.0125 --epochs $epochs --kfac-update-freq $kfac --model $dnn --batch-size $batch_size --lr-decay 25 35 40 45 50 --train-dir /localdata/ILSVRC2012_dataset/train --val-dir /localdata/ILSVRC2012_dataset/val
