PY=/home/comp/20481896/shshi/pytorch1.4/bin/python
#$PY examples/pytorch_imagenet_resnet.py --base-lr 0.0125 --epochs 55 --kfac-update-freq 1 --model resnet50  --batch-size 32 --lr-decay 25 35 40 45 50 --train-dir /home/datasets/ILSVRC2012_dataset/train --val-dir /home/datasets/ILSVRC2012_dataset/val
$PY examples/pytorch_cifar10_resnet.py --base-lr 0.1 --epochs 100 --kfac-update-freq 1 --model resnet32 --lr-decay 35 75 90 --batch-size 32 --dir /home/datasets/cifar10
