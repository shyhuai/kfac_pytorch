#python examples/pytorch_imagenet_resnet.py --base-lr 0.0125 --epochs 55 --kfac-update-freq 0 --model resnet50  --batch-size 32 --lr-decay 25 35 40 45 50 --train-dir /localdata/ILSVRC2012_dataset/train --val-dir /localdata/ILSVRC2012_dataset/val
PY=/home/comp/15485625/pytorch1.4/bin/python
$PY examples/pytorch_imagenet_resnet.py --base-lr 0.0125 --epochs 55 --kfac-update-freq 0 --model resnet50  --batch-size 256 --lr-decay 25 35 40 45 50 \
          --train-dir /home/datasets/imagenet/ILSVRC2012_dataset/train \
          --val-dir /home/datasets/imagenet/ILSVRC2012_dataset/val
