#python examples/pytorch_imagenet_resnet.py --base-lr 0.0125 --epochs 55 --kfac-update-freq 0 --model resnet50  --batch-size 32 --lr-decay 25 35 40 45 50 --train-dir /localdata/ILSVRC2012_dataset/train --val-dir /localdata/ILSVRC2012_dataset/val
python examples/pytorch_imagenet_resnet.py --base-lr 0.0125 --epochs 55 --kfac-update-freq 1 --model resnet50  --batch-size 128 --lr-decay 25 35 40 45 50 \
          --train-dir /home/datasets/imagenet/ILSVRC2012_dataset/train \
          --val-dir /home/datasets/imagenet/ILSVRC2012_dataset/val
