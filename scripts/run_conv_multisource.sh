#!/usr/bin/env bash
# Table "convleak": the adaptive adversary at input-position level, MNIST,
# ResNet-18, five seeds per level, with the discard-and-retrain baseline.
# Each level trains ten ResNet-18s (five full, five without the compromised
# source); about 25 minutes per level on one A100.
source "$(dirname "$0")/_env.sh"

for LEAK in 0 0.25 0.3 0.35 0.4 0.5; do
  $PY conv_multisource.py --dataset mnist --arch resnet18 --epochs 30 --batch 256 --lr 0.1 \
      --seeds 0 1 2 3 4 --unknown-classes 4 7 --compromised-frac 0.25 \
      --poison-leak $LEAK --with-drop $DEV_FLAG
done

$PY summarise.py convleak
