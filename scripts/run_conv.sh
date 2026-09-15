#!/usr/bin/env bash
# Tables "conv" and "convlive", and the clean-model controls of Section
# "Convolutional models": ResNet-18, input-position attribution, four
# datasets, with and without augmentation, three seeds each.
#
# Every call trains three ResNet-18s from scratch (30 epochs, 40 on
# CIFAR-10, batch 256, one-cycle at 0.1); roughly 5 to 25 minutes per call
# on one A100.  Models are not cached.  Set DEVICE=cuda or cpu to override
# the automatic choice.
#
# Output directories are keyed on architecture, augmentation, liveness
# percentile and rule.  The default of --live-pct is 0 (score every
# position); the _lp0 suffix on the archived directories is from when the
# default was 20 and 0 had to be passed explicitly.  Passing --live-pct 0
# explicitly here keeps the archived names.
source "$(dirname "$0")/_env.sh"
COMMON="--arch resnet18 --batch 256 --lr 0.1 --seeds 0 1 2 --poisoned-patch 4 $DEV_FLAG"

# ---- Table "conv": standard training, all positions scored ----------------
$PY conv_attribution.py --dataset mnist   --epochs 30 --live-pct 0 $COMMON
$PY conv_attribution.py --dataset fashion --epochs 30 --live-pct 0 $COMMON
$PY conv_attribution.py --dataset gtsrb   --epochs 30 $COMMON
$PY conv_attribution.py --dataset cifar10 --epochs 40 --live-pct 0 $COMMON
# ---- Table "conv": augmented training ------------------------------------
$PY conv_attribution.py --dataset mnist   --epochs 30 --live-pct 0 --augment $COMMON
$PY conv_attribution.py --dataset fashion --epochs 30 --live-pct 0 --augment $COMMON
$PY conv_attribution.py --dataset gtsrb   --epochs 30 --augment $COMMON
$PY conv_attribution.py --dataset cifar10 --epochs 40 --live-pct 0 --augment $COMMON

# ---- Clean-model controls: same provenance split, no trigger --------------
$PY conv_attribution.py --dataset mnist   --epochs 30 --live-pct 0 --clean-control $COMMON
$PY conv_attribution.py --dataset fashion --epochs 30 --live-pct 0 --clean-control $COMMON
$PY conv_attribution.py --dataset gtsrb   --epochs 30 --clean-control $COMMON
$PY conv_attribution.py --dataset cifar10 --epochs 40 --live-pct 0 --clean-control $COMMON

# ---- Table "convlive": the liveness filter, bottom fifth discarded -------
# These were run before the output key carried the liveness percentile, so
# the archive holds them under the unsuffixed names (ConvAttr_fashion_p4_resnet18,
# ConvAttr_fashion_p4_resnet18_aug, ...) and two of them only in the logs.
# Rerun here they land under *_lp20; summarise.py reads the archived names.
$PY conv_attribution.py --dataset mnist   --epochs 30 --live-pct 20 $COMMON
$PY conv_attribution.py --dataset fashion --epochs 30 --live-pct 20 $COMMON
$PY conv_attribution.py --dataset cifar10 --epochs 40 --live-pct 20 $COMMON
$PY conv_attribution.py --dataset mnist   --epochs 30 --live-pct 20 --augment $COMMON
$PY conv_attribution.py --dataset fashion --epochs 30 --live-pct 20 --augment $COMMON
$PY conv_attribution.py --dataset cifar10 --epochs 40 --live-pct 20 --augment $COMMON

# ---- The bimodality rule on the two augmented failures (Section "Augmentation
# is the remaining boundary"): flags nothing on any seed.
$PY conv_attribution.py --dataset gtsrb   --epochs 30 --augment --rule otsu $COMMON
$PY conv_attribution.py --dataset cifar10 --epochs 40 --live-pct 0 --augment --rule otsu $COMMON

$PY summarise.py conv convlive
