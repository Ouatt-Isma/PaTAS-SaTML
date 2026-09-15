#!/usr/bin/env bash
# Everything, in the order the paper presents it.  The dense experiments and
# the figure take minutes once models are cached; the two convolutional
# scripts are ResNet-18 training runs and take a few hours on one A100.
source "$(dirname "$0")/_env.sh"
S="$ROOT/scripts"
bash $S/run_removal.sh
bash $S/run_patchsize.sh
bash $S/run_localisation.sh
bash $S/run_calibration.sh
bash $S/run_applicability.sh
bash $S/run_multisource.sh
bash $S/run_figure.sh
bash $S/run_conv.sh
bash $S/run_conv_multisource.sh
$PY summarise.py
