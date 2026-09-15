#!/usr/bin/env bash
# Tables "leak" and "leakbase": the adaptive adversary at parameter level.
# Three sources (audited, unvetted owning classes 4 and 7, compromised), a
# fraction of the poison leaked into the audited source.  Every leakage
# level trains its own models (one full model plus two discard-and-retrain
# baselines per seed, cached under results/MultiSource_mnist_128/), then
# replays the attribution for the three criteria at the opinion's budget.
#
# Leak 0 is three seeds; every other level is five, as in the paper.
source "$(dirname "$0")/_env.sh"

$PY eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 2 --unknown-classes 4 7
for LEAK in 0.25 0.3 0.35 0.4 0.5; do
  $PY eval_multisource.py --arch 128 --epochs 20 --seeds 0 1 2 3 4 \
      --unknown-classes 4 7 --poison-leak $LEAK
done

$PY summarise.py leak leakbase
