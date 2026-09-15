#!/usr/bin/env bash
# Table "removal": backdoor removal against the six baselines at two
# poisoning rates, three seeds.  Eight trainings per seed and rate on the
# first run (models are cached under results/Influence_*/models/, after
# which the script is an attribution replay, ~30 s on CPU).
#
# The untrusted source is given the vacuous opinion (0,0,1), the paper's
# default; the cache directory gets the _op001 suffix.
source "$(dirname "$0")/_env.sh"

# 6.5% poisoning: every eligible sample of the untrusted source
$PY influence_baseline.py --dataset mnist --arch 128 --poisoned-patch 4 \
    --seeds 0 1 2 --untrusted-opinion 0,0,1
# 0.97% poisoning: 15% of them
$PY influence_baseline.py --dataset mnist --arch 128 --poisoned-patch 4 \
    --seeds 0 1 2 --untrusted-opinion 0,0,1 --poison-frac 0.15

# The same comparison on Fashion-MNIST and GTSRB, with the believed-
# compromised opinion (0,1,0) as originally run.  Not a table of the paper,
# but these are the backdoored models the applicability table reads.
$PY influence_baseline.py --dataset mnist   --arch 128 --poisoned-patch 4 --seeds 0 1 2
$PY influence_baseline.py --dataset fashion --arch 128 --poisoned-patch 4 --seeds 0 1 2
$PY influence_baseline.py --dataset gtsrb   --arch 128 --poisoned-patch 4 --seeds 0 1 2 --epochs 40

$PY summarise.py removal
