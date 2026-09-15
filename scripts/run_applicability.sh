#!/usr/bin/env bash
# Table "datasets": the applicability condition on MNIST, Fashion-MNIST and
# GTSRB.  Reads the backdoored models of run_removal.sh
# (results/Influence_<dataset>_128_p4/models/full_seed*.pkl) and trains them
# if absent.  The second call is the uncentred share the method section
# quotes for Fashion-MNIST (0.344 vs 0.342).
source "$(dirname "$0")/_env.sh"

$PY applicability.py --datasets mnist fashion gtsrb --seeds 0 1 2
$PY summarise.py datasets
$PY applicability.py --datasets mnist fashion gtsrb --seeds 0 1 2 --no-center
