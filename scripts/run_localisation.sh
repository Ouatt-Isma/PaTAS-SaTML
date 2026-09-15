#!/usr/bin/env bash
# Table "localisation": what the bimodality rule flags across seven
# conditions, three seeds each.  Trains 21 small networks on the first run
# (cached under results/Localisation_mnist_otsu/models/), then replays.
#
# The paper's table uses the vacuous opinion (0,0,1) about the untrusted
# source and the Otsu rule; the script's own default is (0,1,0), so the
# flag is passed explicitly.
source "$(dirname "$0")/_env.sh"

$PY eval_localisation.py --dataset mnist --seeds 0 1 2 --rule otsu --untrusted-opinion 0,0,1

$PY summarise.py localisation
