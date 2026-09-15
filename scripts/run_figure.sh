#!/usr/bin/env bash
# Figure 1: (a) the attributed-trust distribution of a backdoored and a clean
# model (localisation models, seed 0), (b) attack success under the adaptive
# adversary for every criterion (multi-source summaries).  Reads the
# archives of run_localisation.sh and run_multisource.sh; nothing is trained.
source "$(dirname "$0")/_env.sh"

$PY make_fig_trust.py --out results/fig_trust
echo "[run] wrote results/fig_trust.pdf and results/fig_trust.png"
