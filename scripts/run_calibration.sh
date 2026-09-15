#!/usr/bin/env bash
# Table "calibration": sweep of the fixed-deviation constant k over three
# seeds of a backdoored and a clean model.  Attribution replay on cached
# models; about a minute on CPU.
#
# --no-center reproduces the table as printed, which was drafted with the
# believed-compromised opinion (0,1,0) and uncentred attribution (before
# the paper adopted the centred weight and the vacuous opinion).  The
# second call is the same sweep in the paper's final setting, and the third
# is the quieter attacker (0.97% poisoning), where k = 8 finds the trigger
# on one seed of three and k = 4 on all of them (Section "Why the flagging
# constant was abandoned").
source "$(dirname "$0")/_env.sh"

$PY calibrate_k.py --no-center
$PY summarise.py calibration
$PY calibrate_k.py --untrusted-opinion 0,0,1
$PY calibrate_k.py --untrusted-opinion 0,0,1 --poison-frac 0.15 --ks 4 8
