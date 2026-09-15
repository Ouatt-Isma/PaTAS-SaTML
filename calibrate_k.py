"""calibrate_k.py: the flagging constant of the fixed-deviation rule.

Table "calibration" of the paper sweeps the constant k of the fixed-deviation
rule (flag a feature when it sits more than k robust deviations below the
median attributed trust) over three seeds of a backdoored and of a clean
model, reporting how many features are flagged and how many of those are
trigger features.  It is the measurement that made the paper abandon the
constant for the bimodality rule.

The backdoored models are the ones influence_baseline.py trained and cached
under results/Influence_mnist_128_p4/models/ (full_seed*.pkl), which are the
models the table was drafted from; the clean models are the ones
eval_localisation.py cached under results/Localisation_mnist_otsu/models/.
The clean models the printed table used were a cluster-side cache that was
not archived, so the clean column reproduces up to the model instance (with
the archived clean models k = 8 flags 0, 3 and 0 features rather than 0, 0, 0).
Everything here is an attribution replay and runs on CPU in about a minute.

Usage
-----
    python calibrate_k.py --no-center           # the table as printed
    python calibrate_k.py                       # same sweep, centred attribution
    python calibrate_k.py --ks 4 8 --poison-frac 0.15   # the quieter attacker
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "patas_module")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def load_weights(path):
    with open(path, "rb") as fh:
        wd = pickle.load(fh)
    Ws, bs, i = [], [], 1
    while f"W{i}" in wd:
        Ws.append(np.asarray(wd[f"W{i}"], np.float32))
        bs.append(np.asarray(wd[f"b{i}"], np.float32).reshape(-1)); i += 1
    return Ws, bs


def feature_scores(Ws, bs, X, Y, opinion, tail, evidence, center=True):
    """Per-feature attributed trust b + u/2 and the live mask, exactly as
    eval_localisation.py and influence_baseline.py compute them.  With
    ``center=False`` the input layer is weighted by activation magnitude
    rather than by deviation from the trusted mean, which is how the
    attribution was computed when the calibration table was drafted."""
    from attribution import ProvenanceSource, accumulate
    from subjective_logic import bpq_vec
    src = ProvenanceSource(len(X), tail, opinion)
    mu = X[~src.untrusted].mean(0) if center else None
    infl, mass, R, S = accumulate(Ws, bs, X, Y, src, verbose=False, center=mu)
    m0 = mass[0][:-1]
    live = infl[0][:-1].sum(1) > np.percentile(infl[0][:-1].sum(1), 20)
    r = np.divide(evidence * R[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
    s = np.divide(evidence * S[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
    om = bpq_vec(r, s, W=2.0)
    return (om[..., 0] + 0.5 * om[..., 2]).min(1), live


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ks", type=float, nargs="+", default=[5, 8, 12, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--backdoored-models", default="results/Influence_mnist_128_p4/models",
                    help="Directory holding full_seed<s>.pkl, the backdoored "
                         "models influence_baseline.py trained")
    ap.add_argument("--clean-models", default="results/Localisation_mnist_otsu/models",
                    help="Directory holding clean128_128_seed<s>.pkl, the clean "
                         "models eval_localisation.py trained")
    ap.add_argument("--poison-frac", type=float, default=1.0,
                    help="Fraction of eligible untrusted samples poisoned. "
                         "1.0 reads the localisation models; 0.15 reads the "
                         "quieter attacker's models from "
                         "results/Influence_mnist_128_p4_pf0.15_op001/models "
                         "(full_seed*.pkl), which influence_baseline.py trained")
    ap.add_argument("--untrusted-opinion", default="0,1,0",
                    help="Opinion about the untrusted source. The calibration "
                         "table was drafted with the believed-compromised "
                         "opinion (0,1,0) and uncentred attribution, before the "
                         "paper switched to (0,0,1) and centring; both are "
                         "exposed so either setting can be reproduced.")
    ap.add_argument("--no-center", action="store_true",
                    help="Weight the input layer by activation magnitude, not "
                         "deviation from the trusted mean (the setting of the "
                         "table as printed in the paper).")
    ap.add_argument("--untrusted-tail", type=float, default=1.0 / 3.0)
    ap.add_argument("--evidence", type=float, default=50.0)
    ap.add_argument("--epochs", type=int, default=20,
                    help="Used only if a model is missing and has to be trained")
    args = ap.parse_args()

    from NN.datasets import load_data
    from main import DATASET_META
    from attribution import flag_features
    import influence_baseline as IB

    op = tuple(float(v) for v in args.untrusted_opinion.split(","))
    meta = DATASET_META["mnist"]
    pidx = np.array([28 * r + c for r in range(4) for c in range(4)])

    # Backdoored and clean training sets; the clean model sees the same
    # provenance split with nothing poisoned.
    if args.poison_frac >= 1.0:
        Xp, X_test, Yp, Y_test, _ = load_data("mnist", "clean", "clean", poisoned_patch=4)
        bd_path = lambda s: os.path.join(args.backdoored_models, f"full_seed{s}.pkl")
    else:
        Xp, X_test, Yp, Y_test, _ = load_data("mnist", "clean", "clean")
        Xp = np.asarray(Xp, np.float32).copy(); Yp = np.asarray(Yp, np.float32).copy()
        y = Yp.argmax(1); n = len(Xp); a, b = meta["pois_pair"]
        idx = np.array([i for i in range(int(round((1 - args.untrusted_tail) * n)), n)
                        if y[i] in (a, b)])
        sel = np.random.default_rng(9876).choice(idx, int(round(args.poison_frac * len(idx))),
                                                 replace=False)
        Xp[np.ix_(sel, pidx)] = meta["scale_patch"](1.0)
        for i in sel:
            Yp[i] = np.eye(10, dtype=np.float32)[b if y[i] == a else a]
        tag = f"results/Influence_mnist_128_p4_pf{args.poison_frac:g}_op001/models"
        bd_path = lambda s: os.path.join(tag, f"full_seed{s}.pkl")
    Xc, _, Yc, _, _ = load_data("mnist", "clean", "clean")
    Xp, Yp = np.asarray(Xp, np.float32), np.asarray(Yp, np.float32)
    Xc, Yc = np.asarray(Xc, np.float32), np.asarray(Yc, np.float32)
    X_test, Y_test = np.asarray(X_test, np.float32), np.asarray(Y_test, np.float32)

    rows = []
    for seed in args.seeds:
        for model, X, Y, path in (
                ("backdoored", Xp, Yp, bd_path(seed)),
                ("clean", Xc, Yc, os.path.join(args.clean_models, f"clean128_128_seed{seed}.pkl"))):
            Ws, bs = IB.train_model(X, Y, X_test, Y_test, [128], args.epochs, seed, path)
            score, live = feature_scores(Ws, bs, X, Y, op, args.untrusted_tail,
                                         args.evidence, center=not args.no_center)
            order = np.argsort(np.where(live, score, np.inf))
            ranks = [int(np.where(order == p)[0][0]) for p in pidx if live[p]]
            for k in args.ks:
                sel, _ = flag_features(score, live, k, rule="mad")
                rows.append(dict(model=model, seed=seed, k=k, flagged=int(len(sel)),
                                 trigger=int(np.isin(pidx, sel).sum()),
                                 trigger_ranks=[min(ranks), max(ranks)] if ranks else None))
            print(f"[calib] {model:<10} seed {seed}: "
                  + "  ".join(f"k={k:g}: {r['flagged']}({r['trigger']})"
                              for k, r in zip(args.ks, rows[-len(args.ks):]))
                  + (f"   trigger ranks {min(ranks)}-{max(ranks)}" if model == "backdoored" else ""))

    print(f"\n{'k':>4}   {'backdoored: flagged (trigger) per seed':<40}{'clean: flagged per seed'}")
    for k in args.ks:
        b = [r for r in rows if r["k"] == k and r["model"] == "backdoored"]
        c = [r for r in rows if r["k"] == k and r["model"] == "clean"]
        b_s = "   ".join(f"{r['flagged']:>3}({r['trigger']:>2})" for r in b)
        c_s = "   ".join(f"{r['flagged']:>3}" for r in c)
        print(f"{k:>4g}   {b_s:<40}{c_s}")
    out = "results/Calibration_mnist" + ("" if args.poison_frac >= 1.0 else f"_pf{args.poison_frac:g}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(ks=args.ks, seeds=args.seeds, poison_frac=args.poison_frac,
                       untrusted_opinion=args.untrusted_opinion, rule="mad",
                       centered=not args.no_center,
                       rows=rows), fh, indent=2)
    print(f"\n[calib] saved {out}/summary.json")


if __name__ == "__main__":
    main()
