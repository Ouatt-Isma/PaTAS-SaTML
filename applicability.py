"""applicability.py: the condition that decides where the dense defence applies.

Table "datasets" of the paper reports, for MNIST, Fashion-MNIST and GTSRB:

  quiet region     the mean absolute per-pixel deviation of the trusted data
                   over the trigger positions, relative to the median
                   position.  It is computed on the sources the defender
                   trusts, before any model is trained, so the defender can
                   read it off in advance.  "none" when the trigger region
                   is no quieter than a typical position.
  untrusted share  the fraction of a parameter's (centred) influence mass
                   contributed by the untrusted source, S / m.  A feature's
                   share is the maximum over the hidden units it feeds (the
                   most untrusted-driven of its parameters, as eval_repair.py
                   scores it); the table reports the mean over the trigger
                   features against the mean over live features.
  trigger ranks    where the trigger features sit in the ranking of live
                   features by attributed trust (0 = lowest trust).

The table in the paper is seed 0; this script prints every seed and stores
all of them.  The models are the backdoored ones influence_baseline.py
trained and cached under results/Influence_<dataset>_128_p4/models/
full_seed*.pkl, so this is an attribution replay only.  Fashion-MNIST and
GTSRB are downloaded by the loader on first use.  ``--no-center`` gives the
uncentred share the method section quotes for Fashion-MNIST (0.344 vs 0.342).

Usage
-----
    python applicability.py
    python applicability.py --datasets mnist --seeds 0
"""
from __future__ import annotations

import os
import sys
import json
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "patas_module")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fashion", "gtsrb"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--poisoned-patch", type=int, default=4)
    ap.add_argument("--untrusted-tail", type=float, default=1.0 / 3.0)
    ap.add_argument("--evidence", type=float, default=50.0)
    ap.add_argument("--no-center", action="store_true",
                    help="Weight the input layer by magnitude rather than by "
                         "deviation from the trusted mean")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Used only if a cached model is missing (20 for "
                         "MNIST and Fashion-MNIST, 40 for GTSRB, as in "
                         "influence_baseline.py)")
    args = ap.parse_args()

    from NN.datasets import load_data
    from main import DATASET_META, get_lr_mnist, get_lr_gtsrb
    from attribution import ProvenanceSource, accumulate
    from subjective_logic import bpq_vec
    import influence_baseline as IB

    rows = []
    for ds in args.datasets:
        meta = DATASET_META[ds]
        img = meta["img_size"]
        pidx = np.array([img * r + c for r in range(args.poisoned_patch)
                         for c in range(args.poisoned_patch)])
        X, X_test, Y, Y_test, _ = load_data(ds, "clean", "clean",
                                            poisoned_patch=args.poisoned_patch)
        X, Y = np.asarray(X, np.float32), np.asarray(Y, np.float32)
        X_test, Y_test = np.asarray(X_test, np.float32), np.asarray(Y_test, np.float32)
        n = len(X)
        # The untrusted source is believed compromised here so that S/(R+S)
        # is the untrusted share directly; by Proposition 1 the ranking does
        # not depend on this choice.
        src = ProvenanceSource(n, args.untrusted_tail, (0.0, 1.0, 0.0))
        trusted = ~src.untrusted

        # --- the quiet region: needs no model -----------------------------
        dev = np.abs(X[trusted] - X[trusted].mean(0)).mean(0)
        quiet = float(dev[pidx].mean() / np.median(dev))
        print(f"[appl] {ds}: mean absolute per-pixel deviation of trusted data at the "
              f"trigger {dev[pidx].mean():.4f}, median position {np.median(dev):.4f}  ->  "
              f"quiet region {quiet:.3g}x")

        epochs = args.epochs or (40 if ds == "gtsrb" else 20)
        lr = get_lr_gtsrb if ds == "gtsrb" else get_lr_mnist
        cache = f"results/Influence_{ds}_128_p{args.poisoned_patch}/models"
        sh_t, sh_l, rk_lo, rk_hi = [], [], [], []
        for seed in args.seeds:
            Ws, bs = IB.train_model(X, Y, X_test, Y_test, [128], epochs, seed,
                                    os.path.join(cache, f"full_seed{seed}.pkl"), lr)
            mu = None if args.no_center else X[trusted].mean(0)
            infl, mass, R, S = accumulate(Ws, bs, X, Y, src, verbose=False, center=mu)
            m0 = mass[0][:-1]
            live = infl[0][:-1].sum(1) > np.percentile(infl[0][:-1].sum(1), 20)
            share = np.divide(S[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
            f_feat = share.max(1)                         # per input feature
            r = np.divide(args.evidence * R[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
            s = np.divide(args.evidence * S[0][:-1], m0, out=np.zeros_like(m0), where=m0 > 1e-12)
            om = bpq_vec(r, s, W=2.0)
            score = (om[..., 0] + 0.5 * om[..., 2]).min(1)
            order = np.argsort(np.where(live, score, np.inf))
            ranks = [int(np.where(order == p)[0][0]) for p in pidx if live[p]]
            sh_t.append(float(f_feat[pidx].mean())); sh_l.append(float(f_feat[live].mean()))
            rk_lo.append(min(ranks) if ranks else -1); rk_hi.append(max(ranks) if ranks else -1)
            print(f"[appl]   seed {seed}: untrusted share trigger {sh_t[-1]:.3f} vs live "
                  f"{sh_l[-1]:.3f}   trigger ranks {rk_lo[-1]}-{rk_hi[-1]} of {int(live.sum())} live "
                  f"({len(ranks)}/{len(pidx)} trigger features live)")
        # The row of the table is the first seed given (seed 0 in the paper);
        # every seed is kept in the summary.
        rows.append(dict(dataset={"mnist": "MNIST", "fashion": "Fashion-MNIST",
                                  "gtsrb": "GTSRB"}.get(ds, ds),
                         seed=args.seeds[0], quiet_ratio=quiet,
                         share_trigger=sh_t[0], share_live=sh_l[0],
                         rank_min=int(rk_lo[0]), rank_max=int(rk_hi[0]),
                         per_seed=dict(seeds=args.seeds, share_trigger=sh_t,
                                       share_live=sh_l, rank_min=rk_lo, rank_max=rk_hi)))

    print(f"\n(seed {args.seeds[0]})\n{'Dataset':<16}{'Quiet region':>14}{'Untrusted share':>24}{'Trigger ranks':>16}")
    for r in rows:
        q = f"{r['quiet_ratio']:.3g}x" if r["quiet_ratio"] < 0.1 else "none"
        share = f"{r['share_trigger']:.3f} vs {r['share_live']:.3f}"
        ranks = f"{r['rank_min']}--{r['rank_max']}"
        print(f"{r['dataset']:<16}{q:>14}{share:>24}{ranks:>16}")
    out = "results/Applicability" + ("_uncentred" if args.no_center else "")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(seeds=args.seeds, patch=args.poisoned_patch,
                       centered=not args.no_center, rows=rows), fh, indent=2)
    print(f"\n[appl] saved {out}/summary.json")


if __name__ == "__main__":
    main()
