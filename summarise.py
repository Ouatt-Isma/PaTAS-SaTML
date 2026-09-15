"""summarise.py: every table of the paper from the archived JSON summaries.

Each experiment script writes a JSON summary keyed by its configuration into
``results/``.  This script reads those summaries and prints the tables of the
paper in the order they appear, with the same aggregation (mean and population
standard deviation over seeds), so the reported numbers can be checked
without rerunning anything.

    python summarise.py            # every table
    python summarise.py removal    # one table, by its LaTeX label

Two rows of the liveness-filter table (MNIST standard and CIFAR-10 augmented,
bottom fifth discarded) were overwritten on disk by a later run that keyed its
output directory differently; they are recovered from the run logs in
``results/vast_conv_logs/``, which record the same per-seed numbers.
"""
from __future__ import annotations

import os
import re
import sys
import json
import glob
from statistics import mean, pstdev

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(path):
    p = os.path.join(RES, path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def ms(vals, scale=100.0, digits=2):
    vals = [v for v in vals if v == v]           # drop NaN
    if not vals:
        return "---"
    m = mean(vals) * scale
    s = (pstdev(vals) if len(vals) > 1 else 0.0) * scale
    return f"{m:.{digits}f}±{s:.{digits}f}"


def header(label, caption):
    print("\n" + "=" * 78)
    print(f"Table \\ref{{tab:{label}}}: {caption}")
    print("=" * 78)


# ---------------------------------------------------------------------------
def tab_removal():
    header("removal", "backdoor removal at two poisoning rates (influence_baseline.py)")
    arms = [("None", "undefended"), ("Ours", "prune (ours)"),
            ("drop-source", "drop-source"),
            ("activation clustering", "activation clustering"),
            ("spectral signatures", "spectral 15%"),
            ("influence", "influence 15%"), ("random", "random-samples 15%")]
    blocks = [("6.5%", "Influence_mnist_128_p4_op001/summary.json"),
              ("0.97%", "Influence_mnist_128_p4_pf0.15_op001/summary.json")]
    print(f"{'Defence':<24}" + "".join(f"{'clean acc':>14}{'ASR':>14}{'caught':>9}" for _ in blocks))
    print(f"{'':<24}" + "".join(f"{'poisoning ' + lbl:>37}" for lbl, _ in blocks))
    for name, arm in arms:
        line = f"{name:<24}"
        for _, path in blocks:
            d = load(path)
            if d is None:
                line += f"{'(missing)':>37}"; continue
            v = [r for r in d["rows"] if r["arm"] == arm]
            pr = [r["poison_recall"] for r in v if r["poison_recall"] == r["poison_recall"]]
            line += (f"{ms([r['clean_acc'] for r in v]):>14}{ms([r['asr'] for r in v]):>14}"
                     f"{(f'{mean(pr)*100:.1f}%' if pr else '---'):>9}")
        print(line)


def tab_patchsize():
    header("patchsize", "effect of trigger size (eval_repair.py)")
    print(f"{'Trigger':<8}{'Defence':<14}{'Budget':>8}{'clean acc':>12}{'ASR':>9}")
    for p, rows in ((1, [("ours", "attribution", 0.01)]),
                    (4, [("ours", "attribution", 0.02), ("fine-pruning", "finepruning", 0.10)]),
                    (10, [("ours", "attribution", 0.10), ("ours", "attribution", 0.20)])):
        d = load(f"Repair_mnist_128_patch{p}/summary.json")
        if d is None:
            print(f"{p}x{p}   (missing)"); continue
        u = d["undefended"]
        print(f"{f'{p}x{p}':<8}{'none':<14}{'---':>8}{u['clean_acc']*100:>12.2f}{u['asr']*100:>9.2f}")
        for lbl, method, b in rows:
            r = next(r for r in d["rows"] if r["method"] == method and abs(r["budget"] - b) < 1e-9)
            print(f"{'':<8}{lbl:<14}{f'{b*100:.0f}%':>8}{r['clean_acc']*100:>12.2f}{r['asr']*100:>9.2f}")


def tab_calibration():
    header("calibration", "the flagging constant k, fixed-deviation rule (calibrate_k.py)")
    d = load("Calibration_mnist/summary.json")
    if d is None:
        print("results/Calibration_mnist/summary.json not found: run  python calibrate_k.py")
        return
    ks = sorted({r["k"] for r in d["rows"]})
    seeds = sorted({r["seed"] for r in d["rows"]})
    print(f"{'k':>4}   {'Backdoored (flagged (trigger)) seeds ' + '/'.join(map(str, seeds)):<44}"
          f"{'Clean seeds ' + '/'.join(map(str, seeds))}")
    for k in ks:
        b = [next(r for r in d["rows"] if r["k"] == k and r["seed"] == s and r["model"] == "backdoored") for s in seeds]
        c = [next(r for r in d["rows"] if r["k"] == k and r["seed"] == s and r["model"] == "clean") for s in seeds]
        b_s = "   ".join(f"{r['flagged']:>3}({r['trigger']:>2})" for r in b)
        c_s = "   ".join(f"{r['flagged']:>3}" for r in c)
        print(f"{k:>4}   {b_s:<44}{c_s}")


def tab_localisation():
    header("localisation", "what the rule flags across conditions (eval_localisation.py)")
    d = load("Localisation_mnist_otsu/summary.json")
    if d is None:
        print("(missing)"); return
    print(f"{'Condition':<32}{'Flagged':>9}{'Trigger':>10}{'AUROC':>8}{'Cost':>8}")
    seen = []
    for r in d["rows"]:
        if r["key"] not in seen:
            seen.append(r["key"])
    for key in seen:
        rs = [r for r in d["rows"] if r["key"] == key]
        fl = mean(r["flagged"] for r in rs)
        tf = [r["trigger_found"] for r in rs if r["trigger_found"] is not None]
        au = [r["auroc"] for r in rs if r["auroc"] == r["auroc"]]
        cost = mean(r["clean_acc"] - r["masked_acc"] for r in rs) * 100
        tf_s = f"{mean(tf):.0f}/{rs[0]['n_trigger']}" if tf else "---"
        au_s = f"{mean(au):.3f}" if au else "---"
        print(f"{rs[0]['condition']:<32}{fl:>9.1f}{tf_s:>10}{au_s:>8}{cost:>8.2f}")


def tab_datasets():
    header("datasets", "the applicability condition (applicability.py)")
    d = load("Applicability/summary.json")
    if d is None:
        print("results/Applicability/summary.json not found: run  python applicability.py")
        return
    print(f"{'Dataset':<16}{'Quiet region':>14}{'Untrusted share':>24}{'Trigger ranks':>16}")
    for ds in d["rows"]:
        q = ds["quiet_ratio"]
        q_s = f"{q:.3g}x" if q < 0.1 else "none"
        share = f"{ds['share_trigger']:.3f} vs {ds['share_live']:.3f}"
        ranks = f"{ds['rank_min']}--{ds['rank_max']}"
        print(f"{ds['dataset']:<16}{q_s:>14}{share:>24}{ranks:>16}")


def _leak_files(base):
    files = [(0.0, base + ".json")]
    for f in glob.glob(os.path.join(RES, base + "_leak*.json")):
        files.append((float(re.search(r"_leak([0-9.]+)\.json$", f).group(1)),
                      os.path.relpath(f, RES)))
    return sorted(files)


def tab_leak():
    header("leak", "the adaptive adversary, parameter level (eval_multisource.py --poison-leak)")
    crits = [("Verified share", "scalar@thr"), ("Opinion", "opinion@thr"),
             ("Eq. scalarproj", "scalar-proj@thr")]
    print(f"{'Leak':<7}{'seeds':>6}{'Undef.':>8}" + "".join(f"{c:>24}" for c, _ in crits))
    for lk, f in _leak_files("MultiSource_mnist_128/summary_uc4-7_cf0.25_p4"):
        d = load(f)
        if d is None or not any(r["criterion"] == "opinion@thr" for r in d["rows"]):
            continue
        u = mean(r["asr"] for r in d["rows"] if r["criterion"] == "undefended") * 100
        line = f"{f'{lk*100:.0f}%':<7}{len(d['seeds']):>6}{u:>8.2f}"
        for _, c in crits:
            v = [r for r in d["rows"] if r["criterion"] == c]
            rec = mean(r["trigger_recall"] for r in v) * 100 if v else float("nan")
            line += f"{ms([r['asr'] for r in v]) + f' ({rec:.0f})':>24}"
        print(line)


def tab_leakbase():
    header("leakbase", "discard-and-retrain baselines under leakage (eval_multisource.py)")
    want = [0.0, 0.25, 0.5]
    print(f"{'Defence':<26}" + "".join(f"{f'{l*100:.0f}% leak':>12}" for l in want))
    for lbl, c in (("drop compromised source", "drop-compromised"),
                   ("drop all unverified", "drop-unverified")):
        line = f"{lbl:<26}"
        for l in want:
            f = "MultiSource_mnist_128/summary_uc4-7_cf0.25_p4" + (f"_leak{l:g}.json" if l else ".json")
            d = load(f)
            v = [r["asr"] for r in d["rows"] if r["criterion"] == c] if d else []
            line += f"{(f'{mean(v)*100:.2f}' if v else '---'):>12}"
        print(line)


CONV_ROWS = [  # (dataset, training, all-positions dir, bottom-fifth source)
    ("MNIST", "standard", "ConvAttr_mnist_p4_resnet18_lp0", ("log", "queue.log", 0)),
    ("Fashion-MNIST", "standard", "ConvAttr_fashion_p4_resnet18_lp0", ("dir", "ConvAttr_fashion_p4_resnet18")),
    ("GTSRB", "standard", "ConvAttr_gtsrb_p4_resnet18", None),
    ("CIFAR-10", "standard", "ConvAttr_cifar10_p4_resnet18_lp0",
     ("dirs", ["ConvAttr_cifar10_p4", "ConvAttr_cifar10_p4_resnet18"])),
    ("MNIST", "augmented", "ConvAttr_mnist_p4_resnet18_aug_lp0", ("dir", "ConvAttr_mnist_p4")),
    ("Fashion-MNIST", "augmented", "ConvAttr_fashion_p4_resnet18_aug_lp0", ("dir", "ConvAttr_fashion_p4_resnet18_aug")),
    ("GTSRB", "augmented", "ConvAttr_gtsrb_p4_resnet18_aug", None),
    ("CIFAR-10", "augmented", "ConvAttr_cifar10_p4_resnet18_aug_lp0", ("log", "cifar_resnet.log", 0)),
]

_SEED_RE = re.compile(r"\[conv\] seed (\d+): clean ([\d.]+)% attack ([\d.]+)%  \|  flagged (\d+) "
                      r"\((\d+)/(\d+) trigger, ranks (-?\d+)-(-?\d+)\)  ->  masked clean ([\d.]+)% "
                      r"attack ([\d.]+)%")


def _rows_from_log(name, block):
    """Per-seed rows of the ``block``-th run recorded in a conv log."""
    with open(os.path.join(RES, "vast_conv_logs", name), encoding="utf-8") as fh:
        text = fh.read()
    runs = [b for b in text.split("[conv] device")[1:]]
    rows = []
    for m in _SEED_RE.finditer(runs[block]):
        rows.append(dict(seed=int(m.group(1)), clean=float(m.group(2)) / 100,
                         asr=float(m.group(3)) / 100, flagged=int(m.group(4)),
                         trigger_found=int(m.group(5)), n_trigger=int(m.group(6)),
                         masked_clean=float(m.group(9)) / 100,
                         masked_asr=float(m.group(10)) / 100))
    return rows


def _rows(src):
    kind = src[0]
    if kind == "dir":
        d = load(f"{src[1]}/summary.json"); return d["rows"] if d else []
    if kind == "dirs":
        out = []
        for s in src[1]:
            d = load(f"{s}/summary.json")
            if d:
                out += d["rows"]
        return out
    return _rows_from_log(src[1], src[2])


def tab_conv():
    header("conv", "ResNet-18, input-position attribution (conv_attribution.py)")
    print(f"{'Dataset':<15}{'Training':<11}{'clean':>7}{'ASR':>7}{'floor':>7}{'Found':>11}"
          f"{'m.ASR':>8}{'m.clean':>9}{'Share':>12}")
    for ds, tr, d_all, _ in CONV_ROWS:
        d = load(f"{d_all}/summary.json")
        if d is None:
            print(f"{ds:<15}{tr:<11}(missing {d_all})"); continue
        rs = d["rows"]
        n_t = 48 if ds == "CIFAR-10" else 16
        col = lambda k: mean(r[k] for r in rs)
        found = f"{col('trigger_found'):.1f}/{n_t}"
        share = f"{col('share_trigger'):.3f}/{col('share_live'):.3f}"
        print(f"{ds:<15}{tr:<11}{col('clean')*100:>7.2f}{col('asr')*100:>7.2f}"
              f"{col('floor')*100:>7.2f}{found:>11}{col('masked_asr')*100:>8.2f}"
              f"{col('masked_clean')*100:>9.2f}{share:>12}")
    print("\nClean-model controls (no trigger; positions flagged / cost of masking them):")
    for ds, dname in (("MNIST", "ConvAttr_mnist_p4_resnet18_lp0_clean"),
                      ("Fashion-MNIST", "ConvAttr_fashion_p4_resnet18_lp0_clean"),
                      ("GTSRB", "ConvAttr_gtsrb_p4_resnet18_clean"),
                      ("CIFAR-10", "ConvAttr_cifar10_p4_resnet18_lp0_clean")):
        d = load(f"{dname}/summary.json")
        if d is None:
            continue
        rs = d["rows"]
        print(f"  {ds:<15}flagged {mean(r['flagged'] for r in rs):>5.1f}   "
              f"cost {mean(r['clean'] - r['masked_clean'] for r in rs)*100:.2f} pts   "
              f"share {mean(r['share_trigger'] for r in rs):.3f} vs {mean(r['share_live'] for r in rs):.3f}")


def tab_convlive():
    header("convlive", "effect of the liveness filter (conv_attribution.py --live-pct 20 vs 0)")
    print(f"{'':<18}{'bottom fifth discarded':>26}{'all positions scored':>26}")
    print(f"{'':<18}{'found':>14}{'ASR':>12}{'found':>14}{'ASR':>12}")
    for ds, tr, d_all, src in CONV_ROWS:
        if src is None:
            continue
        n_t = 48 if ds == "CIFAR-10" else 16
        a = _rows(src); b = load(f"{d_all}/summary.json")["rows"]
        lbl = ds.replace("-MNIST", "") + (", aug." if tr == "augmented" else "")
        found_a = f"{mean(r['trigger_found'] for r in a):.1f}/{n_t}"
        found_b = f"{mean(r['trigger_found'] for r in b):.1f}/{n_t}"
        print(f"{lbl:<18}{found_a:>14}{mean(r['masked_asr'] for r in a)*100:>12.2f}"
              f"{found_b:>14}{mean(r['masked_asr'] for r in b)*100:>12.2f}")


def tab_convleak():
    header("convleak", "the adaptive adversary at input-position level (conv_multisource.py)")
    crits = [("Verified share", "scalar@thr"), ("Opinion", "opinion@thr"),
             ("Eq. scalarproj", "scalar-proj@thr")]
    print(f"{'Leak':<7}{'Undef.':>8}" + "".join(f"{c:>22}" for c, _ in crits) + f"{'Drop':>8}")
    for f in sorted(glob.glob(os.path.join(RES, "ConvMulti_mnist_resnet18_cf0.25/summary_leak*.json")),
                    key=lambda p: float(re.search(r"leak([0-9.]+)\.json", p).group(1))):
        d = json.load(open(f, encoding="utf-8"))
        lk = d["poison_leak"]
        u = mean(r["asr"] for r in d["rows"] if r["criterion"] == "undefended") * 100
        line = f"{f'{lk*100:.0f}%':<7}{u:>8.2f}"
        for _, c in crits:
            v = [r for r in d["rows"] if r["criterion"] == c]
            rec = mean(r["trigger_recall"] for r in v) * 100
            line += f"{ms([r['asr'] for r in v]) + f' ({rec:.0f})':>22}"
        dr = [r["asr"] for r in d["rows"] if r["criterion"] == "drop-compromised"]
        line += f"{(f'{mean(dr)*100:.2f}' if dr else '---'):>8}"
        print(line)


TABLES = {"removal": tab_removal, "patchsize": tab_patchsize,
          "calibration": tab_calibration, "localisation": tab_localisation,
          "datasets": tab_datasets, "leak": tab_leak, "leakbase": tab_leakbase,
          "conv": tab_conv, "convlive": tab_convlive, "convleak": tab_convleak}


if __name__ == "__main__":
    want = sys.argv[1:] or list(TABLES)
    for t in want:
        if t not in TABLES:
            sys.exit(f"unknown table {t!r}; choose from {', '.join(TABLES)}")
        TABLES[t]()
