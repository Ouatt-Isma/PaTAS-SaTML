# Provenance Is Enough: artifact

Code, driver scripts and result archives for every number in *Provenance Is
Enough: Removing Backdoors From the Parameters an Untrusted Source Shaped*
(SaTML submission). One repository, as the paper's Open Science section
promises: the trust-attribution implementation, the convolutional
input-position variant, the six baselines, one driver script per table, the
figure script, and the JSON summaries those scripts produced for the tables
in the paper.

Two ways to use it:

* **Check the reported numbers without running anything.**
  `python summarise.py` prints every table of the paper from the archived
  summaries under `results/`, with the same aggregation (mean and population
  standard deviation over seeds). `python summarise.py removal` prints one
  table by its LaTeX label.
* **Rerun an experiment.** `scripts/run_<table>.sh` reruns the experiment
  behind one table and prints it; `scripts/run_all.sh` runs everything. The
  dense experiments are minutes on a CPU or a single GPU; the two
  convolutional scripts train ResNet-18s and take a few hours on one A100.

## Layout

```
attribution.py          parameter-level attribution (Eq. mass/r/s), the decision rules
                        (Otsu bimodality, fixed-deviation), ProvenanceSource, MultiSourceProvenance
eval_repair.py          removal at fixed budgets on the framework's poisoned scenario:
                        attribution / influence-share / fine-pruning / random   (Table patchsize)
influence_baseline.py   removal vs drop-source, influence (TracIn), spectral signatures,
                        activation clustering, random, two poisoning rates      (Table removal)
eval_localisation.py    what the rule flags across seven conditions             (Table localisation)
eval_multisource.py     three sources, the adaptive adversary, three criteria   (Tables leak, leakbase)
conv_attribution.py     input-position attribution, ResNet-18, masking          (Tables conv, convlive)
conv_multisource.py     the adaptive adversary at input-position level          (Table convleak)
calibrate_k.py          sweep of the fixed-deviation constant                   (Table calibration)
applicability.py        quiet region, untrusted share, trigger ranks            (Table datasets)
make_fig_trust.py       Figure 1 from the localisation and multi-source archives
summarise.py            every table from results/*.json
subjective_logic.py     re-export of the subjective-logic primitives (bpq_vec, ...)
input_trust.py          the conformity trust source (an alternative attribution.py can use;
                        not used by any table)
patas_module/           the framework: dense trainer (NN/primaryNN.py), dataset loaders and
                        poisoning (NN/datasets.py), cache naming and dataset metadata (main.py),
                        subjective logic (subjective_logic.py), and the socket-based trust
                        propagation runtime (NN/PTAStemplate.py, PTASTemp/, concrete/) that
                        tests/test_mnist_poisoned.py can attach to training
tests/                  test_mnist_poisoned.py / test_gtsrb_pois.py: the framework's poisoned
                        scenario trainers, which produce the caches eval_repair.py reads
scripts/                one run script per table, plus run_all.sh
jobs_slurm/             the SLURM job files that produced the dense archives (provenance only)
results/                JSON summaries for every table, the run logs of the convolutional
                        experiments, the trained dense models, and Figure 1
```

## Environment

Python 3.10 or newer, and

```
pip install -r requirements.txt
```

NumPy, SciPy, scikit-learn, matplotlib, PyTorch and torchvision. Attribution,
flagging and removal for the dense experiments are NumPy on CPU; training
uses PyTorch (CPU is fine for the dense networks, a GPU is needed for the
ResNet-18 runs). The socket-based trust-propagation runtime in
`patas_module` is not exercised by any experiment here except optionally by
`tests/test_mnist_poisoned.py`, which `scripts/run_patchsize.sh` runs with
`--not-ptas` (network training only).

Everything is run from the repository root; the scripts resolve `results/`
and `data/` relative to it (`scripts/_env.sh` takes care of that).
`PYTHON=/path/to/python scripts/run_x.sh` selects an interpreter,
`DEVICE=cuda|cpu` overrides the automatic device choice of the convolutional
scripts.

## Data

MNIST, Fashion-MNIST, GTSRB and CIFAR-10, all downloaded by the loaders in
`patas_module/NN/datasets.py` on first use (torchvision for Fashion-MNIST,
GTSRB and CIFAR-10, cached as `.npz` under `data/`; MNIST from
`~/.keras/datasets/mnist.npz` if present, otherwise TensorFlow if installed,
otherwise scikit-learn's OpenML fetch). Standardisation constants are
computed on the training split and cached beside the data. Poisoning is
applied in memory, so no derived dataset is distributed.

## Where each number comes from

| Table / figure | Script | Archive read by `summarise.py` | Rerun |
|---|---|---|---|
| removal | `influence_baseline.py --untrusted-opinion 0,0,1` [`--poison-frac 0.15`] | `results/Influence_mnist_128_p4_op001/`, `..._pf0.15_op001/` | `scripts/run_removal.sh` |
| patchsize | `eval_repair.py --poisoned-patch {1,4,10}` | `results/Repair_mnist_128_patch{1,4,10}/` | `scripts/run_patchsize.sh` |
| calibration | `calibrate_k.py --no-center` | `results/Calibration_mnist/` | `scripts/run_calibration.sh` |
| localisation | `eval_localisation.py --rule otsu --untrusted-opinion 0,0,1` | `results/Localisation_mnist_otsu/` | `scripts/run_localisation.sh` |
| datasets | `applicability.py` | `results/Applicability/` | `scripts/run_applicability.sh` |
| leak, leakbase | `eval_multisource.py --poison-leak {0,.25,.3,.35,.4,.5}` | `results/MultiSource_mnist_128/summary_uc4-7_cf0.25_p4*.json` | `scripts/run_multisource.sh` |
| conv, convlive | `conv_attribution.py --arch resnet18 [--augment] [--live-pct 20] [--clean-control]` | `results/ConvAttr_*/` and `results/vast_conv_logs/` | `scripts/run_conv.sh` |
| convleak | `conv_multisource.py --poison-leak ... --with-drop` | `results/ConvMulti_mnist_resnet18_cf0.25/` | `scripts/run_conv_multisource.sh` |
| Figure 1 | `make_fig_trust.py` | localisation models + multi-source summaries | `scripts/run_figure.sh` |

Every script writes a JSON summary keyed by its configuration (dataset,
architecture, patch size, poisoning fraction, opinion, leakage, rule,
liveness percentile) into its own directory under `results/`, so reruns with
different settings never overwrite one another. Models trained by the dense
scripts are cached under `results/<experiment>/models/` and reused; the
archive includes them, so every dense script here is an attribution replay
that finishes in under a minute. The convolutional scripts do not cache
models.

Seeds: three per configuration throughout, five per leakage level in the
adaptive-adversary tables (three at zero leakage), as the captions state.
Population standard deviation over seeds.

### What is in `results/`

* `Influence_*`, `Localisation_mnist_otsu`, `MultiSource_mnist_128`: JSON
  summaries plus the trained dense models (`models/*.pkl`) so the attribution
  can be replayed without retraining.
* `Repair_mnist_128_patch*`: JSON summaries only. The scenario models they
  were computed from (`results/NN_Train_mnist_128_trust_trust_PathSize_<p>/nn_model.pkl`)
  are a cache of the framework's trainer that was not carried over;
  `scripts/run_patchsize.sh` retrains them (20 epochs each) before calling
  `eval_repair.py`. Undefended accuracy and attack success of the retrained
  models will differ in the second decimal.
* `ConvAttr_*`, `ConvMulti_*`: JSON summaries of the ResNet-18 runs, and
  `vast_conv_logs/` the console logs of the same runs, which also record the
  per-seed trigger ranks quoted in the text.
* `Calibration_mnist`, `Applicability`: produced by the two helper scripts
  from the archived models.
* `fig_trust.pdf`, `fig_trust.png`: Figure 1 as it appears in the paper.

## Notes a careful reader will want

* **Opinion about the untrusted source.** The paper's default is the vacuous
  opinion (0,0,1) (Section "Experimental Setup"). The scripts' own defaults
  are the believed-compromised opinion (0,1,0), which the first draft used;
  the run scripts pass `--untrusted-opinion 0,0,1` where the table needs it.
  Table patchsize was computed under (0,1,0); at a fixed budget the two give
  the same selection (Proposition 1), so the numbers are unaffected.
* **Table calibration** was drafted with (0,1,0) and *uncentred* attribution,
  before the centred weight of Section "Centring the attribution weight" was
  introduced; `calibrate_k.py --no-center` is that setting and reproduces the
  backdoored column exactly on the archived models. The clean models it was
  computed on were not archived; on the archived clean models
  (`Localisation_mnist_otsu/models/clean128_*`) `k = 8` flags 0, 3 and 0
  features rather than 0, 0, 0. `calibrate_k.py` without `--no-center`
  gives the sweep in the paper's final setting, where the constant fails
  more visibly, which is the point of that subsection.
* **Table datasets** reports seed 0; `applicability.py` prints every seed.
  A feature's untrusted share is the maximum over the parameters it feeds.
* **Table leakbase** in the submitted PDF gives the means over the first
  two seeds of each level (the runs in `jobs_slurm/job_gap_leak_*.sh`). The
  archived summaries hold five seeds per level; over five, drop-compromised
  is 0.03 / 96.83 / 98.41 and drop-unverified 0.17 / 97.23 / 98.68 at
  0 / 25 / 50% leakage. The conclusion is unchanged; the table should be
  updated to the five-seed means.
* **Table convlive.** Two rows (MNIST standard and CIFAR-10 augmented,
  bottom fifth discarded) were overwritten on disk when the output key gained
  the architecture and augmentation suffixes; `summarise.py` recovers them
  from `results/vast_conv_logs/queue.log` and `cifar_resnet.log`, which hold
  the per-seed numbers. The CIFAR-10 standard row of that table pools a
  two-seed run and a one-seed run (`ConvAttr_cifar10_p4`,
  `ConvAttr_cifar10_p4_resnet18`).
* **Directory suffixes.** `_lp0` on the archived `ConvAttr_*` directories
  marks runs that scored every position when the script's default was still
  to discard the bottom fifth; the default is now 0, so a rerun without the
  flag lands in the unsuffixed directory. `scripts/run_conv.sh` passes
  `--live-pct 0` explicitly to keep the archived names.
* **Decision rules.** All dense tables use the Otsu bimodality rule; the
  convolutional tables use the fixed-deviation rule at `k = 8` (the paper
  says why). `--rule` switches either.

## Runtimes

| Script | First run | With cached models |
|---|---|---|
| `run_removal.sh` | ~180 trainings of a 784-128-10 network (12 per seed, rate and dataset), ~1 h GPU | ~30 s per call |
| `run_patchsize.sh` | 3 scenario trainings, ~5 min GPU | ~1 min |
| `run_localisation.sh` | 21 trainings, ~10 min GPU | ~1 min |
| `run_calibration.sh`, `run_applicability.sh`, `run_figure.sh` | replay only | ~1 min each |
| `run_multisource.sh` | 84 trainings, ~1 h GPU | ~3 min |
| `run_conv.sh` | 60 ResNet-18 trainings | ~4 h on one A100 |
| `run_conv_multisource.sh` | 60 ResNet-18 trainings | ~2.5 h on one A100 |

## Provenance of the archives

The dense archives were produced on a SLURM cluster with the job files in
`jobs_slurm/` (paths inside them refer to that cluster's layout and are kept
for the record, not for running); the convolutional archives on a rented
A100 with the commands in `scripts/run_conv.sh` and
`scripts/run_conv_multisource.sh`, logged in `results/vast_conv_logs/`.
