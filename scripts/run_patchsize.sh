#!/usr/bin/env bash
# Table "patchsize": effect of trigger size, with fine-pruning and the
# random control, on the framework's own poisoned-MNIST scenario.
#
# eval_repair.py reads the trained scenario from
# results/NN_Train_mnist_128_trust_trust_PathSize_<p>/nn_model.pkl, which
# tests/test_mnist_poisoned.py produces (--not-ptas trains the network only,
# without the trust-propagation server the framework normally attaches).
# Those model files are not part of this archive, so the first run trains
# them: 20 epochs each, a few minutes on a GPU.
source "$(dirname "$0")/_env.sh"

for P in 1 4 10; do
  if [ ! -f "results/NN_Train_mnist_128_trust_trust_PathSize_${P}/nn_model.pkl" ]; then
    $PY tests/test_mnist_poisoned.py --patch-size $P --epochs 20 --hidden-dim 128 --not-ptas
  fi
  $PY eval_repair.py --dataset mnist --arch 128 --poisoned-patch $P
done

$PY summarise.py patchsize
