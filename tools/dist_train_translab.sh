#!/usr/bin/env bash
set -e
set -x

CONFIG=$1
GPUS=${GPUS:-8}

python -m torch.distributed.launch --nproc_per_node=8 \
  $(dirname "$0")/train_translab.py --config-file $CONFIG ${@:2}