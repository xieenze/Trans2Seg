#!/usr/bin/env bash

set -x

CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=8 \
  $(dirname "$0")/train.py --config-file $CONFIG ${@:2}