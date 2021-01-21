#!/usr/bin/env bash
set -e
set -x

CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=8 \
    $(dirname "$0")/eval.py --config-file $CONFIG ${@:2}