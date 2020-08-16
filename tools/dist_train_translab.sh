#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1

#srun --mpi=pmi2 -p pat_saturn -n1 --gres=gpu:8 --ntasks-per-node=1 --job-name=point_rend --kill-on-bad-exit=1 \
python -m torch.distributed.launch --nproc_per_node=8 \
  $(dirname "$0")/train_translab.py --config-file $CONFIG ${@:3}