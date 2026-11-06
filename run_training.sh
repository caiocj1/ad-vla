#!/usr/bin/env bash
# Train one experiment of ad_vla/conf/experiment on every visible GPU, e.g.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_training.sh examples/sft_ttvlm_v1_qa data.train_dataloader_args.batch_size=2
set -euo pipefail

experiment="${1:-examples/sft_ttvlm_v1_qa}"
shift || true
uv run python scripts/run_training.py "+experiment=${experiment}" "$@"
