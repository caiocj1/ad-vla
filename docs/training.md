# Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/run_training.py +experiment=examples/sft_ttvlm_v1_qa
```

Lightning trains on every visible GPU with DDP. The global batch is the per-GPU batch
(`data.train_dataloader_args.batch_size`) × GPUs × `trainer.accumulate_grad_batches`; the examples
use 64. With one GPU, raise the accumulation to keep it.

## Outputs

```text
$SAVE_PATH/<logging.version>/
  checkpoints/                   Lightning checkpoints (weights, optimizer, progress)
  resolved_training_config.yaml  the full composed config of the run
  hf_ckpt/                       Hugging Face export of the final model, LoRA merged
  best/hf_ckpt/                  export of the best checkpoint, with export.checkpoints: last_and_best
  post_training_export_status.txt
```

`export.checkpoints: last_and_best` also exports the checkpoint with the best `export.best_monitor`
(`eval/ADE` by default). It needs a `ModelCheckpoint` callback on that metric in
`trainer.callbacks`.

Point `model.ckpt_path` at an `hf_ckpt/` to evaluate, test or continue training from it, and FM-VLA's
`model.model_args.backbone_ckpt` at a TextTrajVLM export.

## A typical pipeline

1. **TextTrajVLM** (`examples/sft_ttvlm_v1_qa`): LoRA on Qwen3.5-2B over the driving mix, with
   nuScenes QA, one epoch.
2. **FM-VLA** (`examples/sft_fm_vla`): the action expert on the frozen backbone from step 1, five
   epochs. Only the expert trains, so it is much cheaper per step.
3. Optionally **Flow-GRPO** on closed-loop simulator states (below).

## Resuming

To continue an interrupted run, start it again with the same experiment and `resume_ckpt_path=last`
(the newest checkpoint of this `logging.version`), or a path to a `.ckpt`:

```bash
uv run python scripts/run_training.py +experiment=examples/sft_ttvlm_v1_qa resume_ckpt_path=last
```

Lightning restores the weights, optimizer, schedule and step. The training sampler also continues
from the sample it stopped at, so the epoch neither repeats nor skips samples. A resumed run must
have the same data, batch size, number of GPUs and epochs. It refuses a run that doesn't match.

## Reasoning traces

With `data.reasoning_traces_path` (a folder of `<split>_annotations.jsonl`) and
`prompt.use_cot: true`, TextTrajVLM learns to write a reasoning trace before the trajectory.

## GRPO (TextTrajVLM)

GRPO samples several trajectories per scene, scores them with a reward and pushes the policy towards
the better ones, with a KL penalty to the starting model. Start from an exported SFT checkpoint:

```yaml
# @package _global_
defaults:
  - /prompt: drive_poutine
  - /dataset@data.train.waymo: waymo
  - /dataset@data.val.waymo: waymo
  - override /train_stage: grpo
  - override /model: text_traj_vlm
  - _self_

model:
  ckpt_path: ${data.save_path}/sft_ttvlm_v1_qa/hf_ckpt

logging:
  version: grpo_ttvlm

data:
  train:
    waymo:
      size: [608, 544]
      split: val          # the rater-scored validation frames, for the RFS reward
      subset: ${oc.env:ASSETS_DIR}/indices/waymo_pref_idx.pt
      subset_fraction: [0.0, 0.87]
  val:
    waymo:
      size: [608, 544]
      subset: ${oc.env:ASSETS_DIR}/indices/waymo_pref_idx.pt
      subset_fraction: [0.87, 1.0]
  train_dataloader_args:
    batch_size: 1

trainer:
  accumulate_grad_batches: 32
  val_check_interval: 100
  use_distributed_sampler: false
  max_epochs: 10

train_stage:
  num_traj_samples_per_gpu: 4
  rl_cfg:
    normalize_by_std: true
    kl_beta: 0.001
    reward_fn:
      _target_: ad_vla.rewards.rfs_reward.RFSReward
  generate_cfg:
    temperature: 1.3
  optim_cfg:
    optimizer:
      lr: 1.0e-5
```

Rewards in `ad_vla/rewards/`: `NormL2Reward` (the default, distance to the logged path),
`RFSReward` (Waymo rater feedback score), `BEVReward`, `OccAnyBEVReward`, `AgentCollisionReward` and
`SimPDMReward`.

## Flow-GRPO (FM-VLA)

`train_stage: grpo_flow` applies GRPO to FM-VLA's action expert, with the backbone frozen. Sampling
turns the flow into a stochastic process (`generate_cfg.sampler: sde`) whose per-step log
probabilities give the policy gradient, as in Flow-GRPO. Validation uses the deterministic
deployment sampler (`val_generate_cfg`: zero start, 4 Euler steps).

It trains on closed-loop states from the AlpaSim simulator (`dataset: alpasim_states`), and its
default reward, `SimPDMReward`, is a PDM-style score of each sampled plan against the state's other
agents and lanes: collisions, staying in the drivable corridor, progress, comfort. See
`ad_vla/conf/train_stage/grpo_flow.yaml` for every setting and
[datasets.md](datasets.md#alpasim-states-flow-grpo) for the states.
