# Configuration

Runs are composed by [Hydra](https://hydra.cc) from `ad_vla/conf/`. Each script has a primary config,
and an experiment file adds to it:

| Script | Primary config | Experiment |
|---|---|---|
| `scripts/run_training.py` | `default_training.yaml` | `+experiment=<path under conf/experiment/>` |
| `scripts/run_evaluation.py` | `default_evaluation.yaml` | same |
| `scripts/run_testing.py` | `default_testing.yaml` | same |

```bash
uv run python scripts/run_training.py +experiment=examples/sft_ttvlm_v1_qa trainer.max_epochs=2
uv run python scripts/run_training.py +experiment=examples/sft_ttvlm_v1_qa --cfg job --resolve  # print only
```

Config groups:

| Group | Contents |
|---|---|
| `dataset/` | One dataset adapter each: `waymo`, `navsim`, `nuscenes`, `physical_ai`, `kitscenes`, `nuplan_qa`, `alpasim_states`. |
| `dataset_mix/` | Training mixes of several datasets with their sizes and settings: `v1`, `v1_qa`, `v1_qa_waymo`, `v2_qa`. |
| `model/` | `text_traj_vlm`, `fm_vla`, `qwen_fm_vla`. |
| `prompt/` | `drive_poutine` (driving), `drive_vqa` (driving and any QA), `drive_nearby_obj_det_cot` (lists the five nearest objects and their positions before the trajectory). |
| `train_stage/` | `supervised_train`, `grpo`, `grpo_flow`, `depth_probing`. |
| `trainer/` | Lightning `Trainer` settings: DDP on all visible GPUs, one epoch. Models load in bfloat16. |
| `img_transform/` | `default` (resize), `center_crop_resize`, `camera_rig_augment`. |
| `experiment/` | Complete runs. `examples/` is the place to start. |

Your own experiments can go in `ad_vla/conf/experiment/scratch/` or be named `*_untracked.yaml`;
git ignores both. Every experiment file starts with `# @package _global_`.

## Datasets: `data.train` and `data.val`

Both map a name you choose to a dataset config, read by `ad_vla/dataset/builder.py`. Mount each
dataset in the experiment's `defaults`, or a whole mix from `dataset_mix/`, then state only what
differs:

```yaml
defaults:
  - /prompt: drive_poutine
  - /dataset_mix@data.train: v1_qa
  - /dataset@data.val.waymo: waymo
  - /dataset@data.val.physical_ai: physical_ai
  - /dataset@data.val.nuscenes_qa: nuscenes
  - override /model: text_traj_vlm
  - _self_

data:
  train:
    navsim: {frame_interval: 1}
  primary_val: waymo
  val:
    waymo: {size: [608, 544], subset: ${oc.env:ASSETS_DIR}/indices/waymo_pref_idx.pt}
    physical_ai: {size: [448, 768], clip_ids_path: ${oc.env:PHYSICAL_AI_VAL_CLIP_IDS}}
    nuscenes_qa: {task: qa, size: [384, 672], qa_annotations_path: /path/to/questions.json, max_questions: 1000}
```

A single change also works on the command line, e.g. `data.train.navsim.frame_interval=1`.

The builder reads these keys of an entry and passes everything else to the dataset:

| Key | Meaning |
|---|---|
| `task` | `trajectory` (default) or `qa`, the only switch for QA. `data.val` can hold several QA sets. |
| `size` | Required: `[height, width]`, or `null` for the native resolution. |
| `img_transform` | What runs at that size. Omitted: a plain resize. A name in `conf/img_transform` (`center_crop_resize`), that name with overrides (`{name: center_crop_resize, p_transform: 0.5}`), or a full `{_target_: ...}`. |
| `split` | Dataset split; defaults to `train` in `data.train` and `val` in `data.val`. |
| `repeat` | Training only: how many times the entry appears in the mix. |
| `subset` | Path of a saved list of dataset indices. |
| `subset_fraction` | `[start, end)` share of the set after `subset`, e.g. to split one set into training and validation parts. |
| `max_samples` | Validation during training only: sample cap for this set. The default is `data.val_periodic_samples` (1,000). The final validation uses the whole set. |
| `max_questions`, `subset_seed` | QA sets: a seeded subset of questions, used in both periodic and final validation. Null means all. |
| `dataloader_args` | Overrides `data.val_dataloader_args` for this set. |

With several trajectory validation sets, `data.primary_val` names the one that logs the plain
`eval/*` metrics and picks the best checkpoint. See [evaluation.md](evaluation.md) for the metrics.

Testing takes one dataset the same way, `data.test.<name>` (split `test` by default).

### Image sizes and cameras

Each entry has its own `size`, so datasets with different aspect ratios train together. Use
multiples of 32: each image token covers 32x32 pixels (16-pixel patches merged 2x2), so
`[256, 448]` gives 112 tokens a camera. The model reads the cameras of
`model.model_args.camera_sequence` in that order. The entry's `sensors` must provide them, and
`concat_cameras: true` joins them into one panorama per frame.

### Past frames

`model.model_args.num_past_image_frames` is how many frames the model sees: 1 is the current frame
only, 3 adds two past frames. The datasets then need to return them: `num_past_frames` and
`past_frame_stride` on Waymo (stride in 10 Hz frames, 5 = 0.5 s) and NuPlanQA,
`num_frames` and `frame_spacing_s` on PhysicalAI, `num_history_frames` on NAVSIM. The `drive_vqa`
prompt labels the frames and states their spacing.

## Question answering

`task: qa` makes an entry train on questions. It has one item per frame with questions, repeated
`questions_per_frame` times, and each item draws one question, so an epoch doesn't repeat a question
of a frame. Frames without questions are left out. A QA item has no planning supervision. Its prompt
shows all six labelled current camera views, without trajectory or intent input. QA entries must use
`sensors: all_cams`.

```yaml
defaults:
  - /prompt: drive_poutine
  - /dataset@data.train.nuscenes: nuscenes
  - /dataset@data.train.nuscenes_qa: nuscenes
  - /dataset@data.val.waymo: waymo
  - override /model: text_traj_vlm
  - _self_

data:
  train:
    nuscenes: {size: [448, 768], sensors: front_three}
    nuscenes_qa:
      task: qa
      size: [384, 672]
      sensors: all_cams
      questions_per_frame: 3
      qa_annotations_path: /path/to/NuScenes_train_questions.json
```

`questions_per_frame` sets the QA share of the mix, not a fixed QA share per batch. In a batch,
`batch.qa_mask` marks the QA rows. The drawn question is `sample.qa_pair`, and all of a frame's
questions stay on `sample.qa_pairs`.

Models: `text_traj_vlm` and `qwen_fm_vla` train on QA rows. `fm_vla` doesn't, and neither does
GRPO.

- **nuScenes QA:** see [datasets.md](datasets.md#nuscenes). The `drive_poutine` prompt has a
  nuScenes QA system turn and refuses other QA datasets.
- **Waymo QA:** add `qa_annotations_path` and `clip_index_path` to a `waymo` entry with `task: qa`
  (see `dataset_mix/v1_qa_waymo.yaml`). Waymo's rear cameras are about half as tall as the others;
  `camera_sizes` gives listed cameras their own size, e.g.
  `{REAR_LEFT: [192, 320], REAR: [192, 320], REAR_RIGHT: [192, 320]}`. Needs `prompt: drive_vqa`.
  Spatial and BEV forcing don't support `camera_sizes` yet.
- **NuPlanQA:** about a million questions on 89k frames. One item per question
  (`questions_per_frame: null`) would be half of the `v2_qa` mix; `questions_per_frame: 1` gives one
  item per frame and a different question each epoch.

  ```yaml
  defaults:
    - /dataset@data.train.nuplan_qa: nuplan_qa
    - /dataset@data.val.nuplan_qa: nuplan_qa
  data:
    train:
      nuplan_qa: {questions_per_frame: 1}
    val:
      nuplan_qa:
        qa_annotations_path: ${oc.env:ASSETS_DIR}/annotations/nuplan_qa/nuplanqa_eval_test.json
  ```

**QA validation.** A `task: qa` entry in `data.val` adds a validation loader that flattens the
annotations to one item per question:

```yaml
data:
  val:
    nuscenes_qa:
      task: qa
      split: val
      sensors: all_cams
      size: [384, 672]
      qa_annotations_path: /path/to/NuScenes_val_questions.json
      require_full_past: false
      require_full_future: false
      max_questions: 1000
      subset_seed: 0
      dataloader_args: {batch_size: 4, num_workers: 4}
```

Every annotated sample must exist in the configured split; turning off the full-history and
full-future requirements keeps scenes near their boundaries. The metrics count exact answers and
valid answers over all evaluated questions. Multiple-choice sets (Waymo `validation_mcq.jsonl`,
NuPlanQA evaluation) accept only the option's letter.

## Loaders and optimizer

```yaml
data:
  train_dataloader_args: {batch_size: 4, num_workers: 4}
  val_dataloader_args: {batch_size: 4, num_workers: 0}
```

Batch sizes are per GPU. With `num_workers: 0`, validation loads data inside each rank, which avoids
forking workers after CUDA and DDP start. Leave `prefetch_factor` unset when `num_workers` is 0.
Standalone evaluation uses `data.val_dataloader_args`, and testing `data.test_dataloader_args`.

`trainer.val_check_interval` counts training batches, not optimizer steps. Periodic validation uses
the first 1,000 samples of each trajectory set (`max_samples`); the final validation uses all of
them. Training drops the last incomplete batch.

`train_stage.optim_cfg.parameter_group_lrs` sets separate learning rates for `vlm` and
`action_branch` (Qwen FM-VLA's `action_expert`, `state_encoder`, `action_in_proj` and
`action_out_proj`); null keeps one group.

## Input dropout

`data.drop_cfg` randomly removes inputs during training, so the model also learns to drive without
them: `{past_traj: 0.2, intent: 0.2}`. For FM-VLA, `expert_past_traj` drops the expert's past tokens
independently of the prompt's past. `data.past_traj_truncation_prob` (FM-VLA only) shortens the past of a
share of the samples instead of removing it.

## Common overrides

| Field | Purpose |
|---|---|
| `logging.version` | Run name: the output folder and the logger's run name. |
| `logging.disable_comet` | `true` (default) logs to TensorBoard, `false` to Comet. |
| `model.ckpt_path` | An exported `hf_ckpt/` to evaluate, test or start from. |
| `model.model_args.vlm_name_or_path` | Base VLM when building from pretrained parts, e.g. `Qwen/Qwen3.5-2B`. |
| `model.model_args.past_traj_sampling`, `pred_traj_sampling` | Number and spacing (s) of past and predicted poses. |
| `model.model_args.camera_sequence` | Cameras the model reads, in order. |
| `prompt.use_cot` | Reasoning before the trajectory (needs reasoning traces). |
| `data.<train/val>.<name>.size` | Image size `[height, width]` of an entry. |
| `data.reasoning_traces_path` | Folder of `<split>_annotations.jsonl` reasoning traces for supervised training. |
| `data.drop_cfg` | Input dropout, e.g. `{past_traj: 0.2}`. |
| `data.train_dataloader_args.batch_size` | Training batch per GPU. |
| `trainer.accumulate_grad_batches` | Gradient accumulation. |
| `train_stage.lora_cfg` | LoRA settings; null trains all weights. |
| `train_stage.optim_cfg.optimizer.lr` | Learning rate. |
| `resume_ckpt_path` | `last` to continue an interrupted run (see [training.md](training.md)). |
