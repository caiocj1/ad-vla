# Evaluation and testing

## Metrics

Trajectory metrics compare the predicted and logged future in the ego frame, in metres:

- **ADE**: mean distance between predicted and true poses over the horizon.
- **FDE**: distance at the last pose.
- **RFS** (Waymo rater feedback score, 0 to 10): how close the prediction is to trajectories that
  human raters scored, as in the Waymo E2E challenge. It needs frames with rater trajectories, i.e.
  the Waymo validation subset `waymo_pref_idx.pt`. On other datasets it is -1.
- **ADE_single**: for models that average several samples (FM-VLA with `num_eval_samples > 1`), the
  ADE of a single sample.

The primary validation set (`data.primary_val`) logs `eval/ADE`, `eval/FDE`, `eval/RFS` and
`eval/ADE_single`; these select the best checkpoint. With several trajectory sets, each also logs
`eval/<name>/ADE` and so on (`eval/<name>/RFS` only where it exists). A QA set logs
`eval_qa/accuracy`, the share of exact answers, and `eval_qa/valid_fraction`, the share of answers
in the expected form (a letter for multiple choice). Several QA sets log `eval_qa/<name>/*`.

Predictions and ground truth are compared on each dataset's native time grid. To compare on an
explicit grid, set `data.val_target_sampling: {num_poses: 25, interval_length: 0.2}`: both are
resampled to it before the metrics. The model's output head doesn't change.

## Validation during training

Every `trainer.val_check_interval` training batches, each trajectory set is validated on its first
`max_samples` samples (default 1,000) and each QA set on its `max_questions`. After training, a
final validation runs on the complete sets.

## Standalone evaluation

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_evaluation.py +experiment=examples/eval_ttvlm \
    model.ckpt_path=/path/to/run/hf_ckpt
```

An evaluation config names the checkpoint, the prompt and the sets in `data.val`. The model loads
its saved configuration, so the prompt, the image sizes and the past and future sampling must match
the ones it was trained with (see `examples/eval_ttvlm.yaml`). For KITScenes, add
`- /dataset@data.val.kitscenes: kitscenes` to the defaults with its own size, e.g.
`data.val.kitscenes.size=[544,800]`.

## Testing and Waymo submissions

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/run_testing.py +experiment=examples/test_waymo \
    model.ckpt_path=/path/to/run/hf_ckpt
```

Testing predicts the `data.test` set (split `test` by default) and writes no metrics. For Waymo it
writes a submission for the E2E challenge to `wod_subs/MySubmission_<time>/`, sharded
`mysubmission.binproto-*` files. Their metadata (authors, affiliation, method name, ...) comes from
the `WOD_SUB_*` variables of `.env`. KITScenes predictions go under `kit_subs/`. Run from the
repository root to keep these folders in the checkout.
