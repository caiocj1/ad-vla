# Models

Model presets are in `ad_vla/conf/model/`; the code is in `ad_vla/models/`. Each planner has a
collator that turns samples into model inputs, `compute_loss` for training, and
`predict_from_processed_inputs` for trajectory prediction. Coordinates are in the current ego frame:
x forward, y left, in metres; time intervals are in seconds.

| Preset | Class | What it does |
|---|---|---|
| `text_traj_vlm` | `TextTrajVLM` | Gives Qwen3.5 the camera images and a driving prompt with the past path and intent. Trains on the assistant's text (optionally with reasoning first); generates and parses the future waypoints as text. Also trains and answers QA. |
| `fm_vla` | `FlowMatchingVLA` | A separate transformer expert predicts a flow field conditioned on the prompt's key-value cache of a backbone VLM, usually a frozen, trained TextTrajVLM. Trains with the flow-matching loss and integrates noise into a trajectory at inference. |
| `qwen_fm_vla` | `QwenFlowMatchingVLA` | Joint text-and-action model: Qwen supplies hidden states to an action expert with its own context attention and optional camera-aware geometry. Trains text answers and continuous actions together, and QA rows train only the backbone. Work in progress. |

## TextTrajVLM

The prompt (`conf/prompt/`) states the cameras and frames, the past path
(`past_traj_sampling`: number and spacing of poses), and the driving command. The answer is the
future path, `pred_traj_sampling.num_poses` points at `interval_length` seconds. The examples use
the past 1.5 s (4 poses at 0.5 s) and 5 waypoints at 1 s. With `pad_missing_past_traj: true`, a
shorter past (near the start of a log) is padded, so every prompt has the same format.

Training is the language-model loss on the answer tokens only. LoRA (`train_stage.lora_cfg`, `target_modules: all-linear` in
the examples) adapts the model's linear layers; the export merges it into `hf_ckpt/`.

## FM-VLA

`fm_vla` adds a flow-matching action expert to a backbone VLM:

- `backbone_ckpt` loads the backbone, an exported TextTrajVLM (`backbone_class_path`).
  `expert_cfg.insulate_backbone: true` keeps it frozen, and the expert then trains alone.
- The expert (`expert_cfg`: `hidden_size`, `num_layers`, ...) attends to the backbone's key-value
  cache of the prompt, layer by layer (`kv_layers`; by default the backbone's full-attention layers,
  subsampled to `num_layers`).
- `past_traj_tokens: true` gives the expert its own past-path tokens (`past_traj_sampling`, e.g. 8
  poses at 0.25 s), separate from the prompt's past (`prompt_past_traj_sampling`).
  `past_traj_missing_token: true` stands in for missing poses.
- It predicts `pred_traj_sampling` poses (the example: 20 at 0.25 s, 5 s).

Training details:

- `action_representation: trajs` predicts XY targets divided by `traj_normalization`, or
  standardized with `action_stats` (per-waypoint mean and std, from
  `scripts/utils/compute_action_stats.py`). `acc_curv` predicts acceleration (m/s²) and curvature
  (1/m), converted from trajectories with `ad_vla/utils/action_utils.py`. `polar` needs
  `action_stats`.
- `diff_cfg.num_repeat_steps` repeats each target with independent noise and time. It does not add
  optimizer steps.
- `diff_cfg.num_inference_steps` sets the Euler steps from noise (time 0) to the trajectory (time 1).
  The network predicts the flow velocity.
- `diff_cfg.num_eval_samples` draws several trajectories at evaluation, and `eval_aggregation`
  (`mean` or `medoid`) combines them. `eval_init: zero` starts from zero instead of noise, for one
  deterministic sample. `cuda_graph: true` captures the expert's denoising loop for fast inference.

With a trained TextTrajVLM backbone, FM-VLA plans as well or better in closed loop and is much
faster: a few Euler steps of a small expert instead of generating dozens of tokens.

### Qwen FM-VLA loss weighting

Let `q`, `d` and `a` be the shares of batch rows that are QA, planning, and planning with a valid
future, so `q + d = 1` and `a <= d`. The objective is:

```text
text_loss = d × planning_CE + qa_loss_weight × q × QA_CE
loss = ce_loss_weight × text_loss + diff_loss_weight × a × flow_loss
```

Each CE term is averaged over the supervised tokens of its group, and the flow loss over action
examples, repeats, time steps and channels. QA rows never get action supervision. `fm_vla` uses
the flow loss alone.

The action expert reads the backbone context of the prompt only, not the answer. That context isn't
detached, so the action loss also trains the backbone. `action_expert_cfg.use_cape` turns on
camera-aware conditioning: visual temporal RoPE and intrinsics-based encoding as well as the camera
poses.

## Auxiliary heads

Auxiliary heads add a training loss on the VLM's image tokens. They are set on the training stage
(`train_stage.aux_head`, `aux_loss_weight`, `aux_warmup_steps`), not as a planner:

- **BEV forcing** (`ad_vla.models.aux_heads.bev_forcing.BEVForcingAuxHead`): a frozen SimpleBEV
  model (`bev_model_ckpt_path`) turns the cameras into a bird's-eye-view occupancy map. A
  cross-attention head predicts that map from the image tokens of language-model layer
  `target_layer`, with a binary cross-entropy loss. The Waymo adapter returns fixed-size teacher
  images for SimpleBEV next to the model's images. `train_bev_head_only` freezes the planner. See
  `examples/sft_ttvlm_bev_forcing`.
- **Spatial forcing** (`ad_vla.models.aux_heads.spatial_forcing.SpatialForcingAuxHead`): aligns the
  image tokens of one VLM layer with the features of a frozen VGGT model, resized to each image's
  token grid.

  ```yaml
  train_stage:
    aux_head:
      _target_: ad_vla.models.aux_heads.spatial_forcing.SpatialForcingAuxHead
      target_layer: 22
      add_pos_embedding: true
    aux_loss_weight: 0.25
    aux_warmup_steps: 500
  ```

## Camera calibration and augmentation

Each camera image comes with its calibration (`CalibrationDict`): intrinsics, camera-to-ego rotation
and translation (metres), the original image size, and the axis convention. Canonical camera axes
are optical (x right, y down, z forward). Resizing and cropping update the intrinsics.

`img_transform/camera_rig_augment` crops and resizes, optionally rotates a camera slightly about its
optical centre (a pinhole homography), and jitters colours. Its parameters are drawn per image
unless passed explicitly.

## Initialization and checkpoints

`ad_vla/models/instantiate_model.py` builds the configured class:

| `init_mode` | Behaviour |
|---|---|
| `auto` | With `ckpt_path`, load the saved model; otherwise build from pretrained parts when the class supports it, else from scratch. |
| `from_pretrained` | Load configuration and weights from `ckpt_path`, e.g. an exported `hf_ckpt/`. `model_args` is not applied on top. |
| `pretrained_submodules` | Build the configuration from `vlm_name_or_path` and `model_args`, then load the pretrained parts (the VLM). New heads and experts start untrained. |
| `scratch` | Build the architecture without loading pretrained weights. |

Because `from_pretrained` uses the saved configuration, choose the preset that matches the
checkpoint, and set the prompt and image sizes it was trained with. `config_overrides` changes the
saved configuration before loading, to warm-start a modified architecture: weights the checkpoint
lacks are initialized by the model.
