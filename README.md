# AD-VLA

AD-VLA is a focused research package for supervised trajectory prediction with
a vision-language model. It contains the `TextTrajVLM` training path and an
optional bird's-eye-view (BEV) forcing objective. The repository is deliberately
portable: configuration uses ordinary environment variables and contains no
site-specific scheduler, filesystem, proxy, account, or credential settings.

## What is included

- `TextTrajVLM`, a Qwen image-text backbone trained to emit future XY
  trajectories as text.
- The supervised Lightning training stage, LoRA support, validation metrics,
  and post-training Hugging Face export.
- Waymo E2E, NAVSIM, PhysicalAI-AV, nuScenes, and concat dataset adapters.
- Training, evaluation, and testing entry points.
- `BEVForcingAuxHead`, which uses a frozen SimpleBEV teacher to add spatial
  supervision at an intermediate language-model layer.
- Two ready-to-compose training experiments:
  - `sft_text_traj_vlm`: ordinary trajectory SFT.
  - `sft_text_traj_vlm_bev`: the same SFT run plus BEV forcing.

Unrelated model families, reward-learning stages, annotation pipelines,
notebooks, and cluster-specific launchers are intentionally out of scope.

## Repository map and suggested reading order

1. `ad_vla/conf/experiment/` contains the two top-level experiment variants.
2. `ad_vla/conf/default_training.yaml` assembles datasets, model, training
   stage, trainer, logging, and output settings.
3. `scripts/run_training.py`, `scripts/run_evaluation.py`, and
   `scripts/run_testing.py` are the local entry points.
4. `ad_vla/models/text_traj_vlm.py` contains trajectory prompting, collation,
   assistant-token masking, causal-language loss, generation, and parsing.
5. `ad_vla/train_stages/supervised_train_module.py` combines the model loss
   with an optional auxiliary loss.
6. `ad_vla/models/aux_heads/bev_forcing.py` and
   `ad_vla/models/layers/bev_cross_attention.py` implement BEV forcing.
7. `ad_vla/dataset/` defines the common sample schema and dataset adapters.
8. `resources/physical_ai/val_1000_clip_ids.txt` is the fixed PhysicalAI
   validation subset used for comparable evaluation runs.

Hydra's `defaults` list is the key to reading a run. Start from an experiment
file, follow each referenced config group, and then inspect the resolved
`_target_` classes. Every completed run also writes
`resolved_training_config.yaml`, which is the exact flattened configuration
used for that run.

## Environment setup

The project requires Python 3.12 and CUDA 12.8-compatible PyTorch wheels.
Install the locked environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
cp .env.example .env
```

### CUDA extension installation

`causal-conv1d` and `flash-attn` are compiled CUDA extensions. The lockfile
selects the official prebuilt `causal-conv1d` 1.6.0 wheel matching this
project's Python 3.12, CUDA 12, PyTorch 2.8, and CXX11-ABI stack. This avoids a
local CUDA compilation for that package. `flash-attn` is built after PyTorch is
available, using uv's `no-build-isolation-package` setting in `pyproject.toml`.

For a clean, explicit two-stage installation, use:

```bash
uv sync --dev \
  --no-install-package causal-conv1d \
  --no-install-package flash-attn

uv pip install \
  "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

uv pip install flash-attn==2.8.3 --no-build-isolation
uv sync --dev
```

The final `uv sync` verifies the environment against `uv.lock`; it should not
reinstall extensions whose locked versions are already present. If Python,
PyTorch, CUDA, or ABI settings are changed, select a matching wheel from the
official `causal-conv1d` releases or build it from source instead of using the
wheel above. A working CUDA toolkit and `nvcc` are required to build
`flash-attn`.

At minimum, configure these paths in `.env`:

```dotenv
WAYMO_INDEX_PATH=/path/to/waymo_e2e_index
OUTPUT_DIR=/path/to/training_outputs
```

Set `OPENSCENE_DATA_ROOT` when using NAVSIM and
`PHYSICAL_AI_CACHE_DIR`/`HF_TOKEN` when using PhysicalAI-AV. `HF_HOME` can be
used as the PhysicalAI cache fallback. Comet variables are needed only when
`logging.disable_comet=false`; TensorBoard logging is the default.

## Preparing Waymo E2E data

The dataset adapter reads lightweight pickle indices that point into the
original TFRecord files. Build one index per split:

```bash
uv run python scripts/preprocess_waymo.py \
  --data_path /path/to/waymo_e2e_tfrecords \
  --output_path /path/to/waymo_e2e_index \
  --split train

uv run python scripts/preprocess_waymo.py \
  --data_path /path/to/waymo_e2e_tfrecords \
  --output_path /path/to/waymo_e2e_index \
  --split val
```

The index directory must contain `wod_e2e_train_index.pkl`,
`wod_e2e_val_index.pkl`, and
`val_sequence_name_to_scenario_cluster.json`. Index entries store absolute
TFRecord paths, so regenerate them after moving the raw dataset.

## NAVSIM and PhysicalAI-AV data

For NAVSIM, set `OPENSCENE_DATA_ROOT` to a directory containing
`navsim_logs/trainval` and `sensor_blobs/trainval`. The bundled public split is
loaded automatically. The NAVSIM package is locked from its official GitHub
repository.

PhysicalAI-AV is accessed through `physical-ai-av`. Set `HF_TOKEN` if access to
the dataset is gated and optionally set `PHYSICAL_AI_CACHE_DIR` for its cache.
The repository includes a stable 1,000-clip validation subset at
`resources/physical_ai/val_1000_clip_ids.txt`; pass that path through
`data.val_dataset.clip_ids_path` when reproducing the subset.

The configs under `ad_vla/conf/dataset/` include individual dataset adapters
and public concat variants. For example, an SFT run can replace its training
dataset with `dataset@data.train_dataset=waymo_navsim_concat` and provide both
`WAYMO_INDEX_PATH` and `OPENSCENE_DATA_ROOT`.

## Running ordinary SFT

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_training.py \
  +experiment=sft_text_traj_vlm
```

Or use the convenience wrapper:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_training.sh sft_text_traj_vlm
```

Lightning uses every visible GPU. For example, set
`CUDA_VISIBLE_DEVICES=0,1,2,3` for four local GPUs. Adjust per-device batch size
under `data.train_dataloader_args` and gradient accumulation under
`trainer.accumulate_grad_batches`.

With the locked Qwen3.5/Transformers/FlashAttention versions, keep each
process's train and validation batch size above 1. Batch size 1 can enter a
known Qwen3.5 multidimensional-position-ID bug in the FlashAttention packed
sequence check.

## Evaluation and testing

Evaluate an exported checkpoint on the fixed PhysicalAI validation subset:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_evaluation.py \
  dataset@data.val_dataset=physical_ai \
  model.ckpt_path=/path/to/run/hf_ckpt \
  data.val_dataset.clip_ids_path=resources/physical_ai/val_1000_clip_ids.txt
```

Evaluate on another dataset by selecting its config, such as
`dataset@data.val_dataset=waymo` or `dataset@data.val_dataset=navsim`.

Run the testing path and create a Waymo submission with:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_testing.py \
  model.ckpt_path=/path/to/run/hf_ckpt
```

Waymo predictions are written under `wod_subs/`; KITScenes-style predictions
are written under `kit_subs/`. Run these commands from the repository root if
you want those relative output directories to live inside the checkout.

## Running SFT with BEV forcing

BEV forcing additionally requires the official SimpleBEV source tree and a
pretrained checkpoint directory. SimpleBEV is not distributed as an installable
Python package, so clone it alongside this repository and expose the checkout on
`PYTHONPATH`:

```bash
git clone https://github.com/aharley/simple_bev.git ../simple_bev
export PYTHONPATH="$(realpath ../simple_bev):${PYTHONPATH:-}"
export BEV_MODEL_CKPT_PATH=/path/to/simple_bev_checkpoint
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_training.py \
  +experiment=sft_text_traj_vlm_bev
```

The auxiliary head keeps SimpleBEV frozen, captures VLM image-token hidden
states at `train_stage.aux_head.target_layer`, predicts a forward BEV occupancy
map through learned cross-attention, and adds binary cross-entropy to the
trajectory language loss. Its contribution is controlled by
`train_stage.aux_loss_weight`; `aux_warmup_steps` can ramp that weight from zero.

Important BEV settings:

- `bev_model_ckpt_path`: SimpleBEV checkpoint directory.
- `target_layer`: Qwen language-model layer whose image-token states are used.
- `train_bev_head_only`: freeze the trajectory model and train only the BEV
  prediction head.
- `add_pos_embedding`, `add_residual`, `add_ffn`, `num_layers`, and
  `num_attention_heads`: optional cross-attention variants.

The Waymo adapter preserves resized model images and fixed-size teacher images
with matching calibrations. BEV forcing uses the teacher camera tensors to
produce its frozen target and the Qwen image tokens to predict that target.

## Outputs and checkpoint loading

Runs are written to:

```text
${OUTPUT_DIR}/${logging.version}/
├── checkpoints/                  # Lightning checkpoints
├── resolved_training_config.yaml
├── hf_ckpt/                      # merged Hugging Face model export
└── aux_models/                   # BEV cross-attention weights, when enabled
```

The final exporter restores the saved Lightning state, merges LoRA adapters,
writes `hf_ckpt/`, and separately saves auxiliary-head weights. To resume or
evaluate a trajectory model, point `model.ckpt_path` at the exported
`hf_ckpt/`. The BEV head is a training objective and is not required for normal
trajectory generation.

## Configuration examples

Hydra values can be overridden directly from the command line:

```bash
uv run python scripts/run_training.py \
  +experiment=sft_text_traj_vlm \
  data.train_dataloader_args.batch_size=2 \
  trainer.accumulate_grad_batches=8 \
  logging.version=my_sft_run
```

To inspect the fully composed configuration without training:

```bash
uv run python scripts/run_training.py \
  +experiment=sft_text_traj_vlm \
  --cfg job --resolve
```

Keep credentials and machine-specific paths in `.env` or the process
environment. Do not commit `.env`, checkpoints, dataset indices, or generated
outputs.
