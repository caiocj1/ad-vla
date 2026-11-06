# Installation

## Requirements

- Linux x86-64 with an NVIDIA GPU and a driver that supports CUDA 12.8.
- Python 3.12. uv downloads it if it isn't installed.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.10 or newer.
- About 25 GB of disk for the environment and uv's cache. Point `UV_CACHE_DIR` at a large disk if
  your home directory is small.

The dependencies are pinned in `pyproject.toml` and locked in `uv.lock`: PyTorch 2.8 (CUDA 12.8
wheels), Transformers 5.18, PyTorch Lightning, PEFT and Hydra. FlashAttention 2.8.3 and
causal-conv1d 1.6.0 come as prebuilt wheels for this exact stack (Python 3.12, PyTorch 2.8,
CUDA 12, C++11 ABI), so nothing is compiled.

## Install

```bash
git clone https://github.com/caiocj1/ad-vla.git
cd ad-vla
uv sync --dev          # creates .venv from uv.lock
cp .env.example .env   # see below
```

Run commands with `uv run ...`, or activate the environment with `source .venv/bin/activate`. To put
the environment somewhere else, set `UV_PROJECT_ENVIRONMENT=/path/to/venv` before `uv sync`.

Some dependencies are installed from Git repositories: NAVSIM, Alpamayo (for its action input
projection, used by FM-VLA), VGGT (spatial forcing), SimpleBEV (BEV forcing) and
torchcubicspline. `uv sync` needs access to github.com.

## Environment variables

The scripts read `.env` from the repository root with python-dotenv, and the configs read the
variables with `${oc.env:NAME}`. `.env.example` lists every variable with a comment. You need:

- `SAVE_PATH`: where runs are written.
- `HF_HOME`: the Hugging Face cache. Qwen3.5-2B (about 5 GB) downloads there on the first run.
- The roots of the datasets you use, and the derived files your experiment refers to. See
  [datasets.md](datasets.md).
- `HF_TOKEN`, only for gated models and datasets such as PhysicalAI-AV.

## Check the installation

```bash
uv run pytest tests -m "not gpu_integration and not dataset_integration"   # CPU
uv run pytest tests -m gpu_integration                                        # one GPU
uv run pytest tests -m dataset_integration                                    # needs the dataset roots
```

The GPU integration tests swap the backbone for `tiny-random/qwen3.5` from the Hugging Face Hub and
train each model for a few steps on generated samples. Without Hub access they skip. The dataset
tests skip the datasets whose root variable isn't set.

## Changing the stack

The prebuilt FlashAttention and causal-conv1d wheels only match Python 3.12, PyTorch 2.8 and CUDA 12.
If you change one of these, pick the matching wheels from the
[flash-attention](https://github.com/Dao-AILab/flash-attention/releases) and
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) releases. Change their URLs
under `[tool.uv.sources]` in `pyproject.toml` and run `uv lock`. Building FlashAttention from source
instead needs the CUDA toolkit (`nvcc`) and an installed PyTorch.

## Troubleshooting

- **`flash_attn` import errors or illegal memory accesses:** the wheel doesn't match the installed
  PyTorch. Check `uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- **Training hangs at the start of the first epoch with several dataloader workers:** set
  `TOKENIZERS_PARALLELISM=false` (as in `.env.example`). With `true`, forked workers can deadlock in
  the tokenizer.
- **Hydra `Key ... is not in struct`:** the key doesn't exist in the composed config. Add it with
  `+key=value`, or use `++key=value` to add or override.
