# AD-VLA

Vision-language-action models for end-to-end autonomous driving. A Qwen3.5 vision-language model
reads the camera images, the ego vehicle's recent path and a driving command, and plans the next
seconds of driving. The repository trains, evaluates and tests these planners on Waymo E2E, NAVSIM,
nuScenes, NVIDIA PhysicalAI-AV and KITScenes, with PyTorch Lightning and Hydra configs.

It contains the implementation of
[*Towards Zero-Shot Transfer Across Embodiments For Driving VLAs*](https://arxiv.org/abs/2609.02341)
(WACV 2027): training on several datasets and camera rigs, and BEV-Forcing, an auxiliary loss that
transfers the ground-plane layout of objects from a bird's-eye-view model into the VLA backbone
(`examples/sft_ttvlm_bev_forcing`). If you use the code, please [cite the paper](#citation).

**Two models**

- **TextTrajVLM** writes the future waypoints as text (`(x, y)` pairs in metres, in the ego frame).
  It trains with the language-model loss and can also learn visual question answering on the same
  images (nuScenes QA, Waymo QA, NuPlanQA).
- **FM-VLA** keeps a trained TextTrajVLM frozen and adds a small action expert that reads its
  prompt's key-value cache and generates a continuous trajectory with flow matching. It is much
  faster at inference than writing out waypoints, and can be fine-tuned with reinforcement learning
  in a simulator (Flow-GRPO).

**Training options:** LoRA, mixes of several datasets with their own image sizes, past image frames,
7-class driving intents, BEV and spatial auxiliary losses, GRPO, and exact resume of an interrupted
run.

## Quick start

Requires Linux, Python 3.12, an NVIDIA GPU with drivers for CUDA 12.8, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/caiocj1/ad-vla.git && cd ad-vla
uv sync --dev
cp .env.example .env        # then set SAVE_PATH, HF_HOME and the paths of the datasets you use
```

Check the installation before downloading any dataset. The unit tests run on CPU. The integration
tests train and validate both models for a few steps on one GPU, with a tiny random Qwen model and
generated samples:

```bash
uv run pytest tests -m "not gpu_integration and not dataset_integration"
uv run pytest tests -m gpu_integration
```

Train TextTrajVLM on four GPUs, then evaluate its exported checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python scripts/run_training.py +experiment=examples/sft_ttvlm_v1_qa
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_evaluation.py +experiment=examples/eval_ttvlm
```

Each run writes to `$SAVE_PATH/<logging.version>/`: Lightning checkpoints, the resolved config, and
`hf_ckpt/`, a Hugging Face export that evaluation, testing and FM-VLA load. Metrics go to
TensorBoard (`tensorboard --logdir lightning_logs`) or, with `logging.disable_comet=false`, to Comet.

## Examples

| Experiment (`+experiment=...`) | What it does |
|---|---|
| `examples/sft_ttvlm_v1_qa` | TextTrajVLM on PhysicalAI, Waymo, NAVSIM and nuScenes, plus nuScenes QA. One image frame, the past 1.5 s, 5 waypoints at 1 s. |
| `examples/sft_fm_vla` | FM-VLA expert on the frozen backbone above: 20 poses at 0.25 s (5 s), 16 samples averaged at evaluation. |
| `examples/sft_ttvlm_v2_qa` | TextTrajVLM with 3 frames 0.5 s apart, 7-class intents and Waymo QA. |
| `examples/sft_ttvlm_bev_forcing` | TextTrajVLM on Waymo with a BEV occupancy auxiliary loss from a frozen SimpleBEV model. |
| `examples/eval_ttvlm` | Evaluation on the Waymo rater subset (ADE, FDE, RFS) and 1,000 PhysicalAI clips. |
| `examples/test_waymo` | Predictions on the Waymo E2E test split, written as a challenge submission. |

Any value can be changed on the command line, e.g. `data.train_dataloader_args.batch_size=2` or
`trainer.max_epochs=3`. Add `--cfg job --resolve` to print the composed config without running.

## Documentation

- [Installation](docs/install.md): environment, CUDA wheels, troubleshooting.
- [Datasets](docs/datasets.md): how to download and prepare each dataset and the derived files.
- [Configuration](docs/configuration.md): the Hydra layout, `data.train` / `data.val`, dataset mixes,
  QA, loaders, common overrides.
- [Models](docs/models.md): TextTrajVLM, FM-VLA, auxiliary heads, checkpoints.
- [Training](docs/training.md): SFT, multiple GPUs, resume, GRPO and Flow-GRPO.
- [Evaluation and testing](docs/evaluation.md): metrics, validation sets, Waymo submissions.
- [Adding a dataset](docs/adding_a_dataset.md): the sample format and the dataset interface.

## Hardware

The examples were run on four 80 GB A100 GPUs, in bf16 with FlashAttention 2 (it needs an Ampere or
newer GPU). With three image frames, keep the batch at 2 per GPU and use gradient accumulation.
Evaluation fits on one GPU.

## Repository layout

```text
ad_vla/
  conf/          Hydra configs: datasets, dataset mixes, models, prompts, training stages, experiments
  dataset/       Dataset adapters, the common sample format, the data builder
  models/        TextTrajVLM, FM-VLA, action experts, auxiliary heads
  train_stages/  Lightning modules: supervised training, GRPO, resume
  rewards/       GRPO rewards (RFS, L2, PDM-style simulator score, collisions)
  metrics/       ADE, FDE, Waymo rater feedback score (RFS)
  utils/         Prompts, trajectories, intents, image transforms, camera geometry, plots
scripts/         run_training.py, run_evaluation.py, run_testing.py, data preparation tools
tests/           Unit and integration tests
resources/       Fixed PhysicalAI-AV validation clip list
```

## License

Apache 2.0, see [LICENSE](LICENSE). Datasets and pretrained models keep their own licenses: check
the terms of the Waymo Open Dataset, NAVSIM/OpenScene, nuScenes, PhysicalAI-AV, KITScenes, NuPlanQA
and Qwen before use, and don't redistribute their data.

## Acknowledgements

Built on [Qwen3.5](https://github.com/QwenLM),
[Transformers](https://github.com/huggingface/transformers),
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning),
[Hydra](https://github.com/facebookresearch/hydra), [NAVSIM](https://github.com/autonomousvision/navsim),
the [Waymo Open Dataset](https://waymo.com/open/), [nuScenes](https://www.nuscenes.org/),
[PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
[Alpamayo](https://github.com/NVlabs/alpamayo), [SimpleBEV](https://github.com/aharley/simple_bev),
[VGGT](https://github.com/facebookresearch/vggt) and [NuPlanQA](https://github.com/sungyeonparkk/NuPlanQA).

## Citation

If this code helps your research, please cite:

```bibtex
@inproceedings{azevedo2027zeroshot,
  title         = {Towards Zero-Shot Transfer Across Embodiments For Driving {VLAs}},
  author        = {Azevedo, Caio and Sabatini, Stefano and Hornauer, Sascha and Moutarde, Fabien},
  booktitle     = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year          = {2027},
  eprint        = {2609.02341},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
