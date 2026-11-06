"""Compute per-(waypoint, channel) normalization stats for FlowMatchingVLA diffusion targets.

Uses the same experiment config as training, so the dataset, prediction grid and action
representation match. Prints a YAML block to paste under ``model.model_args.action_stats``.

    python scripts/utils/compute_action_stats.py \
        +experiment=scratch/fmvla/<experiment> [--num_samples 20000] [other hydra overrides]
"""

import argparse
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dotenv import load_dotenv

load_dotenv()

import torch
import yaml
from hydra import compose, initialize_config_dir
from ad_vla.dataset.builder import build_train_dataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ad_vla.dataset.data_types import TrajectorySampling
from ad_vla.models.vlas.fm_vla import initial_speed
from ad_vla.utils.action_utils import POLAR_MIN_SPEED, decode_actions, encode_actions
from ad_vla.utils.traj_utils import resample_tensor

CONF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "ad_vla", "conf"
)


def main():
    """Compute the action statistics on a random subset of the training data and print them."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "overrides", nargs="+", help="Hydra overrides, e.g. +experiment=..."
    )
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(CONF_DIR)):
        cfg = compose("default_training", overrides=args.overrides)

    model_args = cfg.model.model_args
    representation = model_args.get("action_representation", "trajs")
    sampling = TrajectorySampling(**model_args.pred_traj_sampling)

    dataset = build_train_dataset(cfg.data)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[
        : args.num_samples
    ].tolist()

    def collate(samples):
        """Futures resampled to the model's prediction grid, and the initial speeds."""
        trajs = torch.stack(
            [
                resample_tensor(s.fut_traj, s.fut_traj_sampling, sampling)
                for s in samples
            ]
        )
        speeds = torch.stack([initial_speed(s) for s in samples])
        return trajs, speeds

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=32,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    dt = sampling.interval_length
    actions, roundtrip_errors = [], []
    for trajs, speeds in tqdm(loader, desc="Collecting targets"):
        trajs = trajs.double()
        batch_actions = encode_actions(
            trajs, representation, dt=dt, initial_velocity=speeds
        )
        decoded = decode_actions(
            batch_actions, representation, dt=dt, initial_velocity=speeds
        )
        roundtrip_errors.append((decoded - trajs).norm(dim=-1))
        actions.append(batch_actions)
    actions = torch.cat(actions)  # [N, num_poses, 2]
    roundtrip_errors = torch.cat(roundtrip_errors)  # [N, num_poses]

    mean = actions.mean(dim=0)
    std = actions.std(dim=0).clamp_min(1e-4)
    stats = {
        "action_representation": representation,
        "pred_traj_sampling": {
            "num_poses": sampling.num_poses,
            "interval_length": sampling.interval_length,
        },
        "num_samples": len(actions),
        "mean": [[round(v, 5) for v in row] for row in mean.tolist()],
        "std": [[round(v, 5) for v in row] for row in std.tolist()],
    }

    normalized = (actions - mean) / std
    print(
        f"\n{len(actions)} samples, {representation} on {sampling.num_poses} x "
        f"{sampling.interval_length}s. |z| 99th/max: "
        f"{normalized.abs().quantile(0.99).item():.2f}/{normalized.abs().max().item():.1f}"
    )
    print(
        f"Round trip decode(encode(gt)) vs gt: ADE {roundtrip_errors.mean().item():.4f} m, "
        f"max {roundtrip_errors.max().item():.3f} m"
    )
    if representation == "polar":
        slow = (
            actions[..., 1].abs() < POLAR_MIN_SPEED * dt
        )  # heading held (see traj_to_polar)
        backward = (actions[..., 1] < 0) & ~slow
        print(
            f"Slow steps (heading held): {slow.float().mean().item():.1%} of steps; "
            f"fully stationary samples: {slow.all(dim=-1).float().mean().item():.1%}; "
            f"samples with a backward step: {backward.any(dim=-1).float().mean().item():.2%}"
        )
    print("\nPaste under model.model_args:\n")
    print(
        yaml.safe_dump(
            {"action_stats": stats}, default_flow_style=None, sort_keys=False
        )
    )


if __name__ == "__main__":
    main()
