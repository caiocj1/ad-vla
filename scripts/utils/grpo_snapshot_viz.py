"""Probe-state images of a Flow-GRPO checkpoint, as GRPOModule's viz_cfg draws them during training.

For runs started before the in-training images (ad_vla/utils/grpo_viz.py): loads the starting hf_ckpt, swaps in the
model weights of a Lightning checkpoint (``model.*``; none: the starting policy, step 0), draws the same fixed probe
states with G samples of the training sampler and the deployment plan, saves the PNGs and, with --comet-key, logs
them to that (running) Comet experiment as ``probe_state_<k>`` at the checkpoint's global step.
Run on a GPU machine.
"""

from __future__ import annotations

import argparse
import os
import re

import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from ad_vla.dataset.alpasim.alpasim_state_dataset import AlpaSimStateDataset
from ad_vla.models.vlas.fm_vla import FlowMatchingVLA
from ad_vla.rewards.sim_pdm_reward import SimPDMReward
from ad_vla.utils.grpo_viz import (
    front_camera,
    pick_probe_states,
    plan_terms,
    plot_group,
    save_figure,
    state_title,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_STAGE_CFG = os.path.join(REPO, "ad_vla/conf/train_stage/grpo_flow.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base-ckpt", required=True, help="hf_ckpt the run started from"
    )
    parser.add_argument(
        "--lightning-ckpt",
        help="Lightning checkpoint of the run (omit: the starting policy)",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="global step to log (default: from the checkpoint name, else 0)",
    )
    parser.add_argument("--state-list", required=True, help="the run's probe list")
    parser.add_argument("--centerline-dirs", nargs="*", default=None)
    parser.add_argument(
        "--reward-set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="SimPDMReward arguments on top of grpo_flow.yaml (the run's overrides)",
    )
    parser.add_argument("--num-states", type=int, default=20)
    parser.add_argument("--chunk", type=int, default=5, help="states per sampling call")
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        required=True,
        help="PNGs go to <out-dir>/probe_state_<k>/step<step>.png",
    )
    parser.add_argument("--comet-key", help="Comet experiment key to log the images to")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="first delete the probe_state_* images already logged at this step (a re-render)",
    )
    args = parser.parse_args()

    step = args.step
    if step is None:
        match = re.search(r"step=(\d+)", os.path.basename(args.lightning_ckpt or ""))
        step = int(match.group(1)) if match else 0

    stage = OmegaConf.to_container(OmegaConf.load(TRAIN_STAGE_CFG), resolve=False)
    reward_kwargs = {
        k: v for k, v in stage["rl_cfg"]["reward_fn"].items() if k != "_target_"
    }
    for item in args.reward_set:
        key, _, value = item.partition("=")
        reward_kwargs[key] = (
            OmegaConf.create({"v": value})["v"] if value != "null" else None
        )
    reward = SimPDMReward(**reward_kwargs)
    target = reward.target_sampling

    model = (
        FlowMatchingVLA.from_pretrained(
            args.base_ckpt,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        .cuda()
        .eval()
    )
    if args.lightning_ckpt:
        state = torch.load(args.lightning_ckpt, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        weights = {
            k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")
        }
        missing, unexpected = model.load_state_dict(weights, strict=False)
        print(
            f"{args.lightning_ckpt}: {len(weights)} model tensors loaded, {len(missing)} missing, "
            f"{len(unexpected)} unexpected",
            flush=True,
        )
        if unexpected or not weights:
            raise RuntimeError(
                f"checkpoint does not match the model: unexpected {list(unexpected)[:10]}"
            )
        del state

    dataset = AlpaSimStateDataset(
        state_list=args.state_list, centerline_dirs=args.centerline_dirs
    )
    picked = pick_probe_states(dataset, args.num_states)
    collate = model.get_collator(inference_mode=True)
    all_samples, groups, deploys = [], [], []
    for start in range(0, len(picked), args.chunk):
        samples = [dataset[i] for i in picked[start : start + args.chunk]]
        batch = collate(samples)
        inputs = {
            k: v.cuda() if torch.is_tensor(v) else v
            for k, v in batch.model_inputs.items()
        }
        with torch.no_grad():
            group, _ = model.predict_from_processed_inputs(
                **inputs,
                num_traj_samples=args.group_size,
                generate_cfg=stage["generate_cfg"],
                target_sampling=target,
            )
            deploy, _ = model.predict_from_processed_inputs(
                **inputs,
                num_traj_samples=1,
                generate_cfg=stage["val_generate_cfg"],
                target_sampling=target,
            )
        groups.append(
            group.float()
            .reshape(len(samples), args.group_size, target.num_poses, -1)
            .cpu()
        )
        deploys.append(
            deploy.float()
            .reshape(len(samples), -1, target.num_poses, deploy.shape[-1])[:, 0]
            .cpu()
        )
        all_samples += batch.samples
    group, deploy = torch.cat(groups), torch.cat(deploys)

    experiment = None
    if args.comet_key:
        import comet_ml

        load_dotenv(os.path.join(REPO, ".env"))
        experiment = comet_ml.ExistingExperiment(
            api_key=os.environ["COMET_API_KEY"],
            experiment_key=args.comet_key,
            log_env_details=False,
            log_env_gpu=False,
            log_env_cpu=False,
            log_env_host=False,
            log_env_network=False,
            log_env_disk=False,
            auto_metric_logging=False,
            auto_param_logging=False,
            auto_output_logging=False,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            parse_args=False,
            display_summary_level=0,
        )
        if args.replace:
            api_experiment = comet_ml.API(
                api_key=os.environ["COMET_API_KEY"]
            ).get_experiment_by_key(args.comet_key)
            for asset in api_experiment.get_asset_list(asset_type="image"):
                if (
                    asset.get("fileName", "").startswith("probe_state_")
                    and asset.get("step") == step
                ):
                    api_experiment.delete_asset(asset["assetId"])
                    print(f"deleted {asset['fileName']} (step {step})", flush=True)
    for k, sample in enumerate(all_samples):
        ctx = sample.metadata["rl_ctx"]
        rewards, flags = plan_terms(reward, group[k], ctx)
        deploy_reward, _ = reward.score(deploy[k : k + 1], ctx)
        fig = plot_group(
            ctx,
            group[k],
            rewards,
            flags,
            deploy=deploy[k],
            camera=front_camera(sample),
            title=f"probe state {k}, step {step}: " + state_title(sample),
            dt=float(target.interval_length),
            deploy_reward=float(deploy_reward[0]),
        )
        path = save_figure(
            fig, os.path.join(args.out_dir, f"probe_state_{k}", f"step{step:06d}.png")
        )
        print(path, flush=True)
        if experiment is not None:
            experiment.log_image(path, name=f"probe_state_{k}", step=step)
    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
