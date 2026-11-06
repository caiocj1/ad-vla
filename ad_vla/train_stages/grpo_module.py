import torch
import copy
import math
from collections import defaultdict, deque
from hydra.utils import instantiate
from omegaconf import DictConfig

from ad_vla.models.base_traj_planner import BaseTrajPlanner
from ad_vla.dataset.data_types import E2EDataBatch, TrajectorySampling
from ad_vla.train_stages.base_module import BaseTrainingModule


class GRPOModule(BaseTrainingModule):
    """GRPO on a trajectory planner: G rollouts per sample, group-normalized rewards as advantages,
    policy-gradient loss plus KL to a frozen reference.

    Two policy paths:

    * text-trajectory models (TTVLM): rollouts are re-scored token by token by the backbone
      (``_token_policy_terms``), reference = a full frozen copy of the model;
    * models with ``grpo_policy_terms`` (FM-VLA, ``generate_cfg.sampler: sde``): the model returns
      per-denoising-step log-probs and KL itself (Flow-GRPO), reference = ``grpo_reference()``
      (FM-VLA: a frozen copy of the expert; the insulated backbone is shared).

    ``probe_cfg`` (optional): fixed held-out states scored with the deterministic deployment plans
    (``val_generate_cfg``) at the start, every ``every_n_batches`` batches and at each epoch end,
    logged as ``probe/<term>`` and ``probe_<group>/<term>``. Keys: ``dataset`` (a dataset config
    whose samples carry ``metadata["rl_ctx"]``, e.g. ``AlpaSimStateDataset``), ``batch_size``,
    ``num_workers``, ``every_n_batches`` and ``extra_reward_fns`` ({name: reward config}, logged as
    ``probe/reward_<name>``). ``viz_cfg`` (optional): bird's-eye images of training groups and probe
    states (``ad_vla/utils/grpo_viz.py``). Both need a reward with ``score(plans, ctx) -> (reward,
    terms)`` and ``target_sampling`` (``SimPDMReward``); the probe also reads its ``dac`` term. This is
    checked at construction.
    """

    def __init__(
        self,
        model: BaseTrajPlanner,
        optim_cfg: DictConfig,
        rl_cfg: DictConfig,
        num_traj_samples_per_gpu: int,
        generate_cfg: DictConfig | None = None,
        lora_cfg: DictConfig | None = None,
        loss_cfg: DictConfig | None = None,
        aux_head: DictConfig | None = None,
        aux_loss_weight: float = 1.0,
        aux_warmup_steps: int | None = None,
        val_log_every_n_steps: int = 2,
        val_target_sampling: DictConfig | dict | TrajectorySampling | None = None,
        val_generate_cfg: DictConfig | dict | None = None,
        probe_cfg: DictConfig | dict | None = None,
        viz_cfg: DictConfig | dict | None = None,
    ):
        super().__init__(
            model=model,
            optim_cfg=optim_cfg,
            lora_cfg=lora_cfg,
            loss_cfg=loss_cfg,
            aux_head=aux_head,
            aux_loss_weight=aux_loss_weight,
            aux_warmup_steps=aux_warmup_steps,
            val_log_every_n_steps=val_log_every_n_steps,
            val_target_sampling=val_target_sampling,
            val_generate_cfg=val_generate_cfg,
        )

        # Models that score their rollouts themselves (FM-VLA: denoising steps) also say what
        # the reference is (FM-VLA: a copy of the expert only; the backbone is frozen and shared).
        self.policy_hook = hasattr(self.model, "grpo_policy_terms")
        if self.policy_hook:
            self.ref_model = self.model.grpo_reference()
        else:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.generate_cfg = generate_cfg
        self.rollouts_per_gpu = num_traj_samples_per_gpu

        self.kl_beta = rl_cfg.kl_beta
        self.reward_fn = instantiate(rl_cfg.reward_fn)
        self.reward_scale = rl_cfg.reward_scale
        self.normalize_by_std = rl_cfg.normalize_by_std
        # PPO clip range of the per-step ratio; None keeps the plain policy-gradient surrogate
        # exp(logp - logp_old) * A. There is one optimizer step per rollout batch and logp_old comes
        # from the same weights (token path: detached; FM-VLA: at sampling time), so the ratio is ~1
        # and the clip practically never acts; it would with several updates per rollout batch.
        self.clip_range = rl_cfg.get("clip_range")
        self.adv_clip_max = rl_cfg.get("adv_clip_max")
        # Floor on the group reward std in the advantage normalization (None: 1e-6 is added, as before). With the
        # tiered reward (FlowGRPO-Mix-v10, not pursued), groups whose samples all fail at nearly the same time
        # would otherwise blow up tiny reward differences into unit advantages.
        self.adv_std_floor = rl_cfg.get("adv_std_floor")
        # Fraction of the denoising steps that get a loss term (policy-hook models only).
        self.step_fraction = rl_cfg.get("step_fraction", 1.0)

        self.window_size = rl_cfg.window_size
        # Sliding-window means (same window as train/avg_reward) of the per-batch diagnostics,
        # logged as train/avg_<name>: one batch holds only a few states, so the raw per-step
        # values are noisy. Not part of the checkpoint.
        self._diag_windows: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.register_buffer("training_reward_buffer", torch.zeros(self.window_size))
        self.register_buffer("sliding_idx", torch.zeros(1, dtype=torch.long))
        self.register_buffer("window_count", torch.zeros(1, dtype=torch.long))

        # Probe: fixed held-out states (e.g. risky NavTest chunk-0 moments) scored with the
        # deterministic deployment plans (val_generate_cfg) and the reward's terms, at the start,
        # every probe_cfg.every_n_batches training batches and at the end of each epoch. Unlike
        # train/*, the states never change, so a change is a change of the policy.
        self.probe_cfg = probe_cfg
        # probe_cfg.extra_reward_fns {name: reward config}: the deployment plans are also scored with these rewards,
        # logged as probe/reward_<name> (e.g. the previous reward, to compare probe curves across reward changes).
        extra = (probe_cfg or {}).get("extra_reward_fns") or {}
        self.probe_extra_rewards = {
            name: instantiate(cfg) for name, cfg in extra.items()
        }
        self._probe_loader = None
        self._probe_step = None

        # Images (ad_vla/utils/grpo_viz.py), logged to the loggers that take images (Comet) and saved under
        # viz_cfg.save_dir: every viz_cfg.every_n_batches batches, bird's-eye views of the first
        # viz_cfg.num_train_states groups of the batch (the G samples GRPO compares, coloured by reward); at every
        # probe pass, viz_cfg.num_probe_states fixed probe states with G samples of the training sampler and the
        # deployment plan (the same states each time, so the images show the policy changing). Failures only warn.
        self.viz_cfg = viz_cfg
        if probe_cfg is not None or viz_cfg is not None:
            self._check_state_reward(probe_cfg, viz_cfg)
        self._viz_probe_idx = None

    def _check_state_reward(self, probe_cfg, viz_cfg) -> None:
        """The probe and the images score single plans against a state's reward context: fail at
        construction, not at the first probe pass, if the reward cannot do that."""
        users = [
            name
            for name, cfg in (("probe_cfg", probe_cfg), ("viz_cfg", viz_cfg))
            if cfg is not None
        ]
        rewards = {"reward_fn": self.reward_fn}
        rewards.update(
            {
                f"probe_cfg.extra_reward_fns.{k}": v
                for k, v in self.probe_extra_rewards.items()
            }
        )
        for name, fn in rewards.items():
            if (
                not callable(getattr(fn, "score", None))
                or getattr(fn, "target_sampling", None) is None
            ):
                raise ValueError(
                    f"{' and '.join(users)} need a reward with score(plans, ctx) and target_sampling "
                    f"(e.g. SimPDMReward); {name} is {type(fn).__name__}. Set them to null for this reward."
                )

    def _rollout_vlm_inputs(self, extra: dict) -> dict:
        """Backbone inputs for re-scoring the generated rollouts (policy and reference).

        Qwen3.5 in transformers >= 5.18 needs mm_token_type_ids next to image inputs to build
        its multimodal RoPE; the rollouts only carry token ids, so it is rebuilt here the way
        the processor builds it (image placeholder tokens = 1, everything else = 0).
        """
        input_ids = extra["gen_input_ids"]
        img_feat_dim = extra["gen_pixel_values"].shape[-1]
        image_token_id = self.model.vlm.config.image_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": extra["gen_attention_mask"],
            "pixel_values": extra["gen_pixel_values"][None]
            .repeat(self.rollouts_per_gpu, 1, 1)
            .reshape(-1, img_feat_dim),
            "image_grid_thw": extra["gen_image_grid_thw"].repeat_interleave(
                self.rollouts_per_gpu, dim=0
            ),
            "mm_token_type_ids": (input_ids == image_token_id).to(torch.int),
        }

    def training_step(self, batch: E2EDataBatch, batch_idx: int):
        if self.train_aux_head_only():
            raise Exception(
                "Attempting GRPO while train_aux_head_only is True, which is not intended."
            )
        batch_size = len(batch.samples)
        model_inputs = batch.model_inputs

        # Rewards that need a particular trajectory grid declare it; otherwise the ground
        # truth's horizon at 10 Hz.
        target_sampling = getattr(self.reward_fn, "target_sampling", None)
        if target_sampling is None:
            gt_traj_sampling = batch.samples[0].fut_traj_sampling
            target_sampling = TrajectorySampling(
                time_horizon=gt_traj_sampling.time_horizon, interval_length=0.1
            )
        num_timesteps = target_sampling.num_poses

        # 1. Rollout group of predicted trajectories
        # Assumes predict_from_processed_inputs can sample multiple trajs
        self.model.eval()
        with torch.no_grad():
            pred_trajs, extra = self.model.predict_from_processed_inputs(
                **model_inputs,
                num_traj_samples=self.rollouts_per_gpu,
                generate_cfg=self.generate_cfg,
                target_sampling=target_sampling,
            )

            # 2. Compute reward for each trajectory
            pred_trajs = pred_trajs.reshape(
                batch_size, self.rollouts_per_gpu, num_timesteps, -1
            )
            rewards = self.reward_fn(pred_trajs, batch)
            rewards = self.reward_scale * rewards
            every = (
                self.viz_cfg.get("every_n_batches")
                if self.viz_cfg is not None
                else None
            )
            if every and batch_idx % every == 0:
                self._viz_train_groups(batch, pred_trajs, batch_idx)

            # Gather rewards across GPUs for advantage calculation
            groupped_rewards = self.all_gather(
                rewards
            )  # [num_gpu, batch_size, self.rollouts_per_gpu]
            if groupped_rewards.ndim == 2:
                groupped_rewards = groupped_rewards[None]
            groupped_rewards = groupped_rewards.permute(1, 0, 2).flatten(
                1
            )  # [batch_size, group_size]

            advantage = rewards - groupped_rewards.mean(dim=-1, keepdim=True)
            if self.normalize_by_std:
                group_std = groupped_rewards.std(dim=-1, keepdim=True)
                if self.adv_std_floor is not None:
                    advantage = advantage / group_std.clamp_min(self.adv_std_floor)
                else:
                    advantage = advantage / (group_std + 1e-6)
            if self.adv_clip_max is not None:
                advantage = advantage.clamp(-self.adv_clip_max, self.adv_clip_max)
            advantage = advantage.reshape(-1)  # [batch_size * self.rollouts_per_gpu]

        # 3-4. Log-probs of the rollouts under the current policy, and KL to the reference
        self.model.train()
        if self.policy_hook:
            terms = self.model.grpo_policy_terms(
                extra,
                reference=self.ref_model if self.kl_beta > 0 else None,
                step_fraction=self.step_fraction,
            )
        else:
            terms = self._token_policy_terms(extra)
        log_probs, mask = terms["log_probs"], terms["mask"]
        per_token_kl = terms["kl"]

        # 5. Get GRPO loss
        ratio = torch.exp(log_probs - terms["old_log_probs"])
        per_policy_loss = ratio * advantage.unsqueeze(-1)
        if self.clip_range is not None:
            clipped = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            per_policy_loss = torch.minimum(
                per_policy_loss, clipped * advantage.unsqueeze(-1)
            )
        per_kl_loss = self.kl_beta * per_token_kl
        per_token_loss = -(per_policy_loss - per_kl_loss)
        loss = ((per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)).mean()

        # Logging
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch.samples),
            sync_dist=True,
        )
        kl_div = ((per_token_kl * mask).sum(dim=1) / mask.sum(dim=1)).mean()
        self.log(
            "train/kl_div",
            kl_div,
            on_step=True,
            on_epoch=False,
            batch_size=len(batch.samples),
            sync_dist=True,
        )
        self.log(
            "train/group_reward_std",
            groupped_rewards.std(dim=-1).mean(),
            on_step=True,
            on_epoch=False,
            batch_size=len(batch.samples),
            sync_dist=False,
        )
        if self.clip_range is not None:
            clip_frac = ((ratio - 1).abs() > self.clip_range).float()
            self.log(
                "train/clip_frac",
                (clip_frac * mask).sum() / mask.sum(),
                on_step=True,
                on_epoch=False,
                batch_size=len(batch.samples),
                sync_dist=True,
            )
        diagnostics = {
            "kl_div": float(kl_div.detach()),
            "group_reward_std": float(groupped_rewards.std(dim=-1).mean()),
        }
        reward_terms = getattr(self.reward_fn, "last_terms", None)
        if reward_terms:
            for name, value in reward_terms.items():
                self.log(
                    f"train/reward_{name}",
                    value,
                    on_step=True,
                    on_epoch=False,
                    batch_size=len(batch.samples),
                    sync_dist=True,
                )
                diagnostics[f"reward_{name}"] = float(value)
        for name, value in diagnostics.items():
            window = self._diag_windows[name]
            window.append(value)
            self.log(
                f"train/avg_{name}",
                sum(window) / len(window),
                on_step=True,
                on_epoch=False,
                batch_size=len(batch.samples),
                sync_dist=False,
            )

        self.training_buffer_record(rewards.mean(), batch_size=len(batch.samples))

        if self.aux_head is not None:
            loss_aux, _ = self.aux_head.compute_aux_loss(batch)
            aux_loss_name = getattr(self.aux_head, "loss_name", "aux")

            aux_loss_weight = self.current_aux_loss_weight()
            loss += aux_loss_weight * loss_aux

            self.log(
                "train/aux_loss_weight",
                aux_loss_weight,
                on_step=True,
                on_epoch=False,
                batch_size=len(batch.samples),
                sync_dist=False,
            )
            self.log(
                f"train/loss_{aux_loss_name}",
                loss_aux,
                on_step=True,
                on_epoch=False,
                batch_size=len(batch.samples),
                sync_dist=False,
            )

        return loss

    def _token_policy_terms(self, extra: dict) -> dict[str, torch.Tensor]:
        """Per generated token: log-probs under the policy (with grad) and the k3 KL estimate
        to the reference, for text-trajectory models (rollouts re-scored by the backbone)."""
        rollout_inputs = self._rollout_vlm_inputs(extra)
        logits = self.model.vlm(
            **rollout_inputs
        ).logits  # [batch_size * group_size, seq_len, vocab_size]
        logits = logits[
            :, :-1, :
        ]  # [batch_size * group_size, seq_len-1, vocab_size] Logits correspond to next token
        input_ids = extra["gen_input_ids"][
            :, 1:
        ]  # [batch_size * group_size, seq_len-1]
        log_probs = torch.log_softmax(
            logits, dim=-1
        )  # [batch_size * group_size, seq_len-1, vocab_size]
        per_token_logps = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(
            -1
        )  # [batch_size * group_size, seq_len-1]

        prompt_length = extra["prompt_length"]
        per_token_logps = per_token_logps[
            :, prompt_length - 1 :
        ]  # [batch_size * group_size, num_gen_tokens]
        completion_mask = extra["gen_attention_mask"][
            :, prompt_length:
        ]  # [batch_size * group_size, num_gen_tokens]

        # Forward for reference model (to calculate KL divergence)
        self.ref_model.eval()
        with torch.no_grad():
            ref_logits = self.ref_model.vlm(**rollout_inputs).logits

            ref_logits = ref_logits[
                :, :-1, :
            ]  # [batch_size * group_size, seq_len-1, vocab_size] Logits correspond to next token
            ref_log_probs = torch.log_softmax(
                ref_logits, dim=-1
            )  # [batch_size * group_size, seq_len-1, vocab_size]
            ref_per_token_logps = ref_log_probs.gather(
                2, input_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size * group_size, seq_len-1]
            ref_per_token_logps = ref_per_token_logps[
                :, prompt_length - 1 :
            ]  # [batch_size * group_size, num_gen_tokens]

        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )  # [batch_size * group_size, num_gen_tokens]
        return {
            "log_probs": per_token_logps,
            "old_log_probs": per_token_logps.detach(),
            "kl": per_token_kl,
            "mask": completion_mask,
        }

    def training_buffer_record(self, step_reward: torch.Tensor, batch_size: int):
        """
        Record reward in sliding window buffer and log sliding average reward.
        Starts logging once buffer is full to get more stable reward signal.
        """
        idx = self.sliding_idx.item()
        self.training_reward_buffer[idx] = step_reward

        new_idx = (idx + 1) % self.window_size
        self.sliding_idx.fill_(new_idx)
        new_count = min(self.window_count.item() + 1, self.window_size)
        self.window_count.fill_(new_count)

        if new_count >= self.window_size:
            sliding_avg = self.training_reward_buffer.mean()
            self.log(
                "train/avg_reward",
                sliding_avg,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                batch_size=batch_size,
                prog_bar=True,
            )

    # ---------- probe ----------
    def _probe_dataloader(self):
        if self._probe_loader is None:
            self._probe_loader = torch.utils.data.DataLoader(
                instantiate(self.probe_cfg["dataset"]),
                batch_size=self.probe_cfg.get("batch_size", 8),
                num_workers=self.probe_cfg.get("num_workers", 2),
                shuffle=False,
                collate_fn=self.model.get_collator(inference_mode=True),
            )
        return self._probe_loader

    @torch.no_grad()
    def run_probe(self) -> dict[str, float]:
        """Score the probe states once and log ``probe/<term>`` (all states) and
        ``probe_<group>/<term>`` (per ``metadata["probe_group"]``) at the current global step.
        ``ep`` is left out: with one plan per state it is always 1 (so is EP inside ``reward``)."""
        if self.probe_cfg is None or self._probe_step == self.global_step:
            return {}
        self._probe_step = self.global_step
        was_training = self.model.training
        self.model.eval()
        values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        try:
            for batch in self._probe_dataloader():
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                plans, _ = self.model.predict_from_processed_inputs(
                    **batch.model_inputs,
                    num_traj_samples=1,
                    generate_cfg=self.val_generate_cfg,
                    target_sampling=self.reward_fn.target_sampling,
                )
                if plans.dim() == 4:
                    plans = plans[:, 0]
                for b, sample in enumerate(batch.samples):
                    reward, terms = self.reward_fn.score(
                        plans[b : b + 1].float(), sample.metadata["rl_ctx"]
                    )
                    row = {k: float(v) for k, v in terms.items() if k != "ep"}
                    row["reward"] = float(reward[0])
                    if "dac" not in terms:
                        raise KeyError(
                            f"The probe needs a 'dac' reward term; {type(self.reward_fn).__name__} returned {sorted(terms)}."
                        )
                    row["offroad_rate"] = float(terms["dac"] < 1.0)
                    for name, fn in self.probe_extra_rewards.items():
                        row[f"reward_{name}"] = float(
                            fn.score(
                                plans[b : b + 1].float(), sample.metadata["rl_ctx"]
                            )[0][0]
                        )
                    group = sample.metadata.get("probe_group")
                    for key in ["probe"] + ([f"probe_{group}"] if group else []):
                        for name, value in row.items():
                            values[key][name].append(value)
            self._viz_probe_states()
        finally:
            if was_training:
                self.model.train()
        # NaN terms (centreline terms of states without a centreline) are left out of the means.
        metrics = {}
        for key, by_name in values.items():
            for name, vals in by_name.items():
                finite = [v for v in vals if math.isfinite(v)]
                if finite:
                    metrics[f"{key}/{name}"] = sum(finite) / len(finite)
        for key, by_name in values.items():
            metrics[f"{key}/num_states"] = float(len(by_name["reward"]))
        if self.trainer.is_global_zero:
            for logger in self.loggers:
                logger.log_metrics(metrics, step=self.global_step)
        return metrics

    def _log_figure(self, fig, name: str) -> None:
        import os

        from ad_vla.utils.grpo_viz import save_figure

        save_dir = self.viz_cfg.get("save_dir") or os.path.join(
            self.trainer.default_root_dir, "viz"
        )
        path = save_figure(
            fig, os.path.join(save_dir, name, f"step{self.global_step:06d}.png")
        )
        for logger in self.loggers:
            experiment = getattr(logger, "experiment", None)
            if hasattr(experiment, "log_image"):  # Comet
                experiment.log_image(path, name=name, step=self.global_step)

    @torch.no_grad()
    def _viz_train_groups(
        self, batch: E2EDataBatch, pred_trajs: torch.Tensor, batch_idx: int
    ) -> None:
        if not self.trainer.is_global_zero:
            return
        try:
            from ad_vla.utils.grpo_viz import (
                front_camera,
                plan_terms,
                plot_group,
                state_title,
            )

            for b in range(
                min(int(self.viz_cfg.get("num_train_states", 1)), len(batch.samples))
            ):
                sample = batch.samples[b]
                ctx = sample.metadata["rl_ctx"]
                rewards, flags = plan_terms(self.reward_fn, pred_trajs[b].float(), ctx)
                fig = plot_group(
                    ctx,
                    pred_trajs[b],
                    rewards,
                    flags,
                    camera=front_camera(sample),
                    title=f"train batch {batch_idx}, step {self.global_step}: "
                    + state_title(sample),
                    dt=float(self.reward_fn.target_sampling.interval_length),
                )
                self._log_figure(fig, f"train_group_{b}")
        except Exception as exc:  # noqa: BLE001 - images must never stop training
            print(f"[viz] train groups skipped: {exc!r}", flush=True)

    @torch.no_grad()
    def _viz_probe_states(self) -> None:
        num = (
            int(self.viz_cfg.get("num_probe_states", 0))
            if self.viz_cfg is not None
            else 0
        )
        if not num or not self.trainer.is_global_zero:
            return
        try:
            from ad_vla.utils.grpo_viz import (
                front_camera,
                pick_probe_states,
                plan_terms,
                plot_group,
                state_title,
            )

            dataset = self._probe_dataloader().dataset
            if self._viz_probe_idx is None:
                self._viz_probe_idx = pick_probe_states(dataset, num)
            target = self.reward_fn.target_sampling
            collate = self.model.get_collator(inference_mode=True)
            chunk = int(self.viz_cfg.get("probe_chunk", 5))  # states per sampling call
            for start in range(0, len(self._viz_probe_idx), chunk):
                samples = [
                    dataset[i] for i in self._viz_probe_idx[start : start + chunk]
                ]
                batch = self.transfer_batch_to_device(collate(samples), self.device, 0)
                group, _ = self.model.predict_from_processed_inputs(
                    **batch.model_inputs,
                    num_traj_samples=self.rollouts_per_gpu,
                    generate_cfg=self.generate_cfg,
                    target_sampling=target,
                )
                group = group.reshape(
                    len(samples), self.rollouts_per_gpu, target.num_poses, -1
                )
                deploy, _ = self.model.predict_from_processed_inputs(
                    **batch.model_inputs,
                    num_traj_samples=1,
                    generate_cfg=self.val_generate_cfg,
                    target_sampling=target,
                )
                deploy = deploy.reshape(
                    len(samples), -1, target.num_poses, deploy.shape[-1]
                )[:, 0]
                for j, sample in enumerate(batch.samples):
                    k = start + j
                    ctx = sample.metadata["rl_ctx"]
                    rewards, flags = plan_terms(self.reward_fn, group[j].float(), ctx)
                    deploy_reward, _ = self.reward_fn.score(
                        deploy[j : j + 1].float(), ctx
                    )
                    fig = plot_group(
                        ctx,
                        group[j],
                        rewards,
                        flags,
                        deploy=deploy[j],
                        camera=front_camera(sample),
                        title=f"probe state {k}, step {self.global_step}: "
                        + state_title(sample),
                        dt=float(target.interval_length),
                        deploy_reward=float(deploy_reward[0]),
                    )
                    self._log_figure(fig, f"probe_state_{k}")
        except Exception as exc:  # noqa: BLE001 - images must never stop training
            print(f"[viz] probe states skipped: {exc!r}", flush=True)

    def on_train_start(self) -> None:
        self.run_probe()

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        every = (
            self.probe_cfg.get("every_n_batches")
            if self.probe_cfg is not None
            else None
        )
        if every and (batch_idx + 1) % every == 0:
            self.run_probe()

    def on_train_epoch_end(self) -> None:
        self.run_probe()
