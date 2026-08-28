from collections.abc import Callable
from typing import Any
from omegaconf import DictConfig

import torch

from ad_vla.dataset.data_types import E2EDataBatch, E2EDataSample, TrajectorySampling
from ad_vla.utils.traj_utils import resample_tensor


class BaseTrajPlanner:
    """
    Abstract base class for trajectory planning models. Do not instantiate it directly.

    Each model declares its expected input and native output temporal specifications
    via TrajectorySampling objects passed at construction time.

    The predict_trajectories method returns a batched tensor [B, T, 2] at the
    model's native pred_traj_sampling grid. An optional target_sampling parameter
    triggers resampling via the _maybe_resample helper.
    """

    def __init__(
        self,
        past_traj_sampling,
        pred_traj_sampling,
        **kwargs,
    ):
        # Accept both dict and TrajectorySampling for flexibility
        # (dicts come from YAML/JSON configs, objects from direct construction)
        if isinstance(past_traj_sampling, dict):
            past_traj_sampling = TrajectorySampling(**past_traj_sampling)
        if isinstance(pred_traj_sampling, dict):
            pred_traj_sampling = TrajectorySampling(**pred_traj_sampling)
        self.past_traj_sampling = past_traj_sampling
        self.pred_traj_sampling = pred_traj_sampling

    def predict_trajectories(
        self,
        sample: E2EDataSample,
        num_traj_samples: int = 1,
        generate_cfg: DictConfig | None = None,
        target_sampling: TrajectorySampling | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Predict for a single sample, automatically processing with model's collator.

        Args:
            sample (E2EDataSample): Traffic scene which will be input to the model
                Get's processed under the hood by the model's specific collator
            num_traj_samples: Number of trajectories to sample from the model.
            generate_cfg: Optional dict with generation parameters which are model dependent.
            target_sampling (TrajectorySampling): Arbitrary sampling frequency and number of
                poses for output trajectories

        Returns:
            torch.Tensor: Predicted trajectories [num_traj_samples, num_timesteps, 2] at pred_traj_sampling
                (or target_sampling if provided).
            Dict: Extra information (e.g. reasoning trace, success_preds).
        """
        collator = self.get_collator()
        model_inputs = collator([sample]).model_inputs
        return self.predict_from_processed_inputs(
            **model_inputs,
            num_traj_samples=num_traj_samples,
            generate_cfg=generate_cfg,
            target_sampling=target_sampling,
        )

    def predict_from_processed_inputs(
        self,
        num_traj_samples: int = 1,
        generate_cfg: DictConfig | None = None,
        target_sampling: TrajectorySampling | None = None,
        **model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Predict trajectories from model inputs processed by model's collator.

        Args:
            model_inputs: Dictionary with arbitrary arguments depending on the model type.
                The model provides its own data collator with model.get_collator(),
                so one can easily do inference.
                      - MLP might need: 'past_trajs'
                      - VLM might need: 'input_ids', 'pixel_values', 'attention_mask', etc
                These required keys are all present in model_inputs due to collator.
            num_traj_samples: Number of trajectories to sample from the model.
            generate_cfg: Optional dict with generation parameters which are model dependent.
            target_sampling (Optional[TrajectorySampling]): If provided, resamples
                output from the model's native pred_traj_sampling to this grid.

        Returns:
            torch.Tensor: Predicted trajectories [batch_size * num_traj_samples, num_timesteps, 2]
                at pred_traj_sampling (or target_sampling if provided).
            Dict: Extra information (e.g. reasoning trace, success_preds).
        """
        raise NotImplementedError

    def _maybe_resample(
        self,
        pred_trajs: torch.Tensor,
        target_sampling: TrajectorySampling | None = None,
    ) -> torch.Tensor:
        """
        Helper: resample predictions if target_sampling differs from native.

        Models should call this at the end of predict_trajectories to handle
        optional resampling without duplicating logic.

        Args:
            pred_trajs: Batched predictions [B, T, 2] at model's native resolution.
            target_sampling: Optional target grid. If None or same as native, returns as-is.

        Returns:
            torch.Tensor: Trajectories shaped [B, T_target, 2], resampled if needed.
        """
        if target_sampling is not None and target_sampling != self.pred_traj_sampling:
            return resample_tensor(pred_trajs, self.pred_traj_sampling, target_sampling)
        return pred_trajs

    def get_collator(self) -> Callable:
        """
        Model provides its own collator -> simplifies code and input processing

        To do inference on a single specific E2EDataSample:
        ```
        collator = model.get_collator()
        inputs = collator([sample])
        pred_trajs, extra = model.predict_trajectories(**inputs)
        ```
        """
        raise NotImplementedError

    def compute_loss(self, batch: E2EDataBatch) -> torch.Tensor:
        """
        Takes a batch end calculates the loss for training.
        Specific to each model architecture. Used in the SFTModule.
        Expects input batch to come from model's collator, and therefore to have
        correct labels and required inputs.

        Args:
            batch: Full batch of E2E data.

        Returns:
            torch.Tensor: Loss for training.
        """
        raise NotImplementedError
