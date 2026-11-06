"""Interfaces shared by the diffusion / flow-matching samplers of the continuous-action models."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Protocol


class StepFn(Protocol):
    """Model callback used by a sampler: the network evaluated at one state and time."""

    def __call__(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a vector field for the sampler's current state.

        Args:
            x: Current sample, shaped ``[batch, *input_dims]``.
            t: Current time, broadcastable to ``x`` (``[batch, 1, ...]``).

        Returns:
            torch.Tensor: For flow matching, dx/dt with the same shape as x,
                not the final denoised sample.
        """
        ...


class BaseDiffusion(ABC, nn.Module):
    """Base class of a generative sampler over fixed-shape samples (e.g. ``[num_poses, 2]``).

    Args:
        input_dims: Shape of one sample, without the batch dimension.
    """

    def __init__(self, input_dims: tuple[int, ...], **kwargs):
        super().__init__()
        self.input_dims = input_dims

    @abstractmethod
    @torch.no_grad()
    def sample(self):
        """Draw samples by integrating the model (see the subclass for arguments)."""
        raise NotImplementedError
