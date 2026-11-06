"""Rectified-flow (flow-matching) training targets, Euler sampling and Flow-GRPO SDE sampling.

Convention: time t runs from 0 (pure noise) to 1 (data). A training point is
``x_t = t * x + (1 - t) * noise`` and the network regresses the constant velocity
``dx_t/dt = x - noise``. Sampling integrates that velocity field with Euler steps from
``t = 0`` to ``t = 1``.

For RL (Flow-GRPO, Liu et al. 2025, arXiv 2505.05470) the deterministic ODE is replaced by
the SDE with the same marginals,

    dx = [v + sigma_t^2 / (2 (1 - t)) * (t v - x)] dt + sigma_t dw,
    sigma_t = a * sqrt((1 - t) / t),

which is the paper's equation in this file's time convention (paper t = 1 - t here, paper
v = -v here; the bracket uses the score -(x - t v) / (1 - t) of x_t). Each Euler-Maruyama step
is a Gaussian ``N(mean, std^2)`` with ``std = sigma_t * sqrt(dt)``, so a sampled denoising
chain has an exact per-step log-probability (``sde_step_log_prob``). As in the reference
code, sigma at t = 0 (infinite) is taken at the next grid point.
"""

import math

import torch
import torch.nn.functional as F

from ad_vla.models.diffusion.base_diffusion import BaseDiffusion, StepFn


class FlowMatching(BaseDiffusion):
    """Flow-matching objective and deterministic Euler sampler.

    Args:
        input_dims: Shape of one sample without the batch dimension, e.g.
            ``(num_poses, 2)`` for a trajectory.
        num_inference_steps: Default number of Euler steps at sampling time.
        num_repeat_steps: Training copies of each target, each with its own noise and time
            (more loss terms per backbone pass, not more optimizer steps).
    """

    def __init__(
        self,
        input_dims: tuple[int, ...],
        num_inference_steps: int,
        num_repeat_steps: int = 1,
        **kwargs,
    ):
        super().__init__(input_dims, **kwargs)
        self.num_inference_steps = num_inference_steps
        self.num_repeat_steps = num_repeat_steps

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device,
        num_inference_steps: int | None = None,
        return_all_steps: bool = False,
        init: str = "random",
        generator: torch.Generator | None = None,
        **kwargs,
    ):
        """Integrate ``step_fn`` from ``t = 0`` to ``t = 1`` with Euler steps.

        Args:
            batch_size: Number of samples.
            step_fn: Velocity model, see ``StepFn``.
            device: Device of the samples.
            num_inference_steps: Euler steps (None: the constructor's default).
            return_all_steps: Also return every intermediate state and the time grid.
            init: "random" starts from N(0, I) noise; "zero" starts from the noise mean,
                giving one deterministic trajectory per condition.
            generator: Optional RNG for the initial noise (reproducible evaluation).

        Returns:
            Samples ``[batch, *input_dims]``, or with ``return_all_steps`` the states
            ``[batch, steps + 1, *input_dims]`` and the time grid ``[steps + 1]``.
        """
        num_steps = num_inference_steps or self.num_inference_steps
        x0 = self.initial_noise(batch_size, device, init=init, generator=generator)

        # assuming euler sampling for now
        return self._euler(
            x=x0,
            step_fn=step_fn,
            num_inference_steps=num_steps,
            return_all_steps=return_all_steps,
        )

    def initial_noise(
        self,
        batch_size: int,
        device: torch.device,
        init: str = "random",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Starting point of sampling (see `sample`); shared with the CUDA-graph sampler."""
        if init == "random":
            return torch.randn(
                batch_size, *self.input_dims, device=device, generator=generator
            )
        if init == "zero":
            return torch.zeros(batch_size, *self.input_dims, device=device)
        raise ValueError(
            f"Unknown flow-matching init {init!r}; use 'random' or 'zero'."
        )

    def _euler(
        self,
        x: torch.Tensor,
        step_fn: StepFn,
        num_inference_steps: int,
        return_all_steps: bool = False,
    ):
        """Explicit Euler integration on a uniform grid of ``num_inference_steps`` steps."""
        batch_size, device = x.shape[0], x.device
        n_dim = len(self.input_dims)
        timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)

        if return_all_steps:
            all_steps = [x]

        for i in range(num_inference_steps):
            dt = timesteps[i + 1] - timesteps[i]
            dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            t_start = (
                timesteps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            )

            v = step_fn(x=x, t=t_start)

            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)

        if return_all_steps:
            return torch.stack(all_steps, dim=1), timesteps

        return x

    @staticmethod
    def sde_schedule(
        num_steps: int, noise_level: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform time grid ``[num_steps + 1]`` and the SDE noise std of each step ``[num_steps]``.

        The std of step k is ``sigma(t_k) * sqrt(dt)`` with ``sigma(t) = a sqrt((1 - t) / t)``;
        the first step (t = 0, sigma infinite) uses ``t_1`` instead, as Flow-GRPO does.
        """
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        t = timesteps[:-1].clone()
        t[0] = timesteps[1]
        dt = timesteps[1:] - timesteps[:-1]
        sigma = noise_level * torch.sqrt((1.0 - t) / t)
        return timesteps, sigma * torch.sqrt(dt)

    @staticmethod
    def sde_step_mean(
        x: torch.Tensor, v: torch.Tensor, t: float, dt: float, std: float
    ) -> torch.Tensor:
        """Mean of one Euler-Maruyama step of the marginal-preserving SDE from time ``t``.

        ``std = sigma_t * sqrt(dt)``, so ``sigma_t^2 dt = std^2``. With ``std = 0`` this is
        the Euler step ``x + dt * v``.
        """
        return x + dt * v + std**2 / (2.0 * (1.0 - t)) * (t * v - x)

    @staticmethod
    def sde_step_log_prob(
        x_next: torch.Tensor, mean: torch.Tensor, std: float
    ) -> torch.Tensor:
        """Gaussian log-density of ``x_next`` under ``N(mean, std^2 I)``, averaged over all
        non-batch dimensions (Flow-GRPO's reduction; keeps ratios comparable across sizes)."""
        log_prob = (
            -((x_next - mean) ** 2) / (2.0 * std**2)
            - math.log(std)
            - 0.5 * math.log(2.0 * math.pi)
        )
        return log_prob.flatten(1).mean(dim=1)

    @torch.no_grad()
    def sample_sde(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device,
        num_steps: int,
        noise_level: float,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample with the marginal-preserving SDE from N(0, I) noise (training rollouts).

        Returns ``chain`` ``[batch, num_steps + 1, *input_dims]`` (``chain[:, -1]`` is the
        sample), ``timesteps`` ``[num_steps + 1]``, ``stds`` ``[num_steps]`` and the per-step
        ``log_probs`` ``[batch, num_steps]`` of the sampled transitions.
        """
        timesteps, stds = self.sde_schedule(num_steps, noise_level, device)
        x = self.initial_noise(batch_size, device, init="random", generator=generator)
        n_dim = len(self.input_dims)
        chain, log_probs = [x], []
        for k in range(num_steps):
            t, dt, std = (
                float(timesteps[k]),
                float(timesteps[k + 1] - timesteps[k]),
                float(stds[k]),
            )
            t_in = timesteps[k].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            v = step_fn(x=x, t=t_in)
            mean = self.sde_step_mean(x, v, t, dt, std)
            noise = torch.randn(
                mean.shape, device=device, dtype=mean.dtype, generator=generator
            )
            x = mean + std * noise
            log_probs.append(self.sde_step_log_prob(x, mean, std))
            chain.append(x)
        return {
            "chain": torch.stack(chain, dim=1),
            "timesteps": timesteps,
            "stds": stds,
            "log_probs": torch.stack(log_probs, dim=1),
        }

    def chain_step_means(
        self,
        chain: torch.Tensor,
        timesteps: torch.Tensor,
        stds: torch.Tensor,
        step_fn: StepFn,
        steps: list[int] | None = None,
    ) -> torch.Tensor:
        """Recompute the SDE step means for a stored chain with the current ``step_fn``.

        One ``step_fn`` call per requested step (with gradients if enabled); the conditioning
        captured by ``step_fn`` is shared by all calls, so autograd keeps a single copy of it.
        Returns ``[batch, len(steps), *input_dims]``: the mean of the transition
        ``chain[:, k] -> chain[:, k + 1]`` for each ``k`` in ``steps`` (default: all).
        """
        batch_size, n_dim = chain.shape[0], len(self.input_dims)
        steps = list(range(len(stds))) if steps is None else list(steps)
        means = []
        for k in steps:
            x = chain[:, k]
            t_in = timesteps[k].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)
            v = step_fn(x=x, t=t_in)
            means.append(
                self.sde_step_mean(
                    x,
                    v,
                    float(timesteps[k]),
                    float(timesteps[k + 1] - timesteps[k]),
                    float(stds[k]),
                )
            )
        return torch.stack(means, dim=1)

    def construct_training_data(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Noisy training inputs for targets ``x`` ``[batch, *input_dims]``.

        Each target is repeated ``num_repeat_steps`` times with independent ``t ~ U(0, 1)``
        and Gaussian noise. Returns ``x`` (repeated targets), ``noisy_x`` (the model input
        ``t * x + (1 - t) * noise``), ``timesteps`` (``t``, broadcastable to ``x``) and
        ``noise``.
        """
        batch_size = x.shape[0]
        x = x.repeat_interleave(self.num_repeat_steps, dim=0)

        # assuming uniform train timestep sampler for now
        t = torch.rand((batch_size * self.num_repeat_steps,), device=x.device)

        while len(t.shape) < len(x.shape):
            t = t[..., None]

        noise = torch.randn_like(x)
        noisy_x = t * x + (1 - t) * noise
        return {
            "x": x,
            "noisy_x": noisy_x,
            "timesteps": t,
            "noise": noise,
        }

    def compute_loss_from_pred(
        self, training_data: dict[str, torch.Tensor], pred: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error between the predicted velocity and ``x - noise``."""
        x = training_data["x"]
        noise = training_data["noise"]
        target = (x - noise).to(dtype=pred.dtype)
        return F.mse_loss(target, pred)
