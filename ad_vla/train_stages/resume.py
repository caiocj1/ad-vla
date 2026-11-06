"""What continuing a training run needs beyond Lightning's own ``trainer.fit(ckpt_path=...)``.

Lightning restores the weights, optimizer, LR schedule and step counters, but restarts the data from
the top of the shuffled order, and on the sampler's first epoch's order at that. So the samples not
yet reached in that epoch would be skipped, with no error. ``ResumableSampler`` continues instead.
"""

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins.io import TorchCheckpointIO
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import Dataset, DistributedSampler


class FullCheckpointIO(TorchCheckpointIO):
    """Load our own full checkpoints: ``torch.load`` now refuses their non-tensor state by default,
    and Lightning 2.2 doesn't say otherwise. Only for files this project's runs wrote."""

    def load_checkpoint(self, path, map_location=None) -> dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)


class ResumableSampler(DistributedSampler):
    """The order Lightning's own ``DistributedSampler`` gives, that can start partway through.

    Lightning leaves a ``DistributedSampler`` in place, and validation loaders still get its own.
    The ranks are only known once the trainer is set up, see ``ResumeDataPosition``.
    """

    def __init__(self, dataset: Dataset, seed: int = 0) -> None:
        super().__init__(dataset, num_replicas=1, rank=0, shuffle=True, seed=seed)
        self.skip = 0

    def set_ranks(self, num_replicas: int, rank: int) -> None:
        DistributedSampler.__init__(
            self,
            self.dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=self.seed,
        )

    def resume_at(self, epoch: int, samples_done: int) -> None:
        """Set the epoch's order and leave out what this rank already went through."""
        self.set_epoch(epoch)
        self.skip = samples_done

    def __iter__(self):
        indices = list(super().__iter__())[self.skip :]
        self.skip = 0  # only the epoch that was interrupted
        return iter(indices)


class ResumeDataPosition(pl.Callback):
    """Tell the sampler where a checkpoint stopped, and refuse a run that doesn't match it."""

    def __init__(self, sampler: ResumableSampler, batch_size: int) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.saved_total_steps = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage) -> None:
        self.sampler.set_ranks(trainer.world_size, trainer.global_rank)

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        loop = checkpoint["loops"]["fit_loop"]
        # `processed`: a checkpoint saved at the end of a batch hasn't marked it `completed` yet.
        samples_done = (
            loop["epoch_loop.batch_progress"]["current"]["processed"] * self.batch_size
        )
        # Saved on the epoch's last batch: Lightning goes on to the next epoch by itself, and an
        # empty first iterator would end the run instead.
        if samples_done + self.batch_size > len(self.sampler):
            samples_done = 0
        # Lightning repeats the validation the checkpoint ended on, but continues from the last
        # loader's batch count, so it would stop after one batch and log a wrong metric.
        val_progress = loop["epoch_loop.val_loop.batch_progress"]["current"]
        val_progress.update(dict.fromkeys(val_progress, 0))
        epoch = loop["epoch_progress"]["current"]["processed"]
        self.sampler.resume_at(epoch, samples_done)
        rank_zero_info(
            f"Resuming at epoch {epoch}, step {checkpoint['global_step']}: "
            f"skipping {samples_done} samples per GPU."
        )
        schedulers = checkpoint.get("lr_schedulers") or [{}]
        self.saved_total_steps = schedulers[0].get("T_max")

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        total_steps = trainer.estimated_stepping_batches
        if self.saved_total_steps not in (None, total_steps):
            raise ValueError(
                f"The checkpoint was training for {self.saved_total_steps} steps, this run for "
                f"{total_steps}: resume with the same data, batch size, GPUs and epochs."
            )


def resume_checkpoint_path(resume: str | None, run_dir: Path) -> str | None:
    """``resume_ckpt_path`` as a file: ``last`` is the newest checkpoint of this run."""
    if not resume:
        return None
    if resume != "last":
        if not Path(resume).is_file():
            raise FileNotFoundError(f"resume_ckpt_path {resume} doesn't exist.")
        return str(resume)
    candidates = [
        path
        for path in (run_dir / "checkpoints").glob("*.ckpt")
        if not path.name.endswith("-final.ckpt")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint to resume in {run_dir / 'checkpoints'}."
        )
    return str(max(candidates, key=lambda path: path.stat().st_mtime))
