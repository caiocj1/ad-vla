"""An interrupted training continues where it stopped: order, optimizer, schedule and step count."""

import pickle

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ad_vla.train_stages.resume import (
    FullCheckpointIO,
    ResumableSampler,
    ResumeDataPosition,
    resume_checkpoint_path,
)

NUM_SAMPLES, BATCH_SIZE = 40, 4


class Items(Dataset):
    def __len__(self):
        return NUM_SAMPLES

    def __getitem__(self, index):
        return torch.tensor([float(index)])


class Module(pl.LightningModule):
    """Records what it trains on; can crash at one global batch."""

    def __init__(self, crash_at=None):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.seen = []
        self.validations = 0
        self.lrs = []
        self.crash_at = crash_at

    def training_step(self, batch, batch_idx):
        if len(self.seen) == self.crash_at:
            raise KeyboardInterrupt
        self.seen.append((self.current_epoch, batch[:, 0].long().tolist()))
        self.lrs.append(self.trainer.optimizers[0].param_groups[0]["lr"])
        return self.layer(batch / NUM_SAMPLES).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.validations += 1

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def run(
    tmp_path,
    ckpt_path=None,
    crash_at=None,
    epochs=2,
    batch_size=BATCH_SIZE,
    val_every=None,
):
    """Saves at the end of a validation like our configs, or else every 3 batches."""
    pl.seed_everything(0)
    sampler = ResumableSampler(Items())
    loader = DataLoader(Items(), sampler=sampler, batch_size=batch_size, drop_last=True)
    module = Module(crash_at)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        val_check_interval=val_every,
        limit_val_batches=3 if val_every else 0,
        num_sanity_val_steps=0,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                monitor="step",
                mode="max",
                every_n_train_steps=None if val_every else 3,
                save_top_k=1,
                save_on_train_epoch_end=False,
            ),
            ResumeDataPosition(sampler, batch_size),
        ],
    )
    trainer.strategy.checkpoint_io = FullCheckpointIO()
    try:
        val_loader = DataLoader(Items(), batch_size=batch_size)
        trainer.fit(module, loader, val_loader, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        pass
    return module, trainer


def test_sampler_gives_lightnings_own_order_and_skips_only_once():
    plain = DistributedSampler(Items(), num_replicas=2, rank=1, shuffle=True, seed=0)
    sampler = ResumableSampler(Items())
    sampler.set_ranks(2, 1)
    assert len(sampler) == len(plain)
    assert list(sampler) == list(plain)

    plain.set_epoch(3)
    sampler.resume_at(3, 8)
    assert list(sampler) == list(plain)[8:]
    assert list(sampler) == list(plain)


@pytest.mark.parametrize(
    "crash_at, val_every, stopped_at",
    [
        (4, None, 3),  # saved at the end of a batch, first epoch
        (13, None, 12),  # second epoch
        (4, 3, 3),  # saved after a validation, like our configs
        (14, 3, 13),  # second epoch
        (12, 5, 10),  # saved after a validation on the last batch of the epoch
    ],
)
def test_resumed_run_matches_an_uninterrupted_one(
    tmp_path, crash_at, val_every, stopped_at
):
    """10 batches per epoch, 2 epochs."""
    whole, whole_trainer = run(tmp_path / "whole", val_every=val_every)
    run(tmp_path / "parts", crash_at=crash_at, val_every=val_every)

    last = resume_checkpoint_path("last", tmp_path / "parts")
    assert torch.load(last, weights_only=False)["global_step"] == stopped_at

    second, second_trainer = run(
        tmp_path / "parts", ckpt_path=last, val_every=val_every
    )

    assert second_trainer.global_step == whole_trainer.global_step == 20
    # Same batches in the same order as the uninterrupted run, from where the checkpoint stopped.
    assert second.seen == whole.seen[stopped_at:]
    assert second.lrs == pytest.approx(whole.lrs[stopped_at:])
    for a, b in zip(second.parameters(), whole.parameters()):
        assert torch.allclose(a, b)


def test_the_repeated_validation_after_a_resume_is_complete(tmp_path):
    """Lightning repeats the validation the checkpoint ended on: all of it, not one batch."""
    run(tmp_path, crash_at=4, val_every=3)
    last = resume_checkpoint_path("last", tmp_path)
    second, _ = run(tmp_path, ckpt_path=last, val_every=3)
    # Batches 3 (repeated), 6 and 9 of this epoch, then 3, 6 and 9 of the next, 3 batches each.
    assert second.validations == 6 * 3


def test_resume_refuses_a_run_that_does_not_match(tmp_path):
    run(tmp_path, crash_at=4)
    last = resume_checkpoint_path("last", tmp_path)
    with pytest.raises(ValueError, match="same data, batch size"):
        run(tmp_path, ckpt_path=last, batch_size=2)


def test_last_is_the_newest_checkpoint_of_the_run_and_never_the_final_one(tmp_path):
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    (folder / "epoch=0-step=3.ckpt").write_bytes(pickle.dumps(1))
    (folder / "epoch=9-step=9-final.ckpt").write_bytes(pickle.dumps(1))
    assert resume_checkpoint_path("last", tmp_path).endswith("epoch=0-step=3.ckpt")
    assert resume_checkpoint_path(None, tmp_path) is None
    with pytest.raises(FileNotFoundError):
        resume_checkpoint_path("/nowhere.ckpt", tmp_path)
    with pytest.raises(FileNotFoundError):
        resume_checkpoint_path("last", tmp_path / "other")
