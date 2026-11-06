"""Integration tests for dataset __getitem__.

Each test instantiates a real dataset and calls ``dataset[0]``, verifying
that the returned ``E2EDataSample`` has the expected structure.  Tests are
skipped when the required data root is not available (env var unset or path
missing).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import pytest

from ad_vla.dataset.data_types import E2EDataSample


pytestmark = [pytest.mark.dataset_integration]


# ---------------------------------------------------------------------------
# Environment-based skip helpers
# ---------------------------------------------------------------------------

WAYMO_INDEX_PATH = os.environ.get("WAYMO_INDEX_PATH", "")
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT", "")
PHYSICAL_AI_ROOT = os.environ.get(
    "HF_HOME", "" or os.environ.get("PHYSICAL_AI_CACHE_DIR", "")
)
KITSCENES_ROOT = os.environ.get(
    "HF_HOME", "" or os.environ.get("KITSCENES_CACHE_DIR", "")
)
NUSCENES_ROOT = os.environ.get("NUSCENES_ROOT", "")

SKIP_WAYMO = pytest.mark.skipif(
    not WAYMO_INDEX_PATH or not os.path.isdir(WAYMO_INDEX_PATH),
    reason="WAYMO_INDEX_PATH not set or directory does not exist.",
)
SKIP_NAVSIM = pytest.mark.skipif(
    not OPENSCENE_DATA_ROOT or not os.path.isdir(OPENSCENE_DATA_ROOT),
    reason="OPENSCENE_DATA_ROOT not set or directory does not exist.",
)


def _has_hf_token() -> bool:
    """PhysicalAI-AV and KITScenes are gated on the Hugging Face Hub: reading them needs a token."""
    from huggingface_hub import get_token

    return bool(get_token())


SKIP_PHYSICAL_AI = pytest.mark.skipif(
    not PHYSICAL_AI_ROOT or not _has_hf_token(),
    reason="Needs PHYSICAL_AI_CACHE_DIR or HF_HOME, and HF_TOKEN (gated dataset).",
)
SKIP_KITSCENES = pytest.mark.skipif(
    not KITSCENES_ROOT or not _has_hf_token(),
    reason="Needs KITSCENES_CACHE_DIR or HF_HOME, and HF_TOKEN (gated dataset).",
)
SKIP_NUSCENES = pytest.mark.skipif(
    not NUSCENES_ROOT or not os.path.isdir(NUSCENES_ROOT),
    reason="NUSCENES_ROOT not set or directory does not exist.",
)


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


def _assert_valid_sample(sample: E2EDataSample, *, expect_fut_traj: bool = True):
    """Run common structural checks on a returned sample."""
    assert isinstance(sample, E2EDataSample)

    # Metadata
    assert "scenario_id" in sample.metadata
    assert "sampling_freq" in sample.metadata

    # Agent input
    past_traj = sample.agent_input["past_traj"]
    assert past_traj.ndim == 2 and past_traj.shape[1] == 2

    # Cameras
    assert len(sample.cameras) >= 1
    first_timestep = sample.cameras[-1]
    assert len(first_timestep) >= 1
    for _, cam_data in first_timestep.items():
        assert cam_data.image.ndim == 3 and cam_data.image.shape[-1] == 3

    # Future trajectory
    if expect_fut_traj:
        assert sample.fut_traj is not None
        assert sample.fut_traj.ndim == 2 and sample.fut_traj.shape[1] == 2
        assert sample.fut_traj_sampling is not None

    # Intent
    assert sample.intent in {"GO_LEFT", "GO_STRAIGHT", "GO_RIGHT", "UNKNOWN"}


# ---------------------------------------------------------------------------
# Waymo
# ---------------------------------------------------------------------------


@SKIP_WAYMO
def test_waymo_getitem():
    from ad_vla.dataset.waymo.waymo_dataset import WaymoE2EDataset

    dataset = WaymoE2EDataset(
        root=WAYMO_INDEX_PATH,
        split="val",
        sensors="front_cam_only",
    )
    assert len(dataset) > 0

    sample = dataset[0]
    _assert_valid_sample(sample)


# ---------------------------------------------------------------------------
# NAVSIM
# ---------------------------------------------------------------------------


@SKIP_NAVSIM
def test_navsim_getitem():
    from ad_vla.dataset.navsim.navsim_dataset import NAVSIMDataset

    dataset = NAVSIMDataset(
        root=OPENSCENE_DATA_ROOT,
        split="val",
        sensors="front_cam_only",
        max_scenes=5,
    )
    assert len(dataset) > 0

    sample = dataset[0]
    _assert_valid_sample(sample)


# ---------------------------------------------------------------------------
# PhysicalAI
# ---------------------------------------------------------------------------


@SKIP_PHYSICAL_AI
def test_physical_ai_getitem():
    from ad_vla.dataset.physical_ai.physical_ai_dataset import PhysicalAIDataset

    dataset = PhysicalAIDataset(
        root=PHYSICAL_AI_ROOT,
        split="val",
        sensors="front_four_tele",
        stream=True,
    )
    assert len(dataset) > 0

    sample = dataset[0]
    _assert_valid_sample(sample)


# ---------------------------------------------------------------------------
# KITScenes
# ---------------------------------------------------------------------------


@SKIP_KITSCENES
def test_kitscenes_getitem():
    from ad_vla.dataset.kitscenes.kitscenes_dataset import KITScenesDataset

    dataset = KITScenesDataset(
        split="train",
        sensors="front_cam_only",
    )
    assert len(dataset) > 0

    sample = dataset[0]
    _assert_valid_sample(sample)


# ---------------------------------------------------------------------------
# nuScenes
# ---------------------------------------------------------------------------


@SKIP_NUSCENES
def test_nuscenes_getitem():
    from ad_vla.dataset.nuscenes.nuscenes_dataset import NuScenesDataset

    dataset = NuScenesDataset(
        root=NUSCENES_ROOT,
        split="val",
        sensors="front_cam_only",
        max_samples=5,
    )
    assert len(dataset) > 0

    sample = dataset[0]
    _assert_valid_sample(sample)
