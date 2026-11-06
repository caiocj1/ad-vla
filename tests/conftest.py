from __future__ import annotations

import os
import tempfile

from dotenv import load_dotenv


load_dotenv()

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "ad_vla_pytest_mplconfig"),
)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

hf_home = os.environ.get("HF_HOME")
if hf_home:
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu_integration: integration tests that require a CUDA device and external model access.",
    )
    config.addinivalue_line(
        "markers",
        "dataset_integration: integration tests that require local dataset files.",
    )
