"""Central definition of standard camera names and sensor configurations.

Every dataset maps its raw camera identifiers to these standard names.
Sensor configurations reference only standard names, so they are
dataset-agnostic and validated at construction time.
"""

from typing import Literal

# Canonical camera name vocabulary shared across all datasets.
StandardCamera = Literal[
    "FRONT",
    "FRONT_LEFT",
    "FRONT_RIGHT",
    "FRONT_TELE",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR_LEFT",
    "REAR",
    "REAR_RIGHT",
]

# Order of the labeled images in a question-answering prompt.
_QA_CAMERA_ORDER: tuple[StandardCamera, ...] = (
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "FRONT_TELE",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR_LEFT",
    "REAR",
    "REAR_RIGHT",
)


def qa_cameras(frame) -> list[str]:
    """The cameras of one frame that a question-answering prompt shows: all of them."""
    return [name for name in _QA_CAMERA_ORDER if name in frame]


# Sensor configurations: config name -> tuple of standard camera names.
# "all_cams" is a special sentinel (empty tuple) meaning "load every camera
# the dataset has".
SENSOR_CONFIGS: dict[str, tuple[StandardCamera, ...]] = {
    "front_cam_only": ("FRONT",),
    "front_three": ("FRONT_LEFT", "FRONT", "FRONT_RIGHT"),
    "front_four_tele": ("FRONT_LEFT", "FRONT", "FRONT_RIGHT", "FRONT_TELE"),
    "all_cams": (),  # resolved per-dataset to all available cameras
}
