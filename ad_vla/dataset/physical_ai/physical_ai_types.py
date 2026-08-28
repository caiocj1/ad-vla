from typing import Literal


# Raw PhysicalAI-AV camera name type
CameraName = Literal[
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]

# Mapping from raw PhysicalAI camera names → standard camera names
PHYSAI_CAM_LABELS: dict[CameraName, str] = {
    "camera_front_wide_120fov": "FRONT",
    "camera_cross_left_120fov": "FRONT_LEFT",
    "camera_cross_right_120fov": "FRONT_RIGHT",
    "camera_front_tele_30fov": "FRONT_TELE",
    "camera_rear_left_70fov": "REAR_LEFT",
    "camera_rear_right_70fov": "REAR_RIGHT",
    "camera_rear_tele_30fov": "REAR",
}
