NUSCENES_CAM_LABELS: dict[str, str] = {
    "FRONT": "CAM_FRONT",
    "FRONT_LEFT": "CAM_FRONT_LEFT",
    "FRONT_RIGHT": "CAM_FRONT_RIGHT",
    "REAR_LEFT": "CAM_BACK_LEFT",
    "REAR": "CAM_BACK",
    "REAR_RIGHT": "CAM_BACK_RIGHT",
}


_NUSCENES_QA_CAMERAS = (
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "REAR_LEFT",
    "REAR",
    "REAR_RIGHT",
)
_NUSCENES_QA_ANSWERS = frozenset(
    {
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction vehicle",
        "motorcycle",
        "moving",
        "no",
        "not standing",
        "parked",
        "pedestrian",
        "standing",
        "stopped",
        "traffic cone",
        "trailer",
        "truck",
        "with rider",
        "without rider",
        "yes",
    }
)
_NUSCENES_QA_SYSTEM_PROMPT = """You are a surround-view visual question-answering assistant for an ego vehicle. All six labeled camera images show the same moment. Interpret front, back, left, and right relative to the ego vehicle. Answer the user's question with exactly one lowercase answer from the allowed set below. Output only the answer: do not explain, repeat the question, add punctuation, or add any other text.

Allowed answers: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, barrier, bicycle, bus, car, construction vehicle, motorcycle, moving, no, not standing, parked, pedestrian, standing, stopped, traffic cone, trailer, truck, with rider, without rider, yes."""
