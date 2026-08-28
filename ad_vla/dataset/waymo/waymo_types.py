WAYMO_CAM2IDX: dict[str, int] = {
    "UNKNOWN": 0,
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
    "REAR_LEFT": 6,
    "REAR": 7,
    "REAR_RIGHT": 8,
}

WAYMO_IDX2CAM: dict[int, str] = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
    6: "REAR_LEFT",
    7: "REAR",
    8: "REAR_RIGHT",
}

WAYMO_INTENT_DICT: dict[int, str] = {
    0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}

WAYMO_SCENARIO_CLUSTERS: dict[str, int] = {
    "Construction": 0,
    "Cut_ins": 1,
    "Cyclist": 2,
    "Foreign Object Debris": 3,
    "Interections": 4,
    "Multi-Lane Maneuvers": 5,
    "Others": 6,
    "Pedestrian": 7,
    "Single-Lane Maneuvers": 8,
    "Special Vehicles": 9,
}
