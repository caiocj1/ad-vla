"""Turn NuPlanQA's ``nuplan_metadata.pkl`` into a plain index for ``NuPlanQADataset``.

The pickle holds nuplan-devkit objects, so reading it needs the devkit and runs code from the
file. This runs once, on a compute node, with ``python -I`` from a directory without Python files;
training then reads only the ``.npz`` written here (``allow_pickle=False``)::

    cd /tmp && python -I /path/to/ad_vla/scripts/utils/build_nuplanqa_index.py \
        /path/to/nuplan_metadata.pkl /path/to/nuplanqa_index.npz

Per frame it keeps the six image paths (``part_<archive>/<image token>.jpg``), each camera's
intrinsics and camera-to-ego pose (from ``sensor2ego``, the fixed mounting, as NavSim's
``sensor2lidar`` is: nuPlan's lidar sits at the ego origin), the previous frame and the timestamp.
"""

import argparse
import collections
import pickle

import numpy as np

# Order of the cameras in the index arrays.
CAMERAS = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L2", "CAM_R2", "CAM_B0")


def quaternion_to_matrix(q) -> np.ndarray:
    """Rotation matrix of a unit quaternion ``[w, x, y, z]`` (pyquaternion's order, as nuPlan)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def build(metadata: dict) -> dict[str, np.ndarray]:
    tokens = sorted(metadata)
    for token in tokens:
        missing = set(CAMERAS) - set(metadata[token]["cams"])
        if missing:
            raise ValueError(f"Frame {token} has no {sorted(missing)}.")
    n = len(tokens)
    out = {
        "cameras": np.array(CAMERAS),
        "tokens": np.array(tokens),
        "prev": np.array([metadata[t]["prev"] or "" for t in tokens]),
        "scene_tokens": np.array([metadata[t]["scene_token"] for t in tokens]),
        "scenario_types": np.array([metadata[t]["scenario"] or "" for t in tokens]),
        "timestamps": np.array(
            [int(metadata[t]["timestamp"]) for t in tokens], np.int64
        ),
        "image_paths": np.array(
            [[metadata[t]["cams"][c]["data_path"] for c in CAMERAS] for t in tokens]
        ),
        "intrinsics": np.zeros((n, len(CAMERAS), 3, 3), np.float32),
        "cam_translation": np.zeros((n, len(CAMERAS), 3), np.float32),
        "cam_rotation": np.zeros((n, len(CAMERAS), 3, 3), np.float32),
    }
    for i, token in enumerate(tokens):
        cams = metadata[token]["cams"]
        for c, name in enumerate(CAMERAS):
            cam = cams[name]
            out["intrinsics"][i, c] = np.asarray(cam["cam_intrinsic"], np.float64)
            out["cam_translation"][i, c] = np.asarray(
                cam["sensor2ego_translation"], np.float64
            )
            out["cam_rotation"][i, c] = quaternion_to_matrix(cam["sensor2ego_rotation"])
    return out


def report(index: dict[str, np.ndarray]) -> None:
    """What the dataset relies on: one archive per frame, the previous frame's spacing and depth."""
    position = {t: i for i, t in enumerate(index["tokens"])}
    parts = np.array([[p.split("/")[0] for p in row] for row in index["image_paths"]])
    mixed = int((parts != parts[:, :1]).any(axis=1).sum())
    print(f"{len(position):,} frames; {mixed} with cameras in more than one archive")
    print("frames per archive:", dict(sorted(collections.Counter(parts[:, 0]).items())))
    prev_pos = np.array([position.get(p, -1) for p in index["prev"]])
    has_prev = prev_pos >= 0
    named = index["prev"] != ""
    print(f"previous frame named: {named.sum():,}; in the index: {has_prev.sum():,}")
    dt = (index["timestamps"][has_prev] - index["timestamps"][prev_pos[has_prev]]) / 1e6
    print(
        "seconds to the previous frame, percentiles 0/1/50/99/100:",
        np.percentile(dt, [0, 1, 50, 99, 100]).round(3).tolist(),
    )
    same_part = parts[has_prev, 0] == parts[prev_pos[has_prev], 0]
    print(
        f"previous frame in the same archive: {same_part.sum():,} of {has_prev.sum():,}"
    )
    depth = np.zeros(len(prev_pos), np.int64)
    for i in range(len(prev_pos)):
        j, d = prev_pos[i], 0
        while j >= 0 and d < 8:
            j, d = prev_pos[j], d + 1
        depth[i] = d
    print(
        "previous frames in the index (capped at 8):",
        dict(sorted(collections.Counter(depth).items())),
    )
    rot = index["cam_rotation"][0, 0]
    print(
        "first frame CAM_F0 camera-to-ego rotation (column 2 = optical axis):\n",
        rot.round(4),
    )
    print("first frame CAM_F0 translation:", index["cam_translation"][0, 0].round(4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("metadata_pkl")
    parser.add_argument("out_npz")
    args = parser.parse_args()
    with open(args.metadata_pkl, "rb") as f:
        metadata = pickle.load(f)
    index = build(metadata)
    report(index)
    np.savez(args.out_npz, **index)
    print("wrote", args.out_npz)


if __name__ == "__main__":
    main()
