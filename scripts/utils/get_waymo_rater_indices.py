from dotenv import load_dotenv
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

load_dotenv()

import torch
from tqdm import tqdm
import argparse

from ad_vla.dataset.waymo.waymo_dataset import WaymoE2EDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str)
    args = parser.parse_args()

    dataset = WaymoE2EDataset(
        root=os.environ["WAYMO_INDEX_PATH"],
        split="val",
        sensors="all_cams",
    )

    pref_idx = []
    with tqdm(enumerate(dataset), total=len(dataset)) as pbar:
        for i, sample in pbar:
            if hasattr(sample, "pref_trajs"):
                pref_idx.append(i)
                pbar.set_postfix({"n_samples": len(pref_idx), "last_added": i})

    torch.save(pref_idx, os.path.join(args.out_path, "waymo_rater_idxs.pt"))
