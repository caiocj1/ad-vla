import os
from dotenv import load_dotenv
import pickle
import struct
import glob
from tqdm import tqdm
import argparse
import json

load_dotenv()


def create_tfrecord_index(tfrecord_pattern, output_index_path, split):
    """
    Scans TFRecords and saves a list of (filename, offset, length) for every record.
    """
    files = sorted(glob.glob(tfrecord_pattern))
    index = []  # List of (file_path, offset, length)

    print(f"Indexing {len(files)} TFRecord files...")

    for file_path in tqdm(files):
        file_path_abs = os.path.abspath(file_path)

        with open(file_path, "rb") as f:
            while True:
                offset = f.tell()

                header = f.read(12)
                if not header:
                    break

                length = struct.unpack("<Q", header[:8])[0]

                f.seek(length + 4, 1)

                index.append(
                    {"path": file_path_abs, "offset": offset, "length": length}
                )

    print(f"Found {len(index)} samples.")
    os.makedirs(output_index_path, exist_ok=True)
    out_file = os.path.join(output_index_path, f"wod_e2e_{split}_index.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(index, f)
    print(f"Index saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to Waymo TFRecord files"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        help="Split to preprocess",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save processed pickle files",
    )
    args = parser.parse_args()

    filename_pattern = os.path.join(args.data_path, f"{args.split}*.tfrecord*")
    create_tfrecord_index(filename_pattern, args.output_path, args.split)

    if not os.path.exists(
        os.path.join(args.output_path, "val_sequence_name_to_scenario_cluster.json")
    ):
        with open(
            os.path.join(args.data_path, "val_sequence_name_to_scenario_cluster.json"),
            "r",
        ) as f:
            sequence_to_scenario_cluster = json.load(f)

        with open(
            os.path.join(
                args.output_path, "val_sequence_name_to_scenario_cluster.json"
            ),
            "w",
        ) as f:
            json.dump(sequence_to_scenario_cluster, f)
