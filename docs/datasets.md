# Datasets

Every dataset adapter returns the same sample format (see [adding_a_dataset.md](adding_a_dataset.md)).
A dataset is used through its config in `ad_vla/conf/dataset/`, which reads its root from an
environment variable. You only need the datasets your experiment names.

| Dataset | Config | Root variable | Used for |
|---|---|---|---|
| Waymo Open Dataset, End-to-End Driving | `waymo` | `WAYMO_INDEX_PATH` | driving; RFS with rater scores; Waymo QA; test submissions |
| NAVSIM (OpenScene, from nuPlan) | `navsim` | `OPENSCENE_DATA_ROOT` | driving |
| nuScenes | `nuscenes` | `NUSCENES_ROOT` | driving; nuScenes QA |
| NVIDIA PhysicalAI-AV | `physical_ai` | `HF_HOME` or `PHYSICAL_AI_CACHE_DIR` | driving |
| KITScenes LongTail | `kitscenes` | Hugging Face cache | evaluation |
| NuPlanQA | `nuplan_qa` | `NUPLANQA_IMAGES_ROOT` | QA only |
| AlpaSim closed-loop states | `alpasim_states` | `state_dirs` in the config | Flow-GRPO |

Camera sets are chosen per entry with `sensors`: `front_cam_only`, `front_three` (front left, front,
front right), `front_four_tele` (PhysicalAI's three front cameras plus the telephoto one) or
`all_cams`. Camera names are mapped to a common set (`FRONT`, `FRONT_LEFT`, `REAR`, ...), so a model
reads every dataset the same way.

## Derived files: `ASSETS_DIR`, intents, clip lists

The mixes and examples also read a few files made once from the datasets. They live in
`ASSETS_DIR`:

```text
$ASSETS_DIR/
  indices/
    waymo_pref_idx.pt                              Waymo val samples with rater scores (the RFS subset)
    wod_e2e_{train,val,test}_clip_to_dataset_indices.pkl   Waymo clip -> frames, for past frames and QA
    nuplanqa_index.npz                             NuPlanQA frames, cameras and calibration
  annotations/
    nuscenes/NuScenes_{train,val}_questions.json   nuScenes QA
    waymo/{train,validation_mcq}.jsonl             Waymo QA
    nuplan_qa/nuplanqa_{train,eval_test}.json      NuPlanQA
```

Also:

- `INTENT_LABELS_DIR`: 7-class intent tables (`intents_<dataset>_<split>.json`), used by the
  `v2_qa` mix. See [Intent labels](#intent-labels).
- `PHYSICAL_AI_TRAIN_CLIP_IDS` and `PHYSICAL_AI_VAL_CLIP_IDS`: text files of PhysicalAI clip IDs,
  one per line. The repository ships the fixed 1,000-clip validation list,
  `resources/physical_ai/val_1000_clip_ids.txt`. Any training list works if it is disjoint from it;
  the examples use 5,000 clips.

## Waymo E2E

1. Download the End-to-End Driving TFRecords (training, validation, test) from the
   [Waymo Open Dataset](https://waymo.com/open/). This needs an account and the dataset license.
2. Index each split. The index lists `(file, offset, length)` for every frame, so samples are read
   without scanning the records:

   ```bash
   for split in train val test; do
     uv run python scripts/utils/preprocess_waymo.py \
       --data_path /path/to/waymo_e2e_tfrecords --split $split --output_path $WAYMO_INDEX_PATH
   done
   ```

   `--data_path` is the folder that holds the `<split>*.tfrecord*` files and
   `val_sequence_name_to_scenario_cluster.json`. The index stores absolute paths: index again if you
   move the TFRecords.
3. Find the validation frames with rater-scored trajectories. RFS is computed on them:

   ```bash
   uv run python scripts/utils/get_waymo_rater_indices.py --out_path $ASSETS_DIR/indices
   mv $ASSETS_DIR/indices/waymo_rater_idxs.pt $ASSETS_DIR/indices/waymo_pref_idx.pt
   ```

4. For past image frames (`num_past_frames`) or Waymo QA, build the clip index. It reads only each
   record's name, not its images:

   ```bash
   uv run python scripts/utils/build_waymo_clip_index.py --output-dir $ASSETS_DIR/indices
   ```

**Waymo QA** entries read JSONL files with one question per row: `token` (the clip), `frame_index`
(absent for questions about the whole video), `type` (`image` or `video`), `waymo_split`,
`question`, `category`, and either a free-text `answer` or `options` with `answer_index`. Multiple
choice questions are answered with the option's letter. Video questions are put on the clip's last
frame.

## NAVSIM

Download OpenScene and the nuPlan maps as described in the
[NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md).
Set `OPENSCENE_DATA_ROOT` to the folder that holds `navsim_logs/trainval` and
`sensor_blobs/trainval`, and `NUPLAN_MAPS_ROOT` to the maps. The train/val/test log split is
bundled (`ad_vla/dataset/navsim/default_train_val_test_log_split.yaml`); pass `log_split_yaml` to
use another one.

`frame_interval` sets the spacing of samples within a log, in 0.5 s frames, and
`num_history_frames` how much past each sample carries.

## nuScenes

Download nuScenes v1.0-trainval from [nuscenes.org](https://www.nuscenes.org/) and set
`NUSCENES_ROOT`. Driving entries need `num_future_steps` future poses (2 Hz); with
`require_full_past: false` samples near the start of a scene keep a shorter past.

**nuScenes QA** uses the question files of [NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA):
JSON with a `questions` list of `{sample_token, question, answer, num_hop, template_type}`. A QA
entry needs `task: qa`, `sensors: all_cams` and `qa_annotations_path`. Answers are short canonical
words, and validation compares them exactly.

## PhysicalAI-AV

The dataset is [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
on the Hugging Face Hub. Accept its terms there, and set `HF_TOKEN`. It is read through the
`physical-ai-av` package, with these config keys:

- `stream: true` reads clips from the Hub as needed; `false` reads the local cache (`root`,
  `PHYSICAL_AI_CACHE_DIR` or `HF_HOME`), which must hold the clips already.
- `revision` pins a dataset revision. The examples use `b719eea7f0a63619ef51ec7f54178af0937ef050`.
- `clip_ids_path` is a text file of clip IDs.
- `t0_us` is the sample time within each clip, or `sample_t0_us` a list of times for several samples
  per clip.
- `num_history_steps` and `num_future_steps` are counted at 10 Hz. `num_frames` and
  `frame_spacing_s` set the image frames.
- `undistort: true` turns the fisheye images into pinhole views.

## KITScenes

The adapter loads [KIT-MRT/KITScenes-LongTail](https://huggingface.co/datasets/KIT-MRT/KITScenes-LongTail)
with `datasets.load_dataset`, so it downloads into the Hugging Face cache on first use. It is gated:
accept its terms on the Hub and set `HF_TOKEN`. It is used for evaluation; testing writes
KITScenes-style predictions under `kit_subs/`.

## NuPlanQA

[NuPlanQA](https://github.com/sungyeonparkk/NuPlanQA) asks questions about nuPlan frames with six
cameras. It is QA only: its samples have no trajectory, intent or agents.

1. Get `nuplanqa_train.json`, `nuplanqa_eval_test.json` and `nuplan_metadata.pkl` from the NuPlanQA
   repository.
2. Download the nuPlan v1.1 camera archives (sensor blobs) that NuPlanQA's frames come from, from
   [nuplan.org](https://www.nuplan.org/). The archives are tar files despite their `.zip` names.
   Extract each into its own folder, `part_<archive number>`, keeping only the images:

   ```bash
   mkdir -p $NUPLANQA_IMAGES_ROOT/part_32
   tar -xf nuplan-v1.1_train_camera_32.zip -C $NUPLANQA_IMAGES_ROOT/part_32 \
       --strip-components=3 --wildcards '*/CAM_*/*.jpg'
   ```

   Training uses the frames of the folders present. Validation refuses questions whose images are
   missing. The evaluation questions need `val_camera_0` (as `part_0`) and train cameras 41 and 42.
3. Build the frame index from the metadata pickle. Unpickling runs code, so use a trusted file:

   ```bash
   cd /tmp && python -I /path/to/ad-vla/scripts/utils/build_nuplanqa_index.py \
       /path/to/nuplan_metadata.pkl $ASSETS_DIR/indices/nuplanqa_index.npz
   ```

   Run it with the environment's Python, from a folder without Python files (`-I` keeps the
   script's own folder and the current one off the import path).

Frames are 0.6 s apart, not 0.5 s like the other datasets. Samples state their spacing, and the QA
prompt says it, so a model trained on 0.5 s frames is told the truth. Each question comes with the
ego vehicle's speed and steering over the past 1.5 s as text (`ego_state_text`).

## Intent labels

Datasets give a driving command with each sample: by default the dataset's own (Waymo and NAVSIM
commands, or one computed from the future path). With `intent_labels_path`, a sample takes one of 7
classes instead: `FOLLOW_LANE`, `TURN_LEFT`, `TURN_RIGHT`, `LANE_CHANGE_LEFT`, `LANE_CHANGE_RIGHT`,
`U_TURN`, `UNKNOWN`. They come from one geometric rule on the future path,
`ad_vla.utils.intent_utils.intent7_from_path`, the same on every dataset.

A table is a JSON file per dataset and split:

```json
{"classes": ["FOLLOW_LANE", "TURN_LEFT", "TURN_RIGHT", "LANE_CHANGE_LEFT", "LANE_CHANGE_RIGHT", "U_TURN", "UNKNOWN"],
 "labels": {"<sample key>": ["TURN_LEFT", "<source>"]},
 "by_index": {"<Waymo dataset index>": "TURN_LEFT"}}
```

Keys are the NAVSIM frame token, the nuScenes sample token, the Waymo dataset index (`by_index`) or
`<clip_id>__<t0_us>` for PhysicalAI. Tables are worth precomputing because they apply the rule to
the whole logged future, longer than what a sample holds. A sample missing from the table gets the
rule on its own future, so a table with empty `labels` and `by_index` gives the 7 classes without
any precomputation.

## AlpaSim states (Flow-GRPO)

Flow-GRPO trains FM-VLA's action expert on states recorded in closed-loop
[AlpaSim](https://github.com/NVlabs/alpasim) rollouts: the driver's inputs at each step (images,
past path, route) plus the context its reward needs (other agents, lanes). `AlpaSimStateDataset`
reads directories of such states (`state_dirs`, or a `state_list` file). The tools that extract
states from rollouts aren't part of this repository yet.

## Agent boxes

Waymo E2E has no labels for other agents. `agent_boxes_path` can point to an `AgentBoxStore`
directory of 3D detections and tracks, which then fill `sample.agent_info`; some rewards use it. The
examples don't need it.
