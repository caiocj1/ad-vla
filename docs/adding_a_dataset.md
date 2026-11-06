# Adding a dataset

A dataset adapter turns one record of the source into an `E2EDataSample`
(`ad_vla/dataset/data_types.py`). Models, collators, metrics and the builder only see that format,
so a new adapter works with every model and mix.

## The sample format

```python
E2EDataSample(
    metadata={"scenario_id": "...", "sampling_freq": 2.0, ...},
    agent_input={"past_traj": past_xy},   # [T_past, 2], ego frame, oldest first, current pose last
    cameras=[{"FRONT": CameraData(...), "FRONT_LEFT": ...}, ...],  # one dict per frame, oldest first
    intent="GO_STRAIGHT",
    fut_traj=future_xy,                    # [T_future, 2], ego frame; None if unknown
    fut_traj_sampling=TrajectorySampling(num_poses=T_future, interval_length=0.5),
    reasoning_trace=None,
    agent_info=None,
    qa_pairs=None,
)
```

- **Frame.** All positions are in the current ego frame: x forward, y left, metres, with the
  current pose at the origin. `past_traj` ends with the current pose, `(0, 0)`.
- **`metadata`** needs `scenario_id`, a unique ID of the sample, and `sampling_freq`, the rate of
  `past_traj` in Hz. Models resample the past from it. Add anything else that helps you trace a
  sample: scene, timestamp, scenario type.
- **`cameras`** is a list of frames, oldest first, each mapping a standard camera name (`FRONT`,
  `FRONT_LEFT`, `FRONT_RIGHT`, `FRONT_TELE`, `SIDE_LEFT`, `SIDE_RIGHT`, `REAR_LEFT`, `REAR`, `REAR_RIGHT`)
  to `CameraData(image, cal_dict)`. Load images as `[H, W, 3]` uint8 tensors and let the image
  transform resize them. `cal_dict` holds `intrinsics` (3×3), `rotation` (3×3) and `translation`
  (3,), camera to ego in metres, optionally `distortion`, plus `original_img_size` and `is_flu`
  (`False` for the default optical axes).
- **`intent`** is the driving command: `GO_LEFT`, `GO_STRAIGHT`, `GO_RIGHT` or `UNKNOWN`, or one of
  the 7 classes with an intent table (see [datasets.md](datasets.md#intent-labels)).
- **`fut_traj`** with its `fut_traj_sampling` is the training target and the metrics' ground truth.
  Leave it None on test splits.
- **`agent_info`**, optional: other agents' positions, boxes and tracks (`AgentInfo`), for rewards
  and analysis.
- **`qa_pairs`**, optional: `QuestionAnswerPair`s about this sample's images, for `task: qa`
  entries. A QA-only dataset leaves `agent_input` empty and `intent` and `fut_traj` None.

## The adapter

Subclass `BaseDataset` (`ad_vla/dataset/base_dataset.py`) and implement three methods:

```python
from ad_vla.dataset.base_dataset import BaseDataset
from ad_vla.dataset.data_types import E2EDataSample


class MyDataset(BaseDataset):
    def __init__(self, root, split="train", sensors="front_three", img_transform=None,
                 reasoning_traces_path=None, **kwargs):
        super().__init__(root, split, sensors, img_transform, reasoning_traces_path)
        self.records = ...  # an index of the split, cheap to build

    def __len__(self):
        return len(self.records)

    def _get_sample(self, idx):
        return load_record(self.records[idx])  # raw data of one sample

    def _clean_sample(self, record) -> E2EDataSample:
        cameras = {}
        for name in self._resolve_cameras(available_cameras):  # the names `sensors` asks for
            image, cal_dict = self._apply_img_transform(image_of(record, name), calibration_of(record, name), name)
            cameras[name] = {"image": image, "cal_dict": cal_dict}
        return E2EDataSample(metadata=..., agent_input=..., cameras=[cameras], ...)
```

`BaseDataset` provides:

- `_resolve_cameras(available)`: the standard camera names that the `sensors` setting selects.
- `_apply_img_transform(image, cal_dict, name)`: runs the entry's image transform and keeps the
  calibration consistent with resizing and cropping.
- Reasoning traces from `reasoning_traces_path`.
- Precomputed agent boxes from `agent_boxes_path`.

For QA, also implement `qa_pairs_by_index(strict)`: a mapping from dataset index to the questions of
that sample. `strict=True` (validation) must fail when an annotated sample can't be found.

## Config

Add `ad_vla/conf/dataset/my_dataset.yaml`:

```yaml
_target_: ad_vla.dataset.my_dataset.my_dataset.MyDataset
root: ${oc.env:MY_DATASET_ROOT}
sensors: front_three
```

and mount it like any other: `- /dataset@data.train.mine: my_dataset`, with `data.train.mine.size`
set in the experiment. The builder adds the split, the image transform and the subset settings
([configuration.md](configuration.md#datasets-datatrain-and-dataval)).

## Check it

Look at a few samples before training on them:

- The past ends at the origin, and the future starts near it and points forward (positive x).
- Left is positive y: on a left turn the future bends towards +y.
- Camera names match the images: the `FRONT_LEFT` image looks forward and to the left.
- Intrinsics match the image size after the transform.

`tests/test_dataset_getitem.py` shows how the existing adapters are tested on real data, and
`ad_vla.utils.plots.visualize_e2e_sample` draws a sample's cameras and trajectories.
