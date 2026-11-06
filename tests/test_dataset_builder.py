"""data.train / data.val entries: dataset settings, builder settings, and the metrics per set."""

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Subset

from ad_vla.dataset import builder
from ad_vla.dataset.builder import (
    TRANSFORM_DIR,
    check_model_trains_tasks,
    single_entry,
    ValSet,
    build_train_dataset,
    build_val_sets,
    dataset_kwargs,
    primary_val_name,
    reject_old_keys,
    seeded_subset,
)
from ad_vla.train_stages.base_module import BaseTrainingModule


class Items(Dataset):
    def __init__(self, count=5, split="val", img_transform=None, **kwargs):
        self.count, self.split, self.img_transform, self.kwargs = (
            count,
            split,
            img_transform,
            kwargs,
        )

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return index


TARGET = f"{__name__}.Items"


class Model:
    @staticmethod
    def get_collator(inference_mode=False):
        return list


def data_cfg(**data):
    base = {"val_dataloader_args": {"batch_size": 2}, "reasoning_traces_path": None}
    return OmegaConf.create(base | data)


def test_entry_splits_builder_settings_from_dataset_arguments():
    entry = OmegaConf.create(
        {"_target_": TARGET, "count": 3, "size": [4, 6], "subset": "x.pt", "repeat": 2}
    )
    kwargs = dataset_kwargs(entry, "val")
    assert kwargs["split"] == "val" and kwargs["count"] == 3
    assert not {"subset", "repeat", "size"} & set(kwargs)


def transform_of(**entry):
    node = OmegaConf.create({"_target_": TARGET, **entry})
    return dataset_kwargs(node, "val")["img_transform"]


def test_size_alone_is_a_plain_resize():
    transform = transform_of(size=[4, 6])
    assert transform == {
        "_target_": "ad_vla.utils.img_utils.ImageResize",
        "size": [4, 6],
    }


def test_a_named_transform_runs_at_the_entry_size():
    transform = transform_of(size=[4, 6], img_transform="center_crop_resize")
    assert transform["_target_"].endswith("CenterCropResize")
    assert transform["size"] == [4, 6] and transform["p_transform"] == 0.8


def test_a_named_transform_takes_overrides():
    spec = {"name": "center_crop_resize", "p_transform": 0.5}
    transform = transform_of(size=[4, 6], img_transform=spec)
    assert transform["p_transform"] == 0.5 and transform["h_crop_ratio"] == 0.3
    assert "name" not in transform and transform["size"] == [4, 6]


def test_a_full_transform_config_gets_the_entry_size():
    spec = {"_target_": "ad_vla.utils.img_utils.ImageResize"}
    assert transform_of(size=[4, 6], img_transform=spec)["size"] == [4, 6]


def test_every_transform_in_the_config_folder_builds_at_the_entry_size():
    for path in TRANSFORM_DIR.glob("*.yaml"):
        transform = transform_of(size=[4, 6], img_transform=path.stem)
        assert transform["size"] == [4, 6], path.stem
        instantiate(transform)


def test_transform_configs_state_no_size_of_their_own():
    for path in TRANSFORM_DIR.glob("*.yaml"):
        assert "size" not in OmegaConf.load(path), path.stem


def test_a_single_dataset_mapping_holds_exactly_one_entry():
    for entries in (None, OmegaConf.create({}), OmegaConf.create({"a": {}, "b": {}})):
        with pytest.raises(ValueError, match="exactly one"):
            single_entry(entries, "data.test")
    name, entry = single_entry(OmegaConf.create({"a": {"size": [4, 6]}}), "data.test")
    assert name == "a" and entry.size == [4, 6]


def test_native_resolution_is_stated_and_excludes_a_transform():
    assert transform_of(size=None) is None
    with pytest.raises(ValueError, match="needs a size"):
        transform_of(size=None, img_transform="default")


def test_a_missing_size_is_an_error():
    with pytest.raises(ValueError, match="needs size"):
        dataset_kwargs(OmegaConf.create({"_target_": TARGET}), "val", "waymo")


def test_train_entries_keep_order_and_repeat():
    cfg = data_cfg(
        train={
            "a": {"_target_": TARGET, "size": None, "count": 2},
            "b": {"_target_": TARGET, "size": None, "count": 3, "repeat": 3},
        }
    )
    dataset = build_train_dataset(cfg)
    assert len(dataset) == 2 + 3 * 3
    assert [len(part) for part in dataset.datasets] == [2, 3, 3, 3]
    assert dataset.datasets[0].split == "train"


def test_a_single_train_entry_isnt_wrapped():
    cfg = data_cfg(train={"a": {"_target_": TARGET, "size": None}})
    assert isinstance(build_train_dataset(cfg), Items)


def test_qa_train_entry_needs_annotations():
    cfg = data_cfg(train={"a": {"_target_": TARGET, "size": None, "task": "qa"}})
    with pytest.raises(ValueError, match="qa_annotations_path"):
        build_train_dataset(cfg)


def test_primary_is_needed_only_with_several_trajectory_sets():
    one = data_cfg(
        val={
            "a": {"_target_": TARGET, "size": None},
            "q": {"_target_": TARGET, "size": None, "task": "qa"},
        }
    )
    assert primary_val_name(one) == "a"
    two = data_cfg(
        val={
            "a": {"_target_": TARGET, "size": None},
            "b": {"_target_": TARGET, "size": None},
        }
    )
    with pytest.raises(ValueError, match="primary_val"):
        primary_val_name(two)
    two.primary_val = "b"
    assert primary_val_name(two) == "b"
    two.primary_val = "q"
    with pytest.raises(ValueError, match="isn't a trajectory set"):
        primary_val_name(two)


def test_periodic_cap_applies_only_during_training_validation():
    cfg = data_cfg(
        val_periodic_samples=3,
        val={
            "a": {"_target_": TARGET, "size": None, "count": 5},
            "b": {"_target_": TARGET, "size": None, "count": 5, "max_samples": 4},
            "c": {"_target_": TARGET, "size": None, "count": 2},
        },
    )
    sets = {s.name: s for s in build_val_sets(cfg, Model(), training=True)}
    assert len(sets["a"].dataloader(periodic=True).dataset) == 3
    assert len(sets["b"].dataloader(periodic=True).dataset) == 4
    assert len(sets["c"].dataloader(periodic=True).dataset) == 2
    assert len(sets["a"].dataloader().dataset) == 5
    evaluation = build_val_sets(cfg, Model())
    assert len(evaluation[0].dataloader(periodic=True).dataset) == 5


def test_entry_dataloader_args_override_the_default():
    cfg = data_cfg(
        val={
            "a": {
                "_target_": TARGET,
                "size": None,
                "dataloader_args": {"batch_size": 7},
            }
        }
    )
    assert build_val_sets(cfg, Model())[0].dataset.count == 5
    assert build_val_sets(cfg, Model())[0].dataloader_args["batch_size"] == 7


def test_subset_fraction_splits_a_set_without_overlap():
    def entry(fraction):
        return {
            "_target_": TARGET,
            "size": None,
            "count": 200,
            "subset_fraction": fraction,
        }

    head = build_train_dataset(data_cfg(train={"a": entry([0.0, 0.8685])}))
    tail = build_val_sets(data_cfg(val={"a": entry([0.8685, 1.0])}), Model())[0].dataset
    assert len(head) == int(200 * 0.8685) and len(head) + len(tail) == 200
    assert head.indices[-1] + 1 == tail.indices[0]


def test_several_qa_sets_each_log_their_own_metrics():
    module, logged = module_with_sets(["a", "b"], None, qa_names=["a", "b"])
    for name, right in (("a", 1.0), ("b", 0.0)):
        module.qa_val_metrics[name]["correct"].update(torch.tensor(right))
        module.qa_val_metrics[name]["valid"].update(torch.tensor(1.0))
        module.qa_val_metrics[name]["total"].update(torch.tensor(1.0))
        module._qa_val_seen.add(name)
    module.on_validation_epoch_end()
    assert logged["eval_qa/a/accuracy"] == 1.0
    assert logged["eval_qa/b/accuracy"] == 0.0


def test_one_qa_set_keeps_the_plain_metric_names():
    module, logged = module_with_sets(["a"], None, qa_names=["a"])
    module.qa_val_metrics["a"]["correct"].update(torch.tensor(1.0))
    module.qa_val_metrics["a"]["valid"].update(torch.tensor(1.0))
    module.qa_val_metrics["a"]["total"].update(torch.tensor(1.0))
    module._qa_val_seen.add("a")
    module.on_validation_epoch_end()
    assert logged["eval_qa/accuracy"] == 1.0


def _sized_entry(**extra):
    return OmegaConf.create(
        {
            "_target_": TARGET,
            "size": [4, 6],
            "task": "qa",
            "camera_sizes": {"REAR": [2, 6]},
        }
        | extra
    )


def test_camera_sizes_give_listed_cameras_their_own_transform():
    transform = instantiate(dataset_kwargs(_sized_entry(), "val")["img_transform"])
    rear, front = transform.for_camera("REAR"), transform.for_camera("FRONT")
    assert (rear.h, rear.w) == (2, 6)
    assert (front.h, front.w) == (4, 6)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"task": "trajectory"}, "needs task: qa"),
        ({"size": None}, "needs a size"),
        ({"camera_sizes": {"REARR": [2, 6]}}, "unknown cameras"),
    ],
)
def test_camera_sizes_refuses_what_it_cannot_do(extra, message):
    with pytest.raises(ValueError, match=message):
        dataset_kwargs(_sized_entry(**extra), "val")


def test_qa_entries_must_load_every_camera():
    entry = {"_target_": TARGET, "size": None, "task": "qa", "sensors": "front_three"}

    class QAModel(Model):
        predict_qa_answers = None

    with pytest.raises(ValueError, match="all_cams"):
        build_val_sets(data_cfg(val={"q": entry}), QAModel())
    entry["qa_annotations_path"] = "a.json"
    with pytest.raises(ValueError, match="all_cams"):
        build_train_dataset(data_cfg(train={"q": entry}))


def test_qa_batches_stay_recognisable_through_the_qa_loader(monkeypatch):
    monkeypatch.setattr(builder, "QAPairDataset", lambda scenes: scenes)

    class QAModel(Model):
        predict_qa_answers = None

    cfg = data_cfg(
        val={
            "q": {
                "_target_": TARGET,
                "size": None,
                "task": "qa",
                "dataloader_args": {"pin_memory": True},
            }
        }
    )
    assert build_val_sets(cfg, QAModel())[0].dataloader_args["pin_memory"] is False


def test_seeded_subset_is_reproducible_and_keeps_order():
    first = seeded_subset(Items(100), 10, 0)
    second = seeded_subset(Items(100), 10, 0)
    assert isinstance(first, Subset) and first.indices == second.indices
    assert first.indices == sorted(first.indices) and len(first) == 10
    assert seeded_subset(Items(100), 10, 1).indices != first.indices
    dataset = Items(5)
    assert seeded_subset(dataset, None, 0) is dataset
    assert seeded_subset(dataset, 9, 0) is dataset
    with pytest.raises(ValueError):
        seeded_subset(Items(100), 0, 0)


def test_old_fields_are_refused():
    with pytest.raises(ValueError, match="data.val_dataset"):
        reject_old_keys(OmegaConf.create({"val_dataset": {}}))
    reject_old_keys(OmegaConf.create({"val": {}}))


def module_with_sets(names, primary, qa_names=()):
    module = BaseTrainingModule.__new__(BaseTrainingModule)
    torch.nn.Module.__init__(module)
    module.build_metrics()
    module.configure_val_sets(names, primary, qa_names=qa_names)
    logged = {}
    module.log_dict = lambda values, **_: logged.update(values)
    module.log = lambda key, value, **_: logged.update({key: value})
    module._seen_on_any_rank = lambda seen: seen
    return module, logged


def update(metrics, error):
    pred = {"pred_trajs": torch.zeros(1, 1, 3, 2)}
    gt = {"gt_trajs": torch.full((1, 3, 2), float(error))}
    metrics["ADE"].update(pred, gt)
    metrics["FDE"].update(pred, gt)


def test_each_set_logs_its_own_metrics_and_the_primary_also_logs_plain_names():
    module, logged = module_with_sets(["w", "p", "q"], "w", qa_names=["q"])
    assert set(module.other_val_metrics) == {"p"}
    update(module.val_metrics, 1.0)
    update(module.other_val_metrics["p"], 3.0)
    module.on_validation_epoch_end()
    assert logged["eval/ADE"] == pytest.approx(logged["eval/w/ADE"])
    assert logged["eval/p/ADE"] > 2 * logged["eval/ADE"]
    # No rater trajectories here, so RFS is -1: only the plain name, as in earlier runs.
    assert [key for key in logged if key.endswith("RFS")] == ["eval/RFS"]


def test_a_single_set_keeps_the_plain_metric_names():
    module, logged = module_with_sets(["w"], "w")
    update(module.val_metrics, 1.0)
    module.on_validation_epoch_end()
    assert set(logged) == {"eval/ADE", "eval/FDE", "eval/RFS"}


def test_val_set_dataclass_defaults():
    val_set = ValSet("a", "trajectory", Items(5), list, {"batch_size": 2})
    assert len(val_set.dataloader(periodic=True).dataset) == 5


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"size": [4, 6], "task": "qa"}, "no qa_annotations_path"),
        (
            {"size": [4, 6], "task": "qa", "qa_annotations_path": "q", "repeat": 3},
            "repeat",
        ),
        ({"size": [4, 6], "qa_annotations_path": "q"}, "not task: qa"),
        ({"size": [4, 6], "questions_per_frame": 3}, "not task: qa"),
    ],
)
def test_task_qa_is_the_only_switch_for_training_on_questions(entry, message):
    with pytest.raises(ValueError, match=message):
        build_train_dataset(OmegaConf.create({"train": {"a": entry}}))


def test_a_model_without_qa_training_is_refused():
    cfg = OmegaConf.create({"train": {"a": {"task": "qa"}}})
    with pytest.raises(TypeError, match="can't train on"):
        check_model_trains_tasks(cfg, object())
    check_model_trains_tasks(OmegaConf.create({"train": {"a": {}}}), object())


def test_qa_repeat_only_when_every_question_has_its_own_item():
    entry = {"size": [4, 6], "task": "qa", "qa_annotations_path": "q", "repeat": 5}
    builder._check_train_entry(
        "a", OmegaConf.create({**entry, "questions_per_frame": None})
    )
    with pytest.raises(ValueError, match="questions_per_frame: null"):
        builder._check_train_entry(
            "a", OmegaConf.create({**entry, "questions_per_frame": 2})
        )
