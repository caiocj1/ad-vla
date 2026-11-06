import torch
import random

from ad_vla.dataset.camera_config import qa_cameras
from ad_vla.dataset.data_types import (
    CameraData,
    E2EDataSample,
    QuestionAnswerPair,
)
from ad_vla.dataset.nuscenes.nuscenes_types import (
    _NUSCENES_QA_CAMERAS,
    _NUSCENES_QA_SYSTEM_PROMPT,
)


ALL_INPUTS = frozenset(["images", "past_traj", "intent"])
_INPUT_ORDER = ["images", "past_traj", "intent"]


def hf_create_annotate_message(
    system_prompt: str,
    cameras: list[dict[str, CameraData]],
    past_traj: torch.Tensor,
    intent: str,
    camera_sequence: list[str] = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
    num_past_image_frames: int | None = 1,
    fut_traj: torch.Tensor | None = None,
) -> list[list[dict]]:
    num_image_frames = min(len(cameras), num_past_image_frames)

    # If has multiple timesteps, more recent images later
    # Concatenate images to not confuse model
    cam_images = [
        [cameras[t][cam].image for cam in camera_sequence]
        for t in range(-num_image_frames, 0)
    ]
    cam_images = [torch.cat(cams_at_t, dim=1) for cams_at_t in cam_images]

    return [
        # Add system prompt
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        },
        # Add user inputs
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Ego-vehicle images:",
                }
            ]
            + [
                {
                    "type": "image",
                    "image": cam_images[t],
                    "input_data_format": "channels_last",
                }
                for t in range(len(cam_images))
            ]
            + [
                {
                    "type": "text",
                    "text": "Past ego trajectory:"
                    + str(torch.round(past_traj[:, :2], decimals=2).tolist()),
                }
            ]
            + [
                {
                    "type": "text",
                    "text": "High-level intent:" + intent,
                }
            ]
            + (
                [
                    {
                        "type": "text",
                        "text": "Future ego trajectory"
                        + (
                            ":" + str(torch.round(fut_traj[:, :2], decimals=2).tolist())
                        ),
                    }
                ]
                if fut_traj is not None
                else []
            ),
        },
    ]


def normalize_qa_prediction(
    raw_prediction: str, allowed_answers: frozenset[str] | None
) -> str | None:
    """A generated answer in comparable form, or ``None`` if it isn't an allowed answer, which also
    refuses a valid answer with text appended."""
    text = raw_prediction.lower()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.replace("<|im_end|>", " ").strip()
    normalized = text.strip(" \n\t.,:;!?\"'")
    if allowed_answers is None or normalized in allowed_answers:
        return normalized
    return None


class PromptBuilder:
    """
    Wraps a prompt YAML dict and builds system prompts and user content
    dynamically based on the set of active inputs.

    Prompt structured format:
        intro: "You are an expert driver.\\nInput:"
        input_sections:
            images: "- 1 frame of multi-view images ..."
            past_traj: "- past trajectory ..."
            intent: "- Current high-level intent (string)"
        task_text: "Task: Future Trajectory Prediction\\n..."
        task_text_cot: "Task 1: Critical Objects...\\n..."  # optional, used when use_cot=True
        use_cot: false

    In old format, ``build_system_prompt`` always returns the static text
    regardless of ``active_inputs``.  In new structured format, only the
    input_sections whose keys are in ``active_inputs`` are included.
    """

    def __init__(self, yaml_dict: dict):
        self._yaml = yaml_dict
        self._structured = "intro" in yaml_dict

    @property
    def use_cot(self) -> bool:
        return self._yaml.get("use_cot", False)

    def build_system_prompt(
        self,
        active_inputs: frozenset[str] = ALL_INPUTS,
        use_cot: bool | None = None,
        image_interval_s: float | None = None,
    ) -> str:
        """``image_interval_s``: the spacing of the sample's frames, when it isn't the prompt's."""
        if not self._structured:
            return self._yaml["text"]

        use_cot = self.use_cot if use_cot is None else use_cot
        if use_cot and "task_text_cot" in self._yaml:
            task_text = self._yaml["task_text_cot"]
        else:
            task_text = self._yaml["task_text"]

        parts = [self._yaml["intro"].rstrip("\n")]
        input_sections = self._yaml.get("input_sections", {})
        for key in _INPUT_ORDER:
            if key in active_inputs and key in input_sections:
                section = input_sections[key]
                if key == "images":
                    section = self._images_section(section, image_interval_s)
                parts.append(section)
        parts.append(task_text.rstrip("\n"))
        return "\n".join(parts)

    def _images_section(self, section: str, image_interval_s: float | None) -> str:
        """The images line with a sample's own spacing of frames, for a dataset whose frames aren't
        ``image_interval`` apart. Such a sample states it as ``metadata['image_interval_s']``
        (NuPlanQA: 0.6 s); without it the line is unchanged.
        """
        stated = self._yaml.get("image_interval")
        if image_interval_s is None or stated is None:
            return section
        if abs(image_interval_s - stated) < 1e-6:
            return section
        old = f"{stated}s intervals"
        if section.count(old) != 1:
            raise ValueError(
                f"Frames are {image_interval_s:g}s apart, but the images line doesn't state "
                f"'{old}' once to change: {section!r}"
            )
        return section.replace(old, f"{image_interval_s:g}s intervals")

    def build_qa_message(
        self,
        sample: E2EDataSample,
        qa_pair: QuestionAnswerPair,
        num_past_image_frames: int = 1,
    ) -> list[dict]:
        """The question-answering chat: system turn, the sample's latest frames with every camera
        labeled, the question, and the answer as the assistant turn. Several frames are grouped by
        camera, oldest first, each after its offset, like driving's separate cameras.

        A prompt with ``qa_uses_system_prompt`` has one system turn for driving and QA, so what is
        specific to a question goes in the user turn. Other prompts keep the older nuScenes QA
        turns, which fit only nuScenes, so that models trained on them see the same text.
        """
        if not sample.cameras:
            raise ValueError("QA requires a current camera frame.")
        frames = sample.cameras[-num_past_image_frames:]
        cameras = qa_cameras(frames[-1])
        question = f"Question: {qa_pair.question}"
        if self._yaml.get("qa_uses_system_prompt", False):
            system_prompt = self.build_system_prompt(
                ALL_INPUTS,
                use_cot=False,
                image_interval_s=sample.metadata.get("image_interval_s"),
            )
            if qa_pair.answer_hint:
                question += f"\n{qa_pair.answer_hint}"
            if qa_pair.options:
                question += "\nOptions:\n" + "\n".join(qa_pair.options)
        else:
            if qa_pair.options or qa_pair.allowed_answers is None:
                raise ValueError(
                    "This prompt's QA turn only fits nuScenes: use prompt drive_vqa."
                )
            if tuple(cameras) != _NUSCENES_QA_CAMERAS:
                raise ValueError(
                    f"NuScenesQA requires all six surround cameras; got {cameras}."
                )
            system_prompt = _NUSCENES_QA_SYSTEM_PROMPT

        user_content: list[dict] = [{"type": "text", "text": "Surround-view images:"}]
        for camera_name in cameras:
            user_content.append({"type": "text", "text": f"{camera_name} camera:"})
            for t, frame in enumerate(frames):
                if num_past_image_frames > 1:
                    offset = t - len(frames) + 1
                    user_content.append({"type": "text", "text": f"<frame {offset}>"})
                user_content.append(
                    {
                        "type": "image",
                        "image": frame[camera_name].image,
                        "input_data_format": "channels_last",
                    }
                )
        user_content.append({"type": "text", "text": question})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": qa_pair.answer}],
            },
        ]

    def build_user_content(
        self,
        cameras: list[dict[str, CameraData]],
        past_traj: torch.Tensor,
        intent: str,
        active_inputs: frozenset[str],
        camera_sequence: list[str],
        num_past_image_frames: int,
        concat_cameras: bool = True,
    ) -> list[dict]:
        """Build the user-turn content list for the chat message.

        Args:
            concat_cameras: If True (default), camera images at each timestep are
                concatenated horizontally into a single image token. If False, each
                camera is emitted as a separate image token preceded by a text label
                (e.g. "FRONT_LEFT:"), which gives the model explicit camera identity
                at the cost of more tokens. With ``num_past_image_frames > 1``, images are
                grouped by camera, oldest first, each preceded by its frame offset
                (e.g. "FRONT:<frame -1><image><frame 0><image>").
        """
        num_image_frames = min(len(cameras), num_past_image_frames)

        # Collect per-timestep, per-camera images (most-recent frame last)
        cam_images_by_frame: list[list[torch.Tensor]] = [
            [cameras[t][cam].image for cam in camera_sequence]
            for t in range(-num_image_frames, 0)
        ]

        content: list[dict] = [{"type": "text", "text": "Ego-vehicle images:"}]

        if concat_cameras:
            # --- Concatenated mode ---
            # All cameras at a given timestep are merged into one wide image.
            for cams_at_t in cam_images_by_frame:
                content.append(
                    {
                        "type": "image",
                        "image": torch.cat(cams_at_t, dim=1),
                        "input_data_format": "channels_last",
                    }
                )
        else:
            # --- Separate mode ---
            # Each camera gets its own image token, prefixed by a text label so the
            # model knows the spatial position of each view.
            for cam_index, cam_name in enumerate(camera_sequence):
                content.append({"type": "text", "text": f"{cam_name}:"})
                for t, cams_at_t in enumerate(cam_images_by_frame):
                    if num_past_image_frames > 1:
                        frame_offset = t - num_image_frames + 1
                        content.append(
                            {"type": "text", "text": f"<frame {frame_offset}>"}
                        )
                    content.append(
                        {
                            "type": "image",
                            "image": cams_at_t[cam_index],
                            "input_data_format": "channels_last",
                        }
                    )

        if "past_traj" in active_inputs:
            traj_list = past_traj[:, :2].tolist()
            traj_str = str([[round(x, 2) for x in row] for row in traj_list])
            content.append(
                {
                    "type": "text",
                    "text": "Past ego trajectory:" + traj_str,
                }
            )

        if "intent" in active_inputs:
            content.append(
                {
                    "type": "text",
                    "text": "High-level intent:" + intent,
                }
            )

        return content

    def build_message(
        self,
        cameras: list[dict[str, CameraData]],
        past_traj: torch.Tensor,
        intent: str,
        active_inputs: frozenset[str],
        camera_sequence: list[str],
        num_past_image_frames: int,
        reasoning_trace: str | None = None,
        fut_traj: torch.Tensor | None = None,
        inference_mode: bool = False,
        concat_cameras: bool = True,
    ) -> list[dict]:
        """Build the full chat message (system + user + assistant turns)."""
        use_cot = (
            self.use_cot
            if inference_mode
            else (self.use_cot and reasoning_trace is not None)
        )
        system_prompt = self.build_system_prompt(active_inputs, use_cot=use_cot)
        user_content = self.build_user_content(
            cameras,
            past_traj,
            intent,
            active_inputs,
            camera_sequence,
            num_past_image_frames,
            concat_cameras=concat_cameras,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        assistant_content: list[dict] = []
        if self.use_cot and reasoning_trace is not None:
            assistant_content.append({"type": "text", "text": reasoning_trace + "\n"})
        if fut_traj is not None:
            traj_list = fut_traj[:, :2].tolist()
            traj_str = str([[round(x, 2) for x in row] for row in traj_list])
            assistant_content.append(
                {
                    "type": "text",
                    "text": "Future ego trajectory:" + traj_str,
                }
            )
        messages.append({"role": "assistant", "content": assistant_content})

        return messages

    def sample_active_inputs(self, drop_cfg: dict[str, float]) -> frozenset[str]:
        """
        Sample the set of active inputs by randomly dropping inputs.

        Args:
            drop_cfg: mapping of input name to drop probability,
                e.g. {"intent": 0.3} means 30 % chance of dropping intent.
                ``images`` is never dropped.
        """
        active = set(ALL_INPUTS)
        for input_name, drop_prob in drop_cfg.items():
            if input_name == "images":
                continue
            if input_name in active and random.random() < drop_prob:
                active.discard(input_name)
        return frozenset(active)
