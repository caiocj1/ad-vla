from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Any, ClassVar
from collections.abc import Mapping, Iterator
from dataclasses import dataclass
import math
import torch


PROXIMITY_ABS_TOL = 1e-6


@dataclass
class TrajectorySampling:
    """
    Trajectory sampling specification. Defines temporal resolution of a trajectory.

    Defines the temporal resolution of a trajectory. Any 2 of 3 parameters
    define the third:

    - ``num_poses``: number of trajectory waypoints.
    - ``time_horizon``: total time span in seconds.
    - ``interval_length``: time between consecutive waypoints in seconds.

    Adapted from nuplan's ``TrajectorySampling`` dataclass.
    """

    num_poses: int | None = None
    time_horizon: float | None = None
    interval_length: float | None = None

    def __post_init__(self) -> None:
        if self.num_poses and not isinstance(self.num_poses, int):
            raise ValueError(
                f"num_poses was defined but it is not int. Instead {type(self.num_poses)}!"
            )

        if self.time_horizon:
            self.time_horizon = float(self.time_horizon)
        if self.interval_length:
            self.interval_length = float(self.interval_length)

        if self.num_poses and self.time_horizon and not self.interval_length:
            self.interval_length = self.time_horizon / self.num_poses
        elif self.num_poses and self.interval_length and not self.time_horizon:
            self.time_horizon = self.num_poses * self.interval_length
        elif self.time_horizon and self.interval_length and not self.num_poses:
            remainder = math.fmod(self.time_horizon, self.interval_length)
            is_close_to_zero = math.isclose(remainder, 0, abs_tol=PROXIMITY_ABS_TOL)
            is_close_to_interval = math.isclose(
                remainder, self.interval_length, abs_tol=PROXIMITY_ABS_TOL
            )
            if not is_close_to_zero and not is_close_to_interval:
                raise ValueError(
                    "The time horizon must be a multiple of interval length! "
                    f"time_horizon = {self.time_horizon}, interval = {self.interval_length} and remainder is {remainder}"
                )
            self.num_poses = int(self.time_horizon / self.interval_length)
        elif self.num_poses and self.time_horizon and self.interval_length:
            if not math.isclose(
                self.num_poses,
                self.time_horizon / self.interval_length,
                abs_tol=PROXIMITY_ABS_TOL,
            ):
                raise ValueError(
                    "Not valid initialization of sampling class! "
                    f"time_horizon = {self.time_horizon}, "
                    f"interval = {self.interval_length}, num_poses = {self.num_poses}"
                )
        else:
            raise ValueError(
                f"Cannot initialize TrajectorySampling! num_poses = {self.num_poses}, "
                f"interval = {self.interval_length}, time_horizon = {self.time_horizon}"
            )

    def __hash__(self) -> int:
        return hash((self.num_poses, self.time_horizon, self.interval_length))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrajectorySampling):
            return NotImplemented
        return (
            math.isclose(
                float(self.time_horizon),
                float(other.time_horizon),
                abs_tol=PROXIMITY_ABS_TOL,
            )
            and math.isclose(
                float(self.interval_length),
                float(other.interval_length),
                abs_tol=PROXIMITY_ABS_TOL,
            )
            and self.num_poses == other.num_poses
        )


class BaseTensorModel(BaseModel):
    """Base Pydantic model for tensor-based data structures.

    This model allows arbitrary types (e.g. ``torch.Tensor``) and extra
    fields so that dataset- and calibration-specific metadata can be carried
    through without validation errors.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class CalibrationDict(BaseTensorModel):
    """Camera calibration parameters for a single view.

    This object answers two questions about a camera: where it is mounted on
    the ego vehicle, and how rays from the world land on image pixels.

    The stored extrinsics are a camera-to-ego transform:

        ``p_ego = rotation @ p_camera + translation``

    The ego frame is x-forward, y-left, z-up. The normalized camera frame is
    optical/RDF: x points right in the image, y points down in the image, and z
    points forward through the lens. Therefore ``is_flu`` should usually be
    ``False``; legacy FLU camera axes mean x-forward, y-left, z-up in the camera
    frame and should be converted before use by projection/BEV utilities.

    Field meanings:
    - ``translation`` is the camera origin in ego coordinates, in meters. For
      example, positive x means mounted forward of the ego origin, positive y
      means mounted to the left, and positive z means mounted above it.
    - ``rotation`` is a 3x3 matrix whose columns say where the camera's
      right/down/forward axes point in ego coordinates. Equivalently,
      ``rotation[:, 0]`` is camera +x/right expressed in ego xyz,
      ``rotation[:, 1]`` is camera +y/down expressed in ego xyz, and
      ``rotation[:, 2]`` is camera +z/forward expressed in ego xyz.
    - ``intrinsics`` is the 3x3 pinhole matrix
      ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``. ``fx`` and ``fy`` are focal
      lengths in pixels; ``cx`` and ``cy`` are the pixel coordinates where the
      camera's straight-ahead ray hits the image.
    - ``distortion`` optionally stores five lens distortion coefficients
      ``[k1, k2, p1, p2, k3]``. The ``k`` terms model radial bending away from
      the image center, while the ``p`` terms model small tangential offsets.
    """

    intrinsics: torch.Tensor | None = None
    distortion: torch.Tensor | None = None
    translation: torch.Tensor
    rotation: torch.Tensor
    original_img_size: tuple[int, int]
    is_flu: bool

    @field_validator("intrinsics")
    @classmethod
    def validate_intrinsics(cls, v):
        if v is None:
            return v
        if v.ndim != 2 or v.shape[0] != 3 or v.shape[1] != 3:
            raise ValueError("Intrinsics must be a 2D tensor of shape (3, 3)")
        return v

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, v):
        if v.ndim != 2 or v.shape[0] != 3 or v.shape[1] != 3:
            raise ValueError("Rotation must be a 2D tensor of shape (3, 3)")
        return v

    @field_validator("translation")
    @classmethod
    def validate_translation(cls, v):
        if v.ndim != 1 or v.shape[0] != 3:
            raise ValueError("Translation must be a 1D tensor of shape (3,)")
        return v

    @field_validator("distortion")
    @classmethod
    def validate_distortion(cls, v):
        if v is None:
            return v
        if v.ndim != 1 or v.shape[0] != 5:
            raise ValueError("Distortion must be a 1D tensor of shape (5,)")
        return v

    @field_validator("original_img_size")
    @classmethod
    def validate_original_img_size(cls, v):
        if len(v) != 2:
            raise ValueError("Original image size must be a tuple size 2 (h, w)")
        return v


class CameraData(BaseTensorModel):
    """Image and calibration bundle for a single camera view."""

    image: torch.Tensor
    cal_dict: CalibrationDict | None = None

    @field_validator("image")
    @classmethod
    def validate_image(cls, v):
        if v.ndim != 3:
            raise ValueError("Image must be a 3D tensor (H, W, C)")
        if v.shape[-1] != 3:
            raise ValueError("Image must have 3 channels (C=3)")
        return v


class AgentInfo(BaseTensorModel):
    """
    For trajectories of external agents when available.

    Attributes:
        pos: Tensor shaped [N, T, 2] with xy positions of external agents.
        mask: Tensor shaped [N, T] with valid timesteps for each external agent.
        types: List of strings denoting agent type.
    """

    pos: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    types: list[str] | None = None

    @model_validator(mode="after")
    def check_shapes(self):
        if self.pos is None or self.mask is None:
            return self
        if (
            self.pos.shape[0] != self.mask.shape[0]
            or self.pos.shape[1] != self.mask.shape[1]
        ):
            raise ValueError(
                f"Agents' position has shape {self.pos.shape} while mask has shape {self.mask.shape}"
            )
        return self

    @field_validator("pos")
    @classmethod
    def validate_pos(cls, v):
        if v is None:
            return v
        if v.ndim != 3:
            raise ValueError("Agents' position must have 3 dimensions (N, T, 2)")
        if v.shape[-1] != 2:
            raise ValueError(
                "Agents' position must have size 2 in last dimension (N, T, 2)"
            )
        return v

    @field_validator("mask")
    @classmethod
    def validate_mask(cls, v):
        if v is None:
            return v
        if v.ndim != 2:
            raise ValueError("Agents' mask must have 2 dimensions (N, T)")
        return v


@dataclass(frozen=True, slots=True)
class QuestionAnswerPair:
    """One visual question and its short supervised answer.

    The optional NuScenesQA fields are retained for analysis, while training
    only needs ``question`` and ``answer``. A frozen, slotted dataclass keeps the
    hundreds of thousands of annotations considerably lighter than full
    Pydantic objects.

    Args:
        question: Natural-language question about the sample's camera views.
        answer: Exact short text the backbone should produce.
        num_hop: Number of reasoning hops reported by NuScenesQA, when present.
        template_type: NuScenesQA question category, such as ``"status"``.
    """

    question: str
    answer: str
    num_hop: int | None = None
    template_type: str | None = None


class E2EDataSample(BaseTensorModel):
    """Canonical representation of one driving sample and optional tied tasks.

    ``qa_pairs`` belongs to the same scene/sample as the images and trajectory.
    It is annotation data, not metadata: for nuScenes the existing
    ``metadata['scenario_id']`` already identifies the official sample token.
    """

    metadata: dict[str, Any]
    agent_input: dict[str, torch.Tensor]
    cameras: list[dict[str, CameraData]]
    teacher_cameras: list[dict[str, CameraData]] | None = None
    intent: str
    fut_traj: torch.Tensor | None = None  # [T, 2] ground truth trajectory
    fut_traj_sampling: TrajectorySampling | None = None
    reasoning_trace: str | None = None
    agent_info: AgentInfo | None = None
    qa_pairs: tuple[QuestionAnswerPair, ...] | None = None

    @field_validator("agent_input")
    @classmethod
    def validate_agent_inputs(cls, v):
        for ft_name, ft_value in v.items():
            if ft_value.ndim != 2 or ft_value.shape[1] != 2:
                raise ValueError("Agent past ego states must have shape (*, 2)")
        return v

    @field_validator("fut_traj")
    @classmethod
    def validate_fut_traj(cls, v):
        if v is None:
            return v
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"fut_traj must be a torch.Tensor, got {type(v)}")
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"fut_traj must have shape [T, 2], got {v.shape}")
        return v

    # @field_validator("intent")
    # @classmethod
    # def validate_intent(cls, v):
    #     if v not in ["GO_LEFT", "GO_STRAIGHT", "GO_RIGHT", "UNKNOWN"]:
    #         raise ValueError("Invalid driving intent")
    #     return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v):
        if not ("scenario_id" in v.keys() and "sampling_freq" in v.keys()):
            raise ValueError("Required metadata missing")
        return v


class _SamplesCarrier:
    """Carrier object for the sample list in a batch.

    Wrapping the raw list of ``E2EDataSample`` objects in this helper prevents
    PyTorch from trying to pin them in memory, which would otherwise crash
    when ``pin_memory=True`` is enabled on the ``DataLoader``.

    See ``E2EDataBatch.__init__`` and ``E2EDataBatch.items`` for usage.
    """

    __slots__ = ("samples",)

    def __init__(self, samples):
        self.samples = samples


class E2EDataBatch(BaseTensorModel, Mapping):
    _SAMPLES_KEY: ClassVar[str] = "__samples__"
    _QA_MASK_KEY: ClassVar[str] = "__qa_mask__"

    samples: list[E2EDataSample]
    model_inputs: dict[str, Any]
    qa_mask: torch.Tensor | None = None

    def __init__(self, *args, **kwargs):
        """
        Custom init for torch when pin_memory=True.

        Torch will try to do type(batch)({k: pinned(v) for k, v in batch.items()}),
        which is not supported by standard pydantic type initialization so we check
        if the args is just a single mapping and if so we retrive its keys and values
        to initialize pydantic type correctly.
        """
        if len(args) == 1 and isinstance(args[0], Mapping) and not kwargs:
            d = dict(args[0])
            carrier_or_samples = d.pop(
                E2EDataBatch._SAMPLES_KEY, None
            )  # NOTE: class access, not self
            qa_mask = d.pop(E2EDataBatch._QA_MASK_KEY, None)
            samples = (
                carrier_or_samples.samples
                if isinstance(carrier_or_samples, _SamplesCarrier)
                else carrier_or_samples
            )
            super().__init__(samples=samples, model_inputs=d, qa_mask=qa_mask)
            return
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.model_inputs)

    def __getitem__(self, key: str) -> Any:
        return self.model_inputs[key]

    def __iter__(self) -> Iterator[str]:
        # Only expose model keys so model(**inputs) never sees __samples__
        return iter(self.model_inputs)

    def keys(self):
        return self.model_inputs.keys()

    def items(self):
        """
        When torch calls this, it will try to pin samples to memory if they
        are not wrapped in _SamplesCarrier.
        """
        yield (E2EDataBatch._SAMPLES_KEY, _SamplesCarrier(self.samples))
        if self.qa_mask is not None:
            yield (E2EDataBatch._QA_MASK_KEY, self.qa_mask)
        yield from self.model_inputs.items()

    def values(self):
        return self.model_inputs.values()

    @property
    def batch_size(self) -> int:
        return len(self.samples)
