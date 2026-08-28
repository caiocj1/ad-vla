import torch
from typing import Any

from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    PretrainedConfig,
)

from ad_vla.utils.prompts import PromptBuilder, ALL_INPUTS


def _recursive_setattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively set attribute on object and all its children."""
    setattr(obj, attr, value)
    for child in getattr(obj, "children", lambda: [])():
        _recursive_setattr(child, attr, value)


class BaseVLAConfig(PretrainedConfig):
    model_type = "base_vla"

    def __init__(
        self,
        vlm_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        prompt_cfg: dict | None = None,
        camera_sequence: list[str] = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT"],
        num_past_image_frames: int = 1,
        concat_cameras: bool = True,
        pad_missing_past_traj: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.vlm_name_or_path = vlm_name_or_path
        self.camera_sequence = camera_sequence
        self.num_past_image_frames = num_past_image_frames
        self.concat_cameras = concat_cameras
        self.pad_missing_past_traj = pad_missing_past_traj

        # Convert DictConfig → plain dict so HF config serialises cleanly to JSON
        if prompt_cfg is not None and not isinstance(prompt_cfg, dict):
            from omegaconf import OmegaConf

            prompt_cfg = OmegaConf.to_container(prompt_cfg, resolve=True)
        self.prompt_cfg = prompt_cfg

    @property
    def attn_implementation(self):
        return self._attn_implementation


class BaseVLA(PreTrainedModel):
    config_class = BaseVLAConfig
    _supports_flash_attn_2 = True
    base_model_prefix = "vlm"

    def __init__(
        self,
        config: BaseVLAConfig,
        pretrained_modules: dict[str, PreTrainedModel] | None = None,
    ):
        super().__init__(config)

        if pretrained_modules is not None:
            for module in pretrained_modules.values():
                if not isinstance(module, torch.nn.Module):
                    continue
                _recursive_setattr(module, "_is_hf_initialized", True)
        else:
            pretrained_modules = {}

        # When first loading a raw model, should do BaseVLA.from_pretrained_submodules
        # otherwise backbone is initialized with random weights
        if "vlm" in pretrained_modules:
            self.vlm = pretrained_modules["vlm"]
        else:
            vlm_config_obj = AutoConfig.from_pretrained(config.vlm_name_or_path)
            vlm_config_obj._attn_implementation = "flash_attention_2"
            self.vlm = AutoModelForImageTextToText.from_config(vlm_config_obj)
            self.vlm.to(config.dtype if config.dtype is not None else torch.bfloat16)

        # For now, we are using the vanilla processor, can expand to include special tokens later
        self.processor = AutoProcessor.from_pretrained(config.vlm_name_or_path)
        self.processor.tokenizer.padding_side = "left"

        self.prompt_builder = PromptBuilder(config.prompt_cfg)
        self.system_prompt = self.prompt_builder.build_system_prompt(ALL_INPUTS)
        self.use_cot = self.prompt_builder.use_cot

        self.accepts_loss_kwargs = False

        self.camera_sequence = config.camera_sequence
        self.num_past_image_frames = config.num_past_image_frames
        self.concat_cameras = config.concat_cameras
        self.pad_missing_past_traj = config.pad_missing_past_traj

    @classmethod
    def from_pretrained_submodules(
        cls,
        config,
    ):
        """
        Builds model from pretrained backbone.
        """
        pretrained_modules = {}

        vlm = AutoModelForImageTextToText.from_pretrained(
            config.vlm_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        pretrained_modules["vlm"] = vlm

        return cls(config, pretrained_modules=pretrained_modules)

    def get_output_embeddings(self) -> torch.nn.Module:
        """Get the output embeddings of the model."""
        return self.vlm.get_output_embeddings()

    def get_input_embeddings(self) -> torch.nn.Module:
        """Get the input embeddings of the model."""
        return self.vlm.language_model.embed_tokens

    def tie_weights(self, **kwargs) -> None:
        """Delegate weight tying to the nested VLM model."""
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights(**kwargs)
