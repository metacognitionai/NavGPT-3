"""NavGPT3 configuration: Qwen3-VL plus a waypoint regression head."""

from huggingface_hub.dataclasses import strict

from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


@strict
class NavGPT3Config(Qwen3VLConfig):
    r"""
    action_hidden_size (`int`, *optional*, defaults to 512):
        Width of the hidden layers in the action head.
    action_dim (`int`, *optional*, defaults to 24):
        Size of the action output: eight waypoints of (x, y, theta).
    action_num_layers (`int`, *optional*, defaults to 2):
        Number of linear layers in the action head.
    """

    model_type = "navgpt3"

    action_hidden_size: int = 512
    action_dim: int = 24
    action_num_layers: int = 2


# Lets the Auto* loaders used by the processor recognise NavGPT3 checkpoints.
AutoConfig.register("navgpt3", NavGPT3Config, exist_ok=True)

__all__ = ["NavGPT3Config"]
