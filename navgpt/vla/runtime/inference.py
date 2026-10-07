"""Checkpoint loading and prompt formatting for NavGPT3 inference."""

import importlib.util
import os

from ..checkpoint import validate_checkpoint
from .models.modeling_navgpt3 import NavGPT3ForConditionalGeneration
from .models.processing_navgpt3 import NavGPT3Processor

# Processor outputs consumed by NavGPT3ForConditionalGeneration.predict_actions.
MODEL_INPUTS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")


def default_attn_implementation() -> str:
    configured = os.environ.get("NAVGPT_ATTN_IMPLEMENTATION")
    if configured:
        return configured
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"


def load_from_pretrained(model_path, dtype="auto", device="cuda", attn_implementation=None):
    """Load the model and processor, rejecting any missing or unexpected weights."""
    # Check the files before allocating the model; loading info then confirms
    # that every tensor matches this architecture.
    validate_checkpoint(model_path)
    model, loading_info = NavGPT3ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation or default_attn_implementation(),
        output_loading_info=True,
    )
    errors = {key: value for key, value in loading_info.items() if value}
    if errors:
        raise ValueError(f"Checkpoint does not match NavGPT3ForConditionalGeneration: {errors}")
    model.eval()
    return model, NavGPT3Processor.from_pretrained(model_path)


def build_prompt(prompt: str) -> str:
    """Wrap a navigation prompt in the single-turn chat format used in training."""
    text = prompt.replace(" <image>", "<|vision_start|><|image_pad|><|vision_end|>")
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
