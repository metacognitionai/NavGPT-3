"""Qwen3-VL processor that forwards navigation token budgets to the image processor."""

from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor, Qwen3VLProcessorKwargs
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from .image_processing_navgpt3 import NavGPT3ImageProcessor, NavGPT3ImageProcessorKwargs


class NavGPT3ProcessorKwargs(Qwen3VLProcessorKwargs, total=False):
    images_kwargs: NavGPT3ImageProcessorKwargs
    # TypedDict subclasses do not inherit plain class attributes.
    _defaults = Qwen3VLProcessorKwargs._defaults


class NavGPT3Processor(Qwen3VLProcessor):
    valid_processor_kwargs = NavGPT3ProcessorKwargs

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        # Build the components directly rather than through Auto* registries.
        kwargs.pop("subfolder", None)
        return [
            NavGPT3ImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs),
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs),
            Qwen3VLVideoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs),
        ]

    def encode_patches(self, text: str, pixel_values, image_grid_thw, return_tensors="pt") -> BatchFeature:
        """Tokenize ``text`` for images that were already patchified, e.g. from a frame cache."""
        image_inputs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
        replacements = [self.replace_image_token(image_inputs, i) for i in range(len(image_grid_thw))]
        text, _ = self.get_text_with_replacements([text], replacements)
        text_inputs = self.tokenizer(text, padding=False, return_token_type_ids=False)
        text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature({**text_inputs, **image_inputs}, tensor_type=return_tensors)


__all__ = ["NavGPT3Processor"]
