"""Per-image visual token budgets on top of the PIL Qwen2-VL image processor.

A navigation request carries ``steps`` history frames from each of ``view_num``
cameras. The shared budget ``max_nav_vis_tokens`` is split across those images in
proportion to a temporal weight (newer frames weigh more), a camera weight
(front-heaviest by default) and, with the codec enabled, how much each frame
changed since the same camera's previous frame. Each image's budget becomes its
maximum pixel count; the upstream resize and patchify code does the rest.

The PIL backend is used on purpose: it reproduces the resize the checkpoints were
trained and evaluated with. The default torchvision backend resamples differently.
"""

import numpy as np
from PIL import Image

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import make_flat_list_of_images
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorKwargs,
    Qwen2VLImageProcessorPil,
)


def allocate_vision_tokens(
    total_tokens: int,
    num_cameras: int,
    num_timesteps: int,
    temporal_decay: float = 1.0,
    camera_weights: list[float] | None = None,
    min_tokens_per_image: int = 1,
    max_tokens_per_image: int | None = None,
    change_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Split ``total_tokens`` over a ``(num_timesteps, num_cameras)`` image grid.

    Weights are ``exp(temporal_decay * t / (T - 1)) * camera_weights[c]``, times
    ``change_scores[t, c]`` when given. Row 0 is the oldest frame. Every image gets
    between ``min_tokens_per_image`` and ``max_tokens_per_image`` tokens.
    """
    T, C = num_timesteps, num_cameras
    if max_tokens_per_image is None:
        max_tokens_per_image = total_tokens
    if min_tokens_per_image * T * C > total_tokens:
        raise ValueError(
            f"min_tokens_per_image={min_tokens_per_image} x {T} x {C} = "
            f"{min_tokens_per_image * T * C} exceeds total_tokens={total_tokens}."
        )
    if max_tokens_per_image < min_tokens_per_image:
        raise ValueError("max_tokens_per_image must be >= min_tokens_per_image.")

    if T == 1:
        t_weights = np.ones(T)
    else:
        t_weights = np.exp(temporal_decay * np.arange(T, dtype=float) / (T - 1))

    if camera_weights is None:
        c_weights = np.ones(C)
    else:
        if len(camera_weights) != C:
            raise ValueError(f"Length of camera_weights ({len(camera_weights)}) != num_cameras ({C}).")
        c_weights = np.array(camera_weights, dtype=float)
        if np.any(c_weights <= 0):
            raise ValueError("All camera_weights must be positive.")

    weights = np.outer(t_weights, c_weights)
    if change_scores is not None:
        weights = weights * change_scores
    return _allocate_with_constraints(weights, total_tokens, min_tokens_per_image, max_tokens_per_image)


def _allocate_with_constraints(weights: np.ndarray, total: int, lo: int, hi: int) -> np.ndarray:
    """Distribute ``total`` integer tokens by ``weights`` with each cell in ``[lo, hi]``.

    Every cell starts at ``lo``; the rest is shared proportionally, cells that
    exceed ``hi`` are clamped and their surplus is redistributed, and the integer
    rounding remainder goes to the highest-weight cells with headroom.
    """
    shape = weights.shape
    n = weights.size
    flat_w = weights.flatten().astype(float)

    alloc = np.full(n, lo, dtype=int)
    remaining = total - lo * n
    saturated = np.zeros(n, dtype=bool)

    for _ in range(n + 1):
        if remaining == 0:
            break
        active = ~saturated
        w_active = flat_w * active.astype(float)
        w_sum = w_active.sum()
        if w_sum == 0:
            break

        extra_int = np.floor(remaining * w_active / w_sum).astype(int)
        over = (alloc + extra_int) > hi
        newly_saturated = over & ~saturated
        if newly_saturated.any():
            extra_int[over] = hi - alloc[over]
            saturated |= over

        alloc += extra_int
        remaining -= extra_int.sum()
        if not newly_saturated.any():
            break

    if remaining > 0:
        for idx in np.argsort(-flat_w):
            if remaining == 0:
                break
            add = min(remaining, hi - alloc[idx])
            alloc[idx] += add
            remaining -= add

    return alloc.reshape(shape)


def compute_frame_change_scores(images: list, num_timesteps: int, num_cameras: int) -> np.ndarray:
    """Per-frame change scores in ``[0.05, 1]`` from 64x64 thumbnails.

    ``images`` is ordered ``[t0_c0, t0_c1, ..., t1_c0, ...]``. Each camera's first
    frame scores 1.0; later frames score ``5 * mean |thumb - previous thumb| / 255``.
    """
    thumb_size = (64, 64)
    change_scores = np.ones((num_timesteps, num_cameras), dtype=np.float64)
    for c in range(num_cameras):
        prev_thumb = None
        for t in range(num_timesteps):
            img = images[t * num_cameras + c]
            if isinstance(img, Image.Image):
                thumb = np.array(img.convert("RGB").resize(thumb_size, Image.BILINEAR), dtype=np.float32)
            elif isinstance(img, np.ndarray):
                thumb = np.array(Image.fromarray(img).resize(thumb_size, Image.BILINEAR), dtype=np.float32)
            else:
                thumb = np.array(img, dtype=np.float32)

            if t > 0 and prev_thumb is not None:
                diff = np.abs(thumb - prev_thumb).mean() / 255.0
                change_scores[t, c] = np.clip(diff * 5.0, 0.05, 1.0)
            prev_thumb = thumb
    return change_scores


class NavGPT3ImageProcessorKwargs(Qwen2VLImageProcessorKwargs, total=False):
    r"""
    view_num (`int`, *optional*):
        Cameras per history step. Enables per-image token budgets when set.
    steps (`int`, *optional*):
        History steps; images are ordered step-major, camera-minor.
    max_nav_vis_tokens (`int`, *optional*):
        Visual tokens shared by all images in the request.
    temporal_decay (`float`, *optional*, defaults to 1.0):
        Exponential weight toward recent steps.
    camera_weights (`list[float]`, *optional*):
        Per-camera weights; equal when unset.
    min_tokens_per_image (`int`, *optional*, defaults to 256):
        Lower bound on each image's budget.
    max_tokens_per_image (`int`, *optional*):
        Upper bound on each image's budget.
    codec_enabled (`bool`, *optional*, defaults to `False`):
        Also weight each frame by how much it changed since the previous step.
    """

    view_num: int
    steps: int
    max_nav_vis_tokens: int
    temporal_decay: float
    camera_weights: list[float]
    min_tokens_per_image: int
    max_tokens_per_image: int
    codec_enabled: bool


class NavGPT3ImageProcessor(Qwen2VLImageProcessorPil):
    valid_kwargs = NavGPT3ImageProcessorKwargs

    def preprocess(
        self,
        images,
        view_num: int | None = None,
        steps: int | None = None,
        max_nav_vis_tokens: int | None = None,
        temporal_decay: float = 1.0,
        camera_weights: list[float] | None = None,
        min_tokens_per_image: int = 256,
        max_tokens_per_image: int | None = None,
        codec_enabled: bool = False,
        **kwargs,
    ) -> BatchFeature:
        if view_num is None:
            return super().preprocess(images, **kwargs)

        images = make_flat_list_of_images(images)
        if len(images) != steps * view_num:
            raise ValueError(f"Expected {steps * view_num} images (steps={steps}, view_num={view_num}), got {len(images)}.")
        limits = allocate_vision_tokens(
            total_tokens=max_nav_vis_tokens,
            num_cameras=view_num,
            num_timesteps=steps,
            temporal_decay=temporal_decay,
            camera_weights=camera_weights,
            min_tokens_per_image=min_tokens_per_image,
            max_tokens_per_image=max_tokens_per_image,
            change_scores=compute_frame_change_scores(images, steps, view_num) if codec_enabled else None,
        )

        return_tensors = kwargs.pop("return_tensors", None)
        patches, grids = [], []
        for image, limit in zip(images, limits.flatten()):
            image_patches, grid_thw = self.preprocess_with_token_limit(image, int(limit), **kwargs)
            patches.append(image_patches)
            grids.append(grid_thw)
        return BatchFeature(
            data={"pixel_values": np.concatenate(patches), "image_grid_thw": np.array(grids, dtype=np.int64)},
            tensor_type=return_tensors,
        )

    def preprocess_with_token_limit(self, image, token_limit: int, **kwargs) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Patchify one image so it yields at most ``token_limit`` visual tokens."""
        max_pixels = token_limit * (self.patch_size * self.merge_size) ** 2
        size = {"shortest_edge": self.size["shortest_edge"], "longest_edge": max_pixels}
        for key in ("size", "min_pixels", "max_pixels", "return_tensors"):
            kwargs.pop(key, None)
        out = super().preprocess([image], size=size, return_tensors=None, **kwargs)
        return out["pixel_values"], tuple(int(v) for v in out["image_grid_thw"][0])


__all__ = ["NavGPT3ImageProcessor", "allocate_vision_tokens", "compute_frame_change_scores"]
