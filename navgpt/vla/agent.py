"""Stateful multi-view navigation agent around a NavGPT3 checkpoint.

The agent keeps every camera's observation history, samples up to
``max_sample_size`` history steps per inference, splits the visual token budget
over the sampled images and returns eight (x, y, theta) waypoints.

frame_cache
    ``precompute`` patchifies each new frame at every token budget it can be
    assigned, so inference only looks patches up. ``on_demand`` passes the raw
    images through the processor at inference time, as in training.
"""

from __future__ import annotations

import hashlib
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .runtime.embodiments import ROBOT_SCALE_CONFIGS
from .runtime.models.image_processing_navgpt3 import allocate_vision_tokens, compute_frame_change_scores
from .runtime.inference import MODEL_INPUTS, build_prompt, load_from_pretrained

_DEFAULT_CAMERA_WEIGHTS: dict[int, list[float]] = {
    1: [1.0],
    4: [2.0, 1.0, 0.5, 1.0],  # front, right, back, left
    8: [2.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
}

_DEFAULT_CAMERA_NAMES: dict[int, list[str]] = {
    1: ["Front View"],
    4: ["Front View", "Right View", "Back View", "Left View"],
    8: [
        "Front View", "Front-Right View", "Right View", "Back-Right View",
        "Back View", "Back-Left View", "Left View", "Front-Left View",
    ],
}


def sample_indices(n: int, sample_size: int) -> list[int]:
    """Sample up to sample_size indices from [0, n-1], always keeping 0 and n-1."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    if sample_size <= 0:
        raise ValueError("sample_size must be >= 1")
    if n <= sample_size:
        return list(range(n))
    if sample_size == 1:
        return [n - 1]
    pool = list(range(1, n - 1))
    return sorted({0, n - 1} | set(random.sample(pool, sample_size - 2)))


def sample_latest_indices(n: int, sample_size: int) -> list[int]:
    """Return the last sample_size indices from [0, n-1] in order."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    return list(range(max(0, n - sample_size), n))


def recover_absolute_waypoints(waypoints: list, scale: dict) -> list:
    """De-normalise waypoints in place using an embodiment's action ranges."""
    for wp in waypoints:
        wp[0] *= scale["x_range"]
        wp[1] *= scale["y_range"]
        wp[2] *= scale["theta_range"]
    return waypoints


class NavGPTVLA:
    """Multi-view navigation agent.

    Parameters
    ----------
    model_path : str
        NavGPT3 checkpoint directory.
    view_num : int
        Cameras per step, e.g. 4 for front/right/back/left.
    camera_weights, camera_names : list | None
        Per-camera token weights and prompt labels; chosen from view_num when None.
    max_nav_vis_tokens : int
        Visual tokens shared by all sampled images.
    temporal_decay : float
        Exponential weight toward recent steps.
    min_tokens_per_image, max_tokens_per_image : int
        Bounds on each image's token budget.
    max_sample_size : int
        History steps fed to the model per inference.
    sample_mode : str
        ``random`` keeps the first and last step and samples the rest;
        ``latest`` keeps the most recent steps.
    frame_cache : str
        ``precompute`` or ``on_demand``; see the module docstring.
    robot_type : str
        Key into ROBOT_SCALE_CONFIGS for waypoint de-normalisation.
    codec_enabled : bool
        Weight each frame's budget by how much it changed since the previous step.
    """

    PROMPT_TEMPLATE = (
        "Imagine you are a robot programmed for navigation tasks. "
        "You have been given a video of historical observations and an image "
        "of the current observation {}. Your assigned task is: '{}'. "
        "Based on this sequence of images, predict your future trajectory."
    )

    def __init__(
        self,
        model_path: str,
        view_num: int = 4,
        camera_weights: list[float] | None = None,
        camera_names: list[str] | None = None,
        max_nav_vis_tokens: int = 3072,
        temporal_decay: float = 2.0,
        min_tokens_per_image: int = 4,
        max_tokens_per_image: int = 196,
        max_sample_size: int = 16,
        sample_mode: str = "random",
        frame_cache: str = "precompute",
        robot_type: str = "habitat_nav",
        codec_enabled: bool = True,
        attn_implementation: str | None = None,
    ) -> None:
        if robot_type not in ROBOT_SCALE_CONFIGS:
            raise ValueError(f"Unknown robot_type {robot_type!r}; available: {list(ROBOT_SCALE_CONFIGS)}")
        if frame_cache not in ("precompute", "on_demand"):
            raise ValueError("frame_cache must be 'precompute' or 'on_demand'")
        print(f"[NavGPT VLA] view_num={view_num} frame_cache={frame_cache} robot_type={robot_type}")

        self.view_num = view_num
        self.camera_weights = camera_weights or list(_DEFAULT_CAMERA_WEIGHTS.get(view_num, [1.0] * view_num))
        self.camera_names = camera_names or list(
            _DEFAULT_CAMERA_NAMES.get(view_num, [f"View {i}" for i in range(view_num)])
        )
        self.max_nav_vis_tokens = max_nav_vis_tokens
        self.temporal_decay = temporal_decay
        self.min_tokens_per_image = min_tokens_per_image
        self.max_tokens_per_image = max_tokens_per_image
        self.max_sample_size = max_sample_size
        self.sample_mode = sample_mode
        self.frame_cache = frame_cache
        self.codec_enabled = codec_enabled
        self._scale = ROBOT_SCALE_CONFIGS[robot_type]

        self.model, self.processor = load_from_pretrained(model_path, attn_implementation=attn_implementation)

        self.rgb_lists: list[list] = [[] for _ in range(view_num)]
        self.history_digest = hashlib.sha256()
        self.audit_history = False
        self.multi_res_cache: dict = {}
        if frame_cache == "precompute":
            self.possible_token_limits = self._precompute_possible_token_limits()

    # ------------------------------------------------------------------
    # Token budgets and the frame cache
    # ------------------------------------------------------------------

    def _allocate(self, num_steps: int, change_scores=None) -> np.ndarray:
        return allocate_vision_tokens(
            total_tokens=self.max_nav_vis_tokens,
            num_cameras=self.view_num,
            num_timesteps=num_steps,
            temporal_decay=self.temporal_decay,
            camera_weights=self.camera_weights,
            min_tokens_per_image=self.min_tokens_per_image,
            max_tokens_per_image=self.max_tokens_per_image,
            change_scores=change_scores,
        )

    def _precompute_possible_token_limits(self) -> list[list[int]]:
        """Budgets the codec-free allocation can give each camera, for every history length."""
        per_cam: list[set] = [set() for _ in range(self.view_num)]
        for steps in range(1, self.max_sample_size + 1):
            alloc = self._allocate(steps)
            for c in range(self.view_num):
                per_cam[c].update(int(v) for v in alloc[:, c])
        return [sorted(s) for s in per_cam]

    def _cache_step_all_resolutions(self, step_idx: int) -> None:
        """Patchify one step's frames at every precomputed budget, in parallel."""
        image_processor = self.processor.image_processor
        tasks = [
            (cam, limit, self.rgb_lists[cam][step_idx])
            for cam in range(self.view_num)
            for limit in self.possible_token_limits[cam]
        ]

        def _process(task):
            cam, limit, image = task
            return cam, limit, image_processor.preprocess_with_token_limit(image, limit)

        step_cache = {c: {} for c in range(self.view_num)}
        with ThreadPoolExecutor() as executor:
            for cam, limit, result in executor.map(_process, tasks):
                step_cache[cam][limit] = result
        self.multi_res_cache[step_idx] = step_cache

    # ------------------------------------------------------------------
    # Inputs and inference
    # ------------------------------------------------------------------

    def _build_image_describe(self, index_list: list[int]) -> str:
        describe = ""
        for idx in index_list:
            describe += f"Time step {idx} "
            for name in self.camera_names:
                describe += f"{name} <image>\n "
        return describe

    def _images_for(self, index_list: list[int]) -> list:
        return [self.rgb_lists[cam][step] for step in index_list for cam in range(self.view_num)]

    def _build_input_from_cache(self, prompt: str, index_list: list[int]):
        change_scores = None
        if self.codec_enabled:
            change_scores = compute_frame_change_scores(self._images_for(index_list), len(index_list), self.view_num)
        alloc = self._allocate(len(index_list), change_scores)

        patches, grids = [], []
        for pos, step_idx in enumerate(index_list):
            if step_idx not in self.multi_res_cache:
                self._cache_step_all_resolutions(step_idx)
            for cam in range(self.view_num):
                limit = int(alloc[pos, cam])
                cam_cache = self.multi_res_cache[step_idx][cam]
                if limit not in cam_cache:
                    cam_cache[limit] = self.processor.image_processor.preprocess_with_token_limit(
                        self.rgb_lists[cam][step_idx], limit
                    )
                image_patches, grid_thw = cam_cache[limit]
                patches.append(image_patches)
                grids.append(grid_thw)

        return self.processor.encode_patches(
            build_prompt(prompt),
            torch.tensor(np.concatenate(patches), dtype=torch.float32),
            torch.tensor(grids, dtype=torch.long),
        )

    def _build_input_no_cache(self, prompt: str, index_list: list[int]):
        return self.processor(
            text=build_prompt(prompt),
            images=self._images_for(index_list),
            view_num=self.view_num,
            steps=len(index_list),
            max_nav_vis_tokens=self.max_nav_vis_tokens,
            temporal_decay=self.temporal_decay,
            camera_weights=self.camera_weights,
            min_tokens_per_image=self.min_tokens_per_image,
            max_tokens_per_image=self.max_tokens_per_image,
            codec_enabled=self.codec_enabled,
            return_tensors="pt",
        )

    def _model_forward(self, data) -> list:
        """Return the eight raw (normalised) waypoints as [[x, y, theta], ...]."""
        inputs = {key: data[key].to(self.model.device) for key in MODEL_INPUTS}
        with torch.inference_mode():
            pred = self.model.predict_actions(**inputs)[0]
        return pred.to(torch.float32).cpu().numpy().reshape(-1, 3).tolist()

    def predict_inference(self, instruction: str) -> list:
        total = len(self.rgb_lists[0])
        if self.sample_mode == "latest":
            index_list = sample_latest_indices(total, self.max_sample_size)
        else:
            index_list = sample_indices(total, self.max_sample_size)
        prompt = self.PROMPT_TEMPLATE.format(self._build_image_describe(index_list), instruction)

        t0 = time.time()
        if self.frame_cache == "precompute":
            data = self._build_input_from_cache(prompt, index_list)
        else:
            data = self._build_input_no_cache(prompt, index_list)
        print(f"[NavGPT VLA] preprocessing {time.time() - t0:.3f}s steps={len(index_list)} frame_cache={self.frame_cache}")
        return self._model_forward(data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the observation history for a new episode."""
        self.history_digest = hashlib.sha256()
        self.rgb_lists = [[] for _ in range(self.view_num)]
        self.multi_res_cache.clear()
        torch.cuda.empty_cache()

    def observe_history(self, images: list) -> None:
        """Append one step of observations without predicting an action.

        Used for frames passed during Planner-side repair motion, so the next
        inference sees the same history the robot experienced.
        """
        if len(images) != self.view_num:
            raise ValueError(f"Expected {self.view_num} images, got {len(images)}")
        for cam, img in enumerate(images):
            self.rgb_lists[cam].append(img)
            if self.audit_history:
                self.history_digest.update(np.asarray(img).tobytes())

        if self.frame_cache == "precompute":
            step_idx = len(self.rgb_lists[0]) - 1
            self._cache_step_all_resolutions(step_idx)
            # Keep only the steps that can still be sampled.
            if len(self.multi_res_cache) > self.max_sample_size * 2:
                keep = set(range(max(0, step_idx - self.max_sample_size * 2 + 1), step_idx + 1))
                for k in list(self.multi_res_cache):
                    if k not in keep:
                        del self.multi_res_cache[k]

    def act(self, images: list, text: str) -> list:
        """Observe one step and return eight waypoints in metric coordinates."""
        self.observe_history(images)
        t0 = time.time()
        navigation = self.predict_inference(text)
        print(f"[NavGPT VLA] step {len(self.rgb_lists[0])} inference {time.time() - t0:.3f}s")
        return recover_absolute_waypoints(navigation, self._scale)
