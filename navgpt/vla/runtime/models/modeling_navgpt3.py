# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Copyright 2026 The NavGPT-3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NavGPT3VisionModel.interpolate_pos_embed and NavGPT3VisionModel.forward are
# adapted from the Qwen3-VL vision model in Hugging Face Transformers.
"""NavGPT3 model: upstream Qwen3-VL with an MLP waypoint head.

The language model and LM head are the unmodified Transformers Qwen3-VL modules.
The vision tower is upstream except for how it resamples its learned position
embeddings (see ``NavGPT3VisionModel``). ``action_head`` maps the final hidden
state of the last prompt token to eight (x, y, theta) waypoints.
"""

import torch
from torch import nn

from transformers import AutoModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLVisionModel,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.vision_utils import get_vision_attention_seqlens, get_vision_position_ids

from .configuration_navgpt3 import NavGPT3Config


class NavGPT3VisionModel(Qwen3VLVisionModel):
    """Qwen3-VL vision tower with the position-embedding arithmetic used in training.

    The checkpoints were trained and evaluated with Transformers 4.57, which
    resamples the learned position table with bfloat16 bilinear weights and adds the
    four taps one at a time in bfloat16. Transformers 5 computes the same
    interpolation in float32, which changes the encoder output by bfloat16 rounding.
    ``interpolate_pos_embed`` restores the 4.57 arithmetic; the rest of ``forward``
    is the upstream code.
    """

    def interpolate_pos_embed(self, grid_thw: torch.Tensor) -> torch.Tensor:
        side, merge = self.num_grid_per_side, self.config.spatial_merge_size
        indices, weights = [[] for _ in range(4)], [[] for _ in range(4)]
        for _, h, w in grid_thw.tolist():
            h_idx = torch.linspace(0, side - 1, h)
            w_idx = torch.linspace(0, side - 1, w)
            h0, w0 = h_idx.int(), w_idx.int()
            h1, w1 = (h0 + 1).clip(max=side - 1), (w0 + 1).clip(max=side - 1)
            dh, dw = h_idx - h0, w_idx - w0
            for i, (rows, cols, wh, ww) in enumerate(
                ((h0, w0, 1 - dh, 1 - dw), (h0, w1, 1 - dh, dw), (h1, w0, dh, 1 - dw), (h1, w1, dh, dw))
            ):
                indices[i].extend((rows[:, None] * side + cols[None]).flatten().tolist())
                weights[i].extend((wh[:, None] * ww[None]).flatten().tolist())

        table = self.pos_embed.weight
        taps = self.pos_embed(torch.tensor(indices, device=table.device))
        taps = taps * torch.tensor(weights, dtype=table.dtype, device=table.device)[:, :, None]
        pos = taps[0] + taps[1] + taps[2] + taps[3]

        # Repeat over frames and reorder each image into spatial-merge blocks.
        out = []
        sizes = [h * w for _, h, w in grid_thw.tolist()]
        for p, (t, h, w) in zip(pos.split(sizes), grid_thw.tolist()):
            p = p.repeat(t, 1).view(t, h // merge, merge, w // merge, merge, -1)
            out.append(p.permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        return torch.cat(out)

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

        hidden_states = self.patch_embed(hidden_states) + self.interpolate_pos_embed(grid_thw)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)

        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)]
                deepstack_features.append(merger(hidden_states))

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_features,
        )


class NavGPT3Model(Qwen3VLModel):
    def __init__(self, config: NavGPT3Config):
        # Qwen3VLModel.__init__ with the NavGPT3 vision tower.
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = NavGPT3VisionModel._from_config(config.vision_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class NavGPT3ActionHead(nn.Sequential):
    """MLP from the language model hidden size to flattened waypoints."""

    def __init__(self, config: NavGPT3Config):
        in_features = config.text_config.hidden_size
        layers = []
        for _ in range(config.action_num_layers - 1):
            layers += [nn.Linear(in_features, config.action_hidden_size), nn.GELU()]
            in_features = config.action_hidden_size
        layers.append(nn.Linear(in_features, config.action_dim))
        super().__init__(*layers)


class NavGPT3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    config_class = NavGPT3Config
    config: NavGPT3Config

    def __init__(self, config: NavGPT3Config):
        # Qwen3VLForConditionalGeneration.__init__ with the NavGPT3 base model and the
        # action head, and a single post_init once every module exists.
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = NavGPT3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.action_head = NavGPT3ActionHead(config)
        self.post_init()

    def predict_actions(self, **inputs) -> torch.Tensor:
        """Return waypoints of shape ``(batch, action_dim)`` for processor outputs."""
        hidden_states = self.model(**inputs, use_cache=False).last_hidden_state
        return self.action_head(hidden_states[:, -1])


__all__ = ["NavGPT3ActionHead", "NavGPT3ForConditionalGeneration", "NavGPT3Model", "NavGPT3VisionModel"]
