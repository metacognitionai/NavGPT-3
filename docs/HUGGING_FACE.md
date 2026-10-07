# Downloading and loading NavGPT3 checkpoints

The published **NavGPT3-4B** and **NavGPT3-8B** weights use the tensor names of
this source repository's NavGPT3 model. Install the VLA environment
from the [uv or conda guide](../README.md#installation), download a complete
checkpoint bundle, and load its local directory directly.

## Download from Hugging Face

Install the Hub CLI separately from the pinned VLA runtime. Choose one method.

With uv:

```bash
uv tool install huggingface_hub
hf --help
```

Or with conda:

```bash
conda create --yes --name navgpt3-hub --override-channels -c conda-forge python=3.11 pip
conda run --name navgpt3-hub python -m pip install huggingface_hub
conda activate navgpt3-hub
hf --help
```

The weights are on Hugging Face. Download the entire repository so all shards and
tokenizer assets remain together:

| Model | Repository |
|---|---|
| NavGPT3-4B | [`Metacognition-AI/NavGPT3-4B`](https://huggingface.co/Metacognition-AI/NavGPT3-4B) |
| NavGPT3-8B | [`Metacognition-AI/NavGPT3-8B`](https://huggingface.co/Metacognition-AI/NavGPT3-8B) |

```bash
hf download Metacognition-AI/NavGPT3-4B --local-dir /path/to/NavGPT3-4B
hf download Metacognition-AI/NavGPT3-8B --local-dir /path/to/NavGPT3-8B
```

To pin an experiment, add `--revision` with the commit hash from that model
repository and record it with the results. See the official
[Hub download reference](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download).

## Keep the complete inference bundle

The loader uses these files from the checkpoint directory:

| Files | Purpose |
|---|---|
| `config.json` | NavGPT3 architecture and dimensions |
| `model.safetensors.index.json` and every shard it names, or a single `model.safetensors` | Complete model and action-head weights |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer vocabulary and configuration |
| `preprocessor_config.json`, `video_preprocessor_config.json` | Image and video preprocessing |
| `chat_template.jinja` | Tokenizer chat template; the loader builds prompts in the training format itself |

Keep the remaining supplied assets too, including `generation_config.json`,
`special_tokens_map.json`, `added_tokens.json`, `vocab.json`, `merges.txt`, and
any processor configuration. The published model card and license files describe
the checkpoint and its weight-specific terms. The source-code license does not
replace those terms.

The source implementation lives in
[`navgpt/vla/runtime/models/`](../navgpt/vla/runtime/models/). The public class is
`NavGPT3ForConditionalGeneration`, with `model_type: "navgpt3"` and the
Transformers Qwen3-VL text and vision subconfigs (`qwen3_vl_text`,
`qwen3_vl_vision`). Tensor names follow the Transformers Qwen3-VL layout, plus
the action head:

```text
model.visual.blocks.0.attn.proj.weight
model.language_model.layers.0.self_attn.q_proj.weight
lm_head.weight
action_head.0.weight
```

Standard shard filenames remain as provided in the published bundle.

## Validate and start the service

Set the downloaded directory under `checkpoints` in `configs/paths.yaml`
(`navgpt3-4b` or `navgpt3-8b`). The VLA environment must have this source package
installed, as shown in the installation guide. Then run:

```bash
.envs/planner/bin/navgpt check-checkpoint --config configs/experiments/vln/vla_8b_r2r.yaml
.envs/vla/bin/python -m navgpt.vla --config configs/experiments/vln/vla_8b_r2r.yaml
```

Validation is read-only: it checks the current format, shard index, tensor
headers, and required assets. It does not change weights or load the full model.
Starting the service loads the checkpoint and requires a suitable GPU.

## Direct Python loading

Run this in the installed VLA environment:

```python
from navgpt.vla.runtime import load_from_pretrained

model, processor = load_from_pretrained("/path/to/NavGPT3-4B", device="cuda")
```

The loader accepts a **local checkpoint directory**, uses `dtype="auto"` by
default, and rejects missing or unexpected weights. `model.predict_actions(**inputs)`
returns eight normalised (x, y, theta) waypoints for processor outputs; the
waypoints are scaled to metres and radians by the agent. This loads the model;
`navgpt.vla.agent` and the HTTP service handle stateful navigation and action
execution. Use this repository's loader for the current source-and-weights
release. Evaluation uses the same checkpoint through the service; continue
with the [R2R-CE and RxR-CE commands](../README.md#evaluation).

The model source files and tensor names are listed in the README's
[Model Code and Tensor Names](../README.md#model-code-and-tensor-names).
