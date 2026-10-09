# NavGPT-3 usage reference

Start with the [README](../README.md) for installation, data, checkpoints and a
first run. This reference covers experiment configs, commands, the Planner's
tools, the controlled studies and the service interfaces. All commands run from
the repository root.

## Architecture and naming

The repository uses three executable component names:

- **NavGPT Planner** (`navgpt.planner`) is the semantic decision policy. It
  interprets harness-assembled context, selects registered tools, reasons
  about progress and recovery, and makes the final stop decision. It does not
  own the context store or compilation policy.
- **NavGPT Environment** (`navgpt.environment`) owns Habitat simulator state,
  observations, and simulated movement.
- **NavGPT VLA** (`navgpt.vla`) owns low-level multi-view visual-action
  inference using the **NavGPT3** model and processor.

The **NavGPT-3 harness** owns the model-facing interaction framework. A
**NavGPT-3 Runtime** above the harness governs logical-thread lifecycle,
scheduling, permissions, and motion authority in the wider system design.
This repository contains the harness; the deployment runtime is released
separately.

Process boundaries use `api.py`, orchestration uses `runner.py`, model-visible
functions live in `planner/tools/`, provider adapters live in `planner/runtimes/`,
and checkpoint implementation code lives under `vla/runtime/`.

```text
harness context path ──> NavGPT Planner
          ▲                     │
          │ tool results        │ registered tool call
          │                     v
  route/node state <── MCP capability registry
          ▲                     │
          └── NavGPT Environment┼──> NavGPT VLA
                perception/action     waypoint prediction
```

NavGPT Planner runs on Python 3.10+. NavGPT Environment normally runs in the
older Python environment required by `habitat_sim 0.1.7`. NavGPT VLA runs in a
CUDA/PyTorch environment. Keeping these as HTTP-separated processes avoids an
unmaintainable combined dependency stack.

The control model sees only the navigation information and MCP tools selected by
the configured tool set. Benchmark scoring, when used, is an experimental
wrapper around completed runs rather than a model-visible capability.

The Planner chooses the semantic sequence of registered operations. The harness
shapes that behavior through the active tool set, model-visible context,
schema and argument validation, navigation-state contracts, returned evidence,
and termination semantics. NavGPT Environment commits simulator transitions.

Context management belongs to the harness: what the Planner sees, how it is
presented, and where it came from can be inspected outside the model. Every
episode records its briefing, tool calls, tool results and metrics; the Planner
SDK keeps the conversation itself.


### Repository layout

```text
navgpt/
  cli.py          navgpt launch | run | preflight | merge | view | check-data | check-checkpoint
  launch.py       multi-GPU runs: services per GPU, one shard each, merge
  settings.py     experiment config loading and validation
  planner/        briefings, MCP tools, Claude and Codex runtimes, runner, viewer
  environment/    R2R/RxR loading, Habitat observations, actions, maps, metrics
  vla/            NavGPT3 model, processors, inference service, checkpoint check
configs/
  paths.example.yaml   machine paths template
  experiments/         one self-contained config per experiment
docs/             usage reference, dataset and checkpoint guides
```

## Experiment configs

An experiment config is one self-contained YAML file: a `gpus` list and up to six
sections. The shipped configs list every key below in this order. Unknown sections,
keys or values are an error. The run name, and so the output
directory, is the config's path under `configs/experiments/` joined with
underscores (`representation/astra/heading_up_map.yaml` →
`representation_astra_heading_up_map`) unless `name` is set. Omitting `planner`
runs the VLA alone.

**gpus** — the GPUs `navgpt launch` uses, one shard per entry; default `[0]`.
The shipped configs list `[0, 1, 2, 3, 4, 5, 6, 7]`. A GPU listed twice runs two
shards. `navgpt launch --gpus` overrides it ([multi-GPU runs](#multi-gpu-runs)).

**dataset** — which episodes are run.

| Key | Default | Meaning |
|---|---|---|
| `name` | `r2r` | `r2r` or `rxr` |
| `split` | `val_unseen` | split directory, or a label for `episodes_file` |
| `episodes` | `all` | `all`, or indices such as `0-99` or `3,7` |
| `episodes_file` | — | curated episode file: a key under `episodes` in `paths.yaml`, or a path |
| `ground_truth_split` | — | split whose dense ground truth scores an `episodes_file` |
| `ground_truth_file` | — | explicit dense ground-truth file instead |
| `languages`, `roles` | `[en-US, en-IN]`, `[guide]` | RxR instruction selection |

**environment** — the Habitat service.

| Key | Default | Meaning |
|---|---|---|
| `url` | `http://127.0.0.1:9200` | service address |
| `max_steps` | `500` | episode step limit, shared with the runner |
| `forward_view_px` | `512` | forward camera image size |
| `panorama_view_px` | `400` | size of each panorama view |
| `turn_angle_deg`, `step_size_m`, `allow_sliding` | 15, 0.25, true | VLN-CE action settings; the VLA-only configs turn 30° and stop RxR at 400 steps, as VLNCE-EVAL does |

**vla** — the NavGPT3 service.

| Key | Default | Meaning |
|---|---|---|
| `url` | `http://127.0.0.1:8000` | service address |
| `checkpoint` | `navgpt3-8b` | key under `checkpoints` in `paths.yaml`, or a path |
| `visual_token_budget` | `3072` | visual tokens shared across the multi-view history |
| `frame_cache` | `precompute` | `precompute` or `on_demand` image preprocessing |
| `gpu_memory_fraction` | `0` | cap on this process's GPU memory share (0: none) |
| `seed` | `20260910` | base seed for VLA resets in study runs (plus the episode index) |

**planner** — the model that makes decisions.

| Key | Default | Meaning |
|---|---|---|
| `runtime` | `claude` | `claude` (Claude Agent SDK) or `codex` (Codex SDK) |
| `model` | `claude-opus-5` / `gpt-6-astra` | provider model id; the default follows the runtime |
| `effort` | provider default | reasoning effort: `low`, `medium`, `high` or `xhigh` for either runtime, plus `max` (Claude) or `none`/`minimal` (Codex); the shipped configs use `high` |
| `service_tier` | inherited | Codex service tier |
| `tools` | all eight | the tools the Planner can call ([tool sets](#planner-tools)) |
| `max_turns` | `200` | Planner responses per episode |
| `episode_timeout_s` | `2400` | wall-clock limit per episode |
| `max_budget_usd` | 5 (Claude) / null (Codex) | cost cap per episode; Codex reports no cost and needs null |

**harness** — how the Planner sees and drives the robot.

| Key | Default | Meaning |
|---|---|---|
| `interface` | `standard` | how labels, bearings and the map are drawn ([studies](#controlled-studies)) |
| `review_every` | `0` | VLA steps between Planner reviews of a rollout; 0: only when the VLA stops |
| `route_finding` | true | `navigate_relative` follows the walkable route; false walks straight |
| `vla_steps_per_call` | `200` | step cap on one `navigate_by_instruction` call |

**output**

| Key | Default | Meaning |
|---|---|---|
| `dir` | `outputs` | where run directories are written |
| `save_frames` | false | keep observed frames and an action log for `navgpt view` |

Machine paths live in `configs/paths.yaml` (template: `configs/paths.example.yaml`):
`data.r2r`, `data.rxr` and `data.scenes`; curated episode files under `episodes`;
checkpoint directories under `checkpoints`; optionally `codex_bin`, the Codex CLI
used by the Codex runtime; and optionally `python.environment` and `python.vla`,
the interpreters `navgpt launch` starts the services with (default
`.envs/habitat/bin/python` and `.envs/vla/bin/python`). Every command takes
`--paths` to use another file.

## Commands

| Command | Environment | Purpose |
|---|---|---|
| `navgpt launch --config CFG [--gpus LIST] [--episodes SPEC] [--keep-services]` | Planner | start the services on each GPU, run one shard per GPU, merge |
| `python -m navgpt.environment --config CFG [--port P] [--gpu G] [--blind]` | Habitat | serve the experiment's episodes |
| `python -m navgpt.vla --config CFG [--port P] [--gpu G]` | VLA | serve the experiment's checkpoint |
| `navgpt run --config CFG [--episodes SPEC] [--shard I/N] [--env-url U] [--vla-url U]` | Planner | run or resume the experiment |
| `navgpt preflight --config CFG` | Planner | check services, tools and model without running episodes |
| `navgpt merge --config CFG` | Planner | merge the shards of a sharded run |
| `navgpt view [--output-root DIR] [--host H] [--port P] [--run NAME]` | Planner | read-only result browser |
| `navgpt check-data --config CFG` | Planner | check episodes, dense ground truth and scene files |
| `navgpt check-checkpoint --config CFG` | Planner | check the configured checkpoint's files and tensor names |

`navgpt run` and `navgpt launch` resume from an existing `summary.json`: episodes
with a scored record are kept, and the others, including excluded ones, run again.
Use a new experiment name for changed settings.

## Multi-GPU runs

The Environment and VLA services hold one episode at a time, so parallel runs use
one service pair per shard. `navgpt launch` arranges this from the config:

| Shard | GPU | Environment | VLA | Output |
|---|---|---|---|---|
| *i* of *N* | `gpus[i]` | port of `environment.url` + *i* | port of `vla.url` + *i* | `outputs/<experiment>_s<i>` |

Shard *i* runs every *N*-th episode from *i*, so shards mix scenes. After all
shards finish, the launcher merges them into `outputs/<experiment>`, scored like a
single run. With one GPU it runs unsharded straight into `outputs/<experiment>`.
Tool sets without `navigate_by_instruction` start no VLA services.

- **Memory.** Each VLA service loads the full checkpoint on its GPU, next to that
  GPU's Habitat renderer; there is no model parallelism. List a GPU twice only if
  it can hold two of each.
- **Startup.** Services start in their own interpreters (`python` in
  `paths.yaml`), and the launcher waits up to 40 minutes for every `/health` and
  `/info`. A service that exits during startup stops the launch with the end of
  its log.
- **Reuse.** A service already answering on a shard's port is reused if it serves
  the same dataset, split, episode file, interface, camera and motion settings,
  checkpoint and token budget. Anything else on that port is an error.
  `--keep-services` leaves the started services up for the next launch;
  otherwise they are stopped on exit, including after Ctrl-C.
- **Logs.** `outputs/.launch/<experiment>/` holds `environment_<i>.log`,
  `vla_<i>.log` and `shard_<i>.log`. The launcher prints the finished-episode count
  as it changes; `navgpt view` shows the shards while they run.
- **Failures.** If a shard fails, nothing is merged and the command exits
  non-zero; running it again resumes every shard.

The same layout can be started by hand, for example when the services run on a
different host from the Planner:

```bash
CFG=configs/experiments/vln/navgpt3_astra_r2r.yaml
for i in 0 1 2 3; do
  .envs/habitat/bin/python -m navgpt.environment --config $CFG --port $((9200 + i)) --gpu $i &
  .envs/vla/bin/python -m navgpt.vla --config $CFG --port $((8000 + i)) --gpu $i &
done
# once the services report ready:
for i in 0 1 2 3; do
  .envs/planner/bin/navgpt run --config $CFG --shard $i/4 \
    --env-url http://127.0.0.1:$((9200 + i)) --vla-url http://127.0.0.1:$((8000 + i)) &
done
wait
.envs/planner/bin/navgpt merge --config $CFG
```

## Planner tools

The NavGPT-3 harness gives the Planner eight tools:

- `observe_forward()` returns one full-resolution forward view without movement.
- `observe_panorama()` returns simultaneous front, right, back, and left views
  with direction, bearing, clearance, and pose-relative labels rendered into the
  images.
- `observe_map()` refreshes and returns the observed-only map, route status, and
  chronological node table. It never reveals unseen space, the goal, or the
  reference route.
- `navigate_by_instruction(instruction="")` delegates a route instruction to
  NavGPT VLA. The first call passes the full episode instruction; later
  corrective calls pass a self-contained instruction for the remaining route,
  grounded in the returned observations. An omitted argument resumes the active
  route after a scheduled review pause and otherwise uses the episode instruction.
  It moves the simulated agent, then returns sampled route keyframes, endpoint
  views, execution status, and the observed map (for every VLA tool set except the
  one with waypoint return). VLA
  `should_stop` ends only the delegated leg.
- `navigate_relative(turn_deg, distance_m)` performs charged local motion and
  returns realized motion status plus endpoint views.
- `navigate_to_node(target)` performs a charged return to a visited node or a
  waypoint from the latest VLA rollout and returns updated views/map/status.
- `annotate_node(name, caption, place=None, clause=None)` labels an existing
  chronological graph node and records a semantic anchor; it does not create
  topology.
- `terminate_episode(where="here")` optionally returns to a node and then issues
  the irreversible STOP action. Only this Planner decision ends a normal episode.

Two more tools exist for a Planner without the VLA:
`navigate_primitive(actions)` takes discrete moves (0 stop, 1 forward 0.25 m,
2/3 turn 15°), and `observe_node(place)` shows the picture stored for a named place.

`planner.tools` must be one of these sets, each with a matching briefing:

| Tools | Setting |
|---|---|
| `observe_forward`, `navigate_primitive` | camera and primitive moves only (`tools/<planner>/primitive.yaml`) |
| `observe_forward`, `observe_panorama`, `observe_map`, `navigate_primitive`, `navigate_relative`, `terminate_episode` | Planner drives itself (`tools/<planner>/no_vla.yaml`) |
| the previous set plus `observe_node`, `navigate_to_node`, `annotate_node` | Planner drives itself with named places (`tools/<planner>/no_vla_places.yaml`) |
| `navigate_by_instruction`, `observe_forward`, `observe_panorama`, `terminate_episode` | VLA, views and stopping (`tools/<planner>/vla.yaml`)\* |
| the previous set plus `observe_map`, `navigate_to_node`, `annotate_node` | VLA with the map and visited nodes (`tools/<planner>/vla_map.yaml`)\* |
| all eight | NavGPT-3 (`vln/navgpt3_*.yaml`, `tools/<planner>/vla_all.yaml`) |

\* These two use the controlled-study briefing, so they need a
`harness.interface` other than `standard`. Registration, not the briefing, is
the capability boundary: preflight fails when the registered tools differ from
the configured list.



## Controlled studies

The `tools/`, `representation/` and `review/` experiments change one thing at a
time on the R2R subset (`navgpt/planner/tools/ablation.py`). They share one
briefing that names only the configured tools, seed the VLA per episode, and
require the VLA's history-observation and seeded-reset support.

`harness.interface` sets how spatial information is drawn for the Planner:

| Value | What the Planner sees |
|---|---|
| `standard` | the regular NavGPT-3 display, used by the main experiments |
| `reference` | the study display: labels in fixed margins around each view, a north-up map, relative turns |
| `text_labels` | the reference labels given as text instead of drawn |
| `small_digits` | numbers drawn with 3×5-pixel digits instead of 5×7 |
| `heading_up_map` | the map rotates so the robot's heading points up |
| `absolute_bearings` | compass bearings from map-north instead of relative turns |

Camera labels take identical space in every study display. In every display,
including `standard`, the VLA receives the unannotated images.

`harness.review_every` pauses a VLA rollout for Planner review after that many
committed VLA steps. A natural stop wins over a coincident review, and calling
`navigate_by_instruction` without an argument resumes the paused route with its
trace and step cap. Frames passed during the Planner's own repair moves are
added to the VLA's visual history without a prediction. An uncertain VLA
inference invalidates the episode, which is excluded from scoring.

## Planner runtimes

Both adapters drive the same MCP tool server, briefings and turn accounting.

Signing in is covered in the [README](../README.md#planner-sign-in).

- `runtime: claude` uses the Claude Agent SDK, signed in with a Claude
  subscription (`/login` or `CLAUDE_CODE_OAUTH_TOKEN`) or with API, cloud-provider
  or gateway credentials from the environment, which take precedence. It
  enforces `planner.max_budget_usd` per episode.
- `runtime: codex` uses the pinned Codex Python SDK with your local Codex
  sign-in: a ChatGPT plan or an OpenAI API key.
  Codex reports no cost, so `max_budget_usd` must be null; turn and time limits
  still apply. The adapter disables every other configured MCP server and the
  built-in shell, file, browser and search tools, and checks the registered tool
  inventory before the first prompt. Set `codex_bin` in `paths.yaml` when the
  model needs a newer Codex CLI than the one bundled with the SDK.

`preflight` checks the selected runtime. With an API key, the Claude path makes a
4-token model call; with a subscription, it sends one short request through the
CLI. The Codex path checks the local login and the advertised model list without
inference. A spent subscription usage window makes episodes wait and retry every
15 minutes, for up to 6 hours, instead of being scored.

## Service contracts

NavGPT Environment:

| endpoint | purpose |
|---|---|
| `GET /health` | dataset, split, language/role selection, episode count, dense-GT coverage, data fingerprint, step limit, available verbs |
| `POST /call/r2rce__<verb>` | simulator observation/action call |
| `POST /env-panel/field/<name>` | planner-only episode placement |
| `POST /env-panel/action/<name>` | planner-only control action |

NavGPT VLA:

| endpoint | purpose |
|---|---|
| `GET /info` | model identity and inference-service status |
| `POST /reset` | clear temporal state before an episode |
| `POST /act` | one multi-view inference step and waypoint action |
| `POST /observe` | append views to the visual history without predicting (ablation repair) |
| `GET /state` | history, cache and RNG fingerprints for seeded ablation runs |

Both services are stateful at the episode level and are not concurrency-safe:
one service pair serves one run at a time. The `r2rce__` verb prefix is shared
by both datasets; dataset identity comes from `/health`, and the runner refuses
a service whose dataset, split, step limit or ground-truth coverage does not
match the experiment.


## Outputs and viewer

| Path | Contents |
|---|---|
| `summary.json` | run config, per-episode records, aggregate over scored episodes |
| `episode_<i>.jsonl` | event stream: briefing, tool calls and results, metrics |
| `work/ep<i>/raw.jsonl` | raw provider message stream, images stripped |
| `live/ep<i>/` | observed frames and action log (`output.save_frames`) |
| `labels_<i>.json` | node names and captions (tool sets with `annotate_node`) |

An episode is excluded from scoring when it never took a step after an error, when
the provider refused the request, or when an ablation run could not keep a
consistent VLA history; the aggregate and `scored_count` reflect that. `navgpt view`
lists runs under `outputs/` with their aggregates and shows each episode's
trajectory, tool timeline and observations.

## Navigation-state integrity

- The place graph is derived from the append-only walked path. IDs are
  chronological and stable, and teleport-sized jumps do not become graph edges.
- `observed_map` paints only RGB-D-observed cells; unknown space never becomes
  known-free space.
- VLA `should_stop` returns control to the Planner but does not end the episode;
  the Planner decides whether to inspect, continue, return, repair, or terminate.
- A VLA waypoint off the walkable floor becomes one discrete action, as in
  VLNCE-EVAL: a turn toward the VLA's heading change when it exceeds 0.1 rad or
  the previous VLA step left the pose unchanged, otherwise a forward step that
  slides along the obstacle.
- Tool registration, schema validation, movement limits, and state updates are
  enforced independently of the workflow sequence suggested in the briefing.


## Rendering

Real RGB observations require an EGL/OpenGL device. CUDA availability alone is
not enough. Habitat binds its GL context to the thread that creates the
simulator, so all simulator access is serialized through the dedicated simulator
thread in `navgpt/environment/r2r.py`.

`--blind` (on `python -m navgpt.environment`) keeps real episode data, pathfinding, and dynamics but returns synthetic
pixels. It is suitable for API plumbing tests, never for navigation results.


## License and attribution

The source code is released under the Apache License 2.0 in [LICENSE](../LICENSE),
with third-party attribution in [NOTICE](../NOTICE). Model weights and datasets are
distributed separately under their own terms; the source license does not grant
rights to them.

The Planner's evaluation loop, Claude adapter, and MCP tool-server pattern build on
[AgentCanvas](https://github.com/jianzhou0420/AgentCanvas). The NavGPT VLA service files `api.py`, `navigation.py`, and `images.py`
are derived from VLNCE-EVAL, built on EPIC Lab's
[NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE), and keep its MIT license in
[navgpt/vla/LICENSE](../navgpt/vla/LICENSE). The NavGPT3 model and processor classes
extend the Apache-2.0 Qwen3-VL implementation in Hugging Face Transformers.
