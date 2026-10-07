# R2R-CE and RxR-CE data setup

The default evaluation covers **both `val_unseen` benchmarks**: R2R-CE and
**RxR-CE English guide instructions** (`en-US`, `en-IN`). RxR means
*Room-Across-Room*. Both use Matterport3D scenes; they have separate episode
archives and ground-truth paths. The default rollout limit is 500 environment
steps per episode.

Use the [installation guide](../README.md#installation) first. Run the commands
below from the NavGPT-3 repository root. Store large assets wherever your Linux
GPU host has space; the paths in this guide are examples.

## What to download

| Asset | Official source | Needed for |
|---|---|---|
| Matterport3D Habitat scene archive | [Matterport3D access request](https://niessner.github.io/Matterport/#download) | Both benchmarks: scene meshes and navigation meshes |
| `R2R_VLNCE_v1-3_preprocessed.zip` | [Download](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view) · [format and release notes](https://jacobkrantz.github.io/vlnce/data) | R2R-CE episodes and dense ground truth |
| `RxR_VLNCE_v0.zip` | [Download](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) · [official CE dataset instructions](https://github.com/jacobkrantz/VLN-CE#episodes-room-across-room-rxr) | RxR-CE guide/follower episodes and dense ground truth |
| NavGPT3-4B or NavGPT3-8B inference bundle | [Hugging Face download guide](HUGGING_FACE.md) | VLA inference with the published NavGPT3 weights |

Download the dataset archives from their upstream providers and the published
NavGPT3 checkpoint from Hugging Face. Set the local checkpoint directory under
`checkpoints` in `configs/paths.yaml` before VLA evaluation. The source-code license does not grant
rights to datasets or weights; follow each asset's own terms.

### Matterport3D access and scene layout

Request access through the [Matterport3D project](https://niessner.github.io/Matterport/).
The project requires its signed Terms of Use form to obtain the download script.
After approval, use the supplied `download_mp.py` to request **the Habitat
archive**, rather than the full raw capture collection. The legacy downloader
requires Python 2.7; this is separate from all three NavGPT environments.
[Habitat-Sim's versioned dataset instructions](https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/README.md#datasets)
document this command:

```bash
python2.7 /path/to/download_mp.py --task habitat -o /path/to/mp3d-download
```

Extract the archive and arrange the files as follows. `data.scenes` in
`configs/paths.yaml` is the **parent of `mp3d/`**, not the `mp3d/` directory itself:

```text
/path/to/scene-datasets/
└── mp3d/
    ├── 17DRP5sb8fy/
    │   ├── 17DRP5sb8fy.glb
    │   ├── 17DRP5sb8fy.navmesh
    │   └── ...
    └── <other-scan>/
        ├── <other-scan>.glb
        ├── <other-scan>.navmesh
        └── ...
```

Keep the matching `.navmesh` beside every `.glb`. NavGPT's scene loader uses
that same-basename file for navigation and geodesic metrics. Its rendered
fallback can recompute a missing navmesh, but changing navigation geometry can
change results; resolve missing assets before a reported benchmark run. Blind
diagnostics require the saved navmesh and provide synthetic pixels, so they are
not visual-navigation evaluation.

Habitat test scenes, HM3D, and the original Matterport panorama images are not
substitutes for these MP3D Habitat scenes. Only scans referenced by the selected
episodes are needed at runtime; keeping the complete archive avoids missing
scenes when switching splits.

### Episode archives

Download both archives using the table's browser links. For a command-line
download, install [gdown](https://github.com/wkentaro/gdown) in a separate utility
environment. Choose one setup:

```bash
# uv
uv tool install gdown

# Or conda (independent of the old Habitat environment)
conda create --yes --name navgpt-data --override-channels -c conda-forge python=3.11 pip
conda run --name navgpt-data python -m pip install gdown
conda activate navgpt-data
```

Then download and extract into a new dataset directory:

```bash
mkdir -p /path/to/downloads /path/to/datasets
gdown 1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr \
  -O /path/to/downloads/R2R_VLNCE_v1-3_preprocessed.zip
gdown 145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t \
  -O /path/to/downloads/RxR_VLNCE_v0.zip

unzip /path/to/downloads/R2R_VLNCE_v1-3_preprocessed.zip -d /path/to/datasets
unzip /path/to/downloads/RxR_VLNCE_v0.zip -d /path/to/datasets
```

Check the resulting layout rather than adding another directory with the same
name around an archive's existing top-level directory. A Google Drive quota or
confirmation page is not a ZIP file; use the official browser link if the CLI
download fails. No NavGPT-specific annotation preprocessing is needed.

## R2R-CE layout and semantics

Use **v1-3 preprocessed**, whose dense ground-truth files accompany the episode
files. The smaller `R2R_VLNCE_v1-3.zip` has basic episodes but lacks these extra
evaluation assets. Version 1-3 also corrects initial headings relative to older
releases. The official `val_unseen` split has **1,839 episodes across 11 scenes**.
[Dataset format and changelog](https://jacobkrantz.github.io/vlnce/data).

```text
/path/to/datasets/R2R_VLNCE_v1-3_preprocessed/
├── train/
│   ├── train.json.gz
│   └── train_gt.json.gz
├── val_seen/
│   ├── val_seen.json.gz
│   └── val_seen_gt.json.gz
└── val_unseen/
    ├── val_unseen.json.gz
    └── val_unseen_gt.json.gz
```

Each gzip JSON contains an `episodes` array. An episode has `episode_id`,
`scene_id`, `start_position`, `start_rotation`, `goals`, and
`instruction.instruction_text`. `start_rotation` uses `[x, y, z, w]` order.
`reference_path` is the episode's sparse path. Dense ground truth is a separate
mapping keyed by the **string form of `episode_id`**, with a `locations` array
for each episode. The evaluator uses those locations for nDTW; instruction
token IDs and the archive's GloVe embeddings are not needed by NavGPT.

## RxR-CE layout, languages, and roles

Use the **continuous-environment port**, `RxR_VLNCE_v0`. The original
[RxR repository](https://github.com/google-research-datasets/RxR) distributes
graph-navigation annotations as `.jsonl.gz`; those are a different schema and
cannot replace the CE episode files below.

```text
/path/to/datasets/RxR_VLNCE_v0/
├── train/
│   ├── train_guide.json.gz
│   ├── train_guide_gt.json.gz
│   ├── train_follower.json.gz
│   └── train_follower_gt.json.gz
├── val_seen/
│   ├── val_seen_guide.json.gz
│   ├── val_seen_guide_gt.json.gz
│   ├── val_seen_follower.json.gz
│   └── val_seen_follower_gt.json.gz
├── val_unseen/
│   ├── val_unseen_guide.json.gz
│   ├── val_unseen_guide_gt.json.gz
│   ├── val_unseen_follower.json.gz
│   └── val_unseen_follower_gt.json.gz
└── test_challenge/
    └── test_challenge_guide.json.gz
```

The CE files contain an `episodes` array, with the same scene/pose/goal fields
used for R2R. RxR's `instruction` object additionally records `language` and
`instruction_id`. Filter using **`instruction.language`**; ground truth is
indexed by **`episode_id`**, not by `instruction_id` or the array index. The
official loader recognizes `en-US`, `en-IN`, `hi-IN`, and `te-IN`, and the two
annotation roles `guide` and `follower`.
[Official CE schema and filters](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/task.py).

The default follows the
[official RxR English task configuration](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/rxr_vlnce_english_task.yaml):
**guide trajectories, both English variants, 500 environment steps**. Guide and
follower annotations describe different recorded trajectories; select the
corresponding `_guide_gt` or `_follower_gt` file. Do not merge them by instruction
ID. Keep the language and role scope in result labels.

Set `dataset.languages: all` in the experiment config to include Hindi and Telugu
as well as English, or list the language tags to use. `dataset.roles` defaults to
`[guide]`; follower evaluation is a separate selection. NavGPT reads the instruction text directly,
so the upstream baseline's BERT text features, pose traces, and audio files are
not required for inference.

## Configure and verify the files

Point `configs/paths.yaml` at the extracted data (copy it from
`configs/paths.example.yaml`):

```yaml
data:
  r2r: /path/to/datasets/R2R_VLNCE_v1-3_preprocessed
  rxr: /path/to/datasets/RxR_VLNCE_v0
  scenes: /path/to/scene-datasets
episodes:
  opennav100: /path/to/OpenNav_R2R-CE_100_bertidx.json
```

`opennav100` is the 100-episode R2R-CE subset released with
[Open-Nav](https://github.com/YanyuanQiao/Open-Nav), used by the ablations. The
paths may point outside the repository or to symlinks; no copying or
re-tokenization is necessary. Then check each experiment's data. The check uses
only the standard library and does not start Habitat or load model weights:

```bash
.envs/planner/bin/navgpt check-data --config configs/experiments/vln/navgpt3_astra_r2r.yaml
.envs/planner/bin/navgpt check-data --config configs/experiments/vln/navgpt3_astra_rxr.yaml
.envs/planner/bin/navgpt check-data --config configs/experiments/tools/vla_only.yaml
```

The check requires dense ground truth for every episode and a `.glb`/`.navmesh`
pair for every scene, validates annotation structure, IDs and the RxR
language/role selection, and reports the episode count and a dataset
fingerprint. R2R `val_unseen` has 1,839 episodes. For multilingual RxR, set
`dataset.languages` to all four codes in a copy of the RxR config. These checks
establish data readiness, not navigation performance.

## Ground truth and evaluation protocol

The default is local **validation**, not a hidden-test leaderboard submission.
R2R's public test set does not expose goals/reference paths; RxR's
`test_challenge` bundle has no public dense-GT counterpart. Use `val_seen` or
`val_unseen` for local metrics. See the
[R2R release notes](https://jacobkrantz.github.io/vlnce/data) and
[RxR-Habitat task instructions](https://github.com/jacobkrantz/VLN-CE#rxr-habitat-challenge)
for official test submission procedures.

The evaluator requires a ground-truth path for every selected episode. A GT
file from another release, split, or annotation role is not interchangeable.
nDTW aligns the walked path to dense GT locations; a sparse `reference_path`
shown in a visualization does not supply equivalent metric supervision.
[Upstream metric implementation](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/measures.py).

Use 500 as the default environment-step limit for both datasets, as in the
[R2R task configuration](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/vlnce_task.yaml)
and the RxR English configuration above. Environment steps, Planner turns, and
VLA model calls are different budgets. Report changes to any of them alongside
sensor settings, turn angle, sliding behavior, checkpoint identity, split,
language/role filters, selected episode count, and scored count.

The NavGPT harness uses panoramic observations and waypoint execution. The
official RxR-Habitat challenge has its own restricted sensor/action protocol;
its organizers explicitly exclude unmodified panoramic waypoint agents. These
local harness results therefore must not be described as challenge submissions.

## Simulator compatibility and access terms

Use the separate **Python 3.8 / Habitat-Sim 0.1.7** environment in the
[installation guide](../README.md#installation). The
service calls Habitat-Sim directly; Habitat-Lab and VLN-CE baseline training
dependencies are not required. Rendered inference needs a compatible Linux GPU
driver and EGL/OpenGL context. A successful import or blind run does not test
rendered observations. The
[Habitat-Sim 0.1.7 installation reference](https://github.com/facebookresearch/habitat-sim/blob/v0.1.7/README.md#installation)
provides the matching source-build fallback.

Dataset permissions remain with the original providers. Matterport3D requires
its access agreement; VLN-CE documents terms for MP3D-derived task data, while
the original RxR project documents its annotation license. Consult
[VLN-CE's license section](https://github.com/jacobkrantz/VLN-CE#license),
[RxR's license section](https://github.com/google-research-datasets/RxR#license),
and the [Matterport3D Terms of Use](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf)
before redistributing assets. Keep licensed scene files, local credentials,
download archives, and model weights outside the source release.
