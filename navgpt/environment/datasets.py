"""R2R-CE and RxR-CE episode/ground-truth loading without Habitat dependencies."""

import gzip
import hashlib
import json
import math
from pathlib import Path


RXR_LANGUAGES = ("en-US", "en-IN", "hi-IN", "te-IN")
RXR_ROLES = ("guide", "follower")


def selection(value, allowed, default):
    values = value.split(",") if isinstance(value, str) else list(value or default)
    values = [v.strip() for v in values]
    if values in (["all"], ["*"]):
        return list(allowed)
    if not values or len(values) != len(set(values)) or not set(values) <= set(allowed):
        raise ValueError("Expected a comma-separated selection from {} or all".format(allowed))
    return values


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(str(path), "rt", encoding="utf-8") as stream:
        return json.load(stream)


def vector(value, size):
    return (isinstance(value, list) and len(value) == size and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value))


def load_episode_set(data_root=None, split="val_unseen", dataset="r2r",
                     episodes_file=None, gt_file=None, gt_split=None,
                     languages=None, roles=None):
    """Preserve official episode order; index GT by episode_id, never instruction_id.

    The default RxR scope follows VLN-CE's English guide configuration. Missing
    GT is recorded so interactive plumbing can run; scored benchmark evaluation
    requires full coverage. A configured ground-truth file that does not exist is
    always an error.
    """
    if dataset not in ("r2r", "rxr"):
        raise ValueError("dataset must be r2r or rxr")
    language_list = selection(languages, RXR_LANGUAGES, ("en-US", "en-IN")) if dataset == "rxr" else []
    role_list = selection(roles, RXR_ROLES, ("guide",)) if dataset == "rxr" else []
    if not data_root and not episodes_file:
        raise ValueError("data_root or episodes_file is required")
    if episodes_file and dataset == "rxr" and len(role_list) != 1:
        raise ValueError("An explicit RxR episode file must select exactly one annotation role")
    root = Path(data_root) if data_root else None
    stems = [split + "_" + role for role in role_list] if dataset == "rxr" else [split]
    episode_paths = ([Path(episodes_file)] if episodes_file else
                     [root / split / (stem + ".json.gz") for stem in stems])
    parent = gt_split or split
    gt_stems = [parent + "_" + role for role in role_list] if dataset == "rxr" else [parent]
    if gt_file:
        gt_paths = [Path(gt_file)]
    elif root:
        gt_paths = [root / parent / (stem + "_gt.json.gz") for stem in gt_stems]
    else:
        source = Path(episodes_file)
        stem = source.name[:-3] if source.name.endswith(".gz") else source.name
        stem = stem[:-5] if stem.endswith(".json") else stem
        gt_paths = [source.with_name(stem + "_gt.json.gz")]

    episodes = []
    seen = set()
    for index, path in enumerate(episode_paths):
        data = read_json(path)
        entries = data.get("episodes") if isinstance(data, dict) else data
        if not isinstance(entries, list) or not entries:
            raise ValueError("No episodes in {}".format(path))
        for original in entries:
            ep = dict(original)
            instruction = ep.get("instruction") or {}
            if dataset == "rxr":
                if instruction.get("language") not in RXR_LANGUAGES:
                    raise ValueError("RxR episode {} has missing/unknown language".format(ep.get("episode_id")))
                if instruction["language"] not in language_list:
                    continue
                ep["annotation_role"] = role_list[index]
            if "episode_id" not in ep:
                raise ValueError("Episode without episode_id in {}".format(path))
            ep["episode_id"] = str(ep["episode_id"])
            if ep["episode_id"] in seen:
                raise ValueError("Duplicate episode_id {} in selected data".format(ep["episode_id"]))
            seen.add(ep["episode_id"])
            if not vector(ep.get("start_position"), 3) or not vector(ep.get("start_rotation"), 4):
                raise ValueError("Invalid start pose for episode {}".format(ep["episode_id"]))
            if ep.get("goals") and any(not vector(goal.get("position"), 3) for goal in ep["goals"]):
                raise ValueError("Invalid goal position for episode {}".format(ep["episode_id"]))
            if not isinstance(instruction.get("instruction_text"), str) or not instruction["instruction_text"].strip():
                raise ValueError("Missing instruction_text for episode {}".format(ep["episode_id"]))
            scene = str(ep.get("scene_id") or "")
            prefix = "data/scene_datasets/"
            if scene.startswith(prefix):
                scene = scene[len(prefix):]
            if not scene or Path(scene).is_absolute() or ".." in Path(scene).parts:
                raise ValueError("Expected a scene_id relative to scene_root: {!r}".format(scene))
            ep["scene_id"] = scene
            episodes.append(ep)
    if not episodes:
        raise ValueError("No episodes match the selected dataset/languages/roles")

    ground_truth = {}
    for path in gt_paths:
        if not path.is_file() and not gt_file:
            continue
        raw = read_json(path)
        if not isinstance(raw, dict):
            raise ValueError("GT must map episode_id to locations: {}".format(path))
        for key, value in raw.items():
            locations = value.get("locations", []) if isinstance(value, dict) else value
            if locations and (not isinstance(locations, list) or any(not vector(p, 3) for p in locations)):
                raise ValueError("Invalid GT locations for episode {}".format(key))
            key = str(key)
            if key in ground_truth and ground_truth[key] != locations:
                raise ValueError("Conflicting GT for episode {}".format(key))
            ground_truth[key] = locations or []
    covered = sum(bool(ground_truth.get(ep["episode_id"])) for ep in episodes)
    # Binds ordering, instructions, positions and scoring references to results.
    fingerprint = hashlib.sha256(json.dumps(
        {"episodes": episodes, "gt": {ep["episode_id"]: ground_truth.get(ep["episode_id"]) for ep in episodes}},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    return {
        "episodes": episodes, "ground_truth": ground_truth,
        "dataset": dataset, "split": split, "languages": language_list, "roles": role_list,
        "episode_count": len(episodes), "gt_covered": covered,
        "episode_sources": [str(p.resolve()) for p in episode_paths],
        "gt_sources": [str(p.resolve()) for p in gt_paths],
        "dataset_fingerprint": fingerprint,
    }


def check_dataset(dataset, data_root, scene_root, split="val_unseen", episodes_file=None,
                  gt_split=None, gt_file=None, languages=None, roles=None):
    """Check episodes, dense GT and scene files without starting Habitat.

    Returns the dataset summary (counts, sources, fingerprint); raises ValueError
    with the first problem found.
    """
    result = load_episode_set(data_root, split, dataset, episodes_file=episodes_file,
                              gt_file=gt_file, gt_split=gt_split,
                              languages=languages, roles=roles)
    if result["gt_covered"] != result["episode_count"]:
        raise ValueError("Dense GT covers {}/{} episodes".format(
            result["gt_covered"], result["episode_count"]))
    if any(not ep.get("goals") for ep in result["episodes"]):
        raise ValueError("Local evaluation requires goal annotations; use a validation split")
    missing = []
    for scene in sorted({ep["scene_id"] for ep in result["episodes"]}):
        mesh = Path(scene_root) / scene
        for path in (mesh, mesh.with_suffix(".navmesh")):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise ValueError("Missing {} scene files; first paths: {}".format(
            len(missing), ", ".join(missing[:5])))
    return {k: v for k, v in result.items() if k not in ("episodes", "ground_truth")}
