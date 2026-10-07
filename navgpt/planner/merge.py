"""Merge sharded run directories into one summary, as if one worker had run them.

A sharded run writes ``<name>_s0`` … ``<name>_s<N-1>``. The merged summary uses the
same scoring rule (runner.is_scored) and aggregate as an unsharded run, and the
episode files and live frames are copied so the viewer can open the result.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .runner import aggregate, is_scored


def merge_runs(shards: list[Path], out: Path) -> dict:
    by_index: dict[int, dict] = {}
    overlap: set[int] = set()
    config = None
    sources = []
    for shard in shards:
        path = shard / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text())
        config = config or summary.get("config")
        sources.append(shard)
        for episode in summary.get("episodes") or []:
            index = int(episode["index"])
            if index in by_index:
                overlap.add(index)
            by_index[index] = episode
    if not by_index:
        raise ValueError("no shard summaries found among: {}".format(
            ", ".join(map(str, shards)) or "(none)"))

    episodes = [by_index[k] for k in sorted(by_index)]
    scored = [e for e in episodes if is_scored(e)]
    config = dict(config or {}, run_name=out.name, shard=None)
    merged = {
        "run_name": out.name,
        "config": config,
        "merged_from": [str(p) for p in sources],
        "episodes": episodes,
        "scored_count": len(scored),
        "excluded_count": len(episodes) - len(scored),
        "aggregate": aggregate(scored),
        "overlapping_indices": sorted(overlap),
    }
    out.mkdir(parents=True, exist_ok=True)
    for shard in sources:
        for path in shard.glob("episode_*.jsonl"):
            shutil.copy2(path, out / path.name)
        live = shard / "live"
        if live.is_dir():
            for episode_dir in live.iterdir():
                target = out / "live" / episode_dir.name
                if not target.exists():
                    shutil.copytree(episode_dir, target)
    (out / "summary.json").write_text(json.dumps(merged, indent=2))
    return merged
