"""The NavGPT VLA alone — no planner LLM and no MCP tools.

This is the reference for every Planner result that delegates to the VLA, and it
checks the VLA integration itself: view order, quaternion convention and
waypoint frame all show up here first. It makes no Planner calls.

It reuses the Planner runner's placement, `is_scored`, `aggregate` and summary
shape, so its output opens in the same viewer and is directly comparable to the
Planner experiments.

    navgpt run --config configs/experiments/tools/vla_only.yaml
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from .config import RunConfig
from .runner import (EventSink, _write_summary, call_function,
                     format_episodes, is_scored, panel_action, panel_field,
                     PreflightError, select_episodes)


def _reset(cfg: RunConfig) -> None:
    r = requests.post(cfg.vla_url.rstrip("/") + "/reset",
                      json={"success": None}, timeout=300)
    r.raise_for_status()


def run_episode(cfg: RunConfig, index: int, run_dir: Path) -> dict:
    from .tools.vla_client import NavGPTVLAClient, pose_unchanged

    url = cfg.env_url
    panel_field(url, "episode_index", index)
    panel_action(url, "play")
    ep = call_function(url, "{}__reset".format(cfg.verb_prefix), {"trigger": "planner"})
    instruction = str(ep.get("instruction") or "")

    vla = NavGPTVLAClient(cfg.vla_url)
    _reset(cfg)

    sink = EventSink(run_dir / "episode_{}.jsonl".format(index))
    metrics: dict = {}
    steps = 0
    stopped = False
    prev_pose, stuck = None, False
    t0 = time.time()
    try:
        sink.emit("episode_meta", {
            "index": index, "episode_id": ep.get("episode_id"),
            "scene_id": ep.get("scene_id"), "instruction": instruction,
            "geodesic_distance": ep.get("geodesic_distance"),
            "dataset": cfg.dataset, "language": ep.get("language"),
            "instruction_id": ep.get("instruction_id"), "annotation_role": ep.get("annotation_role"),
        })
        sink.emit("session_inputs", {"condition": "vla_only", "model": "navgpt-vla",
                                     "vla_url": cfg.vla_url})

        # NavGPT VLA never emits STOP, so the loop owns stopping entirely:
        # two consecutive collapsed-horizon reads, exactly as VLNCE-EVAL's
        # episode_runner does it.
        for _ in range(cfg.step_budget):
            if time.time() - t0 > cfg.episode_timeout:
                raise TimeoutError("VLA episode exceeded planner.episode_timeout_s")
            pano = call_function(url, "{}__observe_pano".format(cfg.verb_prefix), {})
            views, pose = pano.get("raw_views") or {}, pano.get("pose") or {}
            advice = vla.act(
                views=views, instruction=instruction,
                episode_id=str(ep.get("episode_id") or index),
                position=pose.get("position") or [0.0, 0.0, 0.0],
                rotation_wxyz=pose.get("rotation_wxyz") or [1.0, 0.0, 0.0, 0.0],
                is_stuck=stuck,
            )
            if advice.get("should_stop"):
                out = call_function(url, "{}__step_discrete".format(cfg.verb_prefix),
                                    {"action": 0})
                steps += 1
                stopped = True
                sink.emit("tool_use", {"name": "vla_stop", "input": {}})
                break

            target, rot = advice.get("position"), advice.get("rotation_xyzw")
            if target is None or rot is None:
                raise ValueError("VLA response is missing the predicted position or rotation")
            res = call_function(url, "{}__teleport".format(cfg.verb_prefix),
                                {"position": target, "rotation": rot,
                                 "theta": advice.get("theta", 0.0), "is_stuck": stuck})
            steps += 1
            stuck = pose_unchanged(prev_pose, res.get("pose"))
            prev_pose = res.get("pose")
            sink.emit("tool_use", {"name": "vla_step",
                                   "input": {"navigable": res.get("navigable")}})
            if res.get("terminated") or res.get("truncated"):
                break

        try:
            out = call_function(url, "{}__evaluate".format(cfg.verb_prefix),
                                {"trigger": "planner"})
            metrics = out.get("metrics") or {}
        except Exception as exc:  # noqa: BLE001
            sink.emit("driver_error", {"error": "evaluate failed: {!r}".format(exc)})
        sink.emit("episode_metrics", {"metrics": metrics})

        try:
            traj = call_function(url, "{}__trajectory".format(cfg.verb_prefix),
                                 {"trigger": "planner"})
            sink.emit("episode_trajectory", traj)
        except Exception as exc:  # noqa: BLE001
            sink.emit("driver_error", {"error": "trajectory failed: {!r}".format(exc)})
    finally:
        wall = sink.elapsed
        sink.close()

    return {
        "index": index,
        "episode_id": ep.get("episode_id"),
        "scene_id": ep.get("scene_id"),
        "dataset": cfg.dataset, "language": ep.get("language"),
        "instruction_id": ep.get("instruction_id"), "annotation_role": ep.get("annotation_role"),
        "instruction": instruction,
        "metrics": metrics,
        "agent": {
            "turns": steps,          # one VLA step == one "turn" here
            "tool_calls": {"vla_step": steps},
            "env_steps": steps,
            "called_stop": stopped,
            "end_reason": "stop_called" if stopped else "budget_or_truncated",
            "cost_usd": 0.0,         # no LLM in this condition
            "usage": {},
        },
        "error": None,
        "subtype": "vla_only",
        "wall_s": round(wall, 1),
        "status": "completed",
    }


def run(cfg: RunConfig) -> dict:
    run_dir = (Path(cfg.output_root) / cfg.resolved_run_name()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    records: dict[int, dict] = {}
    if summary_path.exists():
        try:
            prior = json.loads(summary_path.read_text())
            records = {int(e["index"]): e for e in prior.get("episodes", [])}
            print("[vla_only] resuming: {} existing record(s)".format(len(records)))
        except (ValueError, KeyError):
            pass

    try:
        health = requests.get(cfg.env_url.rstrip("/") + "/health", timeout=30).json()
        info = requests.get(cfg.vla_url.rstrip("/") + "/info", timeout=30).json()
    except (requests.RequestException, ValueError) as exc:
        raise PreflightError("Environment ({}) or VLA ({}) unreachable: {}".format(
            cfg.env_url, cfg.vla_url, exc)) from exc
    if health.get("split") != cfg.split:
        raise PreflightError(
            "NavGPT Environment service serves split {!r} but this run is configured for {!r}".format(
                health.get("split"), cfg.split))
    if health.get("dataset") and health["dataset"] != cfg.dataset:
        raise PreflightError("Environment dataset {!r} does not match {!r}".format(health["dataset"], cfg.dataset))
    if (not health.get("blind") and health.get("gt_covered") is not None
            and health["gt_covered"] < health.get("episode_count", 0)):
        raise PreflightError("dense ground truth covers {} of {} episodes".format(
            health["gt_covered"], health.get("episode_count")))
    indices = select_episodes(cfg, int(health.get("episode_count", 0)))
    cfg.dataset_info = {key: health.get(key) for key in (
        "dataset", "split", "languages", "roles", "dataset_fingerprint",
        "episode_count", "gt_covered", "episode_sources", "gt_sources", "caliber")}
    print("[NavGPT VLA] service: {}".format(info))
    print("[vla_only] episodes {}".format(format_episodes(indices)))

    _write_summary(summary_path, cfg, records)
    kept = [i for i in indices if i in records and is_scored(records[i])]
    indices = [i for i in indices if i not in kept]
    if kept:
        print("[vla_only] keeping {} scored episode(s)".format(len(kept)))
    for n, index in enumerate(indices, 1):
        print("[vla_only] --- episode {} ({}/{}) ---".format(index, n, len(indices)))
        try:
            rec = run_episode(cfg, index, run_dir)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
            print("[vla_only]   ERROR: {!r}".format(exc))
            rec = {"index": index, "metrics": {}, "agent": {"env_steps": 0},
                   "error": repr(exc), "status": "errored"}
        records[index] = rec
        m = rec.get("metrics") or {}
        print("[vla_only]   SR={} SPL={} nDTW={} NE={} steps={}".format(
            m.get("success"), m.get("spl"), m.get("ndtw"),
            m.get("distance_to_goal"), (rec.get("agent") or {}).get("env_steps")))
        _write_summary(summary_path, cfg, records)

    summary = _write_summary(summary_path, cfg, records)
    print("\n[vla_only] === aggregate over {} scored / {} total ===".format(
        summary["scored_count"], summary["aggregate"]["episode_count"]))
    for k, v in sorted(summary["aggregate"].items()):
        print("  {:<20} {}".format(k, v))
    print("\n[vla_only] artifacts: {}".format(run_dir))
    return summary
