"""The ``navgpt`` command.

    navgpt launch --config CFG [--gpus LIST] [--episodes SPEC] [--keep-services]
    navgpt run --config CFG [--episodes SPEC] [--shard I/N]
    navgpt preflight --config CFG
    navgpt merge --config CFG
    navgpt view [--output-root DIR]
    navgpt check-data --config CFG
    navgpt check-checkpoint --config CFG

``navgpt launch`` starts the Environment and VLA services on the experiment's
GPUs, runs one shard per GPU and merges them. ``navgpt run`` runs against services
started separately in their own Python environments (``python -m
navgpt.environment`` and ``python -m navgpt.vla``) from the same experiment config.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .settings import SettingsError, check_gpus, load


def _shard(value: str) -> tuple[int, int]:
    try:
        index, count = (int(x) for x in value.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected I/N, e.g. 0/8") from None
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("shard index must be in [0, N)")
    return index, count


def _gpus(value: str) -> list[int]:
    """'0,1,2,3', '0-3' or '0-3,6' -> GPU indices."""
    gpus = []
    try:
        for part in value.split(","):
            lo, _, hi = part.partition("-")
            gpus += range(int(lo), int(hi or lo) + 1)
        return check_gpus(gpus)
    except (ValueError, SettingsError):
        raise argparse.ArgumentTypeError("expected GPU indices such as 0,1,2,3 or 0-7") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navgpt", description="NavGPT-3 evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    def experiment(p):
        p.add_argument("--config", required=True, help="experiment YAML file")
        p.add_argument("--paths", help="machine paths file (default: configs/paths.yaml)")

    launch = sub.add_parser("launch", help="start services on the experiment's GPUs, "
                                           "run one shard per GPU and merge")
    experiment(launch)
    launch.add_argument("--gpus", type=_gpus, help="override gpus, e.g. 0-3 or 0,2,4,6")
    launch.add_argument("--episodes", help="index spec overriding dataset.episodes, e.g. 0-9")
    launch.add_argument("--keep-services", action="store_true",
                        help="leave the services running for the next launch")

    run = sub.add_parser("run", help="run (or resume) an experiment")
    experiment(run)
    run.add_argument("--episodes", help="index spec overriding dataset.episodes, e.g. 0-9")
    run.add_argument("--shard", type=_shard, help="run shard I of N (interleaved episodes)")
    run.add_argument("--env-url", help="override environment.url (one per shard)")
    run.add_argument("--vla-url", help="override vla.url (one per shard)")

    experiment(sub.add_parser("preflight", help="check services, tools and model; spend nothing"))
    experiment(sub.add_parser("merge", help="merge the shards of a sharded run"))
    experiment(sub.add_parser("check-data", help="check episodes, dense GT and scene files"))
    experiment(sub.add_parser("check-checkpoint", help="check the configured NavGPT3 checkpoint"))

    view = sub.add_parser("view", help="browse results in a browser (read-only)")
    view.add_argument("--output-root", default="outputs")
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8080)
    view.add_argument("--run", help="print this run's link on startup")
    return parser


def _report(check, *args, **kwargs) -> int:
    """Run a read-only check and print its JSON summary or its first problem."""
    try:
        result = check(*args, **kwargs)
    except (ValueError, OSError) as exc:
        print("check failed: {}".format(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def _runtime(cfg):
    if cfg.planner_runtime == "codex":
        from .planner.runtimes.codex import CodexPlannerRuntime
        return CodexPlannerRuntime()
    from .planner.runtimes.claude import ClaudePlannerRuntime
    return ClaudePlannerRuntime()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "view":
        from .planner.viewer import serve
        view_args = ["--port", str(args.port), "--host", args.host,
                     "--output-root", args.output_root]
        serve(view_args + (["--run", args.run] if args.run else []))
        return 0

    try:
        settings = load(args.config, args.paths)

        if args.command == "check-data":
            from .environment.datasets import check_dataset
            options = settings.environment_options()
            return _report(check_dataset,
                options["dataset"], options["data_root"], options["scene_root"],
                split=options["split"], episodes_file=options["episodes_file"],
                gt_split=options["gt_split"], gt_file=options["gt_file"],
                languages=options["languages"], roles=options["roles"])

        if args.command == "check-checkpoint":
            from .vla.checkpoint import validate_checkpoint
            return _report(validate_checkpoint, Path(settings.vla_options()["model_path"]))

        if args.command == "launch":
            from .launch import launch
            return launch(settings, args)

        cfg = settings.run_config()
        if args.command == "merge":
            from .planner.merge import merge_runs
            root = Path(cfg.output_root)
            shards = sorted(root.glob(cfg.resolved_run_name() + "_s[0-9]*"))
            try:
                merged = merge_runs(shards, root / cfg.resolved_run_name())
            except ValueError as exc:
                raise SettingsError(str(exc)) from None
            print(json.dumps({"run": merged["run_name"], "shards": len(merged["merged_from"]),
                              "scored": merged["scored_count"],
                              "excluded": merged["excluded_count"],
                              "aggregate": merged["aggregate"]}, indent=2))
            return 0

        if args.command == "run":
            if args.episodes:
                cfg.episodes = args.episodes
            if args.shard:
                cfg.shard = args.shard
            if args.env_url:
                cfg.env_url = args.env_url
            if args.vla_url:
                cfg.vla_url = args.vla_url

        if not settings.has_planner and args.command == "preflight":
            raise SettingsError("preflight checks the Planner; this experiment has none")
        os.environ.update(settings.provider_env())
        from .planner.runner import PreflightError, preflight, run_eval
        try:
            if not settings.has_planner:
                from .planner.vla_only import run as run_vla_only
                run_vla_only(cfg)
            elif args.command == "preflight":
                print(json.dumps(preflight(cfg), indent=2))
            else:
                asyncio.run(run_eval(_runtime(cfg), cfg))
        except PreflightError as exc:
            print("\n[NavGPT Planner] PREFLIGHT FAILED\n  {}\n".format(exc), file=sys.stderr)
            return 2
        return 0
    except SettingsError as exc:
        print("navgpt: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nnavgpt: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
