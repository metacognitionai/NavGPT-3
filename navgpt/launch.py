"""Multi-GPU evaluation from one experiment config.

    navgpt launch --config CFG [--gpus 0,1,2,3] [--episodes SPEC] [--keep-services]

Each entry of the config's ``gpus`` list hosts one Environment service and, when
the experiment uses the VLA, one VLA service, and runs one shard of the episodes
against that pair. Shard ``i`` listens on the ports of ``environment.url`` and
``vla.url`` plus ``i``. When every shard has finished, the shards are merged into
``<output.dir>/<experiment>`` exactly as ``navgpt merge`` does. Listing a GPU twice
places two shards on it.

A service already answering on a shard's port is reused when it serves this
experiment's settings; anything else on that port is an error. Services started
here are stopped on exit unless ``--keep-services`` is given. Running the command
again resumes every shard from its saved episodes.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from .settings import Settings, SettingsError

SERVICE_TIMEOUT_S = 2400   # loading several 8B checkpoints from network storage is slow
POLL_S = 5


def _get(url: str):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError):
        return None


def _shift(url: str, offset: int) -> str:
    parts = urlsplit(url)
    return "{}://{}:{}".format(parts.scheme or "http", parts.hostname, parts.port + offset)


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


def _environment_mismatch(health: dict, options: dict) -> str | None:
    """Why a running Environment cannot serve this experiment, or None."""
    if health.get("blind"):
        return "it renders synthetic pixels (--blind)"
    for key, want in (("dataset", options["dataset"]), ("split", options["split"]),
                      ("interface_variant", options["interface"])):
        if health.get(key) != want:
            return "{} is {!r}, this experiment needs {!r}".format(key, health.get(key), want)
    served = health.get("config") or {}
    for key, want in options["config"].items():
        if served.get(key) != want:
            return "{} is {!r}, this experiment needs {!r}".format(key, served.get(key), want)
    want_sha = (hashlib.sha256(Path(options["episodes_file"]).read_bytes()).hexdigest()
                if options["episodes_file"] else None)
    if health.get("episodes_sha256") != want_sha:
        return "it serves a different episode file"
    return None


def _vla_mismatch(info: dict, options: dict) -> str | None:
    want = os.path.basename(options["model_path"])
    if info.get("model_version") != want:
        return "it serves {!r}, this experiment needs {!r}".format(info.get("model_version"), want)
    for key in ("max_nav_vis_tokens", "frame_cache"):
        if key in options and key in info and info[key] != options[key]:
            return "{} is {!r}, this experiment needs {!r}".format(key, info[key], options[key])
    return None


class Service:
    """One Environment or VLA service: reused if already serving, else started."""

    def __init__(self, kind: str, shard: int, gpu: int, url: str):
        self.kind, self.shard, self.gpu, self.url = kind, shard, gpu, url
        self.proc: subprocess.Popen | None = None
        self.log: Path | None = None
        self.probe = url + ("/health" if kind == "environment" else "/info")

    def __str__(self):
        return "{} {} (GPU {}, {})".format(self.kind, self.shard, self.gpu, self.url)

    def start(self, python: str, config: Path, paths: str | None, logs: Path) -> None:
        self.log = logs / "{}_{}.log".format(self.kind, self.shard)
        cmd = [python, "-m", "navgpt." + self.kind, "--config", str(config),
               "--port", str(urlsplit(self.url).port), "--gpu", str(self.gpu)]
        if paths:
            cmd += ["--paths", paths]
        with open(self.log, "ab") as out:
            # Own session: a Ctrl-C reaches the shards, and the services are
            # stopped (or kept) deliberately afterwards.
            self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _services(settings: Settings, gpus: list[int], needs_vla: bool) -> list[Service]:
    env_url = "http://{host}:{port}".format(**settings.environment_options())
    services = [Service("environment", i, g, _shift(env_url, i)) for i, g in enumerate(gpus)]
    if needs_vla:
        vla_url = "http://{host}:{port}".format(**settings.vla_options())
        services += [Service("vla", i, g, _shift(vla_url, i)) for i, g in enumerate(gpus)]
    return services


def _bring_up(settings: Settings, services: list[Service], paths: str | None,
              logs: Path) -> None:
    env_options = settings.environment_options()
    vla_options = settings.vla_options() if any(s.kind == "vla" for s in services) else None
    missing = []
    for s in services:
        running = _get(s.probe)
        if running is None:
            missing.append(s)
            continue
        why = (_environment_mismatch(running, env_options) if s.kind == "environment"
               else _vla_mismatch(running, vla_options))
        if why:
            raise SettingsError("{} is already in use and cannot be reused: {}. Stop it, or "
                                "move the base port in the config's url.".format(s, why))
        print("[launch] reusing {}".format(s), flush=True)
    python = {kind: settings.interpreter(kind) for kind in {s.kind for s in missing}}
    for s in missing:
        s.start(python[s.kind], settings.source, paths, logs)
        print("[launch] starting {}; log {}".format(s, s.log), flush=True)

    deadline = time.time() + SERVICE_TIMEOUT_S
    waiting = [s for s in services if s.proc]
    while waiting:
        for s in list(waiting):
            if _get(s.probe) is not None:
                print("[launch] ready: {}".format(s), flush=True)
                waiting.remove(s)
            elif s.proc.poll() is not None:
                raise SettingsError("{} exited during startup; last lines of {}:\n{}".format(
                    s, s.log, _tail(s.log)))
        if waiting and time.time() > deadline:
            raise SettingsError("not ready after {} s: {}".format(
                SERVICE_TIMEOUT_S, ", ".join(map(str, waiting))))
        if waiting:
            time.sleep(POLL_S)


def _finished(output_root: Path, names: list[str]) -> int:
    done = 0
    for name in names:
        try:
            done += len(json.loads((output_root / name / "summary.json").read_text())["episodes"])
        except (OSError, ValueError, KeyError):
            pass
    return done


def _run_shards(settings: Settings, cfg, services: list[Service], gpus: list[int],
                args, logs: Path) -> list[str]:
    """Run every shard to completion; return the names of shards that failed."""
    count = len(gpus)
    url = {(s.kind, s.shard): s.url for s in services}
    workers, names = [], []
    for i in range(count):
        cmd = [sys.executable, "-m", "navgpt", "run", "--config", str(settings.source),
               "--env-url", url["environment", i]]
        if ("vla", i) in url:
            cmd += ["--vla-url", url["vla", i]]
        if count > 1:
            cmd += ["--shard", "{}/{}".format(i, count)]
        if args.episodes:
            cmd += ["--episodes", args.episodes]
        if args.paths:
            cmd += ["--paths", args.paths]
        name = cfg.run_name + ("_s{}".format(i) if count > 1 else "")
        log = logs / "shard_{}.log".format(i)
        with open(log, "ab") as out:
            workers.append((name, log, subprocess.Popen(cmd, stdout=out,
                                                        stderr=subprocess.STDOUT,
                                                        stdin=subprocess.DEVNULL)))
        names.append(name)
        print("[launch] shard {} on GPU {} -> {}; log {}".format(i, gpus[i], name, log),
              flush=True)

    output_root = Path(cfg.output_root)
    failed, last = [], -1
    try:
        while any(p.poll() is None for _, _, p in workers):
            done = _finished(output_root, names)
            if done != last:
                print("[launch] {} episodes finished".format(done), flush=True)
                last = done
            time.sleep(30)
    except BaseException:
        for _, _, p in workers:
            p.terminate()
        for _, _, p in workers:
            p.wait()
        raise
    for name, log, p in workers:
        if p.returncode:
            failed.append(name)
            print("[launch] {} failed (exit {}); last lines of {}:\n{}".format(
                name, p.returncode, log, _tail(log)), file=sys.stderr)
    return failed


def launch(settings: Settings, args) -> int:
    from .planner.config import needs_vla

    cfg = settings.run_config()
    gpus = args.gpus if args.gpus is not None else settings.gpus
    uses_vla = not settings.has_planner or needs_vla(cfg.condition)
    logs = Path(cfg.output_root) / ".launch" / cfg.run_name
    logs.mkdir(parents=True, exist_ok=True)
    # A hang-up or kill ends the launch the same way as Ctrl-C: shards and
    # started services are stopped.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))

    services = _services(settings, gpus, uses_vla)
    try:
        _bring_up(settings, services, args.paths, logs)
        failed = _run_shards(settings, cfg, services, gpus, args, logs)
    finally:
        started = [s for s in services if s.proc]
        if args.keep_services:
            if started:
                print("[launch] leaving {} service(s) running".format(len(started)), flush=True)
        else:
            for s in started:
                s.stop()

    if failed:
        print("[launch] {} of {} shard(s) failed; run the same command again to resume "
              "them".format(len(failed), len(gpus)), file=sys.stderr)
        return 1
    root = Path(cfg.output_root)
    if len(gpus) > 1:
        from .planner.merge import merge_runs
        shards = [root / "{}_s{}".format(cfg.run_name, i) for i in range(len(gpus))]
        summary = merge_runs(shards, root / cfg.run_name)
    else:
        summary = json.loads((root / cfg.run_name / "summary.json").read_text())
    print(json.dumps({"run": summary["run_name"], "shards": len(gpus),
                      "scored": summary["scored_count"], "excluded": summary["excluded_count"],
                      "aggregate": summary["aggregate"]}, indent=2))
    return 0
