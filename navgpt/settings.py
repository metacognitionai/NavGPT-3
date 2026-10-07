"""Experiment settings: one YAML file drives the Environment, the VLA and the Planner.

An experiment file is self-contained: up to six sections (dataset, environment,
vla, planner, harness, output), the ``gpus`` used by ``navgpt launch``, and an
optional ``name``. Omitting ``planner`` runs the VLA alone.

Machine-specific paths live in ``configs/paths.yaml`` (copied from
``configs/paths.example.yaml``): dataset roots, scene assets, curated episode
files, checkpoint directories and the service interpreters. Experiments refer to
those entries by key.

Every key is validated, so a misspelled setting fails instead of being ignored.
This module needs only the standard library and PyYAML, so all three component
environments can load it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "configs" / "experiments"

KEYS = {
    "dataset": {"name", "split", "episodes", "episodes_file", "ground_truth_split",
                "ground_truth_file", "languages", "roles"},
    "environment": {"url", "max_steps", "forward_view_px", "panorama_view_px",
                    "turn_angle_deg", "step_size_m", "allow_sliding"},
    "vla": {"url", "checkpoint", "visual_token_budget", "frame_cache",
            "gpu_memory_fraction", "seed"},
    "planner": {"runtime", "model", "effort", "service_tier", "tools", "max_turns",
                "episode_timeout_s", "max_budget_usd"},
    "harness": {"interface", "review_every", "route_finding", "vla_steps_per_call"},
    "output": {"dir", "save_frames"},
}
DOCS = "docs/USAGE.md#experiment-configs"
TOP_LEVEL = set(KEYS) | {"name", "gpus"}
EFFORTS = {"claude": ("low", "medium", "high", "xhigh", "max"),
           "codex": ("none", "minimal", "low", "medium", "high", "xhigh")}
PATH_GROUPS = {"data", "episodes", "checkpoints", "codex_bin", "python"}
# Service interpreters used by `navgpt launch` unless paths.yaml sets `python`.
INTERPRETERS = {"environment": ".envs/habitat/bin/python", "vla": ".envs/vla/bin/python"}


class SettingsError(ValueError):
    pass


def _read(path: Path) -> dict:
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise SettingsError("config file not found: {}".format(path)) from None
    if not isinstance(data, dict):
        raise SettingsError("{} must contain a mapping".format(path))
    return data


def _validate(data: dict, source: Path) -> None:
    unknown = set(data) - TOP_LEVEL
    if unknown:
        raise SettingsError("{}: unknown section(s) {}".format(source, sorted(unknown)))
    for section, allowed in KEYS.items():
        value = data.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise SettingsError("{}: '{}' must be a mapping".format(source, section))
        extra = set(value) - allowed
        if extra:
            raise SettingsError("{}: unknown {} key(s) {}; allowed: {} (see {})".format(
                source, section, sorted(extra), sorted(allowed), DOCS))
    if "gpus" in data:
        check_gpus(data["gpus"], "{}: gpus".format(source))


def check_gpus(gpus, what: str = "gpus") -> list[int]:
    if (not isinstance(gpus, list) or not gpus
            or any(isinstance(g, bool) or not isinstance(g, int) or g < 0 for g in gpus)):
        raise SettingsError("{} must be a non-empty list of GPU indices, e.g. [0, 1, 2, 3]"
                            .format(what))
    return gpus


def find_paths_file(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SettingsError("paths file not found: {}".format(path))
        return path
    for candidate in (Path.cwd() / "configs" / "paths.yaml", REPO_ROOT / "configs" / "paths.yaml"):
        if candidate.is_file():
            return candidate
    return None


def load_paths(explicit: str | Path | None = None) -> dict:
    path = find_paths_file(explicit)
    if path is None:
        return {}
    data = _read(path)
    unknown = set(data) - PATH_GROUPS
    if unknown:
        raise SettingsError("{}: unknown key(s) {}; allowed: {}".format(
            path, sorted(unknown), sorted(PATH_GROUPS)))
    return data


def load(config: str | Path, paths: str | Path | None = None) -> "Settings":
    source = Path(config).resolve()
    data = _read(source)
    _validate(data, source)
    return Settings(source=source, data=data, paths=load_paths(paths))


# The eight-tool NavGPT-3 harness: the default when planner.tools is omitted.
ALL_TOOLS = ["observe_forward", "observe_panorama", "observe_map", "navigate_by_instruction",
             "navigate_relative", "navigate_to_node", "annotate_node", "terminate_episode"]


def tool_set(tools) -> tuple[str, str]:
    """Map a configured tool list to the internal (condition, capability_bundle)
    that registers exactly those tools and briefs the Planner on them."""
    from .planner.runner import EXPECTED_TOOLS
    from .planner.tools.ablation import CAPABILITIES

    known: dict = {frozenset(v): (k, "full") for k, v in EXPECTED_TOOLS.items()}
    for bundle, names in CAPABILITIES.items():
        known.setdefault(frozenset(names), ("planner_vla_memory", bundle))
    every = sorted(set().union(*known))
    unknown = sorted(set(tools) - set(every))
    if unknown:
        raise SettingsError("unknown tool(s) {}; tools are: {}".format(unknown, ", ".join(every)))
    match = known.get(frozenset(tools))
    if match is None:
        options = "\n".join("  - " + ", ".join(sorted(s)) for s in
                            sorted(known, key=lambda s: (len(s), sorted(s))))
        raise SettingsError("planner.tools is not a supported combination; use one of:\n" + options)
    return match


def _url(value: str) -> tuple[str, int]:
    parts = urlsplit(value)
    if not parts.hostname or not parts.port:
        raise SettingsError("service url needs a host and port: {!r}".format(value))
    return parts.hostname, parts.port


@dataclass
class Settings:
    source: Path
    data: dict
    paths: dict

    def section(self, name: str) -> dict:
        return dict(self.data.get(name) or {})

    @property
    def name(self) -> str:
        if self.data.get("name"):
            return str(self.data["name"])
        try:
            rel = self.source.with_suffix("").relative_to(EXPERIMENTS_DIR)
        except ValueError:
            return self.source.stem
        return "_".join(rel.parts)

    @property
    def has_planner(self) -> bool:
        return bool(self.data.get("planner"))

    @property
    def gpus(self) -> list[int]:
        """One shard per entry; each hosts an Environment and a VLA service."""
        return list(self.data.get("gpus") or [0])

    def interpreter(self, service: str) -> str:
        """The Python that runs `navgpt.environment` or `navgpt.vla`."""
        table = self.paths.get("python") or {}
        if not isinstance(table, dict):
            raise SettingsError("python in configs/paths.yaml must map environment and vla "
                                "to interpreters")
        python = Path(table.get(service) or INTERPRETERS[service])
        if not python.is_absolute():
            python = REPO_ROOT / python
        if not python.is_file():
            raise SettingsError(
                "no Python for the {} service at {}: install it as in the README, or set "
                "python.{} in configs/paths.yaml".format(service, python, service))
        return str(python)

    def lookup(self, group: str, key: str, what: str) -> str:
        """Resolve a paths.yaml entry by key; values containing a slash are paths."""
        if "/" in str(key):
            return str(key)
        table = self.paths.get(group) or {}
        if key not in table or not table[key]:
            raise SettingsError(
                "{} '{}' is not set: add it under '{}' in configs/paths.yaml "
                "(see configs/paths.example.yaml)".format(what, key, group))
        return str(table[key])

    # ── Planner / VLA-only run ──

    def run_config(self):
        from .planner.config import (DEFAULT_CODEX_MODEL, DEFAULT_MODEL, INTERFACES,
                                     RunConfig, VLA_ONLY)

        ds, env, vla = self.section("dataset"), self.section("environment"), self.section("vla")
        planner, harness, output = (self.section("planner"), self.section("harness"),
                                    self.section("output"))
        cfg = RunConfig(
            dataset=str(ds.get("name", "r2r")),
            split=str(ds.get("split", "val_unseen")),
            episodes=str(ds.get("episodes", "all")),
            env_url=str(env.get("url", RunConfig.env_url)),
            vla_url=str(vla.get("url", RunConfig.vla_url)),
            step_budget=int(env.get("max_steps", RunConfig.step_budget)),
            ablation_seed=int(vla.get("seed", RunConfig.ablation_seed)),
            run_name=self.name,
            output_root=str(output.get("dir", RunConfig.output_root)),
            live=bool(output.get("save_frames", False)),
        )
        if not self.has_planner:
            cfg.condition = VLA_ONLY
            cfg.max_budget_usd = None
            return cfg

        runtime = str(planner.get("runtime", "claude"))
        if runtime not in ("claude", "codex"):
            raise SettingsError("planner.runtime must be claude or codex, not {!r}".format(runtime))
        cfg.planner_runtime = runtime
        cfg.model = str(planner.get("model") or (DEFAULT_CODEX_MODEL if runtime == "codex"
                                                 else DEFAULT_MODEL))
        cfg.effort = planner.get("effort")
        if cfg.effort is not None and cfg.effort not in EFFORTS[runtime]:
            raise SettingsError("planner.effort for the {} runtime must be one of {}".format(
                runtime, ", ".join(EFFORTS[runtime])))
        cfg.max_turns = int(planner.get("max_turns", cfg.max_turns))
        cfg.episode_timeout = int(planner.get("episode_timeout_s", cfg.episode_timeout))
        budget = planner.get("max_budget_usd", 5.0 if runtime == "claude" else None)
        cfg.max_budget_usd = float(budget) if budget else None
        cfg.tools = list(planner.get("tools") or ALL_TOOLS)
        cfg.condition, cfg.capability_bundle = tool_set(cfg.tools)

        cfg.interface_variant = str(harness.get("interface", "standard"))
        if cfg.interface_variant not in INTERFACES:
            raise SettingsError("harness.interface must be one of {}".format(", ".join(INTERFACES)))
        review_every = harness.get("review_every", 0)
        if isinstance(review_every, bool) or not isinstance(review_every, int) or review_every < 0:
            raise SettingsError("harness.review_every must be a whole number of VLA steps (0: never)")
        cfg.review_every = review_every
        study = cfg.capability_bundle != "full" or cfg.review_every
        if study and cfg.interface_variant == "standard":
            raise SettingsError(
                "this tool set or harness.review_every uses the controlled-study briefing; "
                "set harness.interface to reference (or one of its variants)")
        if cfg.interface_variant != "standard" and cfg.condition != "planner_vla_memory":
            raise SettingsError("study interfaces need the eight-tool VLA tool set "
                                "or one of its reduced sets")
        if "route_finding" in harness:
            cfg.move_routes = bool(harness["route_finding"])
        if "vla_steps_per_call" in harness:
            cfg.vla_max_steps_per_call = int(harness["vla_steps_per_call"])
        return cfg

    def provider_env(self) -> dict:
        """Process environment for the Planner runtime (Codex CLI location, tier)."""
        env = {}
        if self.paths.get("codex_bin"):
            env["NAVGPT_CODEX_BIN"] = str(self.paths["codex_bin"])
        tier = self.section("planner").get("service_tier")
        if tier:
            env["NAVGPT_CODEX_SERVICE_TIER"] = str(tier)
        return env

    # ── services ──

    def environment_options(self) -> dict:
        ds, env = self.section("dataset"), self.section("environment")
        name = str(ds.get("name", "r2r"))
        if name not in ("r2r", "rxr"):
            raise SettingsError("dataset.name must be r2r or rxr")
        host, port = _url(str(env.get("url", "http://127.0.0.1:9200")))
        episodes_file = ds.get("episodes_file")
        if episodes_file:
            episodes_file = self.lookup("episodes", episodes_file, "episode file")
        config = {"max_steps": int(env.get("max_steps", 500))}
        for key, target in (("forward_view_px", "rgb_size"), ("panorama_view_px", "pano_size"),
                            ("turn_angle_deg", "turn_angle_deg"),
                            ("step_size_m", "step_size_m"), ("allow_sliding", "allow_sliding")):
            if key in env:
                config[target] = env[key]
        listify = lambda v: ",".join(v) if isinstance(v, (list, tuple)) else v
        return {
            "dataset": name,
            "split": str(ds.get("split", "val_unseen")),
            "data_root": None if episodes_file and not ds.get("ground_truth_split") else
                         self.lookup("data", name, "dataset root"),
            "scene_root": self.lookup("data", "scenes", "scene root"),
            "episodes_file": episodes_file,
            "gt_split": ds.get("ground_truth_split"),
            "gt_file": ds.get("ground_truth_file"),
            "languages": listify(ds.get("languages", ["en-US", "en-IN"])),
            "roles": listify(ds.get("roles", ["guide"])),
            "interface": str(self.section("harness").get("interface", "standard")),
            "host": host,
            "port": port,
            "config": config,
        }

    def vla_options(self) -> dict:
        vla, output = self.section("vla"), self.section("output")
        host, port = _url(str(vla.get("url", "http://127.0.0.1:8000")))
        options = {
            "model_path": self.lookup("checkpoints", vla.get("checkpoint", "navgpt3-8b"),
                                      "checkpoint"),
            "host": host,
            "port": port,
            "result_path": str(Path(output.get("dir", "outputs")) / ".vla_service"),
        }
        for key, target in (("visual_token_budget", "max_nav_vis_tokens"),
                            ("frame_cache", "frame_cache"),
                            ("gpu_memory_fraction", "gpu_memory_fraction")):
            if key in vla:
                options[target] = vla[key]
        return options
