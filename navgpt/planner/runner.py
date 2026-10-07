"""Episode loop, event sink, and scoring.

The planner runner owns the two things the agent must never see: which episode is
seated, and what it scored. Both ride HTTP calls the agent has no access to.

Main pieces:
  EventSink       the trajectory-file vocabulary
  is_scored        the honest-SR rule — read its docstring
  aggregate
  parse_episodes / format_episodes
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .config import has_places, RunConfig, needs_vla
from .prompts import build_briefing_for, first_prompt_for
from .turns import PlannerTurnCounter, TURN_UNIT

MCP_TOOLS_MODULE = "navgpt.planner.tools.mcp_tools"
MCP_TOOLS_PATH = Path(__file__).resolve().parent / "tools" / "mcp_tools.py"


def mcp_launch(env: dict[str, str]) -> dict[str, Any]:
    """stdio launch spec for the MCP tool server, shared by both provider runtimes."""
    return {"command": sys.executable, "args": [str(MCP_TOOLS_PATH)], "env": env}


# ── Environment HTTP helpers (planner-only; never visible to the model) ──


def panel_field(env_url: str, name: str, value: Any) -> None:
    resp = requests.post(
        "{}/env-panel/field/{}".format(env_url, name), json={"value": value}, timeout=600
    )
    resp.raise_for_status()
    _raise_on_error(resp.json(), "panel_field({})".format(name))


def panel_action(env_url: str, name: str) -> dict[str, Any]:
    resp = requests.post(
        "{}/env-panel/action/{}".format(env_url, name), json={"params": {}}, timeout=600
    )
    resp.raise_for_status()
    body = resp.json()
    _raise_on_error(body, "panel_action({})".format(name))
    return body


def call_function(env_url: str, fn: str, inputs: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        "{}/call/{}".format(env_url, fn), json={"inputs": inputs}, timeout=600
    )
    resp.raise_for_status()
    body = resp.json()
    _raise_on_error(body, fn)
    _raise_on_error(body.get("outputs"), fn)
    return body["outputs"]


def _raise_on_error(body: dict[str, Any], what: str) -> None:
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError("{} failed: {}".format(what, body["error"]))


# ── trajectory file ──


# The API can refuse mid-episode and the SDK reports it as ordinary assistant TEXT
# with subtype "success" and error None. Left alone, such an episode is scored as a
# navigation failure and nothing in the record distinguishes it from bad navigation.
# So the text is matched, recorded, and excluded.
API_ERROR_RE = re.compile(
    r"API Error:\s*\d{3}"          # the SDK's own rendering, e.g. "API Error: 400 …"
    r"|\boverloaded_error\b"
    r"|\brate_limit_error\b",
    re.I)


class EventSink:
    """Wraps one episode's trajectory file. Adapters emit through this only,
    so the curated vocabulary (thinking / assistant_text / tool_use /
    tool_result / system_init / driver_error / result / exit) stays uniform.
    Also tracks the per-tool call counts and the last parsed movement result."""

    def __init__(self, traj_path: Path) -> None:
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = traj_path.open("w")
        self._t0 = time.time()
        self.tool_calls: dict[str, int] = {}
        self.last_step_result: dict[str, Any] | None = None
        self.api_error: str | None = None
        self.review = {}
        self.ablation_error = None
        self.initial_rollout = None
        self.planner_turns = PlannerTurnCounter()

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        self.planner_turns.observe(kind, payload)
        if kind == "tool_use":
            short = str(payload.get("name", "")).rsplit("__", 1)[-1]
            self.tool_calls[short] = self.tool_calls.get(short, 0) + 1
        elif kind == "tool_result":
            for text in payload.get('texts') or []:
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        self.review = data.get('review', self.review)
                        if self.initial_rollout is None:
                            self.initial_rollout = data.get('initial_rollout')
                        self.ablation_error = data.get('ablation_error', self.ablation_error)
                except (ValueError, TypeError):
                    pass
            parsed = self._parse_step_result(payload.get("texts") or [])
            if parsed is not None:
                self.last_step_result = parsed
        elif kind == "assistant_text" and self.api_error is None:
            hit = API_ERROR_RE.search(str(payload.get("text") or ""))
            if hit:
                # keep the FIRST one: it is the refusal that ended the episode, and
                # later turns are just the session unwinding
                self.api_error = str(payload.get("text") or "").strip()[:300]
        self._fh.write(
            json.dumps({"t": round(time.time() - self._t0, 2), "kind": kind, **payload})
            + "\n"
        )
        self._fh.flush()  # a live tail -f must see every event as it happens

    @staticmethod
    def _parse_step_result(texts: list[str]) -> dict[str, Any] | None:
        """The LAST status block in a result, which is the final state.

        Every tool but one emits a single status. `terminate_episode(n)` retraces and
        then stops, so it emits the retrace status (episode_over false) BEFORE the
        terminal one; taking the first would record the episode as never stopped.
        Last-wins is the right rule for "what state did this call leave the episode in".
        """
        found = None
        for text in texts:
            try:
                data = json.loads(text)
            except (ValueError, TypeError):
                continue
            if isinstance(data, dict) and "steps_taken_total" in data:
                found = data
        return found

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def close(self) -> None:
        self._fh.close()


# ── adapter contract ──


@dataclass
class EpisodeContext:
    briefing: str
    first_prompt: str
    model: str
    max_turns: int
    max_budget_usd: float | None
    thinking: dict
    effort: str | None
    betas: list
    mcp_env: dict[str, str]
    workdir: Path
    timeout: int


@dataclass
class SessionOutcome:
    turns: int = 0
    cost_usd: float | None = None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    subtype: str | None = None
    extra: dict = field(default_factory=dict)


# Rate-limit resilience. On the Claude-subscription auth path the throttle comes
# back as a subtype="success" is_error=True result whose text says "temporarily
# limiting requests". The worker backs off exponentially and re-runs the episode
# from scratch (fresh session, env reset); after the last attempt the record is
# excluded via api_error. A spent subscription usage window is handled apart
# (USAGE_LIMIT_MARKERS below).
RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "rate limited",
    "limit exceeded",
    "429",
    "overloaded",
    "too many requests",
    "temporarily limiting",
    "session limit",
    "usage limit",
    "hit your session",
)
RATE_LIMIT_MAX_ATTEMPTS = 6
RATE_LIMIT_BASE_BACKOFF = 30     # seconds, doubles per attempt
RATE_LIMIT_MAX_BACKOFF = 300     # per-backoff cap

# A spent subscription usage window (the Claude Pro/Max five-hour window, a
# ChatGPT plan's Codex limit) is not cleared by a short backoff. The worker waits
# for it to reset and re-runs the episode, rather than excluding every episode
# until the window reopens.
USAGE_LIMIT_MARKERS = ("usage limit", "usage_limit", "session limit", "hit your session",
                       "hit your limit")
USAGE_LIMIT_WAIT = 900           # seconds between retries while the window is spent
USAGE_LIMIT_MAX_WAIT = 6 * 3600  # longer than a five-hour window


def is_rate_limited(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in RATE_LIMIT_MARKERS)


def is_usage_limited(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in USAGE_LIMIT_MARKERS)


def throttle_tag(text: str) -> str | None:
    """'usage_limited', 'rate_limited' or None for an error or reply text."""
    if is_usage_limited(text):
        return "usage_limited"
    if is_rate_limited(text):
        return "rate_limited"
    return None


# ── scoring ──


def is_scored(rec: dict[str, Any]) -> bool:
    """True if the episode is a real, scored navigation attempt that counts
    toward SR.

    INCLUDES a turn-exhausted episode (hit the SDK ``max_turns`` cap without
    calling STOP): it navigated, was evaluated, ``success`` is 0.0 — a
    legitimate FAILURE, not an error to hide (dropping it inflates SR).

    EXCLUDES two kinds of non-attempts:
    - genuine infra failures (timeout / crash) that produced no metrics
      (``metrics == {}`` -> ``success is None``);
    - rate-limit / 'limit exceeded' casualties — errored WITHOUT taking a
      single navigation step (``error`` set + ``env_steps`` 0), so the model
      never really attempted the task. (A genuine turn-exhausted run has
      ``env_steps`` > 0 and stays counted.)
    - API REFUSALS (``api_error`` set), regardless of steps taken. The SDK
      renders these as assistant text with subtype "success" and error None, so
      without this rule they would be scored as navigation failures.
    """
    if rec.get("ablation_error") or (rec.get("agent", {}).get("review") or {}).get("invalid"):
        return False
    if (rec.get("metrics") or {}).get("success") is None:
        return False
    if rec.get("error") and not ((rec.get("agent") or {}).get("env_steps") or 0):
        return False
    # An API refusal is excluded EVEN IF the robot had already taken steps, which is
    # what separates it from turn exhaustion. Turn exhaustion is the agent spending its
    # own budget and failing; this is the episode being cut off half-way through, often
    # after the robot has already moved, so an env_steps test would miss it.
    if rec.get("api_error"):
        return False
    return True


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"episode_count": len(episodes)}
    numeric: dict[str, list[float]] = {}
    for rec in episodes:
        for key, value in (rec.get("metrics") or {}).items():
            if isinstance(value, bool):
                value = float(value)
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
        numeric.setdefault("env_steps", []).append(
            float((rec.get("agent") or {}).get("env_steps", 0))
        )
    for key, values in numeric.items():
        if values:
            agg[key] = round(sum(values) / len(values), 4)
    agg["stop_rate"] = round(
        sum(1 for r in episodes if (r.get("agent") or {}).get("called_stop"))
        / max(1, len(episodes)),
        4,
    )
    return agg


def format_episodes(indices: list[int]) -> str:
    """Inverse of parse_episodes: [7,8,9,14,44] -> '7-9,14,44'."""
    xs = sorted(set(indices))
    if not xs:
        return ""
    out: list[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append("{}-{}".format(start, prev) if start != prev else str(start))
        start = prev = x
    out.append("{}-{}".format(start, prev) if start != prev else str(start))
    return ",".join(out)


def select_episodes(cfg: RunConfig, episode_count: int) -> list[int]:
    """Resolve cfg.episodes ('all' or an index spec) and cfg.shard against the
    Environment's episode count. Shards take every N-th episode so each gets a
    mix of scenes rather than a contiguous, scene-correlated block."""
    spec = str(cfg.episodes).strip().lower()
    indices = list(range(episode_count)) if spec == "all" else parse_episodes(spec)
    if any(i < 0 or i >= episode_count for i in indices):
        raise PreflightError("episode index outside the Environment's {} episodes".format(
            episode_count))
    if cfg.shard:
        index, count = cfg.shard
        indices = indices[index::count]
    return indices


def parse_episodes(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        elif part:
            indices.append(int(part))
    return indices


# ── preflight ──


# Per-condition tool surface, mirroring the registration gate in
# tools/mcp_tools.py. Kept here so preflight can fail once, loudly, instead of
# letting a mis-registered surface produce a plausible-looking run.
# The preflight contract: exactly these tools per profile (docs/USAGE.md).
# The frozen basic profile stops through primitive action 0; other profiles
# expose terminate_episode. The rest follow the capability flags in mcp_tools.py.
EXPECTED_TOOLS = {
    "planner_basic": ["navigate_primitive", "observe_forward"],
    "planner_motion": ["navigate_primitive", "navigate_relative", "observe_forward",
                       "observe_map", "observe_panorama", "terminate_episode"],
    "planner_memory": ["annotate_node", "navigate_primitive", "navigate_relative",
                       "navigate_to_node", "observe_forward", "observe_map", "observe_node",
                       "observe_panorama", "terminate_episode"],
    # the final harness: eight tools
    "planner_vla_memory": ["annotate_node", "navigate_by_instruction", "navigate_relative",
                           "navigate_to_node", "observe_forward", "observe_map",
                           "observe_panorama", "terminate_episode"],
}


def expected_tools(condition, bundle='full'):
    if condition == 'planner_vla_memory':
        from .tools.ablation import CAPABILITIES
        return list(CAPABILITIES[bundle])
    return EXPECTED_TOOLS.get(condition)


# Briefing for the controlled ablations (any non-standard interface). One
# text for every cell, so cells differ only by their registered tools.
ABLATION_FIRST_PROMPT = ("Start by delegating the complete episode instruction to "
                         "navigate_by_instruction().")


def ablation_briefing(tools, instruction):
    return ("Supervise the navigation policy using the complete route instruction. "
            "Delegate the full instruction first, inspect returned route evidence, and explicitly stop at its endpoint. "
            "The graph and route evidence are automatic. Use only registered capabilities: " + ', '.join(tools) +
            ". At each handoff, compare the returned observations and progress with the original route. "
            "For corrective delegation, pass a self-contained remaining route from the observed current position, "
            "including the next supported landmark or direction and the original stopping condition. "
            "Omit completed clauses; do not invent landmarks or change the goal. "
            "Repeat the active instruction (or omit it at a scheduled pause) only for intentional continuation "
            "when it still fits. Visual history survives. You may also correct with other available tools or terminate. "
            "Bearing conventions are stated with each observation; read them before choosing a relative repair. "
            "Node IDs are stable. Do not infer unseen geometry. Instruction: " + instruction)


class PreflightError(RuntimeError):
    pass


def preflight(cfg: RunConfig) -> dict[str, Any]:
    """Fail loudly BEFORE spending money.

    The failure this exists for: if the MCP tool server cannot import its deps, the MCP
    server dies, the model is handed zero tools, and the session still reports
    ``subtype: success, is_error: False`` having spent real tokens. Every
    episode then scores SR=0 and the run looks like a navigation failure
    rather than a config failure, which is why every check below is mandatory.
    """
    if cfg.planner_runtime not in ("claude", "codex"):
        raise PreflightError("unknown Planner runtime: " + cfg.planner_runtime)
    report: dict[str, Any] = {}

    # 1. NavGPT Environment service reachable, correct peer, episodes loaded
    try:
        health = requests.get("{}/health".format(cfg.env_url), timeout=30).json()
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(
            "NavGPT Environment service unreachable at {} ({}). Start it with "
            "`python -m navgpt.environment --config ...`".format(cfg.env_url, exc)
        ) from exc
    if health.get("name") != cfg.verb_prefix:
        raise PreflightError(
            "NavGPT Environment service peer mismatch: /health name={!r} but verb_prefix={!r} — "
            "the wrong server would place the WRONG episodes".format(
                health.get("name"), cfg.verb_prefix
            )
        )
    if not health.get("episode_count"):
        raise PreflightError("NavGPT Environment service reports 0 episodes for split {!r}".format(cfg.split))
    # A server left running from an earlier session serves ITS episode set, and a
    # newly launched one that lost the port bind fails silently behind it. Both
    # produce a run labelled with your split but scored on someone else's
    # episodes, so treat a split mismatch as fatal.
    served = health.get("split")
    if served and served != cfg.split:
        raise PreflightError(
            "NavGPT Environment service at {} is serving split {!r}, but this run is configured "
            "for {!r}. A stale server may still hold the port; restart it with this "
            "experiment's config.".format(cfg.env_url, served, cfg.split)
        )
    if health.get("dataset") and health["dataset"] != cfg.dataset:
        raise PreflightError("Environment dataset {!r} does not match {!r}".format(health["dataset"], cfg.dataset))
    cfg.dataset_info = {key: health.get(key) for key in (
        "dataset", "split", "languages", "roles", "dataset_fingerprint",
        "episode_count", "gt_covered", "episode_sources", "gt_sources", "caliber")}
    # nDTW needs dense ground truth for every episode; a gap would silently score
    # as missing values, so refuse the run instead.
    if (not health.get("blind") and health.get("gt_covered") is not None
            and health["gt_covered"] < health["episode_count"]):
        raise PreflightError("dense ground truth covers {} of {} episodes; check "
                             "dataset.gt_split / gt_file".format(health["gt_covered"],
                                                                 health["episode_count"]))
    if health.get("max_steps") and int(health["max_steps"]) != cfg.step_budget:
        raise PreflightError("Environment step limit {} differs from this run's {}".format(
            health["max_steps"], cfg.step_budget))
    report["env"] = health
    if health.get("blind"):
        report["warning"] = (
            "NavGPT Environment service is in BLIND mode — dynamics and metrics are real, "
            "pixels are synthetic. Navigation results are NOT meaningful."
        )

    # 2. MCP tool server starts and exposes EXACTLY this condition's tool set.
    # Registration is the only real gate (the harness runs bypassPermissions, so
    # allowed_tools cannot withhold a registered tool), which makes an exact
    # match the check that matters: a missing tool silently weakens the
    # condition, and an extra one silently strengthens it. Either way the run
    # would not measure the condition it is labelled with.
    tools = _probe_mcp_tools(cfg)
    from .tools.ablation import validate
    validate(cfg.interface_variant, cfg.capability_bundle, cfg.review_every)
    if (cfg.interface_variant != 'standard' or cfg.capability_bundle != 'full' or cfg.review_every):
        if cfg.condition != 'planner_vla_memory' or cfg.interface_variant == 'standard':
            raise PreflightError('Reduced tool sets and scheduled review use the controlled-study briefing: '
                                 'set harness.interface to reference or one of its variants')
        if health.get('interface_variant') != cfg.interface_variant:
            raise PreflightError('The Environment was started with a different harness.interface; '
                                 'restart it with this experiment config')
        if not cfg.vla_map_enabled():
            raise PreflightError('Ablations require full route maps')
        if cfg.interface_variant != 'standard':
            try:
                info = requests.get(cfg.vla_url.rstrip('/') + '/info', timeout=15).json()
            except (requests.RequestException, ValueError) as exc:
                raise PreflightError(
                    "Ablation runs need NavGPT VLA at {} ({}). Start it with "
                    "`python -m navgpt.vla --config ...`".format(cfg.vla_url, exc)) from exc
            if not info.get('history_observe') or not info.get('seeded_reset'):
                raise PreflightError('VLA must support history-only observation for repair/resume')
    want = expected_tools(cfg.condition, cfg.capability_bundle)
    if want is None:
        raise PreflightError(
            "condition {!r} has no expected tool set — add it to EXPECTED_TOOLS "
            "so a typo cannot silently run a different surface".format(cfg.condition)
        )
    if sorted(tools) != sorted(want):
        missing = sorted(set(want) - set(tools))
        extra = sorted(set(tools) - set(want))
        raise PreflightError(
            "condition {!r} exposed {} — expected {}.{}{}\nIf tools are missing "
            "the MCP server probably failed an import; run it directly:\n"
            "  NAVGPT_CONDITION={} python -m {}".format(
                cfg.condition, sorted(tools), sorted(want),
                " missing={}".format(missing) if missing else "",
                " unexpected={}".format(extra) if extra else "",
                cfg.condition, MCP_TOOLS_MODULE)
        )
    report["mcp_tools"] = sorted(tools)

    # 3. model actually authorized (cheap call)
    if cfg.planner_runtime == "codex":
        from .runtimes.codex import probe_model
        report["model"] = probe_model(cfg)
    else:
        from .runtimes.claude import probe_login, uses_claude_login
        report["model"] = probe_login(cfg.model) if uses_claude_login() else _probe_model(cfg.model)

    # 4. sidecars the chosen condition depends on. Failing here is much cheaper
    #    than discovering it 40 episodes into a sweep.
    from .config import CONDITIONS, needs_vla

    if cfg.condition not in CONDITIONS:
        raise PreflightError(
            "unknown condition {!r} — choose one of {}".format(cfg.condition,
                                                              list(CONDITIONS)))
    report["condition"] = cfg.condition

    if needs_vla(cfg.condition):
        try:
            info = requests.get(cfg.vla_url.rstrip("/") + "/info", timeout=30).json()
        except Exception as exc:  # noqa: BLE001
            raise PreflightError(
                "condition {!r} needs NavGPT VLA at {} but it is "
                "unreachable ({}). Start it with `python -m navgpt.vla --config ...`".format(
                    cfg.condition, cfg.vla_url, exc)
            ) from exc
        report["vla"] = info

    # The pano rig backs observe_panorama / navigate_by_instruction; a server without it
    # would fail per-episode instead of here.
    if cfg.condition != "planner_basic":
        fns = health.get("functions")
        if fns is not None and "{}__observe_pano".format(cfg.verb_prefix) not in fns:
            raise PreflightError(
                "NavGPT Environment service has no {}__observe_pano — it predates the panorama "
                "rig. Restart it so condition {!r} has 4-view observations.".format(
                    cfg.verb_prefix, cfg.condition))
        # The same check, for the place graph: a stale server would hand back the
        # motion surface wearing this condition's name.
        if has_places(cfg.condition):
            if fns is None:
                # Fall back to /manifest, then refuse. A server that will not name
                # its verbs cannot be shown to have the place graph, and "cannot be
                # shown" has to fail here: an absent functions key means the server
                # predates /health carrying one, which predates the graph itself.
                try:
                    fns = requests.get(cfg.env_url.rstrip("/") + "/manifest",
                                       timeout=10).json().get("functions")
                except Exception:  # noqa: BLE001
                    fns = None
            if fns is None:
                raise PreflightError(
                    "NavGPT Environment service at {} does not report its verb list, so it cannot "
                    "be shown to have the place graph — it predates both. Restart "
                    "it, or condition {!r} runs without its place tools."
                    .format(cfg.env_url, cfg.condition))
            need = ["{}__places".format(cfg.verb_prefix),
                    "{}__retrace_to".format(cfg.verb_prefix)]
            missing = [v for v in need if v not in fns]
            if missing:
                raise PreflightError(
                    "NavGPT Environment service is missing {} — it predates the place graph. "
                    "Restart it, or condition {!r} runs without its place tools."
                    .format(", ".join(missing), cfg.condition))
    return report


def _probe_mcp_tools(cfg: RunConfig, timeout: float = 60.0) -> list[str]:
    """Run the MCP tool server as a subprocess and complete an MCP handshake.

    Reads responses incrementally rather than via ``subprocess.run(input=...)``:
    that closes stdin immediately, and the server can shut down on EOF before
    flushing the tools/list reply — a race that shows up as an empty stdout
    even though stderr proves the request was handled.
    """
    import threading

    env = dict(os.environ)
    spec = mcp_launch(mcp_env_for(cfg, live_dir=None))
    env.update(spec["env"])
    proc = subprocess.Popen(
        [spec["command"], *spec["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    stderr_chunks: list[str] = []

    def drain_stderr() -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            stderr_chunks.append(line)

    threading.Thread(target=drain_stderr, daemon=True).start()

    def send(msg: dict) -> None:
        proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    result: dict[str, Any] = {}

    def read_until_id(target: int) -> dict | None:
        while True:
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") == target:
                return msg

    def handshake() -> None:
        send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "preflight", "version": "1"},
            },
        })
        if read_until_id(1) is None:
            return
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        msg = read_until_id(2)
        if msg is not None:
            result["msg"] = msg

    worker = threading.Thread(target=handshake, daemon=True)
    worker.start()
    worker.join(timeout)

    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001 - teardown is best-effort
        proc.kill()

    msg = result.get("msg")
    if msg is None:
        raise PreflightError(
            "MCP tool server never answered tools/list within {}s. Run it directly to see "
            "why:\n  python -m {}\nstderr:\n{}".format(
                timeout, MCP_TOOLS_MODULE, "".join(stderr_chunks).strip()[-2000:]
            )
        )
    if "result" not in msg:
        raise PreflightError("MCP tool server tools/list returned an error: {}".format(msg))
    return [t["name"] for t in msg["result"].get("tools", [])]


def _probe_model(model: str, attempts: int = 3) -> dict[str, Any]:
    """4-token call straight at the API endpoint. Surfaces an unauthorized model as
    a config error instead of 10 failed episodes.

    Retried, because an endpoint can also emit transient errors. Without a retry a
    one-off hiccup aborts an entire unattended run at launch. A genuine authorization
    failure fails identically on every attempt, so retrying costs only a few tokens.
    """

    if any(os.environ.get(flag) for flag in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                                             "CLAUDE_CODE_USE_FOUNDRY")):
        return {"checked": False, "reason": "cloud-provider credentials; not probed"}
    base = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not token:
        return {"checked": False, "reason": "no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in env"}
    headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        headers["authorization"] = "Bearer " + token
    else:
        headers["x-api-key"] = token
    last_body: Any = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.post(
                base.rstrip("/") + "/v1/messages",
                headers=headers,
                json={
                    "model": model,
                    "max_tokens": 4,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001 - network flake shouldn't block a run
            return {"checked": False, "reason": str(exc)}
        body = resp.json() if resp.content else {}
        if body.get("type") == "message":
            out = {"checked": True, "ok": True, "model": model}
            if attempt > 1:
                out["attempts"] = attempt
            return out
        last_body = body
        if attempt < attempts:
            time.sleep(2 * attempt)
    raise PreflightError(
        "model {!r} is not usable at {} after {} attempts: {}".format(
            model, base, attempts, json.dumps(last_body)[:400]
        )
    )


def _reset_vla(cfg: RunConfig, index=0) -> None:
    # Ablation cells seed the VLA per episode so repeated initial rollouts can be
    # paired across cells; standard runs keep the unseeded reset.
    seed = {"seed": cfg.ablation_seed + index} if cfg.interface_variant != "standard" else {}
    resp = requests.post(cfg.vla_url.rstrip("/") + "/reset",
                         json={"success": None, **seed}, timeout=300)
    resp.raise_for_status()


def mcp_env_for(cfg: RunConfig, live_dir: Path | None,
                   episode_id: str | None = None,
                   instruction: str | None = None,
                   labels_path: Path | None = None) -> dict[str, str]:
    env = {
        "NAVGPT_INTERFACE_VARIANT": cfg.interface_variant,
        "NAVGPT_CAPABILITY_BUNDLE": cfg.capability_bundle,
        "NAVGPT_REVIEW_EVERY": str(cfg.review_every),
        "NAVGPT_SERVER_URL": cfg.env_url,
        "NAVGPT_VERB_PREFIX": cfg.verb_prefix,
        "NAVGPT_STEP_BUDGET": str(cfg.step_budget),
        # The MCP tool server registers tools from this; registration is the real gate.
        "NAVGPT_CONDITION": cfg.condition,
        "NAVGPT_VLA_URL": cfg.vla_url,
        "NAVGPT_VLA_MAX_STEPS": str(cfg.vla_max_steps_per_call),
        "NAVGPT_VLA_MAP": "1" if cfg.vla_map_enabled() else "0",
        "NAVGPT_PLACE_LISTING_MAX": str(cfg.place_listing_max),
        "NAVGPT_MOVE_ROUTES": "1" if cfg.move_routes else "0",
        "NAVGPT_MOVE_VIEW": "1" if cfg.move_returns_view else "0",
        "NAVGPT_MAX_KEYFRAMES": str(cfg.max_keyframes),
        "NAVGPT_LABEL_NEAR_M": str(cfg.label_near_m),
    }
    if labels_path is not None:
        # The MCP tool server holds the place names; the NavGPT Environment service has never seen one. Without
        # this, episode_places records the geometry and drops every name and caption.
        env["NAVGPT_PLACE_LABELS"] = str(labels_path)
    if episode_id is not None:
        env["NAVGPT_EPISODE_ID"] = str(episode_id)
    if instruction:
        # vla_navigate defaults to the episode's FULL instruction, so the MCP tool server
        # needs it. Passed as env rather than fetched over HTTP because the
        # driver already has it and an extra reset call has episode semantics.
        env["NAVGPT_INSTRUCTION"] = str(instruction)
    if live_dir is not None:
        env["NAVGPT_LIVE_DIR"] = str(live_dir)
    return env


# ── episode ──


async def run_episode(adapter, cfg: RunConfig, index: int, run_dir: Path) -> dict[str, Any]:
    url = cfg.env_url

    # Place the episode: stage the index, then commit. Blocking HTTP rides
    # to_thread so a cold scene load (~30 s) never stalls the event loop.
    await asyncio.to_thread(panel_field, url, "episode_index", index)
    await asyncio.to_thread(panel_action, url, "play")
    ep = await asyncio.to_thread(
        call_function, url, "{}__reset".format(cfg.verb_prefix), {"trigger": "planner"}
    )
    instruction = str(ep.get("instruction") or "")
    if not instruction:
        raise RuntimeError(
            "reset returned an empty instruction (episode {}) — placement failed?".format(index)
        )

    live_dir = (run_dir / "live" / "ep{:04d}".format(index)) if cfg.live else None
    workdir = run_dir / "work" / "ep{:04d}".format(index)
    workdir.mkdir(parents=True, exist_ok=True)

    # NavGPT VLA keeps per-episode frame history as its temporal context, so
    # this reset is mandatory: skip it and the previous episode's frames leak
    # into this one's sampling window.
    if needs_vla(cfg.condition):
        try:
            await asyncio.to_thread(_reset_vla, cfg, index)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "could not reset NavGPT VLA at {}: {!r}".format(
                    cfg.vla_url, exc)) from exc

    briefing = build_briefing_for(cfg.condition, instruction, cfg.step_budget,
                                  move_routes=cfg.move_routes)
    if cfg.interface_variant != 'standard':
        briefing = ablation_briefing(expected_tools(cfg.condition, cfg.capability_bundle),
                                     instruction)
    # Every tool the briefing names by call syntax has to be one the MCP tool server
    # registers, or the agent is instructed to call something that does not exist.
    named = {m.group(1) for m in re.finditer(r"\b([a-z_][a-z_0-9]*)\(", briefing)}
    want = set(expected_tools(cfg.condition, cfg.capability_bundle) or [])
    promised_but_absent = sorted(
        named & {"navigate_relative", "navigate_to_node", "annotate_node",
                 "observe_panorama", "observe_map", "observe_forward", "navigate_primitive",
                 "navigate_by_instruction", "observe_node",
                 "terminate_episode"} - want)
    if promised_but_absent:
        raise RuntimeError(
            "briefing for condition %r names tools the surface does not have: %s"
            % (cfg.condition, ", ".join(promised_but_absent)))
    first_prompt = (ABLATION_FIRST_PROMPT if cfg.interface_variant != "standard"
                    else first_prompt_for(cfg.condition))

    # Where the MCP tool server mirrors its place labels, so episode_places can carry names.
    labels_path = (run_dir / "labels_{}.json".format(index)
                   if has_places(cfg.condition) else None)

    ctx = EpisodeContext(
        briefing=briefing,
        first_prompt=first_prompt,
        model=cfg.model,
        max_turns=cfg.max_turns,
        max_budget_usd=cfg.max_budget_usd,
        thinking=dict(cfg.thinking),
        effort=cfg.effort,
        betas=list(cfg.betas),
        mcp_env=mcp_env_for(cfg, live_dir, ep.get("episode_id"),
                                  instruction, labels_path=labels_path),
        workdir=workdir,
        timeout=cfg.episode_timeout,
    )

    sink = EventSink(run_dir / "episode_{}.jsonl".format(index))
    metrics: dict[str, Any] = {}
    outcome = SessionOutcome()
    places_total = 0          # bound before the try: an early failure must not
    places_named = 0          # turn a missing counter into a NameError
    clauses_bound = 0
    try:
        sink.emit("episode_meta", {
            "index": index,
            "episode_id": ep.get("episode_id"),
            "scene_id": ep.get("scene_id"),
            "instruction": instruction,
            "geodesic_distance": ep.get("geodesic_distance"),
        })
        sink.emit("session_inputs", {
            "model": cfg.model,
            "condition": cfg.condition,
            "system_prompt": briefing,
            "first_prompt": first_prompt,
            "max_turns": cfg.max_turns,
            "max_budget_usd": cfg.max_budget_usd,
            **adapter.describe(ctx),
        })

        try:
            outcome = await asyncio.wait_for(
                adapter.run(ctx, sink), timeout=cfg.episode_timeout
            )
        except asyncio.TimeoutError:
            outcome = SessionOutcome(error="episode_timeout after {}s".format(cfg.episode_timeout))
            sink.emit("driver_error", {"error": outcome.error})

        outcome.turns = sink.planner_turns.count
        outcome.extra["turn_unit"] = TURN_UNIT
        sink.emit("result", {"result": {
            "usage": outcome.usage, "cost_usd": outcome.cost_usd,
            "turns": outcome.turns, "error": outcome.error,
            "subtype": outcome.subtype, **outcome.extra,
        }})

        # Evaluate while the trajectory file is still open, so metrics land in
        # the log itself. Driver-side; the agent never sees this.
        try:
            out = await asyncio.to_thread(
                call_function, url, "{}__evaluate".format(cfg.verb_prefix),
                {"trigger": "planner"},
            )
            metrics = out.get("metrics") or {}
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
        except Exception as exc:  # noqa: BLE001
            sink.emit("driver_error", {"error": "evaluate failed: {!r}".format(exc)})
        sink.emit("episode_metrics", {"metrics": metrics})

        # Trajectory for the viewer: agent path vs reference vs goal. Lands in
        # the event stream rather than summary.json so a 100-episode summary
        # stays small. Best-effort — a viewer aid, never worth failing a run.
        try:
            traj = await asyncio.to_thread(
                call_function, url, "{}__trajectory".format(cfg.verb_prefix),
                {"trigger": "planner"},
            )
            sink.emit("episode_trajectory", traj)
        except Exception as exc:  # noqa: BLE001
            sink.emit("driver_error", {"error": "trajectory failed: {!r}".format(exc)})

        # The place graph as it finished, for the viewer and for post-hoc analysis:
        # "did it go back to somewhere it named, and was that its own closest
        # approach?" is the question that decides whether the graph helped or just
        # gave it a new way to wander, and it must be answerable without a rerun.
        if has_places(cfg.condition):
            try:
                graph = await asyncio.to_thread(
                    call_function, url, "{}__places".format(cfg.verb_prefix), {})
                # Merge the names/captions/clause bindings the MCP tool server holds. The Environment
                # server never sees a label, so without this the recorded graph is
                # geometry only and "did it caption, and did the caption sit where it
                # thought" is unanswerable without a rerun.
                if labels_path is not None and labels_path.exists():
                    try:
                        labels = json.loads(labels_path.read_text())
                    except Exception:  # noqa: BLE001
                        labels = {}
                    for pl in graph.get("places") or []:
                        lab = labels.get(str(pl.get("id")))
                        if lab:
                            pl["name"] = lab.get("name")
                            pl["caption"] = lab.get("caption")
                            pl["clause"] = lab.get("clause")
                            pl["anchor"] = lab.get("anchor")
                sink.emit("episode_places", graph)
                places_total = len(graph.get("places") or [])
                places_named = sum(1 for pl in (graph.get("places") or [])
                                   if (pl.get("name") or "").strip())
                clauses_bound = len({pl.get("clause")
                                     for pl in (graph.get("places") or [])
                                     if pl.get("clause") is not None})
            except Exception as exc:  # noqa: BLE001
                sink.emit("driver_error", {"error": "places failed: {!r}".format(exc)})
    finally:
        wall = sink.elapsed
        sink.close()

    last = sink.last_step_result or {}
    place_stats = {}
    if has_places(cfg.condition):
        # Named vs merely created is the usage question: a tool that registers
        # and works but is never used shows up here.
        place_stats = {
            "places_total": places_total,
            "places_named": places_named,
            "clauses_bound": clauses_bound,
            "annotate_node_calls": sink.tool_calls.get("annotate_node", 0),
            "navigate_to_node_calls": sink.tool_calls.get("navigate_to_node", 0),
            "terminate_episode_calls": sink.tool_calls.get("terminate_episode", 0),
            "observe_node_calls": sink.tool_calls.get("observe_node", 0),
        }
    return {
        "index": index,
        "episode_id": ep.get("episode_id"),
        "scene_id": ep.get("scene_id"),
        "dataset": cfg.dataset, "language": ep.get("language"),
        "instruction_id": ep.get("instruction_id"), "annotation_role": ep.get("annotation_role"),
        "instruction": instruction,
        "metrics": metrics,
        "agent": {
            "turns": outcome.turns,
            "turn_unit": TURN_UNIT,
            "tool_calls": dict(sink.tool_calls),
            "env_steps": int(last.get("steps_taken_total") or 0),
            "called_stop": last.get("end_reason") == "stop_called",
            "end_reason": last.get("end_reason"),
            "cost_usd": outcome.cost_usd,
            "usage": outcome.usage,
            "review": sink.review,
            "initial_rollout": sink.initial_rollout,
            **place_stats,
        },
        # An API refusal is not a navigation result. Recorded as an error even though
        # the SDK called the session a success, so is_scored() can drop it and it never
        # masquerades as a regression.
        "error": outcome.error or sink.ablation_error or (
            "api_error: {}".format(sink.api_error) if sink.api_error else None),
        "api_error": sink.api_error,
        "ablation_error": sink.ablation_error,
        "subtype": outcome.subtype,
        "wall_s": round(wall, 1),
        "status": ("errored" if (outcome.error or sink.api_error or sink.ablation_error) else "completed"),
    }


# ── run ──


async def run_eval(adapter, cfg: RunConfig) -> dict[str, Any]:
    """Run (or resume) one evaluation. Requested episodes that already have a
    scored record in the run dir's summary are kept; the rest are run."""
    # Absolute: the SDK runs each session with cwd=ctx.workdir, so a relative
    # run dir would make the MCP tool server's NAVGPT_LIVE_DIR resolve inside the
    # per-episode workdir instead of here.
    run_dir = (Path(cfg.output_root) / cfg.resolved_run_name()).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    records: dict[int, dict[str, Any]] = {}
    if summary_path.exists():
        try:
            prior = json.loads(summary_path.read_text())
            records = {int(e["index"]): e for e in prior.get("episodes", [])}
            print("[NavGPT Planner] resuming: {} existing record(s)".format(len(records)))
        except (ValueError, KeyError):
            print("[NavGPT Planner] existing summary unreadable — starting fresh")

    print("[NavGPT Planner] preflight ...")
    report = preflight(cfg)
    indices = select_episodes(cfg, int(report["env"]["episode_count"]))
    print("[NavGPT Planner]   env: split={} episodes={} blind={}".format(
        report["env"].get("split"), report["env"].get("episode_count"),
        report["env"].get("blind"),
    ))
    print("[NavGPT Planner]   MCP tool server tools: {}".format(report["mcp_tools"]))
    print("[NavGPT Planner]   model: {}".format(report["model"]))
    if report.get("warning"):
        print("[NavGPT Planner]   WARNING: {}".format(report["warning"]))

    adapter.prepare(cfg)
    # Publish the run before episode 0 so a monitor sees it at 0/N straight away.
    # Without this the run dir has no summary.json until the first episode
    # finishes, and the viewer — which keys off summary.json — shows nothing,
    # which is indistinguishable from a broken run.
    _write_summary(summary_path, cfg, records)
    kept = [i for i in indices if i in records and is_scored(records[i])]
    indices = [i for i in indices if i not in kept]
    if kept:
        print("[NavGPT Planner] keeping {} scored episode(s)".format(len(kept)))
    print("[NavGPT Planner] running episodes {}".format(format_episodes(indices)))

    for n, index in enumerate(indices, 1):
        print("[NavGPT Planner] --- episode {} ({}/{}) ---".format(index, n, len(indices)))
        attempts, throttles, waited = 0, 0, 0
        while True:
            attempts += 1
            try:
                rec = await run_episode(adapter, cfg, index, run_dir)
            except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
                print("[NavGPT Planner]   ERROR: {!r}".format(exc))
                rec = {
                    "index": index, "metrics": {}, "agent": {"env_steps": 0},
                    "error": throttle_tag(repr(exc)) or repr(exc), "status": "errored",
                }
            tag = throttle_tag(str(rec.get("error") or ""))
            if tag == "usage_limited":
                if waited >= USAGE_LIMIT_MAX_WAIT:
                    rec["api_error"] = "usage limit did not reset within {} h".format(
                        USAGE_LIMIT_MAX_WAIT // 3600)
                    print("[NavGPT Planner]   USAGE LIMIT did not reset — excluding episode")
                    break
                print("[NavGPT Planner]   USAGE LIMIT reached — retrying in {} min".format(
                    USAGE_LIMIT_WAIT // 60))
                await asyncio.sleep(USAGE_LIMIT_WAIT)
                waited += USAGE_LIMIT_WAIT
                continue
            if tag != "rate_limited":
                break
            throttles += 1
            if throttles == RATE_LIMIT_MAX_ATTEMPTS:
                # Out of retries: exclude rather than score a throttle as a nav failure.
                rec["api_error"] = "rate_limited after {} attempts".format(throttles)
                print("[NavGPT Planner]   RATE-LIMITED on every attempt — excluding episode")
                break
            backoff = min(RATE_LIMIT_BASE_BACKOFF * (2 ** (throttles - 1)), RATE_LIMIT_MAX_BACKOFF)
            print("[NavGPT Planner]   RATE-LIMITED (attempt {}/{}) — backing off {}s then retrying"
                  .format(throttles, RATE_LIMIT_MAX_ATTEMPTS, backoff))
            await asyncio.sleep(backoff)  # outside run_episode: excluded from episode_timeout
        if attempts > 1:
            rec["rate_limit_attempts"] = attempts
        if waited:
            rec["usage_limit_wait_s"] = waited
        records[index] = rec
        m = rec.get("metrics") or {}
        print("[NavGPT Planner]   SR={} SPL={} nDTW={} NE={} steps={} turns={} cost=${}".format(
            m.get("success"), m.get("spl"), m.get("ndtw"), m.get("distance_to_goal"),
            (rec.get("agent") or {}).get("env_steps"),
            (rec.get("agent") or {}).get("turns"),
            (rec.get("agent") or {}).get("cost_usd"),
        ))
        _write_summary(summary_path, cfg, records)

    summary = _write_summary(summary_path, cfg, records)
    print("\n[NavGPT Planner] === aggregate over {} scored / {} total ===".format(
        summary["scored_count"], summary["aggregate"]["episode_count"]
    ))
    for k, v in sorted(summary["aggregate"].items()):
        print("  {:<20} {}".format(k, v))
    print("\n[NavGPT Planner] artifacts: {}".format(run_dir))
    return summary


def _write_summary(path: Path, cfg: RunConfig, records: dict[int, dict]) -> dict[str, Any]:
    episodes = [records[k] for k in sorted(records)]
    scored = [r for r in episodes if is_scored(r)]
    summary = {
        "run_name": cfg.resolved_run_name(),
        "config": cfg.to_dict(),
        "episodes": episodes,
        "scored_count": len(scored),
        "excluded_count": len(episodes) - len(scored),
        # Aggregate over SCORED episodes only — see is_scored().
        "aggregate": aggregate(scored),
    }
    path.write_text(json.dumps(summary, indent=2))
    return summary
