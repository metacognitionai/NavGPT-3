"""Run configuration for one Planner (or VLA-only) evaluation.

Values come from an experiment YAML file (see ``navgpt.settings``); the defaults
below are what a field takes when the file leaves it out. Model ids are bare
provider ids. runner.preflight() checks that the configured model is reachable
before an evaluation spends anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_CODEX_MODEL = "gpt-6-astra"

PLANNER_BASIC = "planner_basic"
PLANNER_MOTION = "planner_motion"
PLANNER_MEMORY = "planner_memory"
PLANNER_VLA_MEMORY = "planner_vla_memory"
VLA_ONLY = "vla_only"

CONDITIONS = (
    PLANNER_BASIC,
    PLANNER_MOTION,
    PLANNER_MEMORY,
    PLANNER_VLA_MEMORY,
)

# How spatial information is drawn for the Planner. "standard" is the regular
# NavGPT-3 display; the others are the controlled-study display ("reference")
# and four variants that each change one thing about it.
INTERFACES = ("standard", "reference", "text_labels", "small_digits",
              "heading_up_map", "absolute_bearings")


@dataclass
class RunConfig:
    # ── what to run ──
    dataset: str = "r2r"
    dataset_info: dict = field(default_factory=dict)
    split: str = "val_unseen"
    episodes: str = "all"
    """Episode index spec ('0-9', '3,7', '0-4,10') or 'all'."""
    shard: tuple[int, int] | None = None
    """(index, count): run every count-th selected episode starting at index."""
    planner_runtime: str = "claude"
    """Planner provider adapter: "claude" (Claude Agent SDK) or "codex" (Codex SDK)."""
    model: str = DEFAULT_MODEL
    env_url: str = "http://127.0.0.1:9200"
    vla_url: str = "http://127.0.0.1:8000"
    verb_prefix: str = "r2rce"

    # ── limits ──
    max_turns: int = 200
    step_budget: int = 500
    episode_timeout: int = 2400
    max_budget_usd: float | None = 5.0
    """Per-episode USD ceiling for the Claude runtime. None disables it; the
    Codex runtime reports no cost and requires None."""

    # ── tool surface ──
    tools: list = field(default_factory=list)
    """The Planner's tools as configured (planner.tools); condition and
    capability_bundle are derived from this list by navgpt.settings."""
    condition: str = PLANNER_BASIC
    """Internal name of the tool set and its briefing:
      planner_basic       forward view + primitive actions
      planner_motion      basic + panorama, map, metric motion
      planner_memory      motion + named place memory and retracing
      planner_vla_memory  VLA + motion + place memory
    Registration in the MCP tool server is the real gate: permission_mode is
    bypassPermissions, so allowed_tools cannot withhold a registered tool."""

    # Controlled studies (navgpt/planner/tools/ablation.py). Any interface other
    # than "standard" requires planner_vla_memory.
    interface_variant: str = "standard"
    capability_bundle: str = "full"
    review_every: int = 0
    """Committed VLA steps between Planner reviews; 0 reviews only when the VLA
    stops or reaches its per-call step cap."""
    ablation_seed: int = 20260910

    place_listing_max: int = 24
    """How many places observe_map lists at once, nearest first. When it binds it
    drops the farthest, never renumbers, and `navigate_to_node` still accepts an
    id that was not listed."""
    label_near_m: float = 1.5
    """How near a place must be for a bare annotate_node() to label it."""
    move_routes: bool = True
    """navigate_relative() follows the navmesh route when it is nearly the straight
    line asked for. Off: it walks straight and stops at obstacles, and the
    briefing says so."""
    move_returns_view: bool = True
    """navigate_relative() returns the four views from where it arrives."""

    vla_returns_map: bool | None = None
    """End every navigate_by_instruction result with the top-down map, after the
    four closing views. None follows the condition: on for planner_vla_memory."""
    vla_max_steps_per_call: int = 200
    """Safety bound on one navigate_by_instruction call, in low-level steps of the
    episode budget, so a confused rollout cannot exhaust the episode."""
    max_keyframes: int = 8

    # ── provider ──
    thinking: dict = field(default_factory=lambda: {"type": "adaptive", "display": "summarized"})
    effort: str | None = None
    betas: list = field(default_factory=list)

    # ── output ──
    run_name: str | None = None
    output_root: str = "outputs"
    live: bool = False
    """Store every observed frame and an actions log per episode for the viewer."""

    def vla_map_enabled(self) -> bool:
        """Resolve `vla_returns_map`, whose None means "follow the condition"."""
        if self.vla_returns_map is None:
            return self.condition == PLANNER_VLA_MEMORY
        return bool(self.vla_returns_map)

    def resolved_run_name(self) -> str:
        name = self.run_name or "{}_{}_{}".format(
            self.model.replace(".", "-"), self.condition, self.split)
        if self.shard:
            name += "_s{}".format(self.shard[0])
        return name

    def to_dict(self) -> dict:
        return asdict(self)


def needs_vla(condition: str) -> bool:
    return condition == PLANNER_VLA_MEMORY


def has_places(condition: str) -> bool:
    """Conditions carrying the place graph. Must agree with mcp_tools._HAS_PLACES."""
    return condition in (PLANNER_MEMORY, PLANNER_VLA_MEMORY)
