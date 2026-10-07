"""MCP tool server — the agent's only channel to NavGPT Environment.

Stdio MCP server (FastMCP) forwarding a condition-scoped toolset to a running
navgpt.environment (``POST /call/{fn}``). Registration at the bottom of this
file is the capability boundary. ``planner_basic`` sees two tools; the primary
``planner_vla_memory`` profile sees the eight inspection, delegation, repair,
return, annotation, and termination tools documented in the project README.

Episode selection and metric collection stay driver-side. The agent never
sees SR/SPL, reward, pose, depth, or the goal — that separation is what
makes the eval honest, and it is why this file talks HTTP to the env rather
than importing it.

One MCP tool server process serves one agent session = one episode (the Agent SDK
spawns a fresh stdio server per session), so per-episode step accounting
lives in module globals.

Runs in the AGENT-side interpreter (py>=3.10), not the habitat one — it
needs only ``requests`` + ``pillow``. Note the SDK spawns this with
``command=sys.executable``: if those two imports fail here, the MCP server
dies, the model is handed zero tools, and the session still reports success
while scoring SR=0. ``runner.preflight()`` exists to catch exactly that.

Env vars:
    NAVGPT_SERVER_URL     navgpt.environment base URL (default http://127.0.0.1:9200)
    NAVGPT_VERB_PREFIX    verb namespace (default r2rce)
    NAVGPT_STEP_BUDGET    advisory low-level step budget echoed to the agent
                           (default 500 — the env's max_steps truncates
                           authoritatively regardless)
    NAVGPT_LIVE_DIR       optional dir for live spectating: every observed
                           frame lands as obs_NNNN.png (+ latest.png) and every
                           step call appends a line to actions.log
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP, Image

# The runtime spawns this file BY PATH (runner.mcp_launch),
# so there is no package context and `from .vla_client import ...` raises
# "attempted relative import with no known parent package". That failure is
# invisible to a probe launched as `python -m navgpt.planner.tools.mcp_tools`, which
# DOES have package context.
#
# Import siblings through this helper instead, so the MCP tool server behaves identically
# however it is launched.
def _sibling(module_name, attr):
    try:
        mod = __import__(
            "navgpt.planner.tools." + module_name, fromlist=[attr])
    except ImportError:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(here, module_name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, mod)
        spec.loader.exec_module(mod)
    return getattr(mod, attr)


save_frame = _sibling("frame_store", "save_frame")
read_frame = _sibling("frame_store", "read_frame")
frame_exists = _sibling("frame_store", "frame_exists")

SERVER_URL = os.environ.get("NAVGPT_SERVER_URL", "http://127.0.0.1:9200")
VERB_PREFIX = os.environ.get("NAVGPT_VERB_PREFIX", "r2rce")
STEP_BUDGET = int(os.environ.get("NAVGPT_STEP_BUDGET", "500"))
LIVE_DIR = Path(os.environ["NAVGPT_LIVE_DIR"]) if os.environ.get("NAVGPT_LIVE_DIR") else None

# Which tool surface to expose. Registration is the ONLY reliable gate: the
# harness runs permission_mode="bypassPermissions", so allowed_tools cannot
# withhold a tool that has been registered.
CONDITION = os.environ.get("NAVGPT_CONDITION", "planner_basic")
VLA_URL = os.environ.get("NAVGPT_VLA_URL", "http://127.0.0.1:8000")
VLA_MAX_STEPS = int(os.environ.get("NAVGPT_VLA_MAX_STEPS", "200"))  # 0 = uncapped
# End navigate_by_instruction with the top-down map. The default follows the
# condition, matching RunConfig.vla_map_enabled(); the runner sets it explicitly.
VLA_MAP = os.environ.get(
    "NAVGPT_VLA_MAP",
    "1" if CONDITION == "planner_vla_memory" else "0") not in ("0", "", "false")
PLACE_LISTING_MAX = int(os.environ.get("NAVGPT_PLACE_LISTING_MAX", "24"))
# Walk the navmesh route to the requested point when that route is nearly the
# straight line asked for; a straight walk often falls well short of the distance
# requested.
MOVE_ROUTES = os.environ.get("NAVGPT_MOVE_ROUTES", "1") not in ("0", "", "false")
# Return the four views WITH the move result: a move is almost always followed by
# a look, so arriving and looking are one act rather than two turns.
MOVE_VIEW = os.environ.get("NAVGPT_MOVE_VIEW", "1") not in ("0", "", "false")
MAX_KEYFRAMES = int(os.environ.get("NAVGPT_MAX_KEYFRAMES", "8"))

_HAS_MOVE = CONDITION == "planner_motion"
_HAS_VLA = CONDITION == "planner_vla_memory"
_HAS_PLACES = CONDITION in ("planner_memory", "planner_vla_memory")
_HAS_MOVE = _HAS_MOVE or _HAS_PLACES     # `planner_memory` is the `move` surface plus places
_HAS_PANO = _HAS_MOVE or _HAS_VLA

INTERFACE = os.environ.get('NAVGPT_INTERFACE_VARIANT', 'standard')
BUNDLE = os.environ.get('NAVGPT_CAPABILITY_BUNDLE', 'full')
REVIEW_EVERY = int(os.environ.get('NAVGPT_REVIEW_EVERY', '0'))
_ABLATION = INTERFACE != 'standard'
_CAPABILITIES = _sibling('ablation', 'CAPABILITIES')
_sibling('ablation', 'validate')(INTERFACE, BUNDLE, REVIEW_EVERY)
_scheduler = _sibling('ablation', 'ReviewScheduler')(REVIEW_EVERY)

mcp = FastMCP("env")

_t0 = time.time()
_obs_count = 0
_steps_taken = 0
_episode_over = False
_end_reason = ""


_OBSERVE_DESC = "Look through the robot's forward-facing camera and return the RGB view."

_STEP_DESC = (
    "Execute movement actions in order.\n"
    "Actions: 0 = STOP (permanently ENDS the episode — issue it only when you "
    "believe you have reached the instruction's endpoint), 1 = move forward "
    "0.25 m, 2 = turn left 15 degrees, 3 = turn right 15 degrees.\n"
    "Executes sequentially and halts early if the episode ends. Returns how "
    "many low-level steps were taken and whether the episode is over."
)


# ── driver-invisible plumbing ──


_recent_frames: list[str] = []      # paths written for the most recent observation

# Frames on disk are for VIEWING, never for the model (which gets the PNG bytes
# above). A 400 px render is ~250 KB as PNG and ~30 KB as JPEG q88 with no visible
# difference at thumbnail size. Set NAVGPT_LIVE_FORMAT=png to keep lossless dumps.
LIVE_FORMAT = (os.environ.get("NAVGPT_LIVE_FORMAT") or "jpg").lower()
LIVE_JPEG_QUALITY = int(os.environ.get("NAVGPT_LIVE_JPEG_QUALITY") or "88")

# Which tool is running, so every frame written can say who drew it. Set at the top
# of each tool; the manifest line carries (tool, call number, label, kind).
_tool_ctx: dict[str, Any] = {"tool": None, "call": 0}
_tool_call_counts: dict[str, int] = {}


def _tool_enter(name: str) -> None:
    _tool_call_counts[name] = _tool_call_counts.get(name, 0) + 1
    _tool_ctx["tool"] = name
    _tool_ctx["call"] = _tool_call_counts[name]


def _encode_live(png: bytes) -> tuple[bytes, str]:
    """(bytes, extension) for the on-disk copy of a frame."""
    if LIVE_FORMAT == "png":
        return png, "png"
    try:
        import io
        from PIL import Image as _PILImage
        im = _PILImage.open(io.BytesIO(png))
        im.load()
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=LIVE_JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "jpg"
    except Exception:  # noqa: BLE001 - a missing Pillow must not cost the frame
        return png, "png"


def _live_frame(png: bytes, label: str | None = None, kind: str = "view") -> None:
    """Dump a frame for spectating, and remember where it went.

    Images are written to LIVE_DIR (a run with live output); without it there
    are no stored frames for the captioner or node recall.

    Every frame also gets a line in ``frames.jsonl`` saying which tool drew it and
    what it shows, so a viewer never has to infer the attribution from the tool's
    text output.
    """
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    data, ext = _encode_live(png)
    f = LIVE_DIR / "obs_{:04d}_step{:03d}.{}".format(_obs_count, _steps_taken, ext)
    location = save_frame(f, data)
    save_frame(LIVE_DIR / ("latest." + ext), data)
    _recent_frames.append(location)
    del _recent_frames[:-8]         # one look is at most 5 frames; keep a little slack
    try:
        with (LIVE_DIR / "frames.jsonl").open("a") as fh:
            fh.write(json.dumps({
                "obs": _obs_count, "file": f.name, "step": _steps_taken,
                "tool": _tool_ctx["tool"], "call": _tool_ctx["call"],
                "label": label or kind, "kind": kind,
                "t": round(time.time() - _t0, 1),
            }) + "\n")
    except OSError:
        pass


def _live_log(entry: dict[str, Any]) -> None:
    if LIVE_DIR is None:
        return
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    with (LIVE_DIR / "actions.log").open("a") as fh:
        fh.write(json.dumps({"t": round(time.time() - _t0, 1), **entry}) + "\n")


def _call(function_name: str, inputs: dict[str, Any]) -> dict[str, Any]:
    motion = function_name.rsplit('__', 1)[-1]
    capture = _ABLATION and motion in ('step_hightolow', 'retrace_to', 'retrace_path')
    if capture:
        inputs = dict(inputs, capture_history=True)
    resp = requests.post(
        "{}/call/{}".format(SERVER_URL, function_name),
        json={"inputs": inputs},
        timeout=300,
    )
    if resp.status_code >= 400:
        # The NavGPT Environment service explains itself in the body ({"error": "unknown place 99;
        # places are 0..1"}), and raise_for_status throws all of that away in favour
        # of "500 Server Error". The agent then reads an HTTP code where a sentence
        # was available.
        detail = ""
        try:
            detail = str((resp.json() or {}).get("error") or "")
        except ValueError:
            detail = resp.text[:200]
        raise RuntimeError(detail or "{} {}".format(resp.status_code, resp.reason))
    output = resp.json()["outputs"]
    if capture:
        for frame in output.pop('motion_history', []):
            _lazy_vla().observe_history(frame['views'])
    return output


_vla = None     # NavGPTVLAClient, lazily built
_last_pose = None
_last_scan = None
# Whether the agent is standing where it already stood. Computed env-side from
# the full per-step odometry trail and cached from the last pano payload.
_last_revisit = None
# The episode's FULL instruction, supplied by the driver (NAVGPT_INSTRUCTION).
# vla_navigate falls back to this when called with no argument, which is the
# expected case: NavGPT VLA is handed the whole route.
_instruction = os.environ.get("NAVGPT_INSTRUCTION", "")
_vla_calls = 0
_presentation_by_hash = {}


def _lazy_vla():
    global _vla
    if _vla is None:
        _vla = _sibling("vla_client", "NavGPTVLAClient")(VLA_URL)
    return _vla


def _pano() -> dict[str, Any]:
    """Fetch the 4-view panorama + range scan.

    This is also what grows the top-down map: the NavGPT Environment service unprojects these
    four RGB-D views into its fixed colour grid on every call, so looking is what
    fills the map in.

    Costs zero simulator steps — the four cameras render alongside the agent's
    forward camera in one observation.
    """
    global _last_pose, _last_scan, _last_revisit
    # A caption has to describe the spot it is a caption OF, so the frame list is
    # emptied here and refilled by _live_frame as this observation is decoded.
    # Frames left over from a movement result would show the previous spot.
    _recent_frames.clear()
    out = _call("{}__observe_pano".format(VERB_PREFIX), {})
    if _ABLATION:
        for name, value in (out.get('views') or {}).items():
            _presentation_by_hash[hashlib.sha256(base64.b64decode(value)).hexdigest()] = (out.get('presentation') or {}).get(name, {})
    _last_pose = out.get("pose")
    _last_scan = out.get("scan")
    if "revisiting_earlier_position" in out:
        _last_revisit = out["revisiting_earlier_position"]
    return out


def _clearance_summary(scan, payload=None) -> dict[str, Any]:
    """Prefer the server's own figures when it sent them.

    The server BURNS these numbers into the pictures, so a second computation here
    could disagree with the painted label — the exact failure burning them in was
    meant to remove. It also reports can_walk_m (how far a move would actually get,
    simulated on the navmesh) which is a better answer to "can I go that way" than
    cone clearance is; cone clearance answers "am I about to scrape something".
    """
    if payload:
        walk = payload.get("can_walk_m")
        if walk:
            if INTERFACE == 'absolute_bearings':
                heading = payload['absolute_heading_deg']
                offsets = {'ahead_m': 0, 'right_m': 90, 'behind_m': 180, 'left_m': 270}
                return {'bearing_%03d_m' % ((heading + offsets[k]) % 360): v for k,v in walk.items()}
            return dict(walk)
        clear = payload.get("clearance_m")
        if clear:
            return dict(clear)
    return _clearance_summary_from_scan(scan)


def _clearance_summary_from_scan(scan) -> dict[str, Any]:
    """Calibrated clearance in the four cardinal directions, in metres.

    Depth-derived, so these are measurements rather than impressions — the point
    of the `move` surface. None means that bearing was never observed.
    """
    def wedge(centre_deg):
        if not scan:
            return None
        n = len(scan)
        half = 10
        vals = [scan[(int(centre_deg) + k) % n] for k in range(-half, half + 1)]
        vals = [v for v in vals if v is not None]
        return round(min(vals), 2) if vals else None

    return {"ahead_m": wedge(0), "left_m": wedge(90),
            "behind_m": wedge(180), "right_m": wedge(270)}


def _pano_frames(pano, clear=None, prefix="") -> list[tuple[str, bytes]]:
    """The look, as (caption, png) pairs, one per view.

    ONE renderer for every place a look is shown — observe_panorama, the end of a
    VLA leg, arriving at a place, returning to a waypoint — so their captions
    cannot drift apart.

    `prefix` is what the caption says about WHY you are seeing this ("back at
    place 3", "where NavGPT VLA stopped").
    """
    views = pano.get("views") or {}
    if _ABLATION:
        out = []
        for name in ('front', 'right', 'back', 'left'):
            if not views.get(name):
                continue
            cap = prefix
            out.append((cap.strip(), base64.b64decode(views[name])))
        return out
    clear = clear if clear is not None else _clearance_summary(pano.get("scan"), pano)
    out = []
    for name, label, key in (("front", "ahead", "ahead_m"),
                             ("right", "right (+90 deg right)", "right_m"),
                             ("back", "behind (180 deg)", "behind_m"),
                             ("left", "left (+90 deg left)", "left_m")):
        b64 = views.get(name)
        if not b64:
            continue
        dist = (clear or {}).get(key)
        cap = "{} — clearance {}".format(
            label, "{} m".format(dist) if dist is not None else "unknown")
        out.append(((prefix + " — " + cap) if prefix else cap,
                    base64.b64decode(b64)))
    return out


def _pano_content(pano, clear=None, prefix="") -> list[Any]:
    """_pano_frames as MCP content, captions dropped when they would be empty."""
    content: list[Any] = []
    for cap, png in _pano_frames(pano, clear, prefix):
        _bump_obs(png, (cap.split(" — ")[0] if cap else "view"))
        if cap:
            content.append(cap)
        content.append(Image(data=png, format="png"))
    return content


def _capture_view() -> bytes:
    """Render the current egocentric RGB. Pure read — advances the live-frame
    counter, never the simulator."""
    global _obs_count, _forward_presentation
    _recent_frames.clear()
    outputs = _call("{}__observe_egocentric".format(VERB_PREFIX),
                    {"stamp": _HAS_PANO})
    _forward_presentation = outputs.get("presentation", {})
    png = base64.b64decode(outputs["rgb"])
    if _ABLATION:
        _presentation_by_hash[hashlib.sha256(png).hexdigest()] = _forward_presentation
    _obs_count += 1
    _live_frame(png, "forward view")
    return png


# ── the two tools ──


def observe() -> list:
    _tool_enter("observe_forward")
    if _episode_over:
        return ["episode already over ({}); no image".format(_end_reason)]
    png = _capture_view()
    return [Image(data=png, format="png")]


def step(actions: list[int]) -> list:  # bare `list` => FastMCP unstructured path
    _tool_enter("navigate_primitive")
    global _steps_taken, _episode_over, _end_reason

    if _episode_over:
        return ["episode already over ({}); no more steps possible".format(_end_reason)]
    if not actions:
        return ["no actions given — pass a non-empty list, e.g. navigate_primitive([1])"]

    executed: list[int] = []
    for action in actions:
        try:
            action = int(action)
        except (TypeError, ValueError):
            return ["invalid action {!r} — must be an integer 0-3".format(action)]
        if action not in (0, 1, 2, 3):
            return ["invalid action {} — must be 0, 1, 2 or 3".format(action)]

        outputs = _call("{}__step_discrete".format(VERB_PREFIX), {"action": action})
        _steps_taken += 1
        executed.append(action)

        if action == 0:
            _episode_over = True
            _end_reason = "stop_called"
            break
        if outputs.get("terminated") or outputs.get("truncated"):
            _episode_over = True
            _end_reason = "step_budget_exhausted"
            break

    status = {
        "actions_executed": executed,
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
    }
    if _episode_over:
        status["end_reason"] = _end_reason
    if _HAS_PLACES and not _episode_over:
        line = _places_line()
        if line:
            status["nearest_place"] = line
    _live_log({"step": executed, **status})

    return [json.dumps(status)]


# ══════════════════════════════════════════════════════════════════════
# Richer surfaces — registered per condition (see CONDITION above)
# ══════════════════════════════════════════════════════════════════════


_LOOK_DESC = (
    "Scan the surroundings in ONE call. "
    "Returns four camera views — ahead, right, behind, left."
    "\n"
    "\n"
    "EVERY LABEL IS PAINTED ON THE PICTURE ITSELF, so you never have to work out "
    "which text belongs to which view:\n"
    "  * the direction word on each view, with how far you would actually get "
    "walking that way (measured on the floor plan, sliding along walls exactly as "
    "a real move does — so a small number means genuinely blocked, not just "
    "'something is close');\n"
    "  * a ruler along the bottom of each view giving the BEARING of each column, "
    "written as the exact number navigate_relative() takes (positive = left). This is how you "
    "turn 'the doorway is in the left third of the AHEAD view' into a move: read "
    "the tick under it. A 120-degree view has no single direction, so the word "
    "AHEAD alone cannot tell you where in it to go;\n"
    "  * VIEW n, your position as X and Y in metres from where the episode "
    "started, and MAP-UP — the turn that faces the top of the map. LEFT and "
    "AHEAD mean different world directions after you turn, so MAP-UP is what lets "
    "a view you saw twenty turns ago still be placed;\n"
    "\n"
    "The cameras render simultaneously, so this does not move the robot "
    "and does not change where you stand or which way you face. Use it freely at "
    "any junction or whenever you are unsure — turning to look is never "
    "necessary.\n"
    "\n"
    "For IDENTIFYING a particular object — is that a fire extinguisher, is that "
    "chair the one described — use observe_forward() instead: it returns the forward view "
    "alone at full size, and detail is what it is for."
)

_MOVE_DESC = (
    "Turn to a relative bearing and walk. turn_deg: 0 = straight ahead, "
    "POSITIVE = LEFT, negative = right. distance_m: how far to try to walk.\n"
    "The turn does not use the movement budget; the walking does, one step per 0.25 m "
    "budget, so it costs the same as stepping that far, not less.\n"
    "It walks the floor-plan route to the point you asked for when that route is "
    "nearly the straight line — so a chair or a door frame in the way is walked "
    "around rather than stopped at, and detoured tells you when that "
    "happened. If the only route is a long way round, it does NOT take it: it walks "
    "straight and stops, because that means the way you asked for is genuinely "
    "blocked and you need to know.\n"
    "The result ENDS WITH THE FOUR VIEWS from where you arrive, so you never need a "
    "observe_panorama straight after moving.\n"
    "Returns walked_m against requested_m, and blocked. walked_m is your ACTUAL "
    "displacement: it is often less than you asked for even when blocked is "
    "false, because the robot slides along walls and furniture rather than "
    "stopping dead. blocked=true means genuinely obstructed, never out-of-budget.\n"
    "nearest_obstacle_before_m is a depth measurement along that bearing, and it is "
    "a ROUGH GUIDE ONLY — it is the nearest surface in a 20-degree cone, so a "
    "single chair leg or door frame in that cone reads as a tiny clearance even "
    "though you can walk straight past it; and it is measured at camera height, "
    "so a low step or a bed frame does not appear at all. Do not let a small "
    "value stop you asking for a longer distance: ask, then read walked_m to "
    "learn what actually happened."
)


_MAP_DESC = (
    "A top-down photograph of everywhere you have looked, with the path you have "
    "already walked drawn on it.\n"
    "\n"
    "Cells carry the real colours your cameras measured there, seen from above — "
    "so a red rug reads red, a wooden floor reads brown. The flat slate checker is "
    "territory you have never observed: unknown, not empty — anything your cameras "
    "did see is drawn lighter than it, however dark the surface was.\n"
    "\n"
    "Your route is drawn over it as a line that ages from BLUE to RED: the blue dot "
    "is where the episode started, red is where you have just been, and the RED "
    "ARROW is you — it points the way you currently face. Reading "
    "the colour tells you which way round you walked a loop, not just that you "
    "walked one.\n"
    "\n"
    "Places you can return to are numbered AMBER BADGES, and the place you are "
    "standing on is an amber ring around you — the same numbers the place list "
    "underneath the map uses, so a mark on the picture and a line of the list are "
    "the same thing. That list comes with every map: each place's coordinates in "
    "metres from your start, your own name for it, how far away it is, and what "
    "far walking back to it would be.\n"
    "\n"
    "The map is never rotated to your heading, so a loop looks like a loop across "
    "calls — the arrow moves, the picture does not. You do not need to work out "
    "which way the picture is pointing: the result tells you which way to TURN to "
    "face each place, in the same degrees navigate_relative() takes, and where up-the-map is "
    "relative to you if you want to read the image directly. It also gives the "
    "pixel coordinates of you and of your start, how far you are from the start, "
    "and how big the frame is. The frame is cropped to your route and what you have "
    "seen around it, so it grows as you explore: check frame_extent_m before "
    "comparing two calls by eye.\n"
    "\n"
    "It does not move the robot, and it refreshes the surroundings as it "
    "draws, so you never need a observe_panorama just to update it.\n"
    "\n"
    "Use it to answer the question a forward camera cannot: have I been here "
    "before, and which parts of this floor have I still never looked at? Going "
    "in circles is one of the most common ways this task is failed, and it is "
    "invisible from the camera alone — the result also reports "
    "revisiting_earlier_position outright."
)


_VLA_DESC = (
    "Give a navigation instruction to NavGPT VLA, "
    "which drives the robot along the route for you.\n"
    "\n"
    "WHAT TO PASS: on the first call, the full original episode instruction. "
    "Before a later corrective call, compare the returned views and route evidence "
    "with the original goal. Pass a self-contained instruction for the remaining "
    "route from the observed current position, including the next supported landmark "
    "or direction and the original stopping condition. Omit completed clauses; "
    "do not invent landmarks or change the goal. Reuse the previous instruction "
    "only for intentional continuation when it still fits. With no argument, a "
    "scheduled pause resumes its active instruction; otherwise the episode's "
    "original instruction is used. Always pass corrective instructions explicitly.\n"
    "\n"
    "WHAT IT DOES: drives the robot until it judges the route finished, or until "
    "its step allowance for this call runs out. It moves the robot for real — "
    "those steps come out of your movement budget.\n"
    "\n"
    "WHAT YOU GET BACK: photographs taken along the way, then four views from "
    "wherever the robot ended up (ahead / right / behind / left, each with "
    "measured clearance in metres), then a summary: how many steps it took, how "
    "far it travelled, how far it ended from where it started, how often it hit "
    "something, and vla_suggests_arrival.\n"
    + ("\nAND A MAP, last: a top-down view of everywhere you have looked, your "
       "route ageing from blue at the start to red at the present, and a red "
       "arrow where NavGPT VLA left you, pointing the way you now face. The "
       "photographs say what is around you; the map says where that is "
       "relative to where you began.\n" if VLA_MAP else "") +
    "\n"
    "WHAT IT CANNOT DO: end the episode, and recognise arrival. "
    "vla_suggests_arrival only means it stopped producing movement — it is "
    "wrong in both directions, so never treat it as confirmation. Only your own "
    "STOP ends the episode, and judging whether you are actually at the endpoint "
    "the instruction describes is the part you cannot hand over.\n"
    "\n"
    "HOW TO USE IT: normally one call at the start. Then read the four returned "
    "views against the instruction, walk the last few metres yourself with navigate_relative() "
    "if it finished short or overshot, and STOP when the view matches the "
    "endpoint. The four views it returns are exactly what observe_panorama() gives, so "
    "do not spend a turn on observe_panorama straight afterwards."
)


def look_around() -> list:
    _tool_enter("observe_panorama")
    if _episode_over:
        return ["episode already over ({}); no more looking".format(_end_reason)]

    out = _pano()
    clear = _clearance_summary(out.get("scan"), out)

    content: list[Any] = _pano_content(out, clear)

    status = {
        "can_walk_m": clear,
        "view": out.get("view_index"),   # the number painted on the image
        "you_are_at_xy": out.get("you_are_at_xy"),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
    }
    if _HAS_MOVE and _last_revisit is not None:
        status["revisiting_earlier_position"] = _last_revisit
    content.append(json.dumps(status))
    # The graph, at the moment a decision is made. Without this the place table is
    # reachable only through observe_map(), so the candidates would be off screen exactly
    # when stopping is being decided.
    if _HAS_PLACES:
        cand = _candidates_block()
        if cand:
            content.append(cand)
    _live_log({"observe_panorama": True, "can_walk_m": status.get("can_walk_m")})
    _frame_here()
    _auto_label_start()
    return content


def _auto_label_start() -> None:
    """Node 0 is where the episode began; say so without spending a turn on it."""
    if not _HAS_PLACES or 0 in _place_labels:
        return
    try:
        pos = (_call("{}__agent_state".format(VERB_PREFIX), {}) or {}).get("position")
    except Exception:  # noqa: BLE001
        pos = None
    _place_labels[0] = {
        "name": "start", "caption": "where the episode began", "clause": None,
        "anchor": list(pos) if pos else None,
        "frame": (_recent_frames[-1] if _recent_frames else None),
        "auto": True,
    }
    _dump_labels()


def _detour_m(out: dict[str, Any]) -> float:
    try:
        return round(max(0.0, float(out.get("walked_m") or 0.0)
                         - float(out.get("net_displacement_m") or 0.0)), 2)
    except (TypeError, ValueError):
        return 0.0


def move(turn_deg: float, distance_m: float) -> list:
    _tool_enter("navigate_relative")
    global _steps_taken, _episode_over, _end_reason
    if _episode_over:
        return ["episode already over ({}); no more moves".format(_end_reason)]

    try:
        bearing = float(turn_deg)
        dist = float(distance_m)
    except (TypeError, ValueError):
        return ["invalid arguments — turn_deg and distance_m must be numbers"]
    if dist < 0:
        return ["distance_m must be >= 0 (use turn_deg to turn around)"]

    # Report the clearance we are about to walk into. This is the calibration the
    # surface exists for, so it must describe the pose we are leaving FROM.
    #
    # Whatever look_around() last cached in `_last_scan` can be several moves stale,
    # and then the number is actively wrong. Re-scanning is free (observe_pano
    # renders alongside the agent camera and costs zero simulator steps), so the
    # cached scan is only a fallback.
    predicted = None
    try:
        fresh = _pano().get("scan")
    except Exception:  # noqa: BLE001 - a scan failure must not block the move
        fresh = _last_scan
    if fresh:
        n = len(fresh)
        half = 10
        centre = int(round(bearing)) % n
        vals = [fresh[(centre + k) % n] for k in range(-half, half + 1)]
        vals = [v for v in vals if v is not None]
        predicted = round(min(vals), 2) if vals else None

    out = _call("{}__step_hightolow".format(VERB_PREFIX), {
        "angle_rad": math.radians(bearing),
        "distance_m": dist,
        "route": MOVE_ROUTES,
    })
    _steps_taken = int(out.get("step_count") or _steps_taken)
    if out.get("terminated") or out.get("truncated"):
        _episode_over = True
        _end_reason = "step_budget_exhausted"

    status = {
        "turned_deg": out.get("turned_deg"),
        "requested_m": out.get("requested_m"),
        "walked_m": out.get("walked_m"),
        "blocked": out.get("blocked"),
        # detour_m = ground walked beyond the straight-line displacement. `routed`
        # alone is true for any navmesh walk, which is not "went round something";
        # a detour is when the route actually bent.
        "detour_m": _detour_m(out),
        "detoured": bool(out.get("routed")) and _detour_m(out) > 0.2,
        "nearest_obstacle_before_m": predicted,
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
    }
    if _episode_over:
        status["end_reason"] = _end_reason
    if _HAS_PLACES and not _episode_over:
        line = _places_line()
        if line:
            status["nearest_place"] = line
    _live_log({"move": [bearing, dist], **status})
    _frame_here()

    # ARRIVING AND LOOKING ARE ONE ACT, so they are one call — the same argument
    # `go` already makes. Otherwise the agent pays two turns for one decision and
    # chooses its next move from views taken BEFORE it moved. The pano costs zero
    # simulator steps.
    content: list[Any] = [json.dumps(status)]
    if MOVE_VIEW and _HAS_PANO and not _episode_over:
        try:
            content += _pano_content(_pano())
        except Exception as exc:  # noqa: BLE001 - the view is an aid, not the move
            content.append("(view unavailable after the move: {!r})".format(exc))
    # The place list rides on the MOVE too, not just on look_around. Once moving
    # returns its own views the navigator rarely calls look_around, so attaching the
    # candidates only to look_around would put the graph off screen at the moment
    # stopping is decided.
    if _HAS_PLACES and not _episode_over:
        cand = _candidates_block()
        if cand:
            content.append(cand)
    return content


def _fetch_map() -> tuple[bytes, dict[str, Any]] | tuple[None, str]:
    """The top-down map plus the fields worth reporting, or None if it failed.

    Shared by ``local_map`` and the tail of ``navigate_by_instruction`` so the two cannot
    drift into describing the same picture differently.
    """
    try:
        out = _call("{}__observed_map".format(VERB_PREFIX), {})
        png = base64.b64decode(out["png"])
    except Exception as exc:  # noqa: BLE001 - callers decide how loudly to fail
        # Hand the reason back rather than swallowing it: the text of the failure
        # is what makes it findable in the event log.
        return None, "{}: {}".format(type(exc).__name__, exc)
    fields: dict[str, Any] = {}
    for key, src in (("facing", "facing"),
                     ("you_are_at_px", "you_are_at_px"),
                     ("started_at_px", "started_at_px"),
                     ("straight_line_from_start_m", "straight_line_from_start_m"),
                     ("walked_m", "walked_m"),
                     ("revisiting_earlier_position", "revisiting_earlier_position"),
                     ("seen_area_m2", "seen_area_m2"),
                     ("unobserved_fraction_of_frame", "unobserved_fraction_of_frame"),
                     ("map_extent_m", "frame_extent_m"),
                     ("metres_per_pixel", "meters_per_pixel"),
                     ("frame", "frame"),
                     ("legend", "legend")):
        if out.get(src) is not None:
            fields[key] = out[src]
    if _ABLATION:
        for key in ('places_px', 'presentation_transform', 'absolute_heading_deg', 'up_on_the_map_is', 'caption', 'direction_cues_px'):
            if out.get(key) is not None:
                fields[key] = out[key]
    if _ABLATION:
        _presentation_by_hash[hashlib.sha256(png).hexdigest()] = fields
    return png, fields


# The agent's own words for the places it has been. The server owns the geometry
# and never sees these: room names are perception, and a label the cameras never
# read would be privileged information (MP3D's own annotations are not even
# loaded). `anchor` is the pose the agent stood at when it named the place, which
# is what lets _go_place finish the last metre exactly rather than at whichever
# path point happened to become the node.
_place_labels: dict[int, dict[str, Any]] = {}

# Where to mirror _place_labels so the DRIVER can record them. episode_places is
# built from the NavGPT Environment service's `places` verb, and the server never
# sees a label. The labels live in this process; the driver reads this file at
# episode end and merges.
LABELS_PATH = os.environ.get("NAVGPT_PLACE_LABELS") or ""


def _dump_labels() -> None:
    """Mirror the labels to disk. Never raise: telemetry must not end an episode."""
    if not LABELS_PATH:
        return
    try:
        with open(LABELS_PATH, "w") as fh:
            json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != "frame"}
                       for k, v in _place_labels.items()}, fh)
    except Exception as exc:  # noqa: BLE001 - a lost label is not a lost episode
        _live_log({"dump_labels_error": str(exc)[:120]})


def _fetch_places() -> dict[str, Any] | None:
    try:
        return _call("{}__places".format(VERB_PREFIX), {})
    except Exception:  # noqa: BLE001 - callers report it in their own words
        return None


# One frame per PLACE, recorded whenever the robot looks from it — independent of
# whether the agent ever names it, so observe_node can show any place the robot
# stood at, named or not.
_place_frames: dict[int, str] = {}


PLACE_FRAME_NEAR_M = float(os.environ.get("NAVGPT_PLACE_FRAME_NEAR_M", "1.5"))


def _frames_from_trace(trace) -> None:
    """Give every place the SPECIALIST drove through a picture. Free.

    The right stopping place is typically mid-route, driven through by NavGPT VLA
    during one rollout and never looked at by the agent, so frames captured only
    where the agent looked would leave it without a picture. But the rollout DID
    photograph it: every trace entry carries a front view and the pose it was taken
    from. Matching those poses to places costs nothing and covers the whole route,
    which is the part the endgame decision is actually about.

    Keeps, per place, the frame whose pose is CLOSEST to it, so a picture is never
    attributed to a place the camera was two metres from when a nearer one exists.
    """
    if not (_HAS_PLACES and trace) or LIVE_DIR is None:
        return
    payload = _fetch_places()
    if not payload:
        return
    places = [pl for pl in (payload.get("places") or []) if pl.get("position")]
    if not places:
        return
    best: dict[int, tuple[float, bytes]] = {}
    for fr in trace:
        pos = (fr.get("pose") or {}).get("position")
        png = fr.get("png")
        if not pos or not png:
            continue
        for pl in places:
            wp = pl["position"]
            d = math.hypot(float(pos[0]) - float(wp[0]), float(pos[2]) - float(wp[2]))
            if d > PLACE_FRAME_NEAR_M:
                continue
            pid = int(pl["id"])
            if pid not in best or d < best[pid][0]:
                best[pid] = (d, png)
    if not best:
        return
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        for pid, (d, png) in best.items():
            # never overwrite the agent's own naming frame; that one is deliberate
            if (_place_labels.get(pid) or {}).get("frame"):
                continue
            f = LIVE_DIR / "place_{:03d}.png".format(pid)
            _place_frames[pid] = save_frame(f, png)
    except OSError as exc:  # noqa: BLE001 - a missing picture is not a lost episode
        _live_log({"frames_from_trace_error": str(exc)[:120]})
    else:
        _live_log({"frames_from_trace": sorted(best)})


def _frame_here() -> None:
    """Remember the picture taken at the place the robot is standing on. Free."""
    if not (_HAS_PLACES and _recent_frames):
        return
    payload = _fetch_places()
    if not payload:
        return
    here = payload.get("nearest_place")
    if here is None:
        return
    pl = next((p for p in (payload.get("places") or []) if p["id"] == here), None)
    if pl is None or (pl.get("straight_m") or 99.0) > LABEL_NEAR_M:
        return
    # [-1], not [0]: _recent_frames keeps the last 8 paths ACROSS looks, so after a
    # four-view look on top of a previous one, [0] is a frame from the previous pose.
    _place_frames[int(here)] = _recent_frames[-1]


def _pending_place(payload: dict[str, Any]) -> int | None:
    """The newest place the agent has not captioned yet, if any.

    Derived by comparing ids against the labels the MCP tool server holds, so it needs no
    server state: the graph proposes, the agent names, and "which one still needs a
    name" is just set arithmetic.

    Used only for the NAG ("place 4 still has no name"). It must NOT be used to pick
    what a bare ``annotate_node("name")`` labels — see ``_label_target``.
    """
    ids = [p["id"] for p in payload.get("places") or []]
    unnamed = [i for i in ids if i not in _place_labels]
    return max(unnamed) if unnamed else None


LABEL_NEAR_M = float(os.environ.get("NAVGPT_LABEL_NEAR_M", "1.5"))

def _label_target(payload: dict[str, Any]) -> int | None:
    """Which place a bare ``annotate_node("name", "caption")`` describes.

    Not ``_pending_place`` (the HIGHEST-NUMBERED uncaptioned place, wherever it
    is). Under `planner_vla_memory` NavGPT VLA drives the whole route in one call,
    so the graph gains places 0..N at once; the agent then walks back to the spot
    the last clause describes and names it, and labelling the newest place would
    attach the name to NavGPT VLA's OVERSHOOT node instead.

    A label describes WHERE THE ROBOT IS. So: among the places within
    ``LABEL_NEAR_M`` of the robot prefer an uncaptioned one (which keeps the
    "caption the place you just made" intent), else the nearest of them. Only when
    nothing at all is within reach does it fall back to the nearest place, and the
    caller's ``note`` already says so out loud.
    """
    places = payload.get("places") or []
    near = [p for p in places
            if (p.get("straight_m") if p.get("straight_m") is not None else 99.0)
            <= LABEL_NEAR_M]
    if near:
        uncaptioned = [p for p in near if p["id"] not in _place_labels]
        pool = uncaptioned or near
        return min(pool, key=lambda p: p.get("straight_m") or 0.0)["id"]
    return payload.get("nearest_place")


def _place_line(pl: dict[str, Any], nearest: int | None) -> list[str]:
    label = _place_labels.get(pl["id"])
    name = '"{}"'.format(label["name"]) if label else "(awaiting a caption)"
    here = " <- you are here" if pl["id"] == nearest else ""
    out = ["{:>3}  ({:6.2f},{:6.2f})  {:22} {:5.1f} m away, {} · walk back {} · "
           "visits {} · links {}{}".format(
               pl["id"], pl["xy"][0], pl["xy"][1], name, pl["straight_m"],
               pl.get("turn", "?"),
               "{:.1f} m".format(pl["route_m"]) if pl["route_m"] is not None
               else "no walked route",
               pl["visits"],
               ",".join(str(i) for i in pl["links"]) or "-", here)]
    if label and label.get("clause") is not None:
        out[0] += " · clause {}".format(label["clause"])
    if label and label.get("caption"):
        out.append("       {}".format(label["caption"]))
    return out


def _format_places(payload: dict[str, Any]) -> str:
    """The place table, in one place so no two callers can print it differently."""
    places = payload.get("places") or []
    nearest = payload.get("nearest_place")
    head = ("places ({}) · you are at ({:.2f}, {:.2f}) · x/y are metres along the "
            "map's axes from where you started · every direction below is a turn in "
            "the same degrees navigate_relative() takes, positive = left".format(
                len(places), *(payload.get("you_are_at_xy") or [0.0, 0.0])))
    if _ABLATION:
        head = 'places: x/y metres from start in fixed map axes; relative turns positive left'
    if INTERFACE == 'absolute_bearings':
        head = 'places: x/y metres from start in fixed map axes; bearings clockwise from map-north'
    # nearest first: the decision the table serves is "where can I get back to",
    # and the answer is almost always close by
    ordered = sorted(places, key=lambda pl: pl["straight_m"])
    keep = ordered[:PLACE_LISTING_MAX]
    lines = [head]
    for pl in keep:
        lines += _place_line(pl, nearest)
    if len(ordered) > len(keep):
        rest = [pl["id"] for pl in ordered[len(keep):]]
        lines.append("       ... and {} more, all drawn on the map: {}".format(
            len(rest), ",".join(str(i) for i in rest)))
    if payload.get("note"):
        lines.append("       {}".format(payload["note"]))
    pending = _pending_place(payload)
    if pending is not None and not _ABLATION:
        lines.append('       place {} still has no name — annotate_node("name", '
                     '"caption") records it'.format(pending))
    return "\n".join(lines)


def _places_line() -> str | None:
    """One line for a movement result: the nearest place and what a return costs.

    This rides on every navigate_primitive() and navigate_relative() on purpose: a
    fact the agent must choose to fetch tends to be ignored, and the moment that
    matters is precisely while it is walking away from a place it named.
    """
    payload = _fetch_places()
    if not payload or not payload.get("places"):
        return None
    nearest = payload.get("nearest_place")
    pl = next((p for p in payload["places"] if p["id"] == nearest), None)
    if pl is None:
        return None
    label = _place_labels.get(pl["id"])
    name = '"{}"'.format(label["name"]) if label else "unnamed"
    # Never quote a zero return cost for a place you are not standing on: the
    # route is measured from the graph's cursor, and there is a dead band between
    # the snap radius and the spacing in which the cursor is already "at" a place
    # the robot is still a metre or two from.
    if pl["route_m"] is None:
        cost = "no walked route back"
    else:
        cost = "{} steps to return".format(
            int(math.ceil(max(pl["route_m"], pl["straight_m"]) / 0.25)))
    line = "{} {} at ({:.1f}, {:.1f}) — {:.1f} m away, {}, {}".format(
        pl["id"], name, pl["xy"][0], pl["xy"][1], pl["straight_m"],
        pl.get("turn", "?"), cost)
    pending = _pending_place(payload)
    if pending is not None and pending != pl["id"]:
        line += ' · place {} is new and unnamed'.format(pending)
    elif pending is not None:
        line += ' · unnamed: annotate_node("name", "caption")'
    return line


def local_map() -> list:      # registered as observe_map; see the bottom
    """Top-down colour map of what the cameras have actually seen. See _MAP_DESC.

    Asks the NavGPT Environment service for ``observed_map``, which unprojects the
    four RGB-D pano views into a top-down grid: a cell is painted only where a
    measurement from a pose the agent actually occupied landed in it.

    That keeps the map honest — nothing is disclosed that a depth camera at those
    poses could not have measured, and rooms never looked into stay unknown —
    while surfaces come out in their real colours, which is far easier to match
    against an instruction ("the red rug") than grey blobs.

    Every call takes a fresh pano first. That costs no movement steps and is what
    makes the tool answer about *now*: without it the map would show only what
    the last observe_panorama saw, so calling it after walking would place the
    agent outside its own coverage.
    """
    _tool_enter("observe_map")
    _pano()
    if _last_pose is None:
        return ["no pose available yet"]

    got = _fetch_map()
    if got[0] is None:
        # No fallback renderer on purpose: a substitute that contradicts the tool
        # description reads as confidently as a right map. An honest failure is
        # cheaper.
        return ["map unavailable ({}) — nothing was drawn. Navigate from the "
                "views for now.".format(got[1])]

    # Ordered so the first lines are the ones a lost agent needs — which way am I
    # pointing, how far am I from where I started, have I been here before — and
    # the frame/scale/legend boilerplate comes last.
    png, status = got
    _bump_obs(png, "map", kind="map")
    table = None
    if _HAS_PLACES:
        payload = _fetch_places()
        if payload and payload.get("places"):
            table = _format_places(payload)
    _live_log({"local_map": True,
               "seen_area_m2": status["seen_area_m2"],
               "revisiting": status["revisiting_earlier_position"]})
    out: list[Any] = [Image(data=png, format="png"), json.dumps(status)]
    if table:
        out.append(table)
    return out


def vla_navigate(instruction: str = "") -> list:
    """Delegate a route to NavGPT VLA. See _VLA_DESC.

    The first call uses the full episode instruction. The Planner may explicitly
    revise later instructions using observed progress while preserving the goal.
    An omitted instruction resumes the active route at a scheduled pause, and
    otherwise defaults to the original episode instruction. STOP remains the
    Planner's decision.
    """
    _tool_enter("navigate_by_instruction")
    global _steps_taken, _episode_over, _end_reason, _vla_calls
    _vla_calls += 1
    if _episode_over:
        return ["episode already over ({}); NavGPT VLA cannot run".format(_end_reason)]

    # Preserve a revised route across scheduled pauses when the Planner resumes
    # without arguments. Explicit instructions always take precedence.
    active_route = _scheduler.route if _ABLATION and _scheduler.suspended else ""
    route = (instruction or "").strip() or active_route or (_instruction or "").strip()
    if not route:
        return ["no instruction available — pass the episode's full instruction text"]

    # Bounded by the per-call safety cap AND by what is left of the episode.
    # VLA_MAX_STEPS = 0 means the leg runs until NavGPT VLA is done or the global
    # step budget is spent.
    remaining = max(1, STEP_BUDGET - _steps_taken)
    budget = remaining if VLA_MAX_STEPS <= 0 else max(1, min(VLA_MAX_STEPS, remaining))
    spec = _lazy_vla()

    # Record the WHOLE rollout, then downsample evenly at the end, so the agent
    # judging whether the route followed the instruction is not blind to the
    # middle of it.
    #
    # The front view is already rendered every step for NavGPT VLA, so
    # keeping it costs no extra render — only memory, bounded by the step cap.
    grant = _scheduler.begin(route) if _ABLATION else None
    if _ABLATION and VLA_MAX_STEPS > 0:
        budget = max(1, min(budget, VLA_MAX_STEPS - _scheduler.used))
    trace: list[dict[str, Any]] = _scheduler.trace if _ABLATION else []
    handoff = None
    start_pose = trace[0]["pose"] if trace else None
    travelled = _scheduler.travelled if _ABLATION else 0.0
    blocked_count = _scheduler.blocked_streak if _ABLATION else 0
    pose_unchanged = _sibling("vla_client", "pose_unchanged")
    prev_pose, stuck = None, False      # stuck test between consecutive VLA steps
    arrival = False
    executed = 0

    for i in range(budget):
        pano = _pano()
        views = pano.get("views") or {}
        pose = pano.get("pose") or {}
        if start_pose is None:
            start_pose = pose

        front = views.get("front")
        if front:
            trace.append({
                "step": _scheduler.used if _ABLATION else executed,
                "png": base64.b64decode(front),
                "travelled": travelled,
                "pose": pose,
                "presentation": (pano.get("presentation") or {}).get("front", {}),
                "source_sha256": {k: hashlib.sha256(base64.b64decode(v)).hexdigest()
                    for k,v in (pano.get('raw_views') or {}).items()} if _ABLATION else {},
            })

        try:
            advice = spec.act(
                views=pano.get("raw_views") or {},
                instruction=route,
                episode_id=str(_instruction_id()),
                position=pose.get("position") or [0.0, 0.0, 0.0],
                rotation_wxyz=pose.get("rotation_wxyz") or [1.0, 0.0, 0.0, 0.0],
                is_stuck=stuck,
            )
        except Exception as exc:  # noqa: BLE001 - surface, never crash the episode
            if _ABLATION:
                _scheduler.fail()
                raise RuntimeError('VLA inference failed; history state uncertain, episode must be excluded') from exc
            return [json.dumps({
                "error": "NavGPT VLA unavailable: {}".format(exc),
                "steps_the_vla_took": executed,
                "advice": "NavGPT VLA is down — walk the route yourself with "
                          "observe_panorama(), observe_forward() and navigate_relative()",
            })]

        arrival = bool(advice.get("should_stop"))
        target, rot = advice.get("position"), advice.get("rotation_xyzw")
        if target is None or rot is None:
            if _ABLATION:
                if not arrival:
                    raise RuntimeError('VLA returned neither STOP nor a complete movement')
                handoff = _scheduler.finish(grant)
            break

        before = pose.get("position") or [0.0, 0.0, 0.0]
        res = _call("{}__teleport".format(VERB_PREFIX),
                    {"position": target, "rotation": rot,
                     "theta": advice.get("theta", 0.0), "is_stuck": stuck})
        stuck = pose_unchanged(prev_pose, res.get("pose"))
        prev_pose = res.get("pose")
        executed += 1
        _steps_taken = int(res.get("step_count") or _steps_taken)
        after = (res.get("pose") or {}).get("position") or before
        travelled += math.dist(before, after)
        if not res.get("navigable"):
            blocked_count += 1
        else:
            blocked_count = 0
        if _ABLATION:
            terminal = bool(res.get('terminated') or res.get('truncated'))
            handoff = _scheduler.commit(grant, blocked=not res.get('navigable'),
                                        natural=terminal or arrival or i + 1 >= budget)
            _scheduler.travelled = travelled
        if res.get("terminated") or res.get("truncated"):
            _episode_over = True
            _end_reason = "step_budget_exhausted"
            break
        # NavGPT VLA thinking it has arrived ENDS THE LEG, not the episode.
        # Handing that decision back is the whole point of the surface.
        if arrival or handoff == "scheduled":
            break

    # Evenly spaced across the ACTUAL rollout, so the agent sees the whole route
    # it is being asked to judge — including the middle.
    keyframes: list[tuple[str, bytes]] = []
    if trace:
        k = max(1, min(MAX_KEYFRAMES, len(trace)))
        picks = [round(j * (len(trace) - 1) / max(1, k - 1)) for j in range(k)] if k > 1 else [0]
        seen = set()
        for j in picks:
            if j in seen:
                continue
            seen.add(j)
            fr = trace[j]
            back = travelled - fr["travelled"]
            keyframes.append((
                "waypoint {n}/{tot}: step {s}, {inm:.1f} m into the route, "
                "{back:.1f} m before where it stopped".format(
                    n=len(keyframes) + 1, tot=k, s=fr["step"],
                    inm=fr["travelled"], back=back),
                fr["png"],
            ))
    _set_trace(trace)
    # Every place the rollout drove through now has a photograph, so the endgame
    # comparison covers the route rather than only the spots the agent looked from.
    _frames_from_trace(trace)

    # End the leg with a FULL FOUR-VIEW look, not a single forward frame.
    #
    # A leg moves and turns the robot, so one forward view leaves the agent
    # unable to tell where NavGPT VLA put it. The intermediate keyframes stay front-only: they are a trace of the path
    # walked, where heading context matters far less than at the endpoint, and
    # four views per keyframe would multiply the image budget by four.
    pano = _pano()
    end_clear = _clearance_summary(pano.get("scan"), pano)
    end_frames = _pano_frames(
        pano, end_clear,
        "where NavGPT VLA stopped, after {} steps".format(executed))
    end_pose = pano.get("pose") or {}

    net = 0.0
    if start_pose and end_pose:
        try:
            net = math.dist(start_pose.get("position") or [0, 0, 0],
                            end_pose.get("position") or [0, 0, 0])
        except (TypeError, ValueError):
            net = 0.0

    content: list[Any] = []
    for label, png in keyframes:
        _bump_obs(png, "VLA " + label.split(":")[0])
        content.append(label)
        content.append(Image(data=png, format="png"))
    for label, png in end_frames:
        # captions read "<where it stopped> — <direction> — clearance …"; keep the direction
        parts = [x.strip() for x in (label or "").split(" — ")]
        _bump_obs(png, "stopped: " + (parts[1] if len(parts) > 1 else (parts[0] or "view")))
        content.append(label)
        content.append(Image(data=png, format="png"))

    # Then the map, last, so the agent reads the photographs first and the
    # where-am-I question second. A leg can drive 200 steps; four views of the
    # endpoint say what is around the robot but nothing about how that place
    # relates to where it set off, which is precisely what the agent has to judge
    # before it can decide NavGPT VLA stopped short or overshot.
    map_fields: dict[str, Any] = {}
    if VLA_MAP:
        got = _fetch_map()
        if got[0] is not None:
            map_png, map_fields = got
            _bump_obs(map_png, "map", kind="map")
            content.append(
                "the map after this leg — your route so far, ageing blue (start) "
                "to red (now); the red arrow is where NavGPT VLA left you, "
                "pointing the way you face")
            content.append(Image(data=map_png, format="png"))

    status = {
        "instruction_given": route,
        "steps_the_vla_took": executed,
        "step_budget_for_this_call": budget,
        "distance_travelled_m": round(travelled, 2),
        "net_displacement_m": round(net, 2),
        "blocked_events": blocked_count,
        "vla_suggests_arrival": arrival,
        "arrival_note": (
            "This means NavGPT VLA stopped producing new movement — NOT that "
            "you have arrived. It is unreliable about arrival, in both "
            "directions. Check the four views against the instruction yourself "
            "before you STOP."
        ),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
        "reminder": "the episode is still running — only your own stop ends it",
    }
    if _ABLATION:
        status['handoff'] = handoff or 'natural'
        status['review'] = _scheduler.snapshot()
        if _scheduler.delegation == 1 and _vla_calls == 1:
            # Repeated initial rollouts are paired only after canonical input,
            # executed trace, temporal cache and random-state parity pass.
            canonical = {'trace': [{'pose': f['pose'], 'source': f['source_sha256']} for f in trace],
                         'end_pose': end_pose, 'vla_state': spec.audit_state(),
                         'nodes': [{k: n.get(k) for k in ('id', 'position', 'xy', 'visits', 'links', 'route')}
                                   for n in (_fetch_places() or {}).get('places', [])], 'travelled': travelled}
            status['initial_rollout'] = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        _live_log({'review': status['review'], 'handoff': status['handoff']})
    if map_fields:
        # walked_m, revisiting and the leg's own distance_travelled_m describe
        # different things (whole episode vs this call), so keep both and let the
        # map's version carry the map's names.
        for key in ("caption", "direction_cues_px", "places_px", "presentation_transform", "absolute_heading_deg", "up_on_the_map_is", "frame", "facing", "you_are_at_px", "started_at_px",
                    "straight_line_from_start_m", "revisiting_earlier_position",
                    "seen_area_m2", "legend"):
            if key in map_fields:
                status[key] = map_fields[key]
    elif _HAS_MOVE and _last_revisit is not None:
        status["revisiting_earlier_position"] = _last_revisit
    if _episode_over:
        status["end_reason"] = _end_reason
    content.append(json.dumps(status))
    if _ABLATION:
        places = _fetch_places()
        if places is None:
            raise RuntimeError('Required observed graph unavailable')
        content.append(_format_places(places))
    _live_log({"navigate_by_instruction": route, "steps": executed, "arrival": arrival,
               "map": bool(map_fields)})
    _frame_here()
    return content


_ADD_PLACE_DESC = (
    "Record where you are as a named place, so you can come back to it.\n"
    "\n"
    "name is a short handle you will use later — \"kitchen doorway\", \"pool "
    "corner\". caption is one sentence that will let you recognise it when you have "
    "walked away and forgotten: what is around it, which way the exits go, what "
    "made you notice it. Keep the caption under about 120 characters.\n"
    "\n"
    "New places appear on their own as you cover new ground, and the result of "
    "every move tells you when one has appeared without a name. Name it then, while "
    "you are standing there — a place called \"place 7\" is no use to you later, "
    "and the caption is the whole difference between a map and a memory.\n"
    "\n"
    "You can also name somewhere the moment you decide it matters, without waiting "
    "to be asked: if you think you have arrived at the endpoint the instruction "
    "describes, name it BEFORE you look any further. Then, if the search turns out "
    "to be wrong, you can walk back to it and stop there.\n"
    "\n"
    "The label describes WHERE YOU ARE STANDING. It attaches to the place nearest "
    "you, and the result says which and how far away it was.\n"
    "\n"
    "It does not move the robot."
)

_GO_DESC = (
    "Walk to somewhere you recorded earlier: navigate_to_node(2) by its number, or "
    "navigate_to_node(\"kitchen doorway\") by its name. The robot retraces ground it has "
    "already walked, so the route is one it "
    "has proved is walkable.\n"
    "This is for GOING BACK. To cover new ground use navigate_relative(turn_deg, "
    "distance_m), reading the bearing off the ruler burned under whatever you "
    "want to walk towards.\n"
    "Numbers last the whole episode. observe_map() lists every place with its "
    "coordinates, your name for it, and how far walking back would be."
)
def go(target: Any) -> list:
    """Walk back to a recorded place, or to a waypoint of the last VLA rollout
    ("wp N"). See _GO_DESC."""
    _tool_enter("navigate_to_node")
    if _episode_over:
        return ["episode already over ({}); no more moves".format(_end_reason)]
    text = str(target).strip()
    m_wp = re.match(r"^wp\s*(\d+)$", text, flags=re.I)
    if m_wp:
        # keyframes of the last vla_navigate rollout are addressable nodes too
        return go_back_to_waypoint(int(m_wp.group(1)))
    if text and not text.lstrip("-").isdigit() and len(text) <= 2 and text.isalpha():
        return ["there are no lettered spots on this run — travel with "
                "navigate_relative(turn_deg, distance_m), reading the bearing off the "
                "ruler under whatever you want to walk towards. navigate_to_node() returns to "
                "a numbered place you recorded, or one by name."]
    return _go_place(target)


def add_place(name: str, caption: str = "", place: Any = None,
              clause: Any = None) -> list:
    """Attach the agent's own name and caption to a place. See _ADD_PLACE_DESC."""
    _tool_enter("annotate_node")
    if not str(name).strip():
        return ["a place needs a name — something you will recognise later, like "
                '"kitchen doorway"']
    payload = _fetch_places()
    if not payload or not payload.get("places"):
        return ["no places yet — they appear as you travel"]

    target = None
    if place is None:
        target = _label_target(payload)
    elif isinstance(place, (list, tuple)) and len(place) == 2:
        best, best_d = None, None
        for pl in payload["places"]:
            d = math.dist(pl["xy"], [float(place[0]), float(place[1])])
            if best_d is None or d < best_d:
                best, best_d = pl["id"], d
        target = best
    else:
        try:
            target = int(place)
        except (TypeError, ValueError):
            return ["place must be a number, [x, y], or omitted for where you are"]
    known = [pl["id"] for pl in payload["places"]]
    if target not in known:
        return ["no place {} — places are {}".format(target, known)]

    # Anchor the label to the pose the agent is standing at RIGHT NOW, not to the
    # node — the node sits wherever the path crossed the spacing threshold, which
    # can be a couple of metres from the thing the agent means, and _go_place
    # closes that gap so "the corner of the bar" ends up at the corner of the bar.
    #
    # Ask the server for the pose instead of reading _last_pose: that global only
    # refreshes on a pano, so an agent that walks and then names a place would
    # anchor it to wherever it last LOOKED. agent_state costs no steps.
    anchor = None
    try:
        anchor = (_call("{}__agent_state".format(VERB_PREFIX), {}) or {}).get("position")
    except Exception:  # noqa: BLE001 - an unanchored label still routes to the node
        anchor = (_last_pose or {}).get("position")
    # Which clause of the instruction this place satisfies, if the agent said. This
    # is a RETROSPECTIVE binding — "this is where clause 2 came true" — never a plan
    # to execute. Phrasing it as a plan invites re-execution loops, so nothing
    # anywhere asks what has to become true next.
    clause_n = None
    if clause is not None:
        try:
            clause_n = int(clause)
        except (TypeError, ValueError):
            clause_n = None
    replaced = _place_labels.get(int(target))
    _place_labels[int(target)] = {
        "name": str(name)[:60],
        "caption": str(caption)[:140],
        "clause": clause_n,
        "anchor": list(anchor) if anchor else None,
        # the newest frame at the moment of naming, so recall() can show this place
        # later without carrying an image for every place in every listing. [-1] is
        # the only index guaranteed to come from the current pose — the list keeps 8
        # paths across looks, so [0] could be from the previous one.
        "frame": (_recent_frames[-1] if _recent_frames else None),
    }
    _dump_labels()
    pl = next(p for p in payload["places"] if p["id"] == target)
    status = {
        "recorded_place": target,
        "at_xy": pl["xy"],
        "distance_from_you_m": pl["straight_m"],
        "note": ("recorded where you are standing" if pl["straight_m"] < 0.75 else
                 "the nearest place is {:.1f} m from you; your caption was attached "
                 "to it, and going back will return you to where you stand now"
                 .format(pl["straight_m"])),
        **({"clause": clause_n} if clause_n is not None else {}),
        # Say it when a name is overwritten: a silent overwrite deletes a
        # candidate the endgame might have needed.
        **({"replaced_the_name": replaced.get("name")} if replaced else {}),
    }
    _live_log({"annotate_node": target, "name": str(name)[:60], "clause": clause_n})
    return [json.dumps(status), _format_places(payload)]


def _go_place(place: Any) -> list:
    """Retrace to a numbered/named place along ground already walked."""
    global _steps_taken, _episode_over, _end_reason
    if _episode_over:
        return ["episode already over ({}); cannot move".format(_end_reason)]
    payload = _fetch_places()
    if not payload or not payload.get("places"):
        return ["no places yet — they appear as you travel"]

    known = [pl["id"] for pl in payload["places"]]
    inputs: dict[str, Any] = {}
    if place is None:
        # Never point the agent at an internal helper name: telling it to call a
        # tool that is not on its surface is the same failure mode as a briefing
        # describing an unregistered tool.
        return ["navigate_to_node needs a place: a number ({}), a name you gave with "
                "annotate_node, or [x, y] in metres from your start".format(
                    ", ".join(str(i) for i in known))]
    if isinstance(place, (list, tuple)):
        if len(place) != 2:
            return ["coordinates must be [x, y] in metres from your start"]
        try:
            inputs["xy"] = [float(place[0]), float(place[1])]
        except (TypeError, ValueError):
            return ["coordinates must be two numbers, [x, y]"]
    else:
        target = None
        try:
            target = int(place)
        except (TypeError, ValueError):
            # a name the agent gave: pick the nearest place carrying it, because a
            # big room legitimately spans several places
            want = str(place).strip().lower()
            hits = [pid for pid, lab in _place_labels.items()
                    if lab["name"].strip().lower() == want]
            if not hits:
                return ['no place named "{}" — names so far: {}'.format(
                    place, ", ".join(sorted(l["name"] for l in _place_labels.values()))
                    or "(none yet)")]
            by_id = {pl["id"]: pl for pl in payload["places"]}
            target = min(hits, key=lambda pid: by_id[pid]["straight_m"]
                         if pid in by_id else 1e9)
        if target not in known:
            # Refuse here rather than at the server: the agent gets the list of what
            # it can actually ask for, and no round trip is spent to say so.
            return ["no place {} — the places you have are {}".format(
                target, ", ".join(str(i) for i in known))]
        inputs["place"] = target

    try:
        res = _call("{}__retrace_to".format(VERB_PREFIX), inputs)
    except Exception as exc:  # noqa: BLE001
        return ["cannot go there ({})".format(exc)]

    _steps_taken = int(res.get("step_count") or _steps_taken)
    if res.get("terminated") or res.get("truncated"):
        _episode_over = True
        _end_reason = "step_budget_exhausted"

    # Finish the last stretch on foot. Two reasons it has to happen here:
    #
    # * the server deliberately will not warp it — a recorded route ends within the
    #   snap radius of a place, not on it, and closing that by teleport would be a
    #   free jump of up to a whole spacing;
    # * where the agent NAMED a place is not where the node sits. The node is
    #   wherever the path crossed the spacing threshold; the anchor is the pose the
    #   agent was standing at when it said "the corner of the bar".
    #
    # So walk to the anchor if there is one, and to the place itself otherwise —
    # charged and collision-checked like any other step, never warped.
    approached = None
    label = _place_labels.get(res.get("place"))
    goal_pos = (label or {}).get("anchor")
    if goal_pos is None:
        goal_pos = next((pl.get("position") for pl in payload["places"]
                         if pl["id"] == res.get("place")), None)
    if goal_pos and not _episode_over:
        pose = (res.get("pose") or {}).get("position")
        if pose:
            dx = float(goal_pos[0]) - float(pose[0])
            dz = float(goal_pos[2]) - float(pose[2])
            gap = math.hypot(dx, dz)
            if gap > 0.4:
                heading = float((res.get("pose") or {}).get("heading_rad") or 0.0)
                bearing = math.atan2(-dx, -dz) - heading
                # wrap to (-pi, pi] so a small correction never turns the long way
                bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
                out = _call("{}__step_hightolow".format(VERB_PREFIX),
                            {"angle_rad": bearing, "distance_m": gap})
                _steps_taken = int(out.get("step_count") or _steps_taken)
                approached = round(float(out.get("walked_m") or 0.0), 2)
                if out.get("terminated") or out.get("truncated"):
                    _episode_over = True
                    _end_reason = "step_budget_exhausted"

    pano = _pano()
    clear = _clearance_summary(pano.get("scan"), pano)
    content: list[Any] = []
    content += _pano_content(pano, clear,
                             "back at place {}".format(res.get("place")))

    got = _fetch_map()
    if got[0] is not None:
        _bump_obs(got[0], "map", kind="map")
        content.append("the map, with your route so far and every place on it")
        content.append(Image(data=got[0], format="png"))

    status = {
        "went_to_place": res.get("place"),
        "place_xy": res.get("place_xy"),
        "arrived": res.get("arrived"),
        "route": res.get("route"),
        "retraced_m": res.get("retraced_m"),
        "steps_charged": res.get("steps_charged"),
        # EventSink._parse_step_result sniffs tool results for this key to record
        # the episode's steps, called_stop and end_reason. A movement tool that
        # omits it goes unrecorded.
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
    }
    if approached is not None:
        status["final_approach_m"] = approached
    if res.get("stopped_short_m") is not None:
        status["stopped_short_m"] = res["stopped_short_m"]
    if _episode_over:
        status["end_reason"] = _end_reason
    content.append(json.dumps(status))
    after = _fetch_places()
    if after:
        content.append(_format_places(after))
    _live_log({"navigate_to_node": res.get("place"), "steps_charged": res.get("steps_charged")})
    return content


_trace: list[dict[str, Any]] = []
_trace_keys: list[int] = []


def _set_trace(trace):
    """Keep the last rollout so the agent can return to a waypoint it saw."""
    global _trace, _trace_keys
    _trace = list(trace or [])
    k = max(1, min(MAX_KEYFRAMES, len(_trace)))
    _trace_keys = ([round(j * (len(_trace) - 1) / max(1, k - 1)) for j in range(k)]
                   if k > 1 else [0])


def go_back_to_waypoint(n: Any) -> list:
    """Retrace to waypoint n of the last VLA rollout (reached via navigate_to_node("wp N"))."""
    m_wp = re.match(r"^wp\s*(\d+)$", str(n).strip(), flags=re.I)
    if m_wp:
        n = int(m_wp.group(1))
    global _steps_taken, _episode_over, _end_reason
    if _episode_over:
        return ["episode already over ({}); cannot move".format(_end_reason)]
    if not _trace or not _trace_keys:
        return ["no route recorded yet — call navigate_by_instruction first"]
    try:
        idx = int(n)
    except (TypeError, ValueError):
        return ["n must be a waypoint number, 1..{}".format(len(_trace_keys))]
    if not 1 <= idx <= len(_trace_keys):
        return ["waypoint {} does not exist; there are {}".format(idx, len(_trace_keys))]

    target = _trace[_trace_keys[idx - 1]]
    pose = target["pose"] or {}
    if not pose.get("position"):
        return ["waypoint {} has no recorded pose".format(idx)]
    # The Environment walks the recorded path back to the waypoint and charges one
    # step per 0.25 m retraced, so the budget, path length and SPL see the walk.
    try:
        res = _call("{}__retrace_path".format(VERB_PREFIX),
                    {"position": list(pose["position"]),
                     "rotation_wxyz": pose.get("rotation_wxyz")})
    except Exception as exc:  # noqa: BLE001 - report in words, never crash the episode
        return ["cannot return to waypoint {} ({})".format(idx, exc)]
    _steps_taken = int(res.get("step_count") or _steps_taken)
    if res.get("terminated") or res.get("truncated"):
        _episode_over = True
        _end_reason = "step_budget_exhausted"

    out = _pano()
    clear = _clearance_summary(out.get("scan"), out)
    content: list[Any] = []
    content += _pano_content(out, clear, "at waypoint {}".format(idx))
    status = {
        "returned_to_waypoint": idx,
        "arrived": res.get("arrived"),
        "retraced_m": res.get("retraced_m"),
        "steps_charged": res.get("steps_charged"),
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": _episode_over,
    }
    content.append(json.dumps(status))
    _live_log({"navigate_to_node": idx, **status})
    return content


_ARRIVE_DESC = (
    "END THE EPISODE BY SAYING WHERE YOU ARE STOPPING. This is how you finish; it "
    "is permanent.\n"
    "\n"
    'terminate_episode("here") stops where you stand. terminate_episode(3) walks back to place 3 first '
    "and stops there — the retrace walks back over ground you "
    "already covered.\n"
    "\n"
    "WHY THIS TOOL EXISTS instead of a plain STOP. Ending the episode is a CHOICE "
    "BETWEEN CANDIDATES — where you stand, or somewhere you recorded — rather than a "
    "reflex from wherever the robot happens to be. Both errors cost the same episode "
    "and neither is the safe default: always preferring the place you bound to the "
    "last clause loses the episodes where you were already at the end, and always "
    "preferring where you stand loses the episodes where the route plainly continued "
    "past you. So name your target and say why it beats the other.\n"
    "\n"
    "You do not need to be certain, see the landmark, or be able to name the room. "
    "You need to be within 3 metres of where the last clause points."
)


def arrive(where: Any = "here") -> list:
    """Terminal action for the place-carrying conditions: stop, having named where.

    Profiles without a node graph can only stop where they stand. For
    `planner_memory`/`planner_vla_memory`, ending the episode is a CHOICE BETWEEN
    CANDIDATES rather than a reflex from wherever the robot happens to stand: the
    agent may name a recorded place to retrace to before stopping.

    Telemetry is deliberately identical to a primitive STOP: the same `steps_taken_total`
    key the runner's EventSink keys on, the same `end_reason`, so a run using
    terminate_episode() is scored by exactly the same path as one using action 0.
    """
    _tool_enter("terminate_episode")
    global _steps_taken, _episode_over, _end_reason
    if _episode_over:
        return ["episode already over ({}); nothing left to do".format(_end_reason)]

    content: list[Any] = []
    text = str(where).strip()
    here = text.lower() in ("", "here", "now", "this", "this spot", "current")
    if not here and not _HAS_PLACES:
        return ["this profile has no node graph, so there is nowhere to retrace to — "
                'terminate_episode("here") stops where you stand']
    if not here:
        # Retrace first. _go_place refuses in words when the target is unknown or
        # unreachable; in that case DO NOT stop — a refusal must not silently end the
        # episode at a spot the agent did not choose.
        got = _go_place(where)
        content += got
        blob = " ".join(str(x) for x in got if isinstance(x, str))
        if _episode_over:
            return content
        if ('"went_to_place"' not in blob) or ('"arrived": false' in blob):
            content.append(
                "did NOT stop: the retrace to {!r} did not arrive, so the episode is "
                "still running. Fix the target or call terminate_episode(\"here\").".format(text))
            return content

    outputs = _call("{}__step_discrete".format(VERB_PREFIX), {"action": 0})
    _steps_taken += 1
    _episode_over = True
    _end_reason = "stop_called"
    status = {
        "stopped_at": "here" if here else text,
        "actions_executed": [0],
        "steps_taken_total": _steps_taken,
        "steps_remaining_approx": max(0, STEP_BUDGET - _steps_taken),
        "episode_over": True,
        "end_reason": _end_reason,
        "terminated": bool(outputs.get("terminated")),
    }
    content.append(json.dumps(status))
    _live_log({"arrive": status["stopped_at"], "steps": _steps_taken})
    return content


def _candidates_block(limit: int = 12) -> str | None:
    """EVERY place, on the tail of every look, in route order.

    Otherwise the place table is reachable only by calling observe_map(), so at the
    moment the navigator decides where to stop the graph would be off screen. This
    puts it where the decision is made.

    ALL of them, not a shortlist: the right stopping place is typically mid-route,
    neither the newest nor the closest, so a top-3 shortlist would often hide it.
    Graphs are small, so listing all of them costs a few short lines.

    Route order (by id) rather than by distance, because ids are chronological: the
    list then reads as the journey, which is the frame the instruction is written in.
    Numbers lead, because retrieval is mostly by number.
    """
    payload = _fetch_places()
    if not payload:
        return None
    places = payload.get("places") or []
    if not places:
        return None
    nearest, nearest_d = None, None
    for pl in places:
        d = pl.get("straight_m")
        if d is not None and (nearest_d is None or d < nearest_d):
            nearest, nearest_d = pl["id"], d
    out = []
    for pl in sorted(places, key=lambda pl: pl["id"])[:limit]:
        lab = _place_labels.get(pl["id"]) or {}
        name = lab.get("name") or "unnamed"
        cost = pl.get("route_m")
        out.append("  {}{}: {}{} — {:.1f} m away{}".format(
            pl["id"], " (nearest)" if pl["id"] == nearest else "", name,
            "" if lab.get("clause") is None else " [clause {}]".format(lab["clause"]),
            pl.get("straight_m", 0.0),
            ", {:.0f} steps to return".format(cost / 0.25) if cost else ""))
    extra = ("" if len(places) <= limit else
             "\n  ... and {} more, all on the map".format(len(places) - limit))
    return ("places you have recorded, in the order you reached them — terminate_episode(n) goes "
            "back to one and stops there:\n" + "\n".join(out) + extra)


_RECALL_DESC = (
    "Show the picture that was taken at a place you recorded — observe_node(3).\n"
    "\n"
    "Use it when a caption is not enough to tell two places apart, which is the one "
    "moment an image of a place is worth asking for. It is deliberately NOT attached to "
    "the place list: an image added to this conversation is re-sent on every "
    "following turn, so images per place quickly become expensive. So the list "
    "carries words, the map carries where "
    "everything is, and this carries the pixels only when you ask.\n"
    "It does not move the robot."
)


def recall(place: Any) -> list:
    """The frame a place was recorded at. On demand, never in a listing."""
    _tool_enter("observe_node")
    try:
        pid = int(place)
    except (TypeError, ValueError):
        return ["observe_node takes a place NUMBER, e.g. observe_node(3) — {!r} is not one. The "
                "numbers are in the place list on every look.".format(place)]
    lab = _place_labels.get(pid) or {}
    # the agent's own naming frame if it named the node, else the frame captured
    # from the closest look or rollout pose (every node gets one with live output)
    path = lab.get("frame") or _place_frames.get(pid)
    if not path or not frame_exists(path):
        known = sorted(k for k in set(_place_labels) | set(_place_frames)
                       if ((_place_labels.get(k) or {}).get("frame") or _place_frames.get(k)))
        return ["place {} has no recorded picture{}".format(
            pid, "; pictures exist for {}".format(known) if known else
            " — none of the places have one yet")]
    try:
        png = read_frame(path)
    except OSError as exc:
        return ["could not read the frame for place {}: {}".format(pid, exc)]
    return ["place {} — {}: {}".format(pid, lab.get("name") or "unnamed",
                                       lab.get("caption") or ""),
            Image(data=png, format="png")]


def _instruction_id() -> str:
    """Episode id for NavGPT VLA's per-episode bookkeeping. It only labels
    the server's debug GIF, so a stable per-process value is enough."""
    return os.environ.get("NAVGPT_EPISODE_ID", "episode")


def _bump_obs(png: bytes, label: str | None = None, kind: str = "view") -> None:
    global _obs_count
    _obs_count += 1
    _live_frame(png, label, kind)


def _register(name, description, fn):
    if _ABLATION and name not in _CAPABILITIES[BUNDLE]:
        return
    if not _ABLATION:
        mcp.tool(name=name, description=description)(fn)
        return
    import functools
    if name == 'terminate_episode' and BUNDLE == 'core':
        def stop_here() -> list:
            return arrive('here')
        fn = stop_here
        description = 'End this episode at the current position. This operation cannot return to a node.'
    @functools.wraps(fn)
    def serial(*args, **kwargs):
        with _scheduler.lock:
            correction = name in ('navigate_relative', 'navigate_to_node', 'terminate_episode')
            if correction:
                _scheduler.correction()
            try:
                if _scheduler.failed and name != 'terminate_episode':
                    raise RuntimeError('Ablation episode invalid; terminate and exclude it')
                result = fn(*args, **kwargs)
            except Exception as exc:
                _scheduler.fail()
                return [json.dumps({'ablation_error': str(exc), 'review': _scheduler.snapshot()})]
            if INTERFACE == 'text_labels':
                paired = []
                for item in result:
                    if isinstance(item, Image) and item.data:
                        fields = _presentation_by_hash.get(hashlib.sha256(item.data).hexdigest())
                        if fields:
                            paired.append(json.dumps(fields))
                    paired.append(item)
                result = paired
            result.append(json.dumps({'review': _scheduler.snapshot()}))
            return result
    if name == 'navigate_by_instruction':
        description = ('Execute a navigation instruction using the VLA. Pass the full original route first. '
                       'For later corrections, use current observations to give a self-contained remaining route '
                       'with supported landmarks/directions and the original stopping condition; omit completed clauses. '
                       'At a scheduled review, repeat the active instruction (or omit it) only when it still fits. '
                       'Pass a revised instruction explicitly when correction is needed. Visual history is retained. '
                       'Natural stop/cap returns give the Planner endpoint authority. Review timing is runtime controlled.')
    elif name in ('observe_forward', 'observe_panorama'):
        description = 'Inspect current camera evidence. Labels state the active bearing convention. Observation does not move the robot.'
    elif name == 'observe_map':
        description = 'Inspect the observed route map, stable node IDs and its stated coordinate convention.'
    mcp.tool(name=name, description=description)(serial)


# ── registration: the only real gate ──
#
# The surface uses the systematic <verb>_<object> names (see docs/USAGE.md).
# Python function names below are internal; only the registered name reaches the
# model. The basic profile stops through primitive action 0.
_register("observe_forward", _OBSERVE_DESC, observe)
if CONDITION != "planner_basic":
    _register("terminate_episode", _ARRIVE_DESC, arrive)
if not _HAS_VLA:
    # primitives only where there is no specialist to replace them
    _register("navigate_primitive", _STEP_DESC, step)
if _HAS_PANO:
    _register("observe_panorama", _LOOK_DESC, look_around)
if _HAS_MOVE:
    _register("observe_map", _MAP_DESC, local_map)
    _register("navigate_relative", _MOVE_DESC, move)
if _HAS_PLACES:
    _register("navigate_to_node", _GO_DESC, go)
    _register("annotate_node", _ADD_PLACE_DESC, add_place)
if _HAS_PLACES and not _HAS_VLA:
    _register("observe_node", _RECALL_DESC, recall)
if _HAS_VLA:
    _register("navigate_by_instruction", _VLA_DESC, vla_navigate)


if __name__ == "__main__":
    mcp.run()
