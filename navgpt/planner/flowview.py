"""Episode flow view: trajectory map + turn-by-turn tool flow + the frames the agent saw.

``navgpt.planner.viewer`` serves it at ``/run/<name>/ep/<i>`` straight from the run
directory: nothing is regenerated, and frames stream lazily as thumbnails.

Frame attribution has two sources, in order of trust:

1. ``live/epNNNN/frames.jsonl`` — written by the MCP tool server at the moment each
   frame enters the agent's context, tagged with the tool and a label. Exact.
2. Inference from the tool and its result text (runs without a manifest). It is a
   model of the surface, and the page says when it disagrees with the count on disk.

Nothing here touches the environment service; it reads what the run wrote.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable
from .turns import PlannerTurnCounter

# Image cost in Claude image tokens ~= w*h/750, for the two view layouts and the map.
IMG_TOKENS = {"four": 350, "strip": 617, "observe": 350, "map": 752}

TOOL_COLOUR = {
    "observe_panorama": "#4da3ff", "observe_forward": "#7ee081", "observe_map": "#c08bff",
    "observe_node": "#b98bff",
    "navigate_relative": "#ffb454", "navigate_to_node": "#ff7ab6",
    "navigate_by_instruction": "#5ad4e6", "navigate_primitive": "#ff5f56",
    "annotate_node": "#ffd866", "terminate_episode": "#ff5f56",
}

# What each tool costs and does — keyed by the registered name. The panel shows the
# run's own registered tools; this only supplies the cost/family/one-liner.
TOOL_INFO = {
    "observe_forward":         ("free",             "observe",  "forward camera, full size"),
    "observe_panorama":        ("free",             "observe",  "4 views; clearance, bearing ruler and position burned in"),
    "observe_map":             ("free",             "observe",  "top-down observed map + the node table"),
    "observe_node":            ("free",             "observe",  "the picture stored for a node"),
    "navigate_relative":       ("1 step / 0.25 m",  "navigate", "turn (free) then walk, navmesh-routed; returns 4 views"),
    "navigate_to_node":        ("1 step / 0.25 m",  "navigate", "retrace to a node (id, name or wp<k>) over walked edges; 4 views + map"),
    "navigate_by_instruction": ("the VLA's steps",  "navigate", "specialist drives the whole instruction; 8 keyframes + 4 views + map"),
    "navigate_primitive":      ("1 step / action",  "navigate", "primitives: 0 stop · 1 fwd 0.25 m · 2/3 turn 15°"),
    "annotate_node":           ("free",             "annotate", "name + caption the node under you; stores the exact pose"),
    "terminate_episode":       ("retrace if a node", "terminate", "END the episode here, or back at a recorded node"),
}

VIEW_LABELS = {"four": ["ahead", "right", "behind", "left"]}
FRAME_RE = re.compile(r"obs_(\d+)_step(\d+)\.(png|jpe?g|webp)$")


def run_config(path: str | Path) -> dict:
    """The run's summary.json config, for the knobs that change what the agent saw."""
    p = Path(path)
    for d in (p.parent, p.parent.parent):
        f = d / "summary.json"
        if f.exists():
            try:
                return (json.loads(f.read_text()) or {}).get("config") or {}
            except ValueError:
                pass
    return {}


def load_labels(path: str | Path, index: int | None) -> dict[int, dict]:
    """labels_<i>.json — the agent's own names for places, keyed by place id."""
    if index is None:
        return {}
    f = Path(path).parent / "labels_{}.json".format(index)
    if not f.exists():
        return {}
    try:
        raw = json.loads(f.read_text())
    except ValueError:
        return {}
    out = {}
    for k, v in (raw or {}).items():
        try:
            out[int(k)] = v or {}
        except ValueError:
            continue
    return out


def load_episode(path: str | Path) -> dict:
    path = str(path)
    recs = []
    for line in open(path):
        if not line.strip().startswith("{"):
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    ep: dict[str, Any] = {"path": path, "turns": [], "places": [], "tools": [],
                          "cfg": run_config(path)}
    say, think = [], []
    pending = []          # a list, not one slot: several tools can be called at once
    counter = PlannerTurnCounter()
    for r in recs:
        k = r.get("kind")
        counter.observe(k, r)
        if k == "episode_meta":
            ep.update({key: r.get(key) for key in
                       ("index", "episode_id", "scene_id", "instruction",
                        "geodesic_distance")})
        elif k == "session_inputs":
            ep["condition"] = r.get("condition")
            ep["model"] = r.get("model")
            ep["system_prompt"] = r.get("system_prompt") or ""
        elif k == "system_init":
            ep["tools"] = [t.replace("mcp__env__", "") for t in (r.get("tools") or [])]
            ep["model"] = ep.get("model") or r.get("model")
        elif k == "assistant_text":
            say.append(r.get("text") or "")
        elif k == "thinking":
            think.append(r.get("text") or "")
        elif k == "tool_use":
            pending.append({
                "n": len(ep["turns"]) + len(pending) + 1, "t": r.get("t"),
                "planner_turn": counter.count,
                "tool": (r.get("name") or "").replace("mcp__env__", ""),
                "args": r.get("input") or {},
                "say": "\n".join(s for s in say if s.strip()),
                "think": "\n".join(t for t in think if t.strip()),
                "fields": {}, "captions": []})
            say, think = [], []
        elif k == "tool_result" and pending:
            cur = pending.pop(0)              # results arrive in call order
            for t in [str(x) for x in (r.get("texts") or [])]:
                s = t.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        cur["fields"].update(json.loads(s))
                        continue
                    except ValueError:
                        pass
                if s:
                    cur["captions"].append(s)
            ep["turns"].append(cur)
        elif k == "episode_metrics":
            ep["metrics"] = r.get("metrics") or {}
        elif k == "episode_places":
            ep["places"] = r.get("places") or []
        elif k == "episode_trajectory":
            ep["traj"] = r
        elif k == "driver_error":
            ep.setdefault("driver_errors", []).append(r.get("error"))
        elif k == "result":
            res = r.get("result") or {}
            ep["usage"] = res.get("usage") or {}
            ep["cost"] = res.get("total_cost_usd")
            ep["subtype"] = res.get("subtype")
    for cur in pending:                           # calls with no recorded result
        cur["fields"]["(no result recorded)"] = True
        ep["turns"].append(cur)
    ep["planner_turn_count"] = counter.count
    ep["labels"] = load_labels(path, ep.get("index"))
    for pl in ep["places"]:
        lab = ep["labels"].get(int(pl.get("id", -1)))
        if lab and not pl.get("name"):
            pl["name"] = lab.get("name")
            pl["caption"] = lab.get("caption")
    _annotate(ep)
    return ep


def _xy(pos, start):
    """World -> the agent-facing frame: metres east / north of the start."""
    return (float(pos[0]) - float(start[0]), -(float(pos[2]) - float(start[2])))


def _annotate(ep: dict) -> None:
    """Attach per-turn position and image cost, and episode-level geometry."""
    traj = ep.get("traj") or {}
    path = traj.get("agent_path") or []
    ep["start"] = path[0] if path else (traj.get("start_position") or [0, 0, 0])
    ep["path_xy"] = [_xy(p, ep["start"]) for p in path]
    ep["ref_xy"] = [_xy(p, ep["start"]) for p in (traj.get("reference_path") or [])]
    goals = traj.get("goals") or []
    ep["goal_xy"] = _xy(goals[0], ep["start"]) if goals else None
    ep["success_distance"] = float(traj.get("success_distance") or 3.0)
    for pl in ep["places"]:
        pl["xy_f"] = (pl.get("xy") or _xy(pl["position"], ep["start"]))

    if ep["goal_xy"] and ep["path_xy"]:
        gx, gy = ep["goal_xy"]
        ds = [math.hypot(x - gx, y - gy) for x, y in ep["path_xy"]]
        i = min(range(len(ds)), key=lambda j: ds[j])
        ep["closest"] = {"i": i, "d": ds[i], "xy": ep["path_xy"][i]}
        ep["end"] = {"d": ds[-1], "xy": ep["path_xy"][-1]}

    # A turn's position: steps_taken_total indexes the walked path (one point per
    # movement primitive). Proportional rather than 1:1 because de-duplication drops
    # repeats (a turn in place adds a step but not a point).
    total = max((t["fields"].get("steps_taken_total") or 0) for t in ep["turns"]) \
        if ep["turns"] else 0
    n = len(ep["path_xy"])
    layout = str(ep["cfg"].get("pano_layout") or "four")
    last = ep["path_xy"][0] if ep["path_xy"] else None
    for t in ep["turns"]:
        s = t["fields"].get("steps_taken_total")
        if s is not None and total and n:
            last = ep["path_xy"][min(n - 1, int(round(s / total * (n - 1))))]
        t["at"] = last                # free tools happen where the robot already stands
        t["steps"] = s
        t["imgs"], t["img_tokens"] = _image_cost(t, layout)
    ep["layout"] = layout


def caps_list(turn: dict) -> list[str]:
    return turn.get("captions") or []


def _image_cost(turn: dict, layout: str) -> tuple[int, int]:
    """Images a turn put in the context, and roughly what they cost in tokens.

    Inferred from the tool and the run's pano_layout, because the transcript records
    only the text parts of a result. Used for the token estimate always, and for
    frame attribution only when the run has no frames.jsonl manifest.
    """
    tool = turn["tool"]
    f, caps = turn["fields"], " ".join(turn.get("captions") or [])
    refused = any(
        s in caps for s in ("no route to place", "unknown place", "cannot go there",
                            "no places yet", "no lettered spots",
                            "episode already over", "no more looking",
                            "no more moves", "map unavailable"))
    if refused and tool in ("navigate_to_node", "observe_panorama", "observe_forward",
                            "observe_map", "navigate_by_instruction", "navigate_relative",
                            "terminate_episode"):
        return 0, 0
    per_look = 4 if layout == "four" else 1
    look_tok = IMG_TOKENS["four"] * 4 if layout == "four" else IMG_TOKENS["strip"]
    if tool == "observe_panorama":
        return per_look, look_tok
    if tool == "observe_forward":
        return 1, IMG_TOKENS["observe"]
    if tool == "navigate_primitive":
        return 0, 0                                # text only
    if tool == "observe_node":
        if "no recorded picture" in caps or "no stored frame" in caps:
            return 0, 0
        return 1, IMG_TOKENS["observe"]
    if tool == "terminate_episode":
        # terminate_episode(<node>) walks back and returns the four views + the map;
        # terminate_episode("here") returns text only.
        if any(c.startswith("back at place") for c in caps_list(turn)):
            return per_look + 1, look_tok + IMG_TOKENS["map"]
        return 0, 0
    if tool == "navigate_relative":
        if f.get("walked_m") == 0 and f.get("blocked"):
            return 0, 0
        return per_look, look_tok
    if tool == "observe_map":
        return 1, IMG_TOKENS["map"]
    if tool == "navigate_to_node":
        if "went_to_place" in f:                   # graph retrace: views + map
            return per_look + 1, look_tok + IMG_TOKENS["map"]
        if any(c.startswith("back at") for c in caps_list(turn)) or "retraced_m" in f:
            return per_look, look_tok              # keyframe warp: views only
        return 0, 0                                # refused
    if tool == "navigate_by_instruction":
        wp = sum(1 for c in caps_list(turn) if c.startswith("waypoint"))
        has_map = any("the map" in c for c in caps_list(turn))
        n = wp + per_look + (1 if has_map else 0)
        return n, (wp * IMG_TOKENS["observe"] + look_tok
                   + (IMG_TOKENS["map"] if has_map else 0))
    return 0, 0


# ── the real images the agent saw ──


def live_dir(ep_path: str | Path, index: int | None) -> Path | None:
    if index is None:
        return None
    d = Path(ep_path).parent / "live" / "ep{:04d}".format(int(index))
    return d if d.is_dir() else None


def list_frames(d: Path) -> list[str]:
    def key(name: str):
        m = FRAME_RE.match(name)
        return (int(m.group(1)), int(m.group(2))) if m else (1 << 30, 0)
    return sorted((f for f in os.listdir(d) if FRAME_RE.match(f)), key=key)


def load_manifest(d: Path) -> list[dict]:
    """frames.jsonl: one line per frame, in the order it entered the context."""
    f = d / "frames.jsonl"
    if not f.is_file():
        return []
    out = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _guess_labels(t: dict, n: int, labels: list[str]) -> list[tuple[str, bool]]:
    """(label, is_map) for each of the n frames a turn is inferred to have drawn."""
    out = []
    tool = t["tool"]
    for j in range(n):
        is_map = tool == "observe_map" or (
            tool in ("navigate_to_node", "terminate_episode") and j == n - 1 and n == 5)
        if tool == "observe_forward":
            lab = "forward view"
        elif tool == "observe_node":
            lab = "recalled node"
        elif is_map:
            lab = "map"
        elif tool == "navigate_by_instruction":
            caps = [c for c in caps_list(t) if c.startswith("waypoint")]
            if j < len(caps):
                lab = "VLA wp %s" % caps[j].split(":")[0].split()[-1]
            elif j < len(caps) + len(labels):
                lab = "stopped: " + labels[j - len(caps)]
            else:
                lab = "map"
            is_map = lab == "map"
        else:
            lab = labels[j % len(labels)]
        out.append((lab, is_map))
    return out


def attach_frames(ep: dict, d: Path | None = None) -> None:
    """Give each turn the frames it put in the agent's context.

    Each shot is ``{"label", "file", "is_map", "step"}``; the caller decides how the
    file is delivered (the viewer serves it by URL).
    """
    d = d or live_dir(ep["path"], ep.get("index"))
    ep["frames_found"] = 0
    ep["frames_source"] = None
    for t in ep["turns"]:
        t["shots"] = []
    if not d:
        return
    files = list_frames(d)
    ep["frames_found"] = len(files)
    want = sum(t["imgs"] for t in ep["turns"])

    manifest = load_manifest(d)
    if manifest and all(m.get("file") for m in manifest):
        # exact attribution: frames carry the turn they belong to. Turns are matched
        # by call order per tool, because the manifest knows the tool but not the
        # planner-side turn number.
        ep["frames_source"] = "manifest"
        by_tool: dict[str, list[dict]] = {}
        for t in ep["turns"]:
            by_tool.setdefault(t["tool"], []).append(t)
        cursor: dict[str, int] = {}
        grouped: dict[tuple[str, int], list[dict]] = {}
        for m in manifest:
            tool = str(m.get("tool") or "")
            key = (tool, int(m.get("call") or 0))
            grouped.setdefault(key, []).append(m)
        for (tool, _call), frames in grouped.items():
            turns = by_tool.get(tool) or []
            i = cursor.get(tool, 0)
            if i >= len(turns):
                continue
            turn = turns[i]
            cursor[tool] = i + 1
            for m in frames:
                turn["shots"].append({
                    "label": str(m.get("label") or tool), "file": m["file"],
                    "is_map": bool(m.get("kind") == "map"), "step": m.get("step")})
        ep["frames_aligned"] = (sum(len(t["shots"]) for t in ep["turns"]) == len(files))
        return

    # inference fallback
    ep["frames_source"] = "inferred"
    ep["frames_aligned"] = (want == len(files))
    i = 0
    labels = VIEW_LABELS.get(ep["layout"], ["strip: left | ahead | right | behind"])
    for t in ep["turns"]:
        for lab, is_map in _guess_labels(t, t["imgs"], labels):
            if i >= len(files):
                break
            m = FRAME_RE.match(files[i])
            t["shots"].append({"label": lab, "file": files[i], "is_map": is_map,
                               "step": int(m.group(2)) if m else None})
            i += 1


# ── drawing ──


def svg_map(ep: dict, w: int = 560, h: int = 460, pad: int = 42) -> str:
    pts = list(ep["path_xy"]) + list(ep["ref_xy"])
    if ep["goal_xy"]:
        pts.append(ep["goal_xy"])
    pts += [pl["xy_f"] for pl in ep["places"]]
    if not pts:
        return "<p class=dim>no trajectory recorded</p>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    r = ep["success_distance"]
    lo_x, hi_x = min(xs) - r, max(xs) + r
    lo_y, hi_y = min(ys) - r, max(ys) + r
    span = max(hi_x - lo_x, hi_y - lo_y, 4.0)
    cx, cy = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
    k = (min(w, h) - 2 * pad) / span          # px per metre

    def P(p):
        return (w / 2 + (p[0] - cx) * k, h / 2 - (p[1] - cy) * k)

    out = ['<svg viewBox="0 0 %d %d" class="map">' % (w, h)]
    out.append('<rect width="%d" height="%d" fill="#0d0f13"/>' % (w, h))
    step = 1 if span < 12 else 2 if span < 26 else 5
    g = int(math.floor(lo_x / step)) * step
    while g <= hi_x:
        x = P((g, 0))[0]
        out.append('<line x1="%.1f" y1="0" x2="%.1f" y2="%d" stroke="#1b1f27"/>' % (x, x, h))
        g += step
    g = int(math.floor(lo_y / step)) * step
    while g <= hi_y:
        y = P((0, g))[1]
        out.append('<line x1="0" y1="%.1f" x2="%d" y2="%.1f" stroke="#1b1f27"/>' % (y, w, y))
        g += step

    if len(ep["ref_xy"]) > 1:
        d = " ".join("%.1f,%.1f" % P(p) for p in ep["ref_xy"])
        out.append('<polyline points="%s" fill="none" stroke="#5b6472" '
                   'stroke-width="2.5" stroke-dasharray="6 5"/>' % d)
    if ep["goal_xy"]:
        gx, gy = P(ep["goal_xy"])
        out.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#1e5c2e" '
                   'fill-opacity="0.28" stroke="#3ddc63" stroke-dasharray="4 4"/>'
                   % (gx, gy, r * k))
        out.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="#3ddc63"/>' % (gx, gy))

    pa = ep["path_xy"]
    for i in range(len(pa) - 1):
        f = i / max(1, len(pa) - 2)
        col = "#%02x%02x%02x" % (int(56 + f * 184), int(132 - f * 70), int(255 - f * 193))
        x1, y1 = P(pa[i])
        x2, y2 = P(pa[i + 1])
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
                   'stroke-width="3" stroke-linecap="round"/>' % (x1, y1, x2, y2, col))

    # places: named ones filled amber, auto-created ones outlined
    for pl in ep["places"]:
        px, py = P(pl["xy_f"])
        named = bool(pl.get("name"))
        out.append('<circle cx="%.1f" cy="%.1f" r="10" fill="%s" stroke="#ffc440" '
                   'stroke-width="2"><title>%s</title></circle>'
                   % (px, py, "#5a4410" if named else "#18191f",
                      html.escape("%s %s" % (pl["id"], pl.get("name") or "(unnamed)"))))
        out.append('<text x="%.1f" y="%.1f" class="badge">%s</text>' % (px, py + 3.5, pl["id"]))

    if ep.get("closest"):
        x, y = P(ep["closest"]["xy"])
        out.append('<circle cx="%.1f" cy="%.1f" r="7" fill="none" stroke="#3ddc63" '
                   'stroke-width="2"/>' % (x, y))
    if pa:
        x, y = P(pa[-1])
        out.append('<path d="M%.1f,%.1f l-7,-11 l14,0 z" fill="#ff5f56"/>' % (x, y))
        sx, sy = P(pa[0])
        out.append('<circle cx="%.1f" cy="%.1f" r="5.5" fill="#3884ff"/>' % (sx, sy))

    for t in ep["turns"]:
        if not t.get("at"):
            continue
        x, y = P(t["at"])
        out.append('<circle id="m%d" cx="%.1f" cy="%.1f" r="3" fill="#e8e8ee" '
                   'fill-opacity="0.45" class="tm"/>' % (t["n"], x, y))

    bar = step * k
    out.append('<line x1="%d" y1="%d" x2="%.1f" y2="%d" stroke="#8b93a1" '
               'stroke-width="2"/>' % (pad, h - 16, pad + bar, h - 16))
    out.append('<text x="%.1f" y="%d" class="ax">%d m</text>' % (pad + bar + 6, h - 12, step))
    out.append('<text x="%d" y="18" class="ax">north is up · x east / y north, '
               'metres from the start</text>' % pad)
    out.append("</svg>")
    return "".join(out)


def legend_html() -> str:
    return ('<div class="legend"><span>— — reference route</span>'
            '<span><b style="color:#3884ff">●</b> start</span>'
            '<span><b style="color:#3ddc63">●</b> goal + 3 m circle</span>'
            '<span>blue→red walked path</span>'
            '<span><b style="color:#3ddc63">○</b> closest approach</span>'
            '<span><b style="color:#ff5f56">▲</b> stop</span>'
            '<span><b style="color:#ffc440">●</b> named place · '
            '<b style="color:#ffc440">○</b> auto place</span>'
            '<span>· hover a turn to light where the robot stood</span></div>')


def surface_panel(ep: dict) -> str:
    """The run's own tool surface: what was registered (system_init), how often this
    episode used each tool, and what it costs."""
    registered = sorted(ep.get("tools") or [],
                        key=lambda t: (TOOL_INFO.get(t, ("", "zz", ""))[1], t))
    use = Counter(t["tool"] for t in ep["turns"])
    cond = ep.get("condition") or ep["cfg"].get("condition") or "?"
    rows = []
    for name in registered:
        cost, fam, what = TOOL_INFO.get(name, ("?", "?", ""))
        rows.append(
            '<div class="trow"><span class="tname" style="color:%s">%s</span>'
            '<span class="tcnt">%s</span><span class="tcost">%s</span>'
            '<span class="tfam">%s</span><span class="twhat">%s</span></div>'
            % (TOOL_COLOUR.get(name, "#aaa"), html.escape(name),
               "×%d" % use[name] if use[name] else "·", html.escape(cost),
               html.escape(fam), html.escape(what)))
    unknown = sorted(t for t in use if t not in registered)
    if unknown:
        rows.append('<div class="trow off"><span class="tname">%s</span>'
                    '<span class="twhat">called but not in system_init</span></div>'
                    % html.escape(", ".join(unknown)))
    knobs = ep["cfg"]
    knob_txt = " · ".join(
        "%s %s" % (k, knobs.get(k)) for k in ("max_turns", "step_budget", "pano_layout",
                                              "interface_variant", "review_every",
                                              "vla_returns_map")
        if k in knobs)
    return (
        '<div class="panel"><h3>this run\'s tool surface — %s</h3>'
        '<div class="tl">%s</div>'
        '<div class="loop"><b>model:</b> %s &nbsp; <b>settings:</b> %s</div></div>'
        % (html.escape(str(cond)), "".join(rows),
           html.escape(str(ep.get("model") or knobs.get("model") or "?")),
           html.escape(knob_txt or "defaults")))


def places_panel(ep: dict) -> str:
    if not ep["places"]:
        return ""
    rows = []
    for pl in ep["places"]:
        xy = pl.get("xy_f") or (0, 0)
        rows.append('<div class="prow"><b>%s</b><span>(%.1f, %.1f)</span>'
                    '<span class="pname">%s</span><span class="pcap">%s</span></div>'
                    % (pl["id"], xy[0], xy[1], html.escape(pl.get("name") or ""),
                       html.escape(pl.get("caption") or "")))
    named = sum(1 for p in ep["places"] if p.get("name"))
    return ('<div class="panel"><h3>places — %d nodes, %d named</h3>%s</div>'
            % (len(ep["places"]), named, "".join(rows)))


def _fmt_fields(f: dict) -> str:
    keep = ("walked_m", "requested_m", "blocked", "detoured", "turned_deg", "can_walk_m",
            "nearest_obstacle_before_m", "you_are_at_xy", "view",
            "went_to_place", "place_xy", "arrived", "retraced_m", "route",
            "recorded_place", "at_xy", "distance_from_you_m", "note", "steps_charged",
            "moved_m", "revisiting_earlier_position", "lettered_spots", "nearest_place",
            "steps_the_vla_took", "distance_travelled_m", "net_displacement_m",
            "blocked_events", "vla_suggests_arrival", "travelled_m", "episode_over",
            "end_reason", "stopped_at", "steps_taken_total", "walked_m_total",
            "seen_area_m2", "unobserved_fraction_of_frame", "straight_line_from_start_m")
    out = []
    for k in keep:
        if k not in f:
            continue
        v = f[k]
        if isinstance(v, (dict, list)):
            v = json.dumps(v, separators=(",", ":"))
        v = str(v)
        if len(v) > 150:
            v = v[:150] + "…"
        out.append('<span class="kv"><i>%s</i>%s</span>' % (html.escape(k), html.escape(v)))
    return "".join(out)


def timeline_html(ep: dict, img_src: Callable[[dict, dict], str]) -> str:
    """``img_src(shot, ep)`` returns the src attribute for a frame — a URL or a data URI."""
    out = []
    for t in ep["turns"]:
        col = TOOL_COLOUR.get(t["tool"], "#9aa0aa")
        args = json.dumps(t["args"], separators=(",", ":")) if t["args"] else ""
        if len(args) > 220:
            args = args[:220] + "…"
        say = html.escape(t["say"])[:900]
        think = html.escape(t["think"])[:700]
        out.append('<div class="turn" data-n="%d">' % t["n"])
        shown = t["tool"]
        if shown != t["tool"]:
            args = ("recorded as %s · " % t["tool"]) + args
        out.append('<div class="thead"><span class="tn">T%d</span>'
                   '<span class="pill" style="background:%s">%s</span>'
                   '<code class="args">%s</code>' % (t["planner_turn"], col, shown, html.escape(args)))
        bits = []
        if t.get("steps") is not None:
            bits.append("%s steps used" % t["steps"])
        if t["imgs"]:
            bits.append("%d img · ~%d tok" % (t["imgs"], t["img_tokens"]))
        if t.get("t") is not None:
            bits.append("%.0fs" % t["t"])
        out.append('<span class="meta">%s</span></div>' % " · ".join(bits))
        if say:
            out.append('<div class="say">%s</div>' % say)
        if think:
            out.append('<details class="think"><summary>thinking (%d chars)</summary>'
                       '<div>%s</div></details>' % (len(t["think"]), think))
        f = _fmt_fields(t["fields"])
        if f:
            out.append('<div class="fields">%s</div>' % f)
        for cap in (t.get("captions") or []):
            s = cap.strip()
            if not s:
                continue
            if s.startswith("places ("):
                head = s.split("\n", 1)[0]
                out.append('<details class="places" open><summary>topo map, in words '
                           '— %s</summary><pre>%s</pre></details>'
                           % (html.escape(head.split(" · ")[0]), html.escape(s)))
            else:
                out.append('<div class="cap">%s</div>' % html.escape(s))
        if t.get("shots"):
            out.append('<div class="shots">')
            for s in t["shots"]:
                out.append('<figure class="shot%s"><img loading="lazy" src="%s" alt="%s" '
                           'onclick="zoom(this)"><figcaption>%s</figcaption></figure>'
                           % (" ismap" if s.get("is_map") else "", img_src(s, ep),
                              html.escape(s["label"]), html.escape(s["label"])))
            out.append('</div>')
        out.append("</div>")
    return "".join(out)


def chips_html(ep: dict) -> str:
    m = ep.get("metrics") or {}
    ok = (m.get("success") or 0) >= 1.0
    chips = [
        ("outcome", "SUCCESS" if ok else "FAIL", "#3ddc63" if ok else "#ff5f56"),
        ("final distance", "%.2f m" % (m.get("distance_to_goal") or 0), None),
        ("closest", "%.2f m" % (ep.get("closest", {}).get("d") or 0), None),
        ("walked", "%.1f m" % (m.get("path_length") or 0), None),
        ("steps", "%.0f / %s" % (m.get("steps_taken") or 0, ep["cfg"].get("step_budget", 500)), None),
        ("turns", str(ep["planner_turn_count"]), None),
        ("SPL", "%.3f" % (m.get("spl") or 0), None),
        ("nDTW", "%.3f" % (m.get("ndtw") or 0), None),
        ("route length", "%.1f m" % (ep.get("geodesic_distance") or 0), None),
        ("cost", "$%.2f" % (ep.get("cost") or 0), None),
    ]
    tot_imgs = sum(t["imgs"] for t in ep["turns"])
    tot_tok = sum(t["img_tokens"] for t in ep["turns"])
    chips.append(("images", "%d · ~%dk tok" % (tot_imgs, round(tot_tok / 1000)), None))
    if ep.get("frames_found"):
        src = ep.get("frames_source") or "?"
        aligned = ep.get("frames_aligned")
        chips.append(("frames", "%d on disk · %s%s" % (
            ep["frames_found"], src, "" if aligned else " (count differs)"),
            None if aligned else "#ffb454"))
    if ep.get("driver_errors"):
        chips.append(("driver errors", str(len(ep["driver_errors"])), "#ff5f56"))
    return '<div class="chips">%s</div>' % "".join(
        '<span class="chip"><i>%s</i><b%s>%s</b></span>'
        % (html.escape(k), ' style="color:%s"' % c if c else "", html.escape(v))
        for k, v, c in chips)


def episode_section(ep: dict, img_src: Callable[[dict, dict], str], idx: int = 0,
                    hidden: bool = False) -> str:
    h = ['<section class="ep %s" id="ep%d">' % ("hidden" if hidden else "", idx)]
    h.append(chips_html(ep))
    h.append('<div class="instr"><i>instruction</i>%s</div>'
             % html.escape(ep.get("instruction") or ""))
    h.append('<div class="cols"><div class="left">')
    h.append(svg_map(ep))
    h.append(legend_html())
    h.append(surface_panel(ep))
    h.append(places_panel(ep))
    h.append('</div><div class="right">')
    h.append(timeline_html(ep, img_src))
    h.append('</div></div></section>')
    return "".join(h)


CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0a0b0e;color:#d7dae0;
 font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:14px 20px;border-bottom:1px solid #1b1f27;position:sticky;top:0;
 background:#0a0b0e;z-index:9}
header a{color:#8b93a1;text-decoration:none}
header a:hover{color:#fff}
h1{margin:0 0 8px;font-size:15px;letter-spacing:.02em;color:#fff;font-weight:600}
h3{margin:0 0 8px;font-size:12px;color:#8b93a1;text-transform:uppercase;
 letter-spacing:.08em;font-weight:600}
select{background:#12151b;color:#d7dae0;border:1px solid #262b35;padding:5px 8px;
 border-radius:5px;font:inherit;max-width:88vw}
.hidden{display:none}
.chips{display:flex;flex-wrap:wrap;gap:7px;padding:14px 20px 4px}
.chip{background:#12151b;border:1px solid #1e232c;border-radius:5px;padding:4px 9px}
.chip i{color:#7b8290;font-style:normal;margin-right:6px}
.chip b{color:#e8e8ee;font-weight:600}
.instr{margin:10px 20px 14px;padding:10px 12px;background:#101319;
 border-left:3px solid #4da3ff;border-radius:0 5px 5px 0;color:#c7ccd4}
.instr i{color:#7b8290;font-style:normal;display:block;font-size:11px;
 text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.cols{display:flex;gap:18px;padding:0 20px 40px;align-items:flex-start}
.left{width:600px;flex:0 0 600px;position:sticky;top:96px;max-height:calc(100vh - 110px);
 overflow-y:auto}
.right{flex:1;min-width:0}
.map{width:100%;border:1px solid #1e232c;border-radius:7px;display:block}
.badge{fill:#ffc440;font:600 11px ui-monospace,monospace;text-anchor:middle}
.ax{fill:#6b7280;font:10px ui-monospace,monospace}
.tm{transition:r .1s,fill-opacity .1s}
.tm.hot{r:7;fill:#fff;fill-opacity:1}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 14px;color:#7b8290;
 font-size:11px}
.panel{background:#0e1116;border:1px solid #1a1e26;border-radius:7px;padding:12px;
 margin-bottom:12px}
.tl .trow{display:grid;grid-template-columns:150px 32px 118px 52px 1fr;gap:8px;padding:3px 0;
 border-bottom:1px solid #14181f;font-size:11.5px;align-items:baseline}
.trow.off{opacity:.45}
.tname{font-weight:600}
.tname small{color:#6b7280;font-weight:400;margin-left:4px}
.tcnt{color:#e8e8ee;text-align:right}
.tcost{color:#8b93a1}
.tfam{color:#6b7d99;font-size:10.5px;text-transform:uppercase}
.twhat{color:#aab0ba}
.loop{margin-top:10px;padding-top:9px;border-top:1px solid #1a1e26;color:#8b93a1;
 font-size:11.5px;line-height:1.7}
.loop b{color:#c7ccd4;font-weight:600}
.prow{display:grid;grid-template-columns:26px 90px 150px 1fr;gap:8px;font-size:11px;
 padding:2px 0;border-bottom:1px solid #14181f;color:#aab0ba}
.prow b{color:#ffc440}
.pname{color:#e8e8ee}
.pcap{color:#7b8290;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.turn{border:1px solid #1a1e26;border-left:3px solid #262b35;border-radius:0 6px 6px 0;
 padding:9px 11px;margin-bottom:8px;background:#0d1014}
.turn:hover{border-left-color:#4da3ff;background:#0f1319}
.thead{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.tn{color:#5b6472;min-width:20px}
.pill{color:#0a0b0e;font-weight:700;padding:1px 8px;border-radius:10px;font-size:11px}
.args{color:#8b93a1;font-size:11.5px}
.meta{margin-left:auto;color:#5b6472;font-size:11px}
.say{margin:7px 0 0;color:#dfe3e9;white-space:pre-wrap}
.think{margin-top:6px}
.think summary{color:#6b7280;cursor:pointer;font-size:11px}
.think div{color:#8b93a1;white-space:pre-wrap;margin-top:5px;padding-left:9px;
 border-left:1px solid #262b35}
.fields{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.kv{background:#111520;border:1px solid #1b2130;border-radius:4px;padding:2px 7px;
 font-size:11px;color:#c7ccd4}
.kv i{color:#6b7d99;font-style:normal;margin-right:5px}
.cap{color:#8b93a1;font-size:11px;margin-top:5px}
.places{margin-top:8px;background:#0b0e14;border:1px solid #1b2130;border-radius:5px}
.places summary{cursor:pointer;color:#ffc440;padding:5px 9px;font-size:11px}
.places pre{margin:0;padding:0 9px 9px;color:#c7ccd4;font-size:10.5px;
 white-space:pre-wrap;line-height:1.5;overflow-x:auto}
.shots{display:flex;flex-wrap:wrap;gap:6px;margin-top:9px}
.shot{margin:0}
.shot img{height:104px;border:1px solid #232833;border-radius:4px;display:block;
 cursor:zoom-in;background:#000}
.shot.ismap img{height:160px;border-color:#5a4410}
.shot img:hover{border-color:#4da3ff}
.shot figcaption{color:#6b7280;font-size:10px;margin-top:2px;text-align:center}
#lb{position:fixed;inset:0;background:rgba(6,7,9,.94);display:none;z-index:99;
 align-items:center;justify-content:center;cursor:zoom-out;padding:24px}
#lb.on{display:flex}
#lb img{max-width:96vw;max-height:92vh;border:1px solid #2a2f3a;border-radius:6px}
.dim{color:#6b7280;font-size:11px;margin:8px 0 0}
@media(max-width:1200px){.cols{flex-direction:column}.left{width:100%;flex:none;
 position:static;max-height:none}}
"""

JS = """
function show(v){
  var eps = document.querySelectorAll('.ep');
  eps.forEach(function(e){ e.classList.add('hidden'); });
  var t = document.getElementById('ep' + v);
  if (t) t.classList.remove('hidden');
}
function zoom(img){
  var lb = document.getElementById('lb');
  var big = img.getAttribute('data-full') || img.src;
  lb.querySelector('img').src = big;
  lb.classList.add('on');
}
document.addEventListener('click', function(e){
  if (e.target.id === 'lb' || e.target.parentElement.id === 'lb')
    document.getElementById('lb').classList.remove('on');
});
document.addEventListener('keydown', function(e){
  if (e.key === 'Escape') document.getElementById('lb').classList.remove('on');
});
document.addEventListener('mouseover', function(e){
  var t = e.target.closest('.turn'); if(!t) return;
  var ep = t.closest('.ep');
  ep.querySelectorAll('.tm.hot').forEach(function(m){m.classList.remove('hot');});
  var m = ep.querySelector('#m'+t.dataset.n); if(m) m.classList.add('hot');
});
"""

