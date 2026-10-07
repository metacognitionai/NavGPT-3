"""R2R-CE / RxR-CE environment on habitat-sim 0.1.7 — no habitat-lab.

Episode loading, the discrete VLN-CE action space, and the metric suite,
implemented directly against ``habitat_sim``. Runs in the Habitat
environment (Python 3.8, habitat_sim 0.1.7) and imports nothing outside the
standard library except numpy, quaternion, Pillow and habitat_sim.

Task caliber is the VLN-CE paper default (habitat-lab 0.1.7 defaults), so
SR/SPL here are comparable to published R2R-CE tables:

- actions: STOP / MOVE_FORWARD 0.25 m / TURN_LEFT 15 deg / TURN_RIGHT 15 deg,
  sliding allowed
- RGB at [0, 1.25, 0], hfov 90; success distance 3.0 m; 500-step cap
- metrics: distance_to_goal (geodesic), success (STOP within 3 m), spl,
  oracle_success, ndtw, path_length, steps_taken

BLIND MODE (``blind=True``) skips the Simulator entirely and drives a bare
``habitat_sim.PathFinder``: real episodes, real sliding dynamics, real
geodesic metrics — only the pixels are synthetic. This exists because a
container without an EGL device cannot create a GL context at all, and the
whole agent loop is still worth exercising there.

Data layout:
    <data_root>/{split}/{split}.json.gz
    <data_root>/{split}/{split}_gt.json.gz
    <scene_root>/mp3d/{scan}/{scan}.glb + .navmesh
"""

from __future__ import annotations

import base64
import concurrent.futures
import functools
import logging
import math
import os
import struct
import threading
import zlib

from pathlib import Path

import numpy as np

from .datasets import load_episode_set

log = logging.getLogger("navgpt.environment.r2r")


def on_sim_thread(method):
    """Run this method on the env's single dedicated simulator thread.

    habitat-sim binds its GL context to the thread that constructed the
    ``Simulator``, so calling ``get_sensor_observations()`` from any other
    thread dies with ``GL::Context::current(): no current context``. The HTTP
    server is threaded — one thread per request — so every simulator touch has
    to be funnelled onto one thread.

    This does not show up in blind mode: with no Simulator there is no GL
    context to lose.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return self._submit(method, self, *args, **kwargs)

    return wrapper

ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}

DEFAULTS = {
    "rgb_size": 512,          # what the agent sees; does NOT affect metrics
    "hfov": 90.0,
    "camera_height": 1.25,
    "step_size_m": 0.25,      # FORWARD_STEP_SIZE
    # TURN_ANGLE. Three different values are in play and they are NOT
    # interchangeable:
    #   10  habitat_sim's own ActuationSpec default (always overridden)
    #   15  the VLN-CE paper default, and what the `planner_basic` briefing tells
    #       the agent ("turn left 15 degrees")
    #   30  VLNCE-EVAL's four-view navigation task config, which the NavGPT VLA
    #       is trained and evaluated with
    # Kept at 15 so the `planner_basic` baseline and its prompt stay consistent.
    # The VLA-only configs set environment.turn_angle_deg: 30, as VLNCE-EVAL does;
    # it reaches the VLA only through teleport's non-navigable fallback.
    "turn_angle_deg": 15.0,
    "success_distance": 3.0,
    "max_steps": 500,         # MAX_EPISODE_STEPS
    "seed": 42,
    # Set EXPLICITLY rather than inherited. habitat_sim 0.1.7 happens to default
    # this to True, matching VLN-CE (whose every task yaml sets
    # ALLOW_SLIDING: True), but that should be stated, not inherited.
    # Sliding changes what a blocked forward step does, so it changes path
    # length and therefore SPL; it must not be able to drift silently.
    "allow_sliding": True,
    # panorama rig — matched to the NavGPT VLA's training config so its
    # inputs are in-distribution (VLNCE-EVAL four-view navigation task config)
    "pano_size": 400,
    "pano_hfov": 120.0,
    # Modelled range of the depth camera, in metres. habitat_sim's DEPTH sensor
    # already returns METRES (it is habitat-LAB's DepthSensor that normalises to
    # [0,1] over MIN/MAX_DEPTH, and there is no habitat-lab in this server), so
    # this is a cap, never a scale factor.
    "depth_max": 10.0,
    "scan_bins": 360,         # one range bin per degree of bearing
    # Ranging band, as heights above the FLOOR in metres. The scan keeps only
    # returns inside it, so the floor and the ceiling cannot masquerade as
    # obstacles — this is what makes it behave like a lidar at hip height rather
    # than a camera pointed slightly down. Lower bound clears thresholds and
    # door sills; upper bound stays under the ceiling and hanging lamps.
    # 0.15 m rather than something taller because that is roughly what the
    # navmesh lets the robot climb or slide over: an obstacle below it usually
    # does not block, one above it does. Well clear of the floor plane itself,
    # which sits at exactly 0.0 and would otherwise cap every long range.
    "scan_floor_clear_m": 0.15,
    "scan_ceiling_clear_m": 1.80,
    # top-down colour map. The accumulator grid is sized from the navmesh bounds
    # once, so world coordinates are stable all episode; the DELIVERED frame is
    # then cropped out of it around the route (see observed_map), so two calls
    # share a coordinate system but not an extent.
    # Metres per map CELL. 0.02 balances detail against cost: 0.05 over-reports
    # coverage (every sample rounds up to a 25 cm2 cell), while 0.01 leaves the
    # far field too sparse for map_splat_max_m to bridge and delivers an image
    # large enough to be downscaled by the API before the model sees it. At 0.02
    # the delivered image is 0.02 m/px with no upscale, ~0.57 MP.
    "map_mpp": 0.02,
    "map_ceiling_m": 1.60,    # drop points above this over the floor, or the
                              # ceiling paints over everything below it
    # Subsample RGB-D pixels. A finer stride costs render time for little extra
    # coverage: the holes in the far field are occlusion shadows, not sampling gaps.
    "map_pixel_stride": 4,
    # Cap on how far one sample is splatted, in METRES rather than cells: the
    # physical splat stays fixed and the cell count follows the resolution.
    "map_splat_max_m": 0.10,
    # Target long edge of the delivered PNG, in pixels. The upscale factor is
    # derived from it (1-3x, nearest neighbour) so a small flat and a large
    # multi-storey scan both arrive at a readable size instead of one being a
    # postage stamp. Legibility only: it adds no information, and the reported
    # metres_per_pixel accounts for it.
    "map_target_px": 760,
    # Crop margin around the route + observed region, and a floor on the frame
    # span so the first call (one pose, no coverage) is not a 1x1 image.
    "map_margin_m": 1.5,
    "map_min_span_m": 6.0,
    # ── the place graph (see _derive_places) ──
    # Spacing between places: small enough that a place usually lands within the
    # 3 m success radius of a near-miss endpoint, the case the graph exists to
    # rescue.
    "place_spacing_m": 2.5,
    # Arrival radius for "I am back at a place I know", as a fraction of spacing.
    # BOOKKEEPING ONLY: node count is the same at 0.6 / 0.8 / 0.99, so this does
    # not control node proliferation — only which edges get recorded.
    "place_snap_frac": 0.6,
    # A recorded step longer than this is a teleport, not a walk (real primitives
    # never exceed 0.29 m), so the transition moves `current` without recording an
    # edge — joining the ends would assert connectivity never demonstrated.
    "place_jump_m": 0.5,
    "place_max_nodes": 64,
}


# ── ndtw (normalized Dynamic Time Warping) ──


def _ndtw(path, gt, threshold):
    """nDTW against the ground-truth locations, euclidean, as in VLN-CE."""
    if not path or not gt:
        return 0.0
    n, m = len(path), len(gt)
    p = np.asarray(path, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        for j in range(1, m + 1):
            d = float(np.linalg.norm(p[i - 1] - g[j - 1]))
            cur[j] = d + min(prev[j], prev[j - 1], cur[j - 1])
        prev = cur
    return math.exp(-prev[m] / (m * threshold))


# ── the place graph ──
#
# A topological memory derived from the walked path: nodes where the robot has
# been, edges it has proven walkable by walking them. Failed episodes often end
# further from the goal than a point the robot had already stood on: the agent
# reaches endpoints and fails to commit. A place it can name and return to is the
# missing action.
#
# Everything here is odometry: no goal, no reference path, no simulator
# semantics. It discloses nothing a robot with wheel encoders would not know.


def _derive_places(path, spacing_m=2.5, snap_frac=0.6, jump_m=0.5, max_nodes=64):
    """Derive the place graph from an agent path. Pure function, no simulator.

    Nodes are DERIVED, never stored, because ``_agent_path`` is append-only within
    an episode: the graph is a function of the path, so there is no per-episode
    state to reset and a VLA rollout's teleport hops build it for free.

    Two invariants hold, and both are load-bearing:

    * **Prefix stability.** Node *k* is created by a predicate over ``path[:i+1]``
      alone, and nodes are immutable and append-only, so the node list from a
      later call is always a prefix-extension of an earlier one — an id never
      means two different places within one episode.
      FOUR CHANGES WOULD BREAK IT AND MUST NOT BE MADE: a centred or lookahead
      window in the creation test; a centroid or merge pass; any threshold that
      depends on the finished graph; or hiding a node from the listing on a
      property that changes over time (an id the agent has already used must keep
      working).
    * **Coverage.** Every path point is within ``spacing_m`` of some node — either
      the nearest was already close enough, or a node was created at that very
      point. Consequence: *replaying recorded points can never create a node*, so
      a retrace is graph-neutral by construction.

    ``jump_m`` suppresses edges across teleports. Real movement primitives never
    exceed 0.29 m per recorded point; a ``go_back_to_waypoint`` warp shows up as a
    single long segment. Joining its ends would assert a walkable connection that was never
    demonstrated, so the transition still moves ``current`` but records no edge.

    Returns ``{"nodes": [...], "edges": [...], "current": id}`` where each node is
    ``{"id", "position", "visits", "path_index"}`` and each edge is
    ``{"a", "b", "arc_m", "span"}`` — ``span`` being the path-index range of the
    shortest traversal observed for that pair, which is what lets a retrace replay
    real points instead of cutting a chord through a wall.
    """
    if not path:
        return {"nodes": [], "edges": [], "current": None}

    def xz(a, b):
        return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))

    spacing = float(spacing_m)
    snap = float(snap_frac) * spacing
    jump = float(jump_m)

    nodes = [{"id": 0, "position": list(map(float, path[0])), "visits": 1,
              "path_index": 0}]
    edges = {}
    current = 0
    left_index = 0          # where the robot left `current`
    arc = 0.0
    clean = True            # no teleport warp since leaving `current`

    def link(a, b, length, span):
        key = (a, b) if a < b else (b, a)
        prev = edges.get(key)
        # Keep the SHORTEST traversal ever seen for a pair: that is what makes
        # routing beat naive retracing.
        if prev is None or length < prev["arc_m"]:
            edges[key] = {"arc_m": round(float(length), 3), "span": list(span)}

    for i in range(1, len(path)):
        seg = xz(path[i - 1], path[i])
        arc += seg
        if seg > jump:
            clean = False

        best, best_d = None, None
        for n in nodes:
            d = xz(path[i], n["position"])
            if best_d is None or d < best_d:
                best, best_d = n, d

        if best_d <= snap and best["id"] != current:
            if clean:
                link(current, best["id"], arc, (left_index, i))
            best["visits"] += 1
            current, left_index, arc, clean = best["id"], i, 0.0, True
        elif best_d > spacing and len(nodes) < int(max_nodes):
            new = {"id": len(nodes), "position": list(map(float, path[i])),
                   "visits": 1, "path_index": i}
            nodes.append(new)
            if clean:
                link(current, new["id"], arc, (left_index, i))
            current, left_index, arc, clean = new["id"], i, 0.0, True

    return {
        "nodes": nodes,
        "edges": [{"a": a, "b": b, **v} for (a, b), v in sorted(edges.items())],
        "current": current,
    }


def _map_up_words(up_deg):
    """The turn that faces the top of the map, said in words.

    "MAP-UP: 135 LEFT" reads as what it is: the top of the map is 135 degrees to
    your left. Positive is left everywhere in this surface, so the
    sign becomes a word rather than a convention to remember.
    """
    if _INTERFACE_VARIANT == 'absolute_bearings':
        return 'HEADING %.0f DEG CW FROM MAP-NORTH' % ((float(up_deg)) % 360)
    d = int(round(float(up_deg)))
    if abs(d) <= 12:
        return "MAP-NORTH: AHEAD" if _INTERFACE_VARIANT == "heading_up_map" else "MAP-UP: AHEAD"
    if abs(d) >= 168:
        return "MAP-NORTH: BEHIND" if _INTERFACE_VARIANT == "heading_up_map" else "MAP-UP: BEHIND"
    return "%s: %d %s" % ("MAP-NORTH" if _INTERFACE_VARIANT == "heading_up_map" else "MAP-UP", abs(d), "LEFT" if d > 0 else "RIGHT")


def _relative_turn(dx, dz, heading_rad):
    """Which way to turn to face a world offset, in move()'s own convention.

    Positive is LEFT, exactly as `move(turn_deg)` and `step_hightolow(angle_rad)`
    read it, so the number can be passed straight through with no conversion.

    Relative degrees rather than compass words: the controls take relative turns,
    so an absolute frame on the map forces the agent to convert between the two,
    and it often gets that conversion wrong.
    """
    target = math.atan2(-float(dx), -float(dz))
    rel = math.degrees(target - float(heading_rad))
    rel = (rel + 180.0) % 360.0 - 180.0
    mag = abs(rel)
    if mag <= 12.0:
        return "straight ahead", round(rel)
    if mag >= 168.0:
        return "behind you", round(rel)
    return "{:.0f} deg {}".format(mag, "left" if rel > 0 else "right"), round(rel)


def _place_route(graph, start_id, goal_id):
    """Cheapest walked route between two places: Dijkstra over edge arcs.

    Returns ``(ids, metres)`` or ``(None, None)`` if the goal is unreachable —
    which happens exactly when every path between them ran through a teleport
    warp whose edge was suppressed, and refusing is then the honest answer.
    """
    if start_id == goal_id:
        return [start_id], 0.0
    adj = {}
    for e in graph["edges"]:
        adj.setdefault(e["a"], []).append((e["b"], e["arc_m"]))
        adj.setdefault(e["b"], []).append((e["a"], e["arc_m"]))
    dist = {start_id: 0.0}
    prev = {}
    seen = set()
    while True:
        node = min((n for n in dist if n not in seen), key=lambda n: dist[n],
                   default=None)
        if node is None:
            return None, None
        if node == goal_id:
            break
        seen.add(node)
        for nxt, w in adj.get(node, ()):
            if nxt in seen:
                continue
            if nxt not in dist or dist[node] + w < dist[nxt]:
                dist[nxt] = dist[node] + w
                prev[nxt] = node
    route = [goal_id]
    while route[-1] != start_id:
        route.append(prev[route[-1]])
    route.reverse()
    return route, round(dist[goal_id], 3)


# ── minimal PNG encoder (stdlib only) ──


def _encode_png(rgb):
    """Encode an HxWx3 uint8 array as PNG using zlib + struct only."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = arr.shape[0], arr.shape[1]
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


# ── a 5x7 bitmap font, for every label burned into a picture ──
#
# Labels are drawn with numpy alone, so they look the same on every machine
# whatever fonts are installed. Glyph geometry is 5 wide by 7 tall, rows
# MSB-left over five bits, so 0b11111 is a full row.
#
# 5x7 rather than 3x5 because these numbers drive decisions — which bearing to
# turn to, where the robot is, which look this is — so an ambiguous digit is not a
# cosmetic problem. At 3x5, six digit pairs differ by a single bit (0/8, 3/9,
# 5/6, 5/9, 6/8, 8/9); at 5x7 the minimum Hamming distance between digits is 6
# bits of 35.

# Spatial-representation ablation for this service; each ablation cell owns a
# dedicated Environment.
_INTERFACE_VARIANT = os.environ.get('NAVGPT_INTERFACE_VARIANT', 'standard')
from .presentation import compact_digit, camera_canvas, rotate_map

_GLYPH_W, _GLYPH_H = 5, 7

_GLYPHS = {
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b10001, 0b00001, 0b00010, 0b00100, 0b00100, 0b00100),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    # Letters, for the words burned into the labels.
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "I": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111),
    "J": (0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "Q": (0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "X": (0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
    "-": (0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000),
    "+": (0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000),
    ".": (0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100),
    ":": (0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000),
    "/": (0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000),
    " ": (0, 0, 0, 0, 0, 0, 0),
}


def _digits(img, text, cx, cy, scale, rgb):
    """Blit `text` centred on (cx, cy) in output pixels. Digits only."""
    scale = max(1, int(scale))
    glyph_w, glyph_h, gap = _GLYPH_W * scale, _GLYPH_H * scale, scale
    total = len(text) * glyph_w + (len(text) - 1) * gap
    x0 = int(round(cx - total / 2.0))
    y0 = int(round(cy - glyph_h / 2.0))
    H, W = img.shape[0], img.shape[1]
    for ch in text:
        if _INTERFACE_VARIANT == 'small_digits' and ch in '0123456789':
            compact_digit(img, ch, x0, y0, glyph_w, glyph_h, rgb)
            x0 += glyph_w + gap
            continue
        rows = _GLYPHS.get(ch)
        if rows is None:
            x0 += glyph_w + gap
            continue
        for r, bits in enumerate(rows):
            for c in range(_GLYPH_W):
                if not (bits >> (_GLYPH_W - 1 - c)) & 1:
                    continue
                ys, xs = y0 + r * scale, x0 + c * scale
                ye, xe = min(H, ys + scale), min(W, xs + scale)
                if ys < H and xs < W and ye > 0 and xe > 0:
                    img[max(0, ys):ye, max(0, xs):xe] = rgb
        x0 += glyph_w + gap


def _text_width(text, scale):
    """Width in pixels of `text` blitted by _digits at `scale`."""
    return max(0, len(text) * (_GLYPH_W + 1) * scale - scale)


def _fit_scale(text, width, lo=2, hi=9):
    """Largest glyph scale at which `text` still fits inside `width`."""
    for s in range(hi, lo, -1):
        if _text_width(text, s) <= width - 4 * s:
            return s
    return lo


def _text_bars(img, lines, y0=0, scales=None, fg=(245, 245, 250),
               bg=(10, 12, 16)):
    """Burn several lines, each auto-fitted, as one bar. Returns the y past it.

    Several lines rather than one because _fit_scale floors at 2 rather than
    shrinking forever, so a long stamp on one line would overflow the image. A
    cut-off stamp is worse than a small one: it still looks like a complete label.
    """
    y = y0
    for i, line in enumerate(lines):
        s = None if not scales else scales[i]
        y = _text_bar(img, line, y, s, fg, bg)
    return y


def _wrap_text(text, width, scale):
    """Break `text` into lines that each fit `width` at `scale`. Never drops a word.

    Silent truncation is the failure this exists to prevent: _fit_scale floors at
    scale 2, so an unwrapped long stamp would run off the edge and still look like
    a complete label. Wrapping is honest; truncating is not.
    """
    words, lines, cur = str(text).upper().split(" "), [], ""
    for w in words:
        cand = (cur + " " + w) if cur else w
        if cur and _text_width(cand, scale) > width - 4 * scale:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _text_bar(img, text, y0=0, scale=None, fg=(245, 245, 250), bg=(10, 12, 16)):
    """Burn text into a solid bar across the width of `img`, wrapping if needed.

    The label lives in the PIXELS, not in a neighbouring text block. This is the
    one thing a long multi-turn context cannot degrade: after eighty
    near-identical indoor photographs, a caption that merely SITS BESIDE an image
    is bound to it only by position, and position is what a long context keeps
    worst. "LEFT 2.1M" painted on the picture travels with the picture.
    Returns the y coordinate just past the bar.
    """
    text = str(text).upper()
    H, W = img.shape[0], img.shape[1]
    scale = scale or _fit_scale(text, W)
    pad = max(2, scale)
    lines = _wrap_text(text, W, scale)
    row_h = _GLYPH_H * scale + pad
    bar_h = row_h * len(lines) + pad
    y1 = min(H, max(0, y0) + bar_h)
    img[max(0, y0):y1, :] = bg
    for i, line in enumerate(lines):
        _digits(img, line, 2 * scale + _text_width(line, scale) / 2.0,
                y0 + pad + i * row_h + (_GLYPH_H * scale) / 2.0, scale, fg)
    return y0 + bar_h


def _labels_at(img, items, y0=None, scale=None, fg=(255, 236, 170),
               bg=(10, 12, 16)):
    """Burn short labels at given x positions, in a bar across the image.

    This is the bearing ruler, and it exists because a 120-degree panel labelled
    "AHEAD" hides the question that actually decides the next action. The open
    corridor in a bedroom view is not "ahead", it is 45 degrees to the left of
    ahead, and no amount of reading the word AHEAD recovers that. A tick printed
    UNDER THE COLUMN it belongs to turns the bearing into something read off the
    picture rather than estimated from it.
    """
    if not items:
        return
    H, W = img.shape[0], img.shape[1]
    if scale is None:
        # Fit to the SMALLEST GAP BETWEEN ADJACENT TICKS, not to W/len(items).
        # Ruler ticks are tan-spaced, not evenly spaced: on a 120-degree panel the
        # +-40 ticks sit at 0.26 and 0.74 of the width, so the real gap is 0.24 W
        # while W/3 is 0.33 W. Sizing to the average lets adjacent ticks run
        # together, and a clipped sign still looks like a number.
        xs = sorted(float(x) for x, _t in items)
        gap = min([b - a for a, b in zip(xs, xs[1:])] or [float(W)])
        gap = min(gap, W / float(len(items)))
        widest = max(len(str(t)) for _x, t in items)
        scale = 2
        for s in range(5, 2, -1):
            if _text_width("X" * widest, s) <= gap - 6:
                scale = s
                break
    pad = max(2, scale)
    bar_h = _GLYPH_H * scale + 2 * pad
    y0 = H - bar_h if y0 is None else y0
    img[max(0, y0):min(H, y0 + bar_h), :] = bg
    for x, text in items:
        text = str(text).upper()
        half = _text_width(text, scale) / 2.0
        cx = min(W - half - scale, max(half + scale, float(x)))
        _digits(img, text, cx, y0 + bar_h / 2.0, scale, fg)


def _blind_frame(size, text_rows):
    """A deterministic placeholder frame. Encodes state as coarse colour bands
    so a human tailing the run can see the agent is actually moving, while
    carrying no navigational information the model could exploit."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = (24, 24, 32)
    band = max(1, size // max(1, len(text_rows) * 2))
    for i, val in enumerate(text_rows):
        y0 = i * band * 2
        y1 = min(size, y0 + band)
        img[y0:y1, :] = (
            (int(val) * 37) % 256,
            (int(val) * 91) % 256,
            (int(val) * 151) % 256,
        )
    return img


class R2RCEEnv:
    """One live episode at a time. Every public method is lock-guarded: the
    HTTP server is threaded but habitat_sim is not reentrant.

    The simulator is rebuilt only on scene change — R2R-CE splits are
    scene-clustered, so consecutive episodes usually reuse the live sim.
    """

    PANO_VIEWS = 4
    # Yaw index -> name. +pi/2 about +Y is counter-clockwise = left.
    PANO_NAMES = ("front", "left", "back", "right")

    def __init__(
        self,
        scene_root,
        data_root=None,
        split="val_unseen",
        episodes_file=None,
        gt_file=None,
        gt_split=None,
        blind=False,
        config=None,
        dataset="r2r",
        languages=None,
        roles=None,
    ):
        """Two ways to source episodes:

        - ``data_root`` + ``split`` — the standard layout,
          ``<data_root>/<split>/<split>.json.gz`` with a sibling
          ``<split>_gt.json.gz``.
        - ``episodes_file`` — a standalone episode file (``.json`` or
          ``.json.gz``), e.g. a curated 100-episode evaluation subset. Such
          files usually ship no GT of their own, so point ``gt_file`` at one,
          or ``gt_split`` at the parent split whose GT covers them.

        No path has a default. They are deployment facts, not code constants.
        """
        if not scene_root:
            raise ValueError("scene_root is required (data.scenes in configs/paths.yaml)")
        if not data_root and not episodes_file:
            raise ValueError(
                "need either data_root (+ split) or episodes_file "
                "(see configs/paths.example.yaml)"
            )
        self.dataset = dataset
        self.languages = languages
        self.roles = roles
        self._dataset_info = {}
        self.data_root = data_root
        self.scene_root = scene_root
        self.episodes_file = episodes_file
        self.gt_file = gt_file
        self.gt_split = gt_split
        self.blind = bool(blind)
        self.config = dict(DEFAULTS)
        if config:
            self.config.update(config)

        self._lock = threading.RLock()
        # Single worker: owns the GL context for the process's whole lifetime.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="r2rce-sim"
        )
        self._tls = threading.local()
        self._sim = None
        self._pathfinder = None
        self._scene_path = None

        self._split = split
        self._episodes = []
        self._gt = {}
        self._gt_coverage = 0

        self._ep = None
        self._ep_index = -1
        # Staged by the env panel's episode_index field, committed by `play`.
        self._pending_index = None
        # fixed-frame top-down colour accumulator (see _accumulate_colour)
        self._cmap_sum = None
        self._cmap_n = None
        self._waypoints = []
        # A single monotonic counter over EVERY image the agent is shown, painted on
        # each one, so an image lifted out of a long context dates itself and any
        # two of them can be put in order. Two photographs of the same corridor
        # twenty turns apart are otherwise indistinguishable. It counts views, not
        # look_around calls — local_map and go() render views too, so tying it to
        # one tool would have made the numbers repeat.
        self._view_index = 0
        self._cmap_lo = (0.0, 0.0)
        self._cmap_shape = (0, 0)
        self._step_index = 0
        self._stop_called = False
        self._done = False
        self._collisions = 0
        self._agent_path = []
        # Teleport fallbacks are the only way turn_angle_deg affects a
        # VLA-driven episode; both counts are reported with the metrics.
        self._teleports = 0
        self._teleport_fallbacks = 0

        # blind-mode agent state (no Simulator to hold it for us)
        self._pos = None
        self._rot = None

        self.set_split(split)

    # ── simulator-thread dispatch ──

    def _submit(self, fn, *args, **kwargs):
        """Run fn on the simulator thread, blocking for its result.

        Re-entrant: a method already executing there (e.g. ensure_live calling
        set_episode_by_index) runs inline instead of submitting to the
        single-worker pool, which would deadlock on itself.
        """
        if getattr(self._tls, "on_sim_thread", False):
            return fn(*args, **kwargs)

        def wrapped():
            self._tls.on_sim_thread = True
            try:
                return fn(*args, **kwargs)
            finally:
                self._tls.on_sim_thread = False

        return self._executor.submit(wrapped).result()

    def set_split(self, split):
        """Load official R2R or role/language-filtered RxR data atomically."""
        with self._lock:
            loaded = load_episode_set(
                data_root=self.data_root, split=split, dataset=self.dataset,
                episodes_file=self.episodes_file, gt_file=self.gt_file,
                gt_split=self.gt_split, languages=self.languages, roles=self.roles,
            )
            self._split = split
            self._episodes = loaded.pop("episodes")
            self._gt = loaded.pop("ground_truth")
            self._dataset_info = loaded
            self._gt_coverage = loaded["gt_covered"]
            # A changed split must never keep a live episode from the previous set.
            self._ep = None
            self._ep_index = -1
            self._pending_index = None
            log.info("dataset=%s split=%s episodes=%d gt=%d/%d", self.dataset, split,
                     len(self._episodes), self._gt_coverage, len(self._episodes))
            if self._gt_coverage < len(self._episodes):
                log.warning("Incomplete dense GT: benchmark evaluation will refuse this dataset")
            return dict(loaded)

    @property
    def episode_count(self):
        return len(self._episodes)

    def stage_episode_index(self, index):
        """Stage an index without seating it — the panel's field/action split.
        Placement stays a two-step commit so a half-configured cascade cannot
        silently score the previous episode again."""
        with self._lock:
            self._pending_index = int(index)

    def staged_episode_index(self):
        with self._lock:
            return self._pending_index

    # ── simulator ──

    def _scene_file(self, ep):
        # scene_id is "mp3d/<scan>/<scan>.glb", joined against the scene root
        return os.path.join(self.scene_root, str(ep["scene_id"]))

    def _open_scene(self, scene_path):
        """Load navmesh (+ Simulator unless blind). Idempotent per scene."""
        import habitat_sim

        if scene_path == self._scene_path and (self._pathfinder or self._sim):
            return

        self._close_scene()
        navmesh = os.path.splitext(scene_path)[0] + ".navmesh"

        if self.blind:
            if not os.path.isfile(navmesh):
                raise FileNotFoundError("navmesh missing (blind mode needs it): " + navmesh)
            pf = habitat_sim.PathFinder()
            pf.load_nav_mesh(navmesh)
            if not pf.is_loaded:
                raise RuntimeError("failed to load navmesh: " + navmesh)
            self._pathfinder = pf
            self._scene_path = scene_path
            return

        cfg = self.config
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.gpu_device_id = int(os.environ.get("NAVGPT_GPU_DEVICE_ID", "0"))
        sim_cfg.allow_sliding = bool(cfg["allow_sliding"])

        # ── the agent's own forward camera (fixed: the `planner_basic` baseline
        # must stay comparable) ──
        rgb = habitat_sim.SensorSpec()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [int(cfg["rgb_size"]), int(cfg["rgb_size"])]
        rgb.position = [0.0, float(cfg["camera_height"]), 0.0]
        rgb.parameters["hfov"] = str(cfg["hfov"])
        specs = [rgb]

        # ── panorama rig: 4 RGB + 4 depth at 400x400 hfov 120 ──
        #
        # Four SIMULTANEOUS cameras, cloned the way VLN-CE synthesises them,
        # with yaws 0, pi/2, pi, 3pi/2. Because
        # they render in the same get_sensor_observations() call, a full 360
        # scan costs ZERO simulator steps, so looking around never competes with
        # moving for the step budget.
        #
        # Resolution and hfov match what the NavGPT VLA was trained on
        # (400x400, hfov 120) so its inputs are in-distribution. Yaw -> name is
        # rgb_1=left, rgb_2=back, rgb_3=right (+pi/2 about +Y is CCW = left),
        # consistent with the VLNCE-EVAL agent, simulator adapter, and episode runner.
        #
        # Depth is what makes `move` calibrated rather than a guess: it yields a
        # real range scan, so clearances are metres rather than vibes. It also
        # carries the RGB-D pairs that `_accumulate_colour` unprojects into the
        # top-down colour map behind `local_map`.
        pano_px = int(cfg["pano_size"])
        pano_hfov = float(cfg["pano_hfov"])
        for cam in range(self.PANO_VIEWS):
            yaw = 2.0 * math.pi * cam / self.PANO_VIEWS
            for kind in ("color", "depth"):
                spec = habitat_sim.SensorSpec()
                spec.uuid = ("pano_rgb_%d" % cam) if kind == "color" else ("pano_depth_%d" % cam)
                spec.sensor_type = (habitat_sim.SensorType.COLOR if kind == "color"
                                    else habitat_sim.SensorType.DEPTH)
                spec.resolution = [pano_px, pano_px]
                spec.position = [0.0, float(cfg["camera_height"]), 0.0]
                spec.parameters["hfov"] = str(pano_hfov)
                spec.orientation = [0.0, yaw, 0.0]
                specs.append(spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=float(cfg["step_size_m"])),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=float(cfg["turn_angle_deg"])),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=float(cfg["turn_angle_deg"])),
            ),
        }

        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        if os.path.isfile(navmesh):
            self._sim.pathfinder.load_nav_mesh(navmesh)
        else:
            log.warning("no navmesh at %s — recomputing", navmesh)
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height = 1.5
            settings.agent_radius = 0.1
            if not self._sim.recompute_navmesh(self._sim.pathfinder, settings):
                raise RuntimeError("recompute_navmesh failed for " + scene_path)
        self._sim.seed(int(cfg["seed"]))
        self._pathfinder = self._sim.pathfinder
        self._scene_path = scene_path

    def _close_scene(self):
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the real error
                pass
        self._sim = None
        self._pathfinder = None
        self._scene_path = None

    @on_sim_thread
    def _close_on_sim_thread(self):
        with self._lock:
            self._close_scene()

    def close(self):
        """Tear down the simulator on its own thread, then retire the thread.
        Not decorated: the executor cannot shut itself down from inside."""
        try:
            self._close_on_sim_thread()
        finally:
            self._executor.shutdown(wait=True)

    # ── episode placement ──

    @on_sim_thread
    def set_episode_by_index(self, index):
        import habitat_sim
        import quaternion  # noqa: F401 — registers np.quaternion

        with self._lock:
            if not self._episodes:
                return {"error": "no episodes loaded"}
            if index < 0 or index >= len(self._episodes):
                return {
                    "error": "index {} out of range (0..{})".format(
                        index, len(self._episodes) - 1
                    )
                }

            ep = self._episodes[index]
            scene_path = self._scene_file(ep)
            if not os.path.isfile(scene_path):
                return {"error": "scene mesh missing: " + scene_path}
            self._open_scene(scene_path)

            pos = np.asarray(ep["start_position"], dtype=np.float32)
            r = ep["start_rotation"]  # dataset order [x, y, z, w]
            rot = np.quaternion(r[3], r[0], r[1], r[2])

            if self.blind:
                self._pos, self._rot = pos, rot
            else:
                state = habitat_sim.AgentState()
                state.position = pos
                state.rotation = rot
                self._sim.get_agent(0).set_state(state)

            self._ep = ep
            self._ep_index = index
            self._step_index = 0
            self._stop_called = False
            self._done = False
            self._collisions = 0
            # Per-episode, like every other counter here. Accumulating these
            # across a 100-episode run would report the run's total as each
            # episode's value.
            self._teleports = 0
            self._teleport_fallbacks = 0
            self._agent_path = [list(map(float, pos))]
            # a new episode may be a new scene; drop the accumulated colour map
            self._cmap_sum = None
            self._cmap_n = None
            self._view_index = 0

            log.info(
                "episode %d (id=%s) scene=%s%s",
                index, ep.get("episode_id"), ep.get("scene_id"),
                " [blind]" if self.blind else "",
            )
            return self._meta()

    def _meta(self):
        ep = self._ep or {}
        return {
            "episode_id": str(ep.get("episode_id", "")),
            "scene_id": str(ep.get("scene_id", "")),
            "instruction": (ep.get("instruction") or {}).get("instruction_text", ""),
            "language": (ep.get("instruction") or {}).get("language"),
            "instruction_id": (ep.get("instruction") or {}).get("instruction_id"),
            "annotation_role": ep.get("annotation_role"),
            "dataset": self.dataset,
            "geodesic_distance": float((ep.get("info") or {}).get("geodesic_distance", 0.0)),
            "step_count": self._step_index,
            "done": self._done,
        }

    @on_sim_thread
    def ensure_live(self):
        """Reset semantics: a live episode is read untouched; a finished one is
        re-armed in place.

        Placement is the driver's job (stage episode_index, then `play`), and
        the driver always does that before calling reset. If reset arrives with
        nothing ever seated this falls back to index 0, so a caller that skips
        placement quietly re-scores episode 0 rather than erroring. Stage
        explicitly; don't rely on this."""
        with self._lock:
            live = self._ep is not None and not self._done and (self._sim or self._pathfinder)
            if live:
                return self._meta()
            index = self._ep_index if self._ep_index >= 0 else 0
        return self.set_episode_by_index(index)

    # ── agent state ──

    def _agent_state(self):
        if self.blind:
            return self._pos, self._rot
        state = self._sim.get_agent(0).get_state()
        return np.asarray(state.position), state.rotation

    def _forward_vector(self, rot):
        import quaternion

        return quaternion.as_rotation_matrix(rot) @ np.array([0.0, 0.0, -1.0])

    # ── transition ──

    @on_sim_thread
    def step(self, action):
        import numpy as _np
        import quaternion  # noqa: F401

        with self._lock:
            if self._ep is None:
                return {"error": "no live episode — set an episode first"}
            if self._done:
                return {
                    "terminated": True,
                    "truncated": False,
                    "step_count": self._step_index,
                    "info": {"note": "episode already done"},
                    "metrics": None,
                }

            action = int(action)
            if action == 0:
                self._stop_called = True
                self._done = True
            elif action in (1, 2, 3):
                prev, _ = self._agent_state()
                prev = _np.asarray(prev, dtype=_np.float64)
                if self.blind:
                    self._blind_move(action)
                else:
                    self._sim.step({1: "move_forward", 2: "turn_left", 3: "turn_right"}[action])
                    self._record_motion_frame()
                pos, _ = self._agent_state()
                pos = _np.asarray(pos, dtype=_np.float64)
                if action == 1 and float(_np.linalg.norm(pos - prev)) < 1e-4:
                    self._collisions += 1
                # Record a location ONLY when the position actually changed —
                # matching VLN-CE's NDTW.update_metric.
                # A turn in place leaves the position identical, and these agents
                # batch turns heavily (step([2]*6)), so appending unconditionally
                # fills the path with duplicates. DTW aligns over ~len(path)
                # steps while nDTW normalizes by len(gt), so duplicates inflate
                # the DTW cost and collapse nDTW. path_length / oracle_success /
                # distance_to_goal are unaffected either way; nDTW is not.
                point = list(map(float, pos))
                if not self._agent_path or point != self._agent_path[-1]:
                    self._agent_path.append(point)
            else:
                return {"error": "invalid action {} (expected 0-3)".format(action)}

            self._step_index += 1
            truncated = False
            if not self._done and self._step_index >= int(self.config["max_steps"]):
                truncated = True
                self._done = True

            info = {
                "action": action,
                "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
                "step_count": self._step_index,
                "collisions": self._collisions,
                "episode_id": str(self._ep.get("episode_id", "")),
            }
            metrics = self._metrics() if self._done else None
            if metrics is not None:
                info["metrics"] = metrics
            return {
                "terminated": bool(self._stop_called),
                "truncated": truncated,
                "step_count": self._step_index,
                "info": info,
                "metrics": metrics,
            }

    def _blind_move(self, action):
        """Sliding forward motion / in-place turn, using the same pathfinder
        primitive habitat's own move_forward uses (``try_step``), so blind
        trajectories are dynamically faithful."""
        import quaternion

        if action == 1:
            fwd = self._forward_vector(self._rot)
            target = self._pos + float(self.config["step_size_m"]) * fwd.astype(np.float32)
            self._pos = np.asarray(
                self._pathfinder.try_step(self._pos, target), dtype=np.float32
            )
            return
        deg = float(self.config["turn_angle_deg"])
        sign = 1.0 if action == 2 else -1.0  # left is +Y in habitat
        half = math.radians(deg) / 2.0
        turn = np.quaternion(math.cos(half), 0.0, sign * math.sin(half), 0.0)
        self._rot = self._rot * turn

    @on_sim_thread
    def call_with_motion_history(self, handler, inputs):
        self._capture_motion_history = True
        self._motion_history = []
        try:
            result = handler(self, inputs)
            result['motion_history'] = self._motion_history
            return result
        finally:
            self._capture_motion_history = False
            self._motion_history = []

    def _record_motion_frame(self):
        if not getattr(self, '_capture_motion_history', False) or self.blind:
            return
        observations = self._sim.get_sensor_observations()
        self._motion_history.append({'views': {
            name: base64.b64encode(_encode_png(np.asarray(
                observations['pano_rgb_%d' % cam], dtype=np.uint8)[..., :3])).decode('ascii')
            for cam, name in enumerate(self.PANO_NAMES)}})

    # ── perception ──

    @on_sim_thread
    def observe(self, stamp=False):
        """The forward camera. `stamp` burns the pose into the pixels.

        OFF BY DEFAULT, and that default is load-bearing: `planner_basic` is the
        two-tool baseline condition, and altering the pixels it sees would silently
        change what its results mean. Only the
        richer surfaces, whose briefings already describe burned-in labels, ask
        for the stamp.
        """
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}

            size = int(self.config["rgb_size"])
            if self.blind:
                rgb = _blind_frame(size, [self._step_index, self._collisions, len(self._agent_path)])
            else:
                obs = self._sim.get_sensor_observations()
                rgb = np.asarray(obs["rgb"], dtype=np.uint8)
                if rgb.ndim == 3 and rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]

            pos, rot = self._agent_state()
            presentation = {}
            if stamp and not self.blind:
                self._view_index += 1
                hx, hy = self._place_xy(pos)
                _up_word, up_deg = _relative_turn(0.0, -1.0, self._yaw_of(rot))
                labels = ["FORWARD VIEW", "VIEW %d   X %.1f Y %.1f   %s" %
                          (self._view_index, hx, hy, _map_up_words(up_deg))]
                if _INTERFACE_VARIANT != 'standard':
                    rgb, presentation = camera_canvas(rgb, labels, [], _INTERFACE_VARIANT, _text_bars, _labels_at)
                else:
                    _text_bars(rgb, labels, scales=[4, 2])
            # The top-down colour map is fed from observe_pano, not here: this
            # verb renders one camera and no depth, so it has nothing to
            # unproject.
            return {
                "presentation": presentation,
                "rgb": base64.b64encode(_encode_png(rgb)).decode("ascii"),
                "pose": {
                    "position": list(map(float, np.asarray(pos))),
                    "orientation": [
                        float(rot.x), float(rot.y), float(rot.z), float(rot.w),
                    ],
                },
                "instruction_text": (self._ep.get("instruction") or {}).get(
                    "instruction_text", ""
                ),
                "step_index": self._step_index,
                "blind": self.blind,
            }

    # ── panorama + range scan ──

    def _yaw_of(self, rot):
        """Agent heading as a bearing in radians, CCW-positive, 0 = its forward."""
        import quaternion

        fwd = self._forward_vector(rot)
        # habitat: -Z is forward, +X is right. Bearing measured CCW from forward.
        return math.atan2(-float(fwd[0]), -float(fwd[2]))

    def _range_scan(self, depth_views):
        """Turn the 4 depth images into ONE 360-bin range scan, in metres.

        This is what makes `move` calibrated. Each depth column corresponds to a
        known bearing through the pinhole model, so the nearest surface in that
        direction is a real distance, not an estimate from RGB.

        Floor and ceiling are excluded by HEIGHT, not by a fixed slice of image
        rows. A fixed slice does not work at this geometry: 25% of a 400 px image
        at hfov 120 reaches 23 degrees below the horizon, where the floor sits
        2.9 m away, so open space beyond ~2.9 m would read as the floor and the
        scan would silently cap there. Reprojecting each
        pixel and keeping a waist-height band makes the scan what it claims to
        be: a 2-D lidar sweep at roughly hip height, reporting HORIZONTAL range.

        Bins are absolute bearings in the AGENT's frame: index 0 = straight
        ahead, increasing counter-clockwise (left), one bin per degree. Unseen
        bearings stay None so a consumer can tell "nothing there" from
        "never looked".
        """
        cfg = self.config
        nbins = int(cfg["scan_bins"])
        dmax = float(cfg["depth_max"])
        scan = [None] * nbins
        half_hfov = math.radians(float(cfg["pano_hfov"])) / 2.0

        for cam in range(self.PANO_VIEWS):
            d = depth_views.get(cam)
            if d is None:
                continue
            arr = np.asarray(d, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.squeeze()
            h, w = arr.shape[0], arr.shape[1]
            # habitat_sim depth is already in metres, measured along the camera
            # axis — see config["depth_max"].
            metres = np.where(arr <= 1e-6, np.nan, arr)   # 0 = no return

            # Reproject: pixel row -> height relative to the camera, axial depth
            # -> horizontal range. f is in pixels, from the hfov. With Z the axial
            # depth, X = (x/f)Z and Y = (y/f)Z, so height is Y and the horizontal
            # range is sqrt(X^2 + Z^2) = Z*sqrt(1 + (x/f)^2) — the COLUMN term,
            # multiplied. (Dividing by the row term instead would shrink readings
            # by up to 2x at the frame edges.)
            f = (w / 2.0) / math.tan(half_hfov)
            vv = np.arange(h, dtype=np.float32)[:, None]
            uu = np.arange(w, dtype=np.float32)[None, :]
            y_over_z = -((vv - (h - 1) / 2.0) / f)         # +up
            x_over_z = (uu - (w - 1) / 2.0) / f            # +right
            height = y_over_z * metres                     # metres above camera
            horiz = metres * np.sqrt(1.0 + x_over_z ** 2)

            cam_y = float(cfg["camera_height"])
            keep = ((height > (float(cfg["scan_floor_clear_m"]) - cam_y))
                    & (height < (float(cfg["scan_ceiling_clear_m"]) - cam_y)))
            horiz = np.where(keep, horiz, np.nan)
            # Cap at the modelled camera range rather than dropping longer
            # returns: the simulator can see across a whole building, a depth
            # camera cannot. A bin reading dmax means "at least dmax", which is
            # what a real sensor reports, and it keeps "nothing within range"
            # distinguishable from None = "never looked that way".
            horiz = np.minimum(horiz, dmax)
            # +inf for "nothing in the band", not NaN: nanmin over an all-NaN
            # column is a RuntimeWarning per column per pano, which buries the
            # server log in noise for a case that is completely normal (a column
            # of pure ceiling, or open sky through a window).
            horiz = np.where(np.isnan(horiz), np.inf, horiz)
            col_min = horiz.min(axis=0)

            cam_yaw = 2.0 * math.pi * cam / self.PANO_VIEWS
            tan_half = math.tan(half_hfov)
            for u in range(w):
                r = col_min[u]
                if not np.isfinite(r):
                    continue
                # pinhole: normalised image x -> angle off the camera axis.
                # +x in image is to the RIGHT, so bearing is the negative.
                nx = (2.0 * (u + 0.5) / w) - 1.0
                off = math.atan(nx * tan_half)
                bearing = cam_yaw - off
                idx = int(round(math.degrees(bearing))) % nbins
                cur = scan[idx]
                if cur is None or r < cur:
                    scan[idx] = round(float(r), 3)
        return scan

    @staticmethod
    def clearance_at(scan, turn_deg, width_deg=20.0):
        """Nearest surface within a wedge around a bearing, metres, or None.

        A wedge rather than a single ray: a robot 0.2 m wide cannot thread a gap
        that only one ray sees through.
        """
        if not scan:
            return None
        n = len(scan)
        half = max(1, int(width_deg / 2))
        centre = int(round(turn_deg)) % n
        vals = [scan[(centre + k) % n] for k in range(-half, half + 1)]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return round(min(vals), 2) if vals else None

    # The four views, in the order a head turns rather than the order the cameras
    # are indexed. `strip` stitches in THIS order, so left-of-picture is
    # left-of-robot and the wrap-around (BEHIND) sits at the far end behind a wider
    # seam. PANO_NAMES stays camera order because NavGPT VLA's rig depends on it.
    SWEEP = (("left", "LEFT"), ("front", "AHEAD"), ("right", "RIGHT"),
             ("back", "BEHIND"))
    SWEEP_BEARING = {"front": 0, "left": 90, "back": 180, "right": 270}

    def _walkable_m(self, pos, rot, bearing_rad, max_m=8.0, step=0.25):
        """How far the robot would ACTUALLY GET walking along a bearing, metres.

        Used for the labels instead of cone clearance: a door frame clipping the
        20-degree cone can label a view straight down an open corridor
        "AHEAD 0.4M". A number painted on a picture is trusted much
        harder than one in a JSON field, so painting a misleading one is worse than
        painting none at all.

        The estimator is the navmesh's own `try_step`, which is what a forward
        primitive does — including sliding along walls — so this number is the
        OUTCOME OF THE ACTION THE TICK INVITES, simulated without moving the robot
        or spending a step. Returned as advance along the requested bearing, which
        is the quantity comparable to a move()'s distance_m.

        Marching `is_navigable` outward and stopping at the first failure would be
        wrong: the navmesh is eroded by the agent radius, so a doorway threshold
        reads non-navigable while the robot walks straight through it. Cone clearance stays in the payload as clearance_m; it
        still answers "am I about to scrape something".
        """
        if self._pathfinder is None:
            return None
        fwd = self._forward_vector(rot)
        fx, fz = float(fwd[0]), float(fwd[2])
        norm = math.hypot(fx, fz) or 1.0
        fx, fz = fx / norm, fz / norm
        lx, lz = -fz, fx
        cb, sb = math.cos(bearing_rad), math.sin(bearing_rad)
        return self._walkable_from(pos, cb * fx + sb * lx, cb * fz + sb * lz,
                                   max_m, step)

    def _walkable_from(self, pos, dx, dz, max_m=8.0, step=0.25):
        """The same march, from any point, along any world direction."""
        if self._pathfinder is None:
            return None
        start = np.array([float(pos[0]), float(pos[1]), float(pos[2])],
                         dtype=np.float32)
        cur, along = start, 0.0
        try:
            for _i in range(int(max_m / step)):
                d = along + step
                tgt = np.array([start[0] + dx * d, start[1], start[2] + dz * d],
                               dtype=np.float32)
                nxt = np.asarray(self._pathfinder.try_step(cur, tgt),
                                 dtype=np.float32)
                moved = math.hypot(float(nxt[0] - cur[0]), float(nxt[2] - cur[2]))
                if moved < step * 0.4:          # wedged: sliding has stopped paying
                    break
                cur = nxt
                along = ((float(cur[0]) - float(start[0])) * dx
                         + (float(cur[2]) - float(start[2])) * dz)
        except Exception:  # noqa: BLE001 - a probe, never a blocker
            pass
        return round(max(0.0, along), 2)

    @on_sim_thread
    def observe_pano(self):
        """4 RGB views + a calibrated 360-degree range scan + pose.

        Costs ZERO simulator steps: the four cameras render in the same
        get_sensor_observations() call as the agent's forward camera.

        The four views are returned as separate images. Every view carries its
        direction, its measured clearance, the look number and the robot's (x, y)
        burned into its own pixels, so an image pulled out of a long context
        still says what it is.
        """
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            if self.blind:
                size = int(self.config["pano_size"])
                views = {
                    name: base64.b64encode(_encode_png(
                        _blind_frame(size, [self._step_index, i, len(self._agent_path)])
                    )).decode("ascii")
                    for i, name in enumerate(self.PANO_NAMES)
                }
                pos, rot = self._agent_state()
                return {
                    "views": views, "raw_views": views,
                    "scan": [None] * int(self.config["scan_bins"]),
                    "pose": self._pose_dict(pos, rot), "blind": True,
                    "step_index": self._step_index,
                    "revisiting_earlier_position": self._revisiting(),
                }

            obs = self._sim.get_sensor_observations()
            views, depths, arrays = {}, {}, {}
            for cam, name in enumerate(self.PANO_NAMES):
                rgb = np.asarray(obs["pano_rgb_%d" % cam], dtype=np.uint8)
                if rgb.ndim == 3 and rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
                arrays[name] = rgb.copy()
                depths[cam] = obs.get("pano_depth_%d" % cam)

            raw_arrays = {name: a.copy() for name, a in arrays.items()}
            # Unannotated views: what NavGPT VLA receives. `views` carry the
            # labels drawn for the Planner.
            raw_views = {name: base64.b64encode(_encode_png(a)).decode('ascii') for name,a in raw_arrays.items()}
            presentation_fields = {}
            pos, rot = self._agent_state()
            scan = self._range_scan(depths)
            # ONE clearance computation, whose results are both burned into the
            # pictures and reported as fields. Two implementations would let the
            # painted label and the JSON disagree, which is exactly the failure
            # burning the label in is meant to remove.
            clear = {n: self.clearance_at(scan, b)
                     for n, b in self.SWEEP_BEARING.items()}

            self._view_index += 1
            hx, hy = self._place_xy(pos)
            # MAP-UP ties the picture to the map and to memory. LEFT / AHEAD /
            # RIGHT / BEHIND are robot-relative, so "the dining room was ahead"
            # silently stops being true the moment the robot turns — two looks twenty
            # turns apart can label the same world direction differently and nothing
            # on either picture says so. MAP-UP is the turn that would face the top of
            # the map, in the degrees move() takes, so it makes every panel's world
            # direction recoverable from the panel itself.
            _up_word, up_deg = _relative_turn(0.0, -1.0, self._yaw_of(rot))
            stamp = "VIEW %d   X %.1f Y %.1f   %s" % (
                self._view_index, hx, hy, _map_up_words(up_deg))

            # Ruler ticks: three per panel, at the panel centre and +-40 deg,
            # each printing THE EXACT ARGUMENT move() takes (positive = left) and
            # how far the navmesh says you could walk that way. Panels overlap by
            # 30 deg each side, so the twelve ticks cover the full circle without a
            # gap, and each sits under the pixel column it describes.
            hfov = math.radians(float(self.config["pano_hfov"]))
            rulers, walk = {}, {}
            for name, _word in self.SWEEP:
                cam_yaw = math.radians(self.SWEEP_BEARING[name])
                items = []
                for rel_deg in (40, 0, -40):
                    bearing = cam_yaw + math.radians(rel_deg)
                    m = self._walkable_m(pos, rot, bearing)
                    deg = int(round(math.degrees(bearing)))
                    deg = ((deg + 180) % 360) - 180
                    # u as a FRACTION of the width, so the tick lands in the right
                    # column whatever resolution the panel is delivered at
                    u_frac = 0.5 - (math.tan(math.radians(rel_deg))
                                    / (2.0 * math.tan(hfov / 2.0)))
                    # BEARING ONLY. Bearing and metres in one tick run together
                    # at small panel sizes; the walkable distance straight ahead
                    # is printed once per view, in its label.
                    tick = ('%.0f' % ((-math.degrees(self._yaw_of(rot)) - deg) % 360)
                            if _INTERFACE_VARIANT == 'absolute_bearings' else '%+d' % deg)
                    items.append((u_frac, tick))
                    if rel_deg == 0:
                        walk[name] = m
                rulers[name] = items

            def label_of(name, word):
                d = walk.get(name)
                if _INTERFACE_VARIANT == 'absolute_bearings':
                    word = 'BEARING %.0f' % ((-math.degrees(self._yaw_of(rot)) - self.SWEEP_BEARING[name]) % 360)
                return "%s %s" % (word, ("%.1fM" % d) if d is not None else "?")

            for name, word in self.SWEEP:
                img = arrays[name]
                if _INTERFACE_VARIANT != 'standard':
                    img, presentation_fields[name] = camera_canvas(raw_arrays[name],
                        [label_of(name, word), stamp], rulers[name], _INTERFACE_VARIANT, _text_bars, _labels_at)
                    views[name] = base64.b64encode(_encode_png(img)).decode('ascii')
                    continue
                _labels_at(img, [(u * img.shape[1], t)
                                 for u, t in rulers[name]])
                _text_bars(img, [label_of(name, word), stamp],
                           scales=[4, 2])
                views[name] = base64.b64encode(
                    _encode_png(img)).decode("ascii")
            # This is the ONLY place the colour map grows: a pano render is the
            # only observation carrying RGB *and* depth for all four cameras, so
            # the map records exactly where the agent has looked. The walked
            # trail comes from odometry instead (self._agent_path, appended every
            # step), so moving without looking still shows up as a path.
            try:
                self._accumulate_colour(raw_arrays, depths, pos, rot)
            except Exception:  # noqa: BLE001 - the map is an aid, never a blocker
                log.exception("colour projection failed")
            return {
                "views": views,
                "view_index": self._view_index,
                "you_are_at_xy": [hx, hy],
                "up_on_the_map_is_deg": None if _INTERFACE_VARIANT == "absolute_bearings" else up_deg,
                # the same numbers that are painted on the pictures, from the same
                # computation.
                # can_walk_m is what the labels show and what a move should be
                # planned against; clearance_m is the nearest surface in a cone,
                # which answers a different question (see _walkable_m).
                "can_walk_m": {"ahead_m": walk.get("front"),
                               "left_m": walk.get("left"),
                               "right_m": walk.get("right"),
                               "behind_m": walk.get("back")},
                "nearest_obstacle_m": {"ahead_m": clear.get("front"),
                                       "left_m": clear.get("left"),
                                       "right_m": clear.get("right"),
                                       "behind_m": clear.get("back")},
                "raw_views": raw_views,
                "presentation": presentation_fields,
                "absolute_heading_deg": (-math.degrees(self._yaw_of(rot))) % 360,
                "bearing_ruler": {n: [t for _u, t in items]
                                  for n, items in rulers.items()},
                "scan": scan,
                "pose": self._pose_dict(pos, rot),
                "blind": False,
                "step_index": self._step_index,
                "revisiting_earlier_position": self._revisiting(),
            }

    def _pose_dict(self, pos, rot):
        """Agent pose. WXYZ quaternion, because that is what the NavGPT VLA
        /act endpoint wants (its reply then comes back XYZW — the asymmetry is
        real and documented in VLNCE-EVAL's own client)."""
        return {
            "position": list(map(float, np.asarray(pos))),
            "rotation_wxyz": [float(rot.w), float(rot.x), float(rot.y), float(rot.z)],
            "heading_rad": self._yaw_of(rot),
        }

    @on_sim_thread
    def agent_state(self):
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            pos, rot = self._agent_state()
            out = self._pose_dict(pos, rot)
            out["step_index"] = self._step_index
            out["done"] = self._done
            return out

    # ── higher-level motion ──

    def _routed_walk(self, distance_m):
        """Walk the navmesh route to a point `distance_m` straight ahead.

        Returns (walked_m, blocked, True) if it routed, or None to let the caller
        fall back to straight-line primitives.

        WHY: a straight walk stops on contact, so a chair in a doorway costs a metre
        of intent and many moves come up well short of the distance asked for.
        Routing spends the same steps and arrives.

        WHY THE DETOUR CAP: unbounded routing would happily walk the agent around a
        wall into a different room to honour the request. So the route is taken only
        when it is nearly the straight line the agent actually asked for
        (<= 1.35x + 0.3 m);
        anything longer means the way ahead is genuinely blocked, which is
        information the agent needs rather than something to route around.

        Charging is per 0.25 m of arc, identical to the primitives, and the path is
        recorded at that resolution so the place graph stays connected.
        """
        import habitat_sim

        if self._pathfinder is None:
            return None
        cfg = self.config
        step_m = float(cfg["step_size_m"])
        pos, rot = self._agent_state()
        fwd = self._forward_vector(rot)
        fx, fz = float(fwd[0]), float(fwd[2])
        n = math.hypot(fx, fz) or 1.0
        fx, fz = fx / n, fz / n
        want = np.array([float(pos[0]) + fx * distance_m, float(pos[1]),
                         float(pos[2]) + fz * distance_m], dtype=np.float32)
        try:
            target = np.asarray(self._pathfinder.snap_point(want), dtype=np.float32)
            if not np.all(np.isfinite(target)):
                return None
            sp = habitat_sim.ShortestPath()
            sp.requested_start = np.asarray(pos, dtype=np.float32)
            sp.requested_end = target
            if not self._pathfinder.find_path(sp) or len(sp.points) < 2:
                return None
            arc = float(sp.geodesic_distance)
            if not math.isfinite(arc) or arc > 1.35 * float(distance_m) + 0.3:
                return None            # genuinely blocked ahead — say so instead
        except Exception:  # noqa: BLE001 - never let the aid break the move
            return None

        max_steps = int(cfg["max_steps"])
        placed = np.asarray(pos, dtype=np.float32)
        walked, charge = 0.0, 0
        for q in [np.asarray(v, dtype=np.float32) for v in sp.points[1:]]:
            leg = math.hypot(float(q[0]) - float(placed[0]),
                             float(q[2]) - float(placed[2]))
            if leg < 1e-6:
                continue
            need = int((walked + leg) / step_m) - charge
            if self._step_index + need >= max_steps:
                return (round(walked, 3), True, True)
            self._step_index += need
            charge += need
            walked += leg
            subs = max(1, int(math.ceil(leg / step_m - 1e-9)))
            for i in range(1, subs + 1):
                t = i / float(subs)
                pt = [float(placed[0]) + (float(q[0]) - float(placed[0])) * t,
                      float(placed[1]) + (float(q[1]) - float(placed[1])) * t,
                      float(placed[2]) + (float(q[2]) - float(placed[2])) * t]
                if not self._agent_path or pt != self._agent_path[-1]:
                    self._agent_path.append(pt)
            placed = q
        st = self._sim.get_agent(0).get_state()
        st.position = placed
        self._sim.get_agent(0).set_state(st)
        self._record_motion_frame()
        return (round(walked, 3), False, True)

    @on_sim_thread
    def step_hightolow(self, angle_rad, distance_m, route=False):
        """Rotate by angle_rad (CCW-positive), then walk up to distance_m.

        `route` walks the navmesh route to the requested point instead of a straight
        line, but ONLY when that route is nearly straight (see _routed_walk). It is
        opt-in, so the default keeps straight-line semantics.

        The turn is applied as a pose change, then
        ``int(distance // step_size)`` MOVE_FORWARD primitives run with sliding.
        Every primitive counts against the step budget, so this is a convenience
        over `step`, not a discount.

        Returns how far it ACTUALLY got, which is the point — `blocked` means
        obstructed, never out-of-budget, so the agent can tell a wall from a
        spent budget.
        """
        import quaternion

        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            if self._done:
                return {"error": "episode already done"}

            cfg = self.config
            start_pos, _ = self._agent_state()
            start_pos = np.asarray(start_pos, dtype=np.float64)

            # apply the turn as one pose update rather than N turn primitives:
            # a 90 degree turn should not cost 6 steps when the agent asked for
            # a single calibrated motion.
            if abs(float(angle_rad)) > 1e-6:
                half = float(angle_rad) / 2.0
                turn = np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)
                if self.blind:
                    self._rot = self._rot * turn
                else:
                    st = self._sim.get_agent(0).get_state()
                    st.rotation = st.rotation * turn
                    self._sim.get_agent(0).set_state(st)
                    self._record_motion_frame()

            ksteps = int(max(0.0, float(distance_m)) // float(cfg["step_size_m"]))
            walked, blocked = 0.0, False
            routed = False
            if route and not self.blind and ksteps > 0:
                got = self._routed_walk(float(distance_m))
                if got is not None:
                    walked, blocked, routed = got
            for _ in range(0 if routed else ksteps):
                if self._step_index >= int(cfg["max_steps"]):
                    break
                prev, _ = self._agent_state()
                prev = np.asarray(prev, dtype=np.float64)
                if self.blind:
                    self._blind_move(1)
                else:
                    self._sim.step("move_forward")
                    self._record_motion_frame()
                pos, _ = self._agent_state()
                pos = np.asarray(pos, dtype=np.float64)
                moved = float(np.linalg.norm(pos - prev))
                self._step_index += 1
                point = list(map(float, pos))
                if not self._agent_path or point != self._agent_path[-1]:
                    self._agent_path.append(point)
                walked += moved
                if moved < 1e-4:
                    self._collisions += 1
                    blocked = True
                    break

            truncated = False
            if not self._done and self._step_index >= int(cfg["max_steps"]):
                truncated = True
                self._done = True

            pos, rot = self._agent_state()
            net = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - start_pos))
            return {
                "requested_m": round(float(distance_m), 3),
                "walked_m": round(walked, 3),
                "net_displacement_m": round(net, 3),
                "turned_deg": round(math.degrees(float(angle_rad)), 1),
                "blocked": blocked,
                "routed": routed,
                "steps_used": ksteps,
                "step_count": self._step_index,
                "terminated": bool(self._stop_called),
                "truncated": truncated,
                "pose": self._pose_dict(pos, rot),
            }

    @on_sim_thread
    def teleport(self, position, rotation_xyzw, theta=0.0, is_stuck=False):
        """Place the agent at a world pose — how the NavGPT VLA's TELEPORT
        actions execute.

        Mirrors VLNCE-EVAL's patched TeleportAction: if the target is not
        navigable it does NOT silently succeed, it takes one discrete action
        instead. ``theta`` is the VLA's heading change for the step (negative
        turns left) and ``is_stuck`` says the previous VLA step kept its pose
        (vla_client.pose_unchanged). When |theta| > 0.1 rad or the agent is
        stuck it turns toward theta; otherwise it moves forward, sliding along
        whatever blocks it. Counts one step either way.
        """
        import quaternion

        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            if self._done:
                return {"error": "episode already done"}

            target = np.asarray(position, dtype=np.float32)
            navigable = bool(self._pathfinder.is_navigable(target)) if self._pathfinder else False
            fallback = None
            self._teleports += 1

            if navigable:
                x, y, z, w = [float(v) for v in rotation_xyzw]
                rot = np.quaternion(w, x, y, z)
                if self.blind:
                    self._pos, self._rot = target.astype(np.float32), rot
                else:
                    st = self._sim.get_agent(0).get_state()
                    st.position = target
                    st.rotation = rot
                    self._sim.get_agent(0).set_state(st)
                    self._record_motion_frame()
            else:
                # Not navigable: one real primitive instead of warping into
                # geometry.
                theta = float(theta or 0.0)
                turn = abs(theta) > 0.1 or bool(is_stuck)
                if turn and theta < 0:
                    fallback = "turn_left"
                elif turn and theta > 0:
                    fallback = "turn_right"
                else:
                    fallback = "move_forward"
                action = {"turn_left": 2, "turn_right": 3, "move_forward": 1}[fallback]
                self._teleport_fallbacks += 1
                if self.blind:
                    self._blind_move(action)
                else:
                    self._sim.step(fallback)
                    self._record_motion_frame()

            self._step_index += 1
            pos, rot = self._agent_state()
            point = list(map(float, np.asarray(pos)))
            if not self._agent_path or point != self._agent_path[-1]:
                self._agent_path.append(point)

            truncated = False
            if not self._done and self._step_index >= int(self.config["max_steps"]):
                truncated = True
                self._done = True

            return {
                "navigable": navigable,
                "fallback": fallback,
                "step_count": self._step_index,
                "terminated": bool(self._stop_called),
                "truncated": truncated,
                "pose": self._pose_dict(pos, rot),
            }

    # ── metrics ──

    def _geodesic(self, a, b):
        import habitat_sim

        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(a, dtype=np.float32)
        path.requested_end = np.asarray(b, dtype=np.float32)
        if self._pathfinder is not None and self._pathfinder.find_path(path):
            return float(path.geodesic_distance)
        raise ValueError("No navigable geodesic path between scoring positions")

    def _metrics(self):
        cfg = self.config
        ep = self._ep
        goal = (ep.get("goals") or [{}])[0].get("position")
        if goal is None:
            return {}

        cur = self._agent_path[-1] if self._agent_path else ep["start_position"]
        d_goal = self._geodesic(cur, goal)
        success = float(self._stop_called and d_goal <= cfg["success_distance"])

        oracle = 0.0
        for p in self._agent_path:
            if self._geodesic(p, goal) <= cfg["success_distance"]:
                oracle = 1.0
                break

        path_len = 0.0
        for i in range(1, len(self._agent_path)):
            path_len += float(
                np.linalg.norm(
                    np.asarray(self._agent_path[i]) - np.asarray(self._agent_path[i - 1])
                )
            )

        gd = self._geodesic(ep["start_position"], goal)
        spl = success * (gd / max(path_len, gd) if max(path_len, gd) > 0 else 1.0)

        gt = self._gt.get(str(ep.get("episode_id", "")), [])
        ndtw = _ndtw(self._agent_path, gt, cfg["success_distance"]) if gt else None

        return {
            "distance_to_goal": round(d_goal, 4),
            "success": success,
            "spl": round(spl, 4),
            "oracle_success": oracle,
            "ndtw": round(ndtw, 4) if ndtw is not None else None,
            "sdtw": round(success * ndtw, 4) if ndtw is not None else None,
            "path_length": round(path_len, 4),
            "steps_taken": float(self._step_index),
            "stop_called": float(self._stop_called),
            "collisions": float(self._collisions),
            "teleports": float(self._teleports),
            "teleport_fallbacks": float(self._teleport_fallbacks),
        }

    @on_sim_thread
    def evaluate(self):
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            return self._metrics()

    # ── colour projection ────────────────────────────────────────────────
    #
    # Every pano render carries RGB and depth for all four cameras, so each pixel
    # can be unprojected to a world point and painted into a top-down cell. That
    # turns the map from flat occupancy into what the robot has actually seen,
    # in colour, which is far easier to match against an instruction ("the red
    # rug", "the blue pool") than grey blobs.
    #
    # The grid is sized ONCE from the navmesh bounds and never re-fitted, so the
    # map keeps a fixed frame for the whole episode: successive calls are
    # directly comparable instead of silently rescaling as coverage grows.

    def _cmap_init(self):
        if self._cmap_sum is not None:
            return
        lo, hi = self._pathfinder.get_bounds()
        mpp = float(self.config["map_mpp"])
        self._cmap_lo = (float(lo[0]), float(lo[2]))
        w = int(math.ceil((float(hi[0]) - float(lo[0])) / mpp)) + 1
        h = int(math.ceil((float(hi[2]) - float(lo[2])) / mpp)) + 1
        self._cmap_sum = np.zeros((h, w, 3), dtype=np.float32)
        self._cmap_n = np.zeros((h, w), dtype=np.int32)
        self._cmap_shape = (h, w)

    def _accumulate_colour(self, rgbs, depths, pos, rot):
        """Unproject 4 RGB-D views into the fixed top-down colour grid."""
        if self._pathfinder is None or not self._pathfinder.is_loaded:
            return
        import quaternion

        self._cmap_init()
        cfg = self.config
        mpp = float(cfg["map_mpp"])
        dmax = float(cfg["depth_max"])
        half = math.radians(float(cfg["pano_hfov"])) / 2.0
        floor_y = float(pos[1])
        keep_lo, keep_hi = floor_y - 0.30, floor_y + float(cfg["map_ceiling_m"])
        stride = int(cfg["map_pixel_stride"])
        R_agent = quaternion.as_rotation_matrix(rot)
        # The cameras sit camera_height above the agent's feet, and `pos` is the
        # feet. Unprojecting from `pos` would put every point that much too low and
        # the whole floor would fall below keep_lo.
        cam_pos = (np.asarray(pos, dtype=np.float32)
                   + np.array([0.0, float(cfg["camera_height"]), 0.0],
                              dtype=np.float32))
        h_g, w_g = self._cmap_shape

        for cam, name in enumerate(self.PANO_NAMES):
            rgb = rgbs.get(name)
            d = depths.get(cam)
            if rgb is None or d is None:
                continue
            rgb = np.asarray(rgb, dtype=np.uint8)
            dep = np.asarray(d, dtype=np.float32)
            if dep.ndim == 3:
                dep = dep.squeeze()
            rgb = rgb[::stride, ::stride, :3]
            dep = dep[::stride, ::stride]          # metres; see config["depth_max"]
            H, W = dep.shape
            fx = (W / 2.0) / math.tan(half)
            fy = (H / 2.0) / math.tan(half)
            uu, vv = np.meshgrid(np.arange(W), np.arange(H))
            xn = (uu - (W - 1) / 2.0) / fx
            yn = -((vv - (H - 1) / 2.0) / fy)
            # habitat depth is the distance along the camera's -Z axis
            # Beyond the modelled camera range is not a measurement, so it is
            # not painted: rooms only visible across a 20 m atrium stay unknown.
            good = (dep > 1e-3) & (dep < dmax)
            if not good.any():
                continue
            pc = np.stack([xn * dep, yn * dep, -dep], axis=-1)[good]
            cols = rgb[good].astype(np.float32)
            yaw = 2.0 * math.pi * cam / self.PANO_VIEWS
            cy, sy = math.cos(yaw), math.sin(yaw)
            R_cam = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            world = pc @ (R_agent @ R_cam).T + cam_pos
            band = (world[:, 1] > keep_lo) & (world[:, 1] < keep_hi)
            world, cols = world[band], cols[band]
            if not len(world):
                continue
            col_i = np.rint((world[:, 0] - self._cmap_lo[0]) / mpp).astype(np.int32)
            row_i = np.rint((world[:, 2] - self._cmap_lo[1]) / mpp).astype(np.int32)
            ok = (col_i >= 0) & (col_i < w_g) & (row_i >= 0) & (row_i < h_g)
            col_i, row_i, cols = col_i[ok], row_i[ok], cols[ok]

            # A sampled pixel is an AREA sample, not a point: at range d it
            # covers about d * dtheta metres, where dtheta is the angle between
            # sampled columns. Past ~5 m that is wider than a map cell, so
            # painting one cell per sample would leave the far field as a fan of
            # single-pixel spokes with unknown between them. Splat each sample
            # over the cells it actually covers instead.
            rng = np.linalg.norm(world[ok] - cam_pos, axis=1)
            dtheta = (2.0 * half) / max(1, W)
            rmax = int(max(1, round(float(cfg["map_splat_max_m"]) / mpp)))
            radius = np.clip(np.rint(rng * dtheta / mpp / 2.0),
                             0, rmax).astype(np.int32)

            sum_flat = self._cmap_sum.reshape(-1, 3)
            n_flat = self._cmap_n.reshape(-1)
            offs = range(-rmax, rmax + 1)
            for dr in offs:
                for dc in offs:
                    reach = max(abs(dr), abs(dc))
                    m = radius >= reach if reach else slice(None)
                    r2, c2, cl = row_i[m] + dr, col_i[m] + dc, cols[m]
                    inside = (r2 >= 0) & (r2 < h_g) & (c2 >= 0) & (c2 < w_g)
                    if not inside.any():
                        continue
                    flat = r2[inside] * w_g + c2[inside]
                    cl = cl[inside]
                    for ch in range(3):
                        np.add.at(sum_flat[:, ch], flat, cl[:, ch])
                    np.add.at(n_flat, flat, 1)

    def _revisiting(self, radius_m=1.0, ignore_recent_m=3.0):
        """Is the agent back where it already stood, more than a few metres ago?

        Odometry only — no goal, no reference path — so it is fair game for a
        tool. It answers the question a forward camera cannot: a classic failure
        is re-entering a junction without recognising it.

        ``ignore_recent_m`` of trailing path is excluded, or every step would
        trivially be "near where I just was". Distances are straight-line in the
        x/z plane; y is the floor.
        """
        path = self._agent_path
        if len(path) < 4:
            return False
        p = np.asarray(path, dtype=np.float64)
        seg = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1)
        # walk back from the end until ignore_recent_m of travel is excluded
        run, cut = 0.0, len(path) - 1
        while cut > 0 and run < float(ignore_recent_m):
            run += float(seg[cut - 1])
            cut -= 1
        if cut <= 0:
            return False
        here = p[-1, [0, 2]]
        d = np.linalg.norm(p[:cut, [0, 2]] - here, axis=1)
        return bool(d.min() < float(radius_m))

    # ── the place graph, as the agent sees it ──

    def _graph(self):
        """The derived place graph for the live episode, with this env's config."""
        cfg = self.config
        return _derive_places(self._agent_path,
                              spacing_m=float(cfg["place_spacing_m"]),
                              snap_frac=float(cfg["place_snap_frac"]),
                              jump_m=float(cfg["place_jump_m"]),
                              max_nodes=int(cfg["place_max_nodes"]))

    def _place_xy(self, position):
        """World xz -> the agent-facing frame: metres east/north of the START.

        This is the only coordinate frame the agent ever sees. Habitat's raw x/z
        never leaves the server, and the z sign is flipped exactly once, here, so
        that "+y" and "up on the map" mean the same thing everywhere — in this
        table, in `you_are_at_xy`, and on the drawn map. The start is (0, 0) by
        construction, which gives every number a meaning without a legend.
        """
        start = self._agent_path[0] if self._agent_path else [0.0, 0.0, 0.0]
        # + 0.0 turns -0.0 into 0.0: the start must read (0.0, 0.0), not (0.0, -0.0)
        return [round(float(position[0]) - float(start[0]), 2) + 0.0,
                round(-(float(position[2]) - float(start[2])), 2) + 0.0]

    @on_sim_thread
    def places(self):
        """The place graph: where the robot has been, and how those places connect.

        Odometry only. There is no goal, no reference path and no simulator
        semantics in here — a place is a spot the robot occupied, an edge is a
        stretch it actually walked, and the labels belong to the agent, not to us
        (MP3D's room annotations are not even loaded; see _open_scene).

        Distances are two numbers on purpose: `straight_m` is how far away the
        place is, `route_m` is what walking back to it would cost, and they differ
        by a factor of 3 often enough that the agent needs both to decide.
        """
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            if not self._agent_path:
                return {"error": "no poses yet"}
            graph = self._graph()
            nodes, here = graph["nodes"], self._agent_path[-1]
            _, rot_now = self._agent_state()
            out = []
            for n in nodes:
                xy = self._place_xy(n["position"])
                dx = float(n["position"][0]) - float(here[0])
                dz = float(n["position"][2]) - float(here[2])
                word, deg = _relative_turn(dx, dz, self._yaw_of(rot_now))
                route, route_m = _place_route(graph, graph["current"], n["id"])
                out.append({
                    "id": n["id"],
                    "xy": xy,
                    "position": [round(float(v), 3) for v in n["position"]],
                    "visits": n["visits"],
                    "straight_m": round(math.hypot(dx, dz), 2),
                    "route_m": route_m,
                    "route": route,
                    # in move()'s units: positive is left, so it can be passed
                    # to move(turn_deg, ...) with no conversion
                    "turn": ('bearing %.0f deg CW from map-north' % (math.degrees(math.atan2(dx, -dz)) % 360)
                             if _INTERFACE_VARIANT == 'absolute_bearings' else word),
                    "turn_deg": None if _INTERFACE_VARIANT == 'absolute_bearings' else deg,
                    "links": sorted(
                        e["b"] if e["a"] == n["id"] else e["a"]
                        for e in graph["edges"]
                        if n["id"] in (e["a"], e["b"])),
                })
            nearest = min(out, key=lambda r: r["straight_m"]) if out else None
            res = {
                "places": out,
                "you_are_at_xy": self._place_xy(here),
                # `current` is the edge-bookkeeping cursor, not "where you are":
                # between the snap radius and the spacing there is a dead band in
                # which no transition fires, so it can sit up to a spacing away.
                # The agent is told `nearest_place`, recomputed here, instead.
                "nearest_place": nearest["id"] if nearest else None,
                "nearest_place_m": nearest["straight_m"] if nearest else None,
                "spacing_m": float(self.config["place_spacing_m"]),
                # No compass words anywhere: x/y name the picture's own axes, and
                # every direction is a turn the agent can hand straight to move().
                "frame": "x/y are metres along the map's own axes (x right, y up) "
                         "measured from where you started, so the start is (0, 0); "
                         "turn is which way to turn to face the place, in the same "
                         "degrees move() takes (positive = left)",
            }
            if len(out) <= 1:
                res["note"] = ("places appear as you travel; you have not yet moved "
                               "far enough from the start for a second one")
            return res

    @on_sim_thread
    def retrace_to(self, place=None, xy=None):
        """Walk back to a place along the ground the robot already covered.

        Why this lives in NavGPT Environment rather than as tool-side teleport hops:

        * A hop between two places is a CHORD. The walked arc can be several times
          the chord, so a chord can cut through a wall — and ``teleport``'s
          navigability guard cannot catch it, because the endpoint is navigable:
          the robot stood there.
        * The recorded path would then contain the chord, not the route, which
          makes ``path_length`` shorter than the truth and flatters SPL.
        * A step charge kept outside this server is overwritten by the next
          ``step_count`` it reports.

        So the route is replayed through the recorded points themselves, and the
        cost is charged to ``_step_index`` at the same 0.25 m per step
        ``step_hightolow`` uses — which means the env's own ``max_steps``
        truncation and the number the agent reads cannot disagree. Running
        out mid-route stops where the budget ran out and says so.

        ``_collisions`` is deliberately untouched: every point replayed here was
        traversed once already, so there is no contact to hide.
        """
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            if self._done:
                return {"error": "episode already over"}
            if not self._agent_path:
                return {"error": "no poses yet"}

            graph = self._graph()
            nodes = graph["nodes"]
            target = None
            if place is not None:
                for n in nodes:
                    if n["id"] == int(place):
                        target = n
                        break
                if target is None:
                    return {"error": "unknown place {}; places are 0..{}".format(
                        int(place), len(nodes) - 1)}
            elif xy is not None:
                # xy arrives already in the agent frame, so compare in it
                spacing = float(self.config["place_spacing_m"])
                best, best_d = None, None
                for n in nodes:
                    nxy = self._place_xy(n["position"])
                    d = math.hypot(nxy[0] - float(xy[0]), nxy[1] - float(xy[1]))
                    if best_d is None or d < best_d:
                        best, best_d = n, d
                if best is None or best_d > spacing:
                    return {"error": "no place within {} m of ({}, {}); nearest is "
                                     "{:.1f} m away".format(spacing, xy[0], xy[1],
                                                            best_d or 0.0)}
                target = best
            else:
                return {"error": "give either place (an id) or xy"}

            route, route_m = _place_route(graph, graph["current"], target["id"])
            if route is None:
                return {"error": "no route to place {} — every path the robot took "
                                 "between here and there crossed a teleport, so it "
                                 "has never walked it".format(target["id"])}

            # Expand the route into the recorded points of each hop, oriented so
            # the replay runs in the direction of travel.
            path = self._agent_path
            points = []
            for a, b in zip(route, route[1:]):
                span = next((e["span"] for e in graph["edges"]
                             if {e["a"], e["b"]} == {a, b}), None)
                if span is None:
                    continue
                seg = path[span[0]:span[1] + 1]
                if not seg:
                    continue
                # The edge key is unordered, so orient the recorded stretch by
                # whichever end starts nearer the place being left.
                start_pos = next(n["position"] for n in nodes if n["id"] == a)
                def _to_start(q):
                    return math.hypot(float(q[0]) - float(start_pos[0]),
                                      float(q[2]) - float(start_pos[2]))
                if _to_start(seg[0]) > _to_start(seg[-1]):
                    seg = list(reversed(seg))
                points.extend(seg)
            points.append(target["position"])

            import quaternion  # noqa: F401 - registers np.quaternion

            step_m = float(self.config["step_size_m"])
            max_steps = int(self.config["max_steps"])
            walked, charged, truncated = 0.0, 0, False
            prev = list(map(float, path[-1]))
            placed = prev
            for pt in points:
                pt = list(map(float, pt))
                d = math.hypot(pt[0] - prev[0], pt[2] - prev[2])
                # Charge on the CUMULATIVE arc, never per segment. Recorded points
                # sit ~0.21 m apart (habitat's forward primitive with sliding), so
                # int(0.21 / 0.25) is zero and a per-segment charge would bill
                # almost nothing.
                want = int((walked + d) / step_m)
                if self._step_index + (want - charged) >= max_steps:
                    truncated = True
                    break
                self._step_index += want - charged
                charged = want
                walked += d
                if not self._agent_path or pt != self._agent_path[-1]:
                    self._agent_path.append(pt)
                placed, prev = pt, pt
            # round the tail up, the way a robot pays for a part-step
            if not truncated and walked > charged * step_m + 1e-9:
                self._step_index += 1
                charged += 1

            # Face along the last stretch replayed: free, exact, and what a robot
            # that walked this route would be looking at. _agent_path stores no
            # rotations, so inventing one from a stale record would be worse.
            _, rot = self._agent_state()
            if walked > 1e-6 and len(self._agent_path) > 1:
                back = self._agent_path[-2]
                fx, fz = placed[0] - float(back[0]), placed[2] - float(back[2])
                if math.hypot(fx, fz) > 1e-6:
                    yaw = math.atan2(-fx, -fz)      # -Z is forward in habitat
                    half = yaw / 2.0
                    rot = np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)

            target_pos = np.asarray(placed, dtype=np.float32)
            if self.blind:
                self._pos, self._rot = target_pos, rot
            else:
                st = self._sim.get_agent(0).get_state()
                st.position = target_pos
                st.rotation = rot
                self._sim.get_agent(0).set_state(st)
                self._record_motion_frame()

            if self._step_index >= max_steps:
                truncated = True
                self._done = True

            arrived = math.hypot(placed[0] - float(target["position"][0]),
                                 placed[2] - float(target["position"][2])) < 0.51
            pos, rot = self._agent_state()
            out = {
                "arrived": bool(arrived and not truncated),
                "place": target["id"],
                "place_xy": self._place_xy(target["position"]),
                "route": route,
                "route_m": route_m,
                "retraced_m": round(walked, 2),
                "steps_charged": charged,
                "step_count": self._step_index,
                "terminated": bool(self._stop_called),
                "truncated": truncated,
                "pose": self._pose_dict(pos, rot),
            }
            if truncated:
                out["stopped_short_m"] = round(
                    math.hypot(placed[0] - float(target["position"][0]),
                               placed[2] - float(target["position"][2])), 2)
            return out

    @on_sim_thread
    def retrace_path(self, position, rotation_wxyz=None):
        """Walk back along the recorded path to a point the robot passed.

        The VLA-rollout counterpart of ``retrace_to``: the target is a pose the
        robot occupied (a waypoint of the last rollout), so the walk replays the
        recorded path backwards from where the robot stands to the most recent
        visit of that point. Steps are charged on the cumulative arc at
        ``step_size_m`` per step and every replayed point is appended to the
        path, exactly as in ``retrace_to``, so ``max_steps``, ``path_length`` and
        SPL see the real walk. Running out mid-way stops where the budget ran
        out. On arrival the robot takes the recorded heading, so the views match
        the waypoint's.
        """
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            if self._done:
                return {"error": "episode already over"}
            path = self._agent_path
            if not path:
                return {"error": "no poses yet"}
            target = [float(v) for v in position]
            dists = [math.hypot(float(q[0]) - target[0], float(q[2]) - target[2])
                     for q in path]
            best = min(dists)
            if best > 0.5:
                return {"error": "that point is {:.1f} m from anywhere on the recorded "
                                 "path".format(best)}
            # the most recent visit, so the walk back is the shortest replay
            index = max(i for i, d in enumerate(dists) if d <= best + 1e-6)
            points = [list(map(float, q)) for q in reversed(path[index:-1])]

            step_m = float(self.config["step_size_m"])
            max_steps = int(self.config["max_steps"])
            walked, charged, truncated = 0.0, 0, False
            prev = list(map(float, path[-1]))
            placed = prev
            for pt in points:
                d = math.hypot(pt[0] - prev[0], pt[2] - prev[2])
                # charge on the cumulative arc, as retrace_to does
                want = int((walked + d) / step_m)
                if self._step_index + (want - charged) >= max_steps:
                    truncated = True
                    break
                self._step_index += want - charged
                charged = want
                walked += d
                if pt != self._agent_path[-1]:
                    self._agent_path.append(pt)
                placed, prev = pt, pt
            if not truncated and walked > charged * step_m + 1e-9:
                self._step_index += 1
                charged += 1

            _, rot = self._agent_state()
            if not truncated and rotation_wxyz is not None:
                w, x, y, z = [float(v) for v in rotation_wxyz]
                rot = np.quaternion(w, x, y, z)
            elif walked > 1e-6 and len(self._agent_path) > 1:
                back = self._agent_path[-2]
                fx, fz = placed[0] - float(back[0]), placed[2] - float(back[2])
                if math.hypot(fx, fz) > 1e-6:
                    half = math.atan2(-fx, -fz) / 2.0      # -Z is forward in habitat
                    rot = np.quaternion(math.cos(half), 0.0, math.sin(half), 0.0)

            target_pos = np.asarray(placed, dtype=np.float32)
            if self.blind:
                self._pos, self._rot = target_pos, rot
            else:
                st = self._sim.get_agent(0).get_state()
                st.position = target_pos
                st.rotation = rot
                self._sim.get_agent(0).set_state(st)
                self._record_motion_frame()

            if self._step_index >= max_steps:
                truncated = True
                self._done = True
            pos, rot = self._agent_state()
            return {
                "arrived": not truncated,
                "retraced_m": round(walked, 2),
                "steps_charged": charged,
                "step_count": self._step_index,
                "terminated": bool(self._stop_called),
                "truncated": truncated,
                "pose": self._pose_dict(pos, rot),
            }

    @on_sim_thread
    def observed_map(self):
        """What the robot has actually seen, in colour, on a fixed global frame.

        Cells are painted with the mean RGB of the pixels that landed in them,
        unprojected from the four RGB-D cameras. A cell is coloured only if a
        measurement fell in it, so nothing is disclosed that the depth cameras
        did not observe, and rooms never looked into stay unknown.

        The frame is CROPPED to the route plus what has been observed around it,
        with a small margin. A whole-building frame would leave the agent's
        surroundings as a speck in a mostly unobserved image. Cropping costs
        cross-call comparability, so ``frame`` reports the extent in metres and
        the corner offset, and the agent is told the frame follows the route.

        North stays up. The map is never rotated to the agent's heading: a frame
        that spins with the robot makes successive calls incomparable in the one
        dimension that matters for spotting a loop. Heading is shown by the
        arrow marker instead.

        Takes no arguments on purpose. Resolution is ``config["map_mpp"]`` for the whole episode,
        because a caller changing it mid-run is exactly the reframing this map is
        built to avoid.
        """
        with self._lock:
            if self._pathfinder is None or not self._pathfinder.is_loaded:
                return {"error": "no navmesh loaded"}
            if not self._agent_path:
                return {"error": "no poses yet"}
            self._cmap_init()
            mpp = float(self.config["map_mpp"])
            h, w = self._cmap_shape
            n = self._cmap_n
            img = np.zeros((h, w, 3), dtype=np.uint8)
            # Never observed, as a faint checker rather than flat black. Flat
            # black is what a dark surface also looks like, and the two mean
            # opposite things: the occlusion shadow behind a row of chairs is
            # unknown floor, not a black object. Kept light enough that the shape
            # of what HAS been mapped stays visible around the unknown. Measured
            # cells are floored brighter than this below, so the invariant holds:
            # nothing observed is ever as dark as unknown.
            img[...] = (46, 52, 63)
            yy, xx = np.mgrid[0:h, 0:w]
            img[((yy // 4) + (xx // 4)) % 2 == 0] = (58, 66, 79)

            seen = n > 0
            if seen.any():
                cnt = np.maximum(n, 1)[..., None].astype(np.float32)
                mean = self._cmap_sum / cnt
                # Gamma rather than gain-and-offset: a linear gain washes surfaces
                # out and clips the bright end, which costs exactly what the colour
                # map is for: telling one surface from another. Gamma lifts the
                # shadows and leaves hue and contrast alone.
                mean = 255.0 * np.power(np.clip(mean, 0, 255) / 255.0, 0.85)
                # Floor the brightness of MEASURED cells, so "observed and very
                # dark" never renders like "never observed" — otherwise a beam's
                # shadow on the floor looks exactly like a hole in the map. Anything
                # the cameras saw reads as at least this dark grey, and only unknown
                # is darker than it. Tracks the
                # unknown checker above: raise one and this must rise too.
                mean = np.maximum(mean, 88.0)
                img[seen] = np.clip(mean, 0, 255)[seen].astype(np.uint8)

                # Close single-cell holes. Far-field samples are area samples
                # splatted over the cells they cover, and where two splats leave a
                # one-cell gap the result is a checkerboard that reads as texture
                # rather than as missing data. A cell surrounded on 5+ of 8 sides
                # takes the mean of those neighbours. Interpolation BETWEEN
                # measurements only — never across a real frontier, and it does
                # not count toward seen_area_m2, which stays measured-cells-only.
                col = np.where(seen[..., None], mean, 0.0).astype(np.float32)
                msk = seen.astype(np.float32)
                acc_c = np.zeros_like(col)
                acc_m = np.zeros_like(msk)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        acc_c += np.roll(np.roll(col, dy, 0), dx, 1)
                        acc_m += np.roll(np.roll(msk, dy, 0), dx, 1)
                fill = (~seen) & (acc_m >= 5)
                # np.roll wraps, so the frame's own border would sample from the
                # opposite edge. Never fill the border.
                fill[0, :] = fill[-1, :] = False
                fill[:, 0] = fill[:, -1] = False
                if fill.any():
                    img[fill] = np.clip(
                        acc_c[fill] / acc_m[fill][..., None], 0, 255).astype(np.uint8)

            # Everything above is the colour grid, one cell per measurement
            # bucket. Everything below is drawn AFTER the upscale, in final
            # pixels: markers rendered at grid resolution and then magnified come
            # out as mosaic blocks. Shapes drawn at output scale stay shapes.
            seen_cells = int(seen.sum())

            # Crop to the route plus its observed surroundings. The union matters:
            # cropping to observed cells alone can clip the start of a long route
            # whose early coverage has scrolled away, and cropping to the route
            # alone throws away the context either side of it.
            content = seen | (fill if seen.any() else np.zeros_like(seen))
            ys, xs = np.nonzero(content)
            path_cells = [(int((float(p[2]) - self._cmap_lo[1]) / mpp),
                           int((float(p[0]) - self._cmap_lo[0]) / mpp))
                          for p in self._agent_path]
            pys = [c[0] for c in path_cells]; pxs = [c[1] for c in path_cells]
            # The route is never clipped — the whole walked path must be on the
            # map, because "have I been here before" is the question this tool
            # exists to answer.
            y0, y1 = min(pys), max(pys)
            x0, x1 = min(pxs), max(pxs)
            # Observed cells extend the frame, clipped to the 2nd-98th percentile
            # so the sparsest whiskers of the splat fan do not set the extent. A
            # panorama splats far-field area samples out to the depth limit, so
            # the raw observed bbox reaches tens of metres down every open
            # sightline. A high unobserved fraction is not a framing bug:
            # corridor-shaped coverage is non-convex, and ANY bounding box around
            # an L or a cross is mostly empty.
            if ys.size:
                q = np.percentile(ys, [2.0, 98.0])
                y0 = min(y0, int(q[0])); y1 = max(y1, int(q[1]))
                q = np.percentile(xs, [2.0, 98.0])
                x0 = min(x0, int(q[0])); x1 = max(x1, int(q[1]))
            margin = int(round(float(self.config["map_margin_m"]) / mpp))
            y0 = max(0, y0 - margin); x0 = max(0, x0 - margin)
            y1 = min(h - 1, y1 + margin); x1 = min(w - 1, x1 + margin)
            # Never let the crop collapse: a first call has one pose and no
            # coverage, and a 1x1 frame is useless.
            min_cells = int(round(float(self.config["map_min_span_m"]) / mpp))
            if y1 - y0 + 1 < min_cells:
                cy = (y0 + y1) // 2
                y0 = max(0, cy - min_cells // 2); y1 = min(h - 1, y0 + min_cells - 1)
            if x1 - x0 + 1 < min_cells:
                cx = (x0 + x1) // 2
                x0 = max(0, cx - min_cells // 2); x1 = min(w - 1, x0 + min_cells - 1)

            img = img[y0:y1 + 1, x0:x1 + 1]
            crop_seen = content[y0:y1 + 1, x0:x1 + 1]
            ch, cw = img.shape[0], img.shape[1]
            # Origin of the CROPPED frame in world metres, so marker maths below
            # stays in world coordinates and simply shifts.
            lo = (self._cmap_lo[0] + x0 * mpp, self._cmap_lo[1] + y0 * mpp)

            # Recompute magnification from the cropped size, not the navmesh, or a
            # small crop would come back as a small image.
            k = int(max(1, min(8, round(float(self.config["map_target_px"])
                                        / max(ch, cw)))))
            if k > 1:
                img = np.repeat(np.repeat(img, k, axis=0), k, axis=1)
            H, W = img.shape[0], img.shape[1]
            ppm = k / mpp                                  # pixels per metre

            def to_px(pt):
                return ((float(pt[0]) - lo[0]) * ppm,
                        (float(pt[2]) - lo[1]) * ppm)

            def disc(x, y, r, rgb):
                r = max(0.5, r)
                x0, x1 = int(math.floor(x - r)), int(math.ceil(x + r))
                y0, y1 = int(math.floor(y - r)), int(math.ceil(y + r))
                for yy in range(max(0, y0), min(H, y1 + 1)):
                    for xx in range(max(0, x0), min(W, x1 + 1)):
                        if (xx - x) ** 2 + (yy - y) ** 2 <= r * r:
                            img[yy, xx] = rgb

            def stroke(p0, p1, r, rgb):
                x0, y0 = p0
                x1, y1 = p1
                steps = int(max(abs(x1 - x0), abs(y1 - y0), 1) * 2)
                for t in range(steps + 1):
                    disc(x0 + (x1 - x0) * t / steps,
                         y0 + (y1 - y0) * t / steps, r, rgb)

            def polygon(poly, rgb):
                """Even-odd fill of a small closed polygon, in output pixels."""
                xs = [q[0] for q in poly]
                ys = [q[1] for q in poly]
                for yy in range(max(0, int(min(ys))), min(H, int(max(ys)) + 2)):
                    for xx in range(max(0, int(min(xs))), min(W, int(max(xs)) + 2)):
                        inside = False
                        j = len(poly) - 1
                        for i in range(len(poly)):
                            xi, yi = poly[i]
                            xj, yj = poly[j]
                            if (yi > yy) != (yj > yy) and \
                               xx < xi + (yy - yi) * (xj - xi) / ((yj - yi) or 1e-9):
                                inside = not inside
                            j = i
                        if inside:
                            img[yy, xx] = rgb

            EDGE = (18, 20, 26)          # dark keyline, so markers read on any colour
            START = (56, 132, 255)       # blue: where the episode began
            END = (240, 62, 62)          # red: where the robot is now
            PLACE = (255, 196, 64)       # amber: a place in the graph

            # The path is drawn as a time gradient, blue at the start fading to red
            # at the present, so the ORDER it was walked is visible. On a
            # single-colour trail a loop is just a loop; graded, it shows which way
            # round the loop went and which end is recent — which is the whole
            # question the agent asks when it suspects it is going in circles.
            #
            # Keylines are drawn TIGHT — a one-pixel outline, never a scaled-up
            # copy of the shape beneath, which would read as separate map content.
            pts = [to_px(p) for p in self._agent_path]
            segs = max(1, len(pts) - 1)
            for a, b in zip(pts, pts[1:]):
                stroke(a, b, 0.055 * ppm + 1.0, EDGE)
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                t = i / segs
                stroke(a, b, 0.055 * ppm,
                       tuple(int(round(START[c] + (END[c] - START[c]) * t))
                             for c in range(3)))

            # where it started: a blue dot, the same blue the path begins with
            disc(pts[0][0], pts[0][1], 0.20 * ppm + 1.0, EDGE)
            disc(pts[0][0], pts[0][1], 0.20 * ppm, START)

            _, rot = self._agent_state()
            fwd = self._forward_vector(rot)
            fx, fz = float(fwd[0]), float(fwd[2])
            norm = math.hypot(fx, fz) or 1.0
            fx, fz = fx / norm, fz / norm
            rx, rz = -fz, fx                       # perpendicular, in the xz plane
            here = self._agent_path[-1]

            # where it is now: a red ARROW, pointing where the robot faces. One
            # shape answers both "where am I" and "which way am I facing", which
            # is what a lost agent actually asks.
            def arrow(scale):
                # Tip, two barbs, and a notched tail, in world metres around the
                # current pose, then rotated onto (fx, fz).
                spec = [(1.00, 0.00), (-0.55, 0.62), (-0.28, 0.00), (-0.55, -0.62)]
                poly = []
                for along, across in spec:
                    ax = along * scale
                    cx_ = across * scale
                    dx = fx * ax + rx * cx_
                    dz = fz * ax + rz * cx_
                    poly.append(to_px((here[0] + dx, 0.0, here[2] + dz)))
                return poly

            # ── numbered place badges ──
            #
            # A second, distinct layer: the trail says where the robot WALKED, the
            # badges say where the PLACES are, and the id printed on each one is
            # what ties a row of the text table to a mark on the picture without
            # anybody counting pixels. Amber because the rest of the palette is
            # spoken for — blue is the start, red is now.
            #
            # Drawn after the trail (a badge on the route must not be buried) and
            # before the arrow (the robot must never be buried by a badge).
            graph = self._graph()
            places_px = {}
            nearest_id, nearest_d = None, None
            for n in graph["nodes"]:
                d = math.hypot(float(n["position"][0]) - float(here[0]),
                               float(n["position"][2]) - float(here[2]))
                if nearest_d is None or d < nearest_d:
                    nearest_id, nearest_d = n["id"], d
            def ring(x, y, radius, rgb, weight):
                """A circle outline. Used for the place the robot is standing on,
                which a filled disc cannot mark because the arrow is drawn over it.
                A ring encircles the arrow instead of hiding under it."""
                for i in range(72):
                    a = 2.0 * math.pi * i / 72.0
                    disc(x + radius * math.cos(a), y + radius * math.sin(a),
                         weight, rgb)

            for n in graph["nodes"]:
                cx_px, cy_px = to_px(n["position"])
                places_px[n["id"]] = [int(round(cx_px)), int(round(cy_px))]
                r = 0.17 * ppm
                if n["id"] == nearest_id and nearest_d is not None and nearest_d < 0.7:
                    # standing on it: ring the robot rather than draw under it
                    ring(cx_px, cy_px, 0.46 * ppm, EDGE, 0.035 * ppm + 1.5)
                    ring(cx_px, cy_px, 0.46 * ppm, PLACE, 0.035 * ppm)
                else:
                    # the nearest place gets a white keyline: "which place am I at"
                    # is what navigate_to_node needs answered, and the picture can say it
                    # more directly than the table can
                    outline = (255, 255, 255) if n["id"] == nearest_id else EDGE
                    disc(cx_px, cy_px, r + 2.0, outline)
                    disc(cx_px, cy_px, r, PLACE)

            polygon(arrow(0.62), EDGE)
            polygon(arrow(0.50), END)

            # Ids go on LAST, and step aside when the robot is standing on the
            # place: the arrow is drawn over the badge it sits on and would hide a
            # digit painted before it. Size the digit to the BADGE, not to the
            # image: _GLYPH_H rows at this scale is ~1.2x the badge radius, so it
            # fits.
            glyph = max(2, int(round(0.17 * ppm * 1.2 / float(_GLYPH_H))))
            for n in graph["nodes"]:
                cx_px, cy_px = to_px(n["position"])
                if math.hypot(float(n["position"][0]) - float(here[0]),
                              float(n["position"][2]) - float(here[2])) < 0.7:
                    cy_px -= 0.62 * ppm            # just outside the ring
                if _INTERFACE_VARIANT != "text_labels":
                    _digits(img, str(n["id"]), cx_px, cy_px, glyph, EDGE)

            # Where "up the map" is, expressed as a turn — the one number that ties
            # the picture to the body. A turn rather than an absolute heading, so
            # the agent never has to reason in compass degrees and then convert.
            up_word, up_deg = _relative_turn(0.0, -1.0, self._yaw_of(rot))

            # YOUR OWN FRAME, DRAWN ON THE MAP: L / R / B ticks around the arrow.
            #
            # Reading "the living room is toward the bottom-left of the map" and
            # converting it into a turn is arithmetic across two frames, which
            # otherwise tempts the agent to spend steps turning on the spot to
            # check. Three letters beside the arrow make the conversion a glance.
            yaw = self._yaw_of(rot)
            direction_cues = {}
            for lbl, off in (("L", 90.0), ("R", -90.0), ("B", 180.0)):
                a_ = yaw + math.radians(off)
                # +x is east on the map, -z is up, matching to_px and _place_xy
                tx = float(here[0]) - math.sin(a_) * 1.15
                tz = float(here[2]) - math.cos(a_) * 1.15
                tpx, tpy = to_px([tx, here[1], tz])
                gl = max(2, int(round(0.15 * ppm / float(_GLYPH_H))))
                shown = str(int(round(-math.degrees(a_))) % 360) if _INTERFACE_VARIANT == 'absolute_bearings' else lbl
                tw = _text_width(shown, gl)
                img[max(0, int(tpy - _GLYPH_H * gl / 2) - 2):
                    int(tpy + _GLYPH_H * gl / 2) + 2,
                    max(0, int(tpx - tw / 2) - 2):int(tpx + tw / 2) + 2] = EDGE
                shown = str(int(round(-math.degrees(a_))) % 360) if _INTERFACE_VARIANT == 'absolute_bearings' else lbl
                direction_cues[shown] = [float(tpx), float(tpy)]
                if _INTERFACE_VARIANT != "text_labels":
                    _digits(img, shown, tpx, tpy, gl, (255, 255, 255))

            # A bar APPENDED below the picture, never painted over it: the view
            # number so a map from twenty turns ago can be placed in the sequence,
            # the position, the turn to the top of the map, and a scale bar of a
            # known length in metres so a distance can be read off the picture
            # without doing arithmetic on metres_per_pixel.
            left = "VIEW %d   X %.1f Y %.1f   %s" % (
                self._view_index + 1, *self._place_xy(here),
                _map_up_words(up_deg))
            # the scale bar claims the right-hand end, so the stamp is fitted to
            # what is left rather than to the whole width — otherwise the two
            # collide on a narrow crop, which is how a legible number becomes an
            # unreadable overlap
            two_m = int(round(2.0 * ppm))
            room = W - min(two_m, W // 3) - 70
            bs = max(2, min(4, _fit_scale(left, max(80, room))))
            bar_h = _GLYPH_H * bs + 2 * max(2, bs)
            bar = np.zeros((bar_h, W, 3), dtype=np.uint8)
            bar[:, :] = (10, 12, 16)
            if _INTERFACE_VARIANT != 'text_labels':
                _digits(bar, left, 2 * bs + _text_width(left, bs) / 2.0,
                        bar_h / 2.0, bs, (245, 245, 250))
            if two_m + 60 < W - _text_width(left, bs) - 30:
                x1 = W - 8 - _text_width("2M", bs) - 6
                x0 = max(_text_width(left, bs) + 30, x1 - two_m)
                bar[bar_h // 2 - 1:bar_h // 2 + 2, x0:x1] = (245, 245, 250)
                bar[bar_h // 2 - 4:bar_h // 2 + 5, x0:x0 + 2] = (245, 245, 250)
                bar[bar_h // 2 - 4:bar_h // 2 + 5, x1 - 2:x1] = (245, 245, 250)
                _digits(bar, "2M", x1 + 4 + _text_width("2M", bs) / 2.0,
                        bar_h / 2.0, bs, (245, 245, 250))
            if _INTERFACE_VARIANT != 'standard':
                # Every representation has the same footer dimensions, including
                # longer absolute-bearing labels. Text never changes map scale.
                bar = np.zeros((128, W, 3), dtype=np.uint8)
                bar[:] = (10, 12, 16)
                if _INTERFACE_VARIANT != 'text_labels':
                    _text_bars(bar[:96], [left], scales=[2])
                if two_m + 64 < W:
                    bar[110:113, 12:12+two_m] = (245, 245, 250)
                    if _INTERFACE_VARIANT != 'text_labels':
                        _digits(bar, '2M', 40+two_m, 111, 2, (245, 245, 250))
            img = np.vstack([img, bar])
            self._view_index += 1
            H = img.shape[0]

            points = {'here': list(to_px(here)), 'start': list(pts[0])}
            points.update({'cue:' + key: value for key,value in direction_cues.items()})
            points.update({'node:' + str(key): value for key,value in places_px.items()})
            transform = None
            if _INTERFACE_VARIANT != 'standard':
                img, points, transform = rotate_map(img, points, self._yaw_of(rot) if _INTERFACE_VARIANT == 'heading_up_map' else 0)
                H,W = img.shape[:2]
                places_px = {key: points['node:' + str(key)] for key in places_px}
            start = self._agent_path[0]
            from_start = math.hypot(float(here[0]) - float(start[0]),
                                    float(here[2]) - float(start[2]))
            return {
                "png": base64.b64encode(_encode_png(img)).decode("ascii"),
                "meters_per_pixel": round(mpp / k, 4),
                "shape": [H, W],
                # up_word already reads as a phrase ("behind you", "straight
                # ahead", "57 deg left"), so nothing is appended to it.
                "caption": (left + "; numbered nodes and body-frame direction cues are given by their pixel coordinates; scale bar 2M when drawn") if _INTERFACE_VARIANT == 'text_labels' else None,
                "up_on_the_map_is": (None if _INTERFACE_VARIANT == 'absolute_bearings' else
                    'ahead (map rotates with heading)' if _INTERFACE_VARIANT == 'heading_up_map' else
                    "{} ({:+d} deg, positive = left)".format(up_word, up_deg)),
                # Pixel coordinates of the two markers, because the agent reasons
                # in them whether or not they are supplied, and the server can
                # state what it would otherwise estimate from the image. Origin is
                # the image's top-left; combine
                # with metres_per_pixel to convert a pixel gap into metres.
                "you_are_at_px": points["here"],
                "started_at_px": points["start"],
                "presentation_transform": transform,
                "direction_cues_px": {key: points["cue:" + key] for key in direction_cues},
                "absolute_heading_deg": (-math.degrees(self._yaw_of(rot))) % 360,
                # Pixel coordinates are only valid for THIS image: the frame is
                # cropped to the route and grows as the route does. The durable
                # handle is the metric one — xy in metres from the start, which the
                # place table carries and which never shifts.
                "places_px": places_px,
                "you_are_at_xy": self._place_xy(here),
                "straight_line_from_start_m": round(from_start, 2),
                # The CROPPED extent.
                "frame_extent_m": [round(cw * mpp, 2), round(ch * mpp, 2)],
                "seen_area_m2": round(seen_cells * mpp * mpp, 1),
                "seen_cells": seen_cells,
                # Deliberately "of the frame", not "of the building", so nobody
                # reads it as a coverage percentage. Computed over the CROPPED
                # frame, which is the image the agent is actually looking at.
                "unobserved_fraction_of_frame": round(
                    1.0 - float(crop_seen.sum()) / float(max(ch * cw, 1)), 4),
                "walked_m": round(float(sum(
                    float(np.linalg.norm(np.asarray(b)[[0, 2]] - np.asarray(a)[[0, 2]]))
                    for a, b in zip(self._agent_path, self._agent_path[1:]))), 2),
                "revisiting_earlier_position": self._revisiting(),
                "frame": ("observed map rotated to heading-up; metric x/y remain fixed world axes" if _INTERFACE_VARIANT == "heading_up_map" else "cropped to your route and what you have seen around "
                          "it, plus a small margin, so it re-fits as you explore; "
                          "the top of the map is a fixed world direction and "
                          "the map is never rotated to your heading"),
                "frame_origin_m": [round(lo[0], 2), round(lo[1], 2)],
                "legend": "colour = what your cameras saw there; flat slate "
                          "checker = never observed (unknown, not empty) and is "
                          "always darker than anything you have seen; the walked "
                          "path runs blue -> red over time, so red is where you have "
                          "just been; blue dot = where you started; red ARROW = "
                          "where you are now, and it points the way you face; numbered "
                          "amber badges are the places you can return to, and the "
                          "place you are standing on is an amber RING around you. "
                          "Pixel coordinates count from the top-left of the "
                          "image; multiply a pixel gap by metres_per_pixel for "
                          "metres",
            }

    @on_sim_thread
    def trajectory(self):
        """Where the agent actually went, versus where it should have gone.

        Driver-only, exactly like evaluate(): this is ground truth about the
        route and the goal, so it must never be reachable from the agent's tool
        surface. Coordinates are habitat world frame (y is up), so a top-down
        plot uses the x/z components.
        """
        with self._lock:
            if self._ep is None:
                return {"error": "no live episode"}
            ep = self._ep
            goals = [g.get("position") for g in (ep.get("goals") or []) if g.get("position")]
            return {
                "episode_id": str(ep.get("episode_id", "")),
                "scene_id": str(ep.get("scene_id", "")),
                "agent_path": [list(map(float, p)) for p in self._agent_path],
                # GT locations used by nDTW; falls back to the dataset's coarse
                # reference_path when this episode has no GT entry.
                "reference_path": [
                    list(map(float, p))
                    for p in (
                        self._gt.get(str(ep.get("episode_id", "")))
                        or ep.get("reference_path")
                        or []
                    )
                ],
                "goals": [list(map(float, g)) for g in goals],
                "start_position": list(map(float, ep.get("start_position") or [])),
                "success_distance": float(self.config["success_distance"]),
                "geodesic_distance": float(
                    (ep.get("info") or {}).get("geodesic_distance", 0.0)
                ),
                "stop_called": bool(self._stop_called),
            }

    # ── introspection ──

    def health(self):
        with self._lock:
            return {
                **self._dataset_info,
                "name": "r2rce",
                "interface_variant": _INTERFACE_VARIANT,
                "gpu_device_id": int(os.environ.get("NAVGPT_GPU_DEVICE_ID", "0")),
                "status": "ok",
                "max_steps": self.config["max_steps"],
                "blind": self.blind,
                "split": self._split,
                "episode_count": len(self._episodes),
                "episode_index": self._ep_index,
                "gt_paths": len(self._gt),
                "gt_covered": self._gt_coverage,
                "caliber": {
                    "turn_angle_deg": self.config["turn_angle_deg"],
                    "step_size_m": self.config["step_size_m"],
                    "allow_sliding": self.config["allow_sliding"],
                    "success_distance": self.config["success_distance"],
                    "max_steps": self.config["max_steps"],
                },
                "episodes_file": self.episodes_file,
                "episodes_sha256": (__import__('hashlib').sha256(Path(self.episodes_file).read_bytes()).hexdigest()
                                    if self.episodes_file else None),
                "gt_sha256": (__import__('hashlib').sha256(Path(self.gt_file).read_bytes()).hexdigest()
                              if self.gt_file else None),
                "config": dict(self.config),
            }
