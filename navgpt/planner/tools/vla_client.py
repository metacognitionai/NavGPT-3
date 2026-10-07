"""HTTP client for the NavGPT VLA service.

The protocol is ported from VLNCE-EVAL rather than re-derived,
because its wire format has several asymmetries that are invisible until results
are quietly bad:

- All nine ``/act`` keys are REQUIRED (the server indexes them with ``data['x']``,
  so a missing key is a 500).
- ``global_compass`` goes out as **WXYZ**; the ``action_args["rotation"]`` that
  comes back is **XYZW**. Same quaternion, opposite convention, one call apart.
- Wire view order is **front, right, back, left** — NOT the front/left/back/right
  of the sensor indices. The model weights its cameras unequally (front heaviest,
  back lightest), so swapping left and right does not error, it just quietly
  gives the back-view token budget to a side view.
- ``navigation`` is ``8 x [x, y, theta]``: **agent-relative, cumulative, on a
  0.125 m grid**, ``+x`` forward, ``+y`` LEFT, ``+theta`` counter-clockwise,
  metres and radians. Cumulative means waypoint k is the displacement from the
  pose you just sent, not from waypoint k-1.
- The server **never emits STOP** — it always returns a TELEPORT. Deciding to
  stop is entirely the caller's job.
- It is **stateful**: every ``/act`` appends to a per-camera frame history that
  is the model's temporal context, and ``/reset`` is the only thing that clears
  it. Skip the reset and the previous episode's frames leak into this one.
- It is **not concurrency-safe** (Flask threaded, unsynchronised shared state),
  so calls must be serialised per server.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import requests

log = logging.getLogger("navgpt.planner.vla")

# Stop heuristic thresholds, as in VLNCE-EVAL's episode runner.
# The model under-predicts stopping, so this is what actually ends its episodes
# in the reference implementation.
LAST_WP_STOP_MAGNITUDE = 0.25
LAST_WP_STOP_THETA = math.pi / 6

# Habitat local frame: +X right, +Y up, -Z forward.
_FORWARD_AXIS = 2


def _qmul(a, b):
    """Quaternion product, both WXYZ."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _qrot(q, v):
    """Rotate vector v by WXYZ quaternion q, via v + 2w(u x v) + 2(u x (u x v))."""
    w, x, y, z = q
    ux, uy, uz = x, y, z
    vx, vy, vz = v
    # u x v
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    # u x (u x v)
    dx, dy, dz = uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx
    return (vx + 2.0 * w * cx + 2.0 * dx,
            vy + 2.0 * w * cy + 2.0 * dy,
            vz + 2.0 * w * cz + 2.0 * dz)


def pose_unchanged(before, after) -> bool:
    """VLNCE-EVAL's stuck test between two consecutive VLA steps: some position
    coordinate and the whole heading quaternion are exactly unchanged. Height
    rarely changes, so in practice this means the step kept the heading, as a
    blocked forward step does."""
    if not before or not after:
        return False
    return (any(a == b for a, b in zip(before["position"], after["position"]))
            and list(before["rotation_wxyz"]) == list(after["rotation_wxyz"]))


def compose_relative(position, rotation_wxyz, rel_x, rel_y, rel_theta):
    """Compose an agent-relative (x forward, y left, theta CCW) offset onto a
    world pose, exactly as NavGPT VLA's own server does.

    As in the VLNCE-EVAL server:

        dx, dy   = x, -y
        dtheta   = -theta
        local    = [dy, 0, -dx]              # habitat local frame
        p_new    = R(q) . local + p
        q_new    = q * quat(cos(theta/2), 0, sin(theta/2), 0)

    Deliberately pure-Python: the MCP tool server must stay importable in an environment
    with nothing but the stdlib plus requests/Pillow, so no numpy-quaternion here.

    Returns ``(position_xyz, rotation_xyzw)`` — the OUTPUT quaternion order is
    XYZW, matching what the server returns and what ``teleport`` expects, while
    the input is WXYZ.
    """
    q = tuple(float(v) for v in rotation_wxyz)
    dx, dy = float(rel_x), -float(rel_y)
    local = (dy, 0.0, -dx)
    rx, ry, rz = _qrot(q, local)
    p_new = [float(position[0]) + rx, float(position[1]) + ry, float(position[2]) + rz]

    half = float(rel_theta) / 2.0
    q_new = _qmul(q, (math.cos(half), 0.0, math.sin(half), 0.0))
    return p_new, [q_new[1], q_new[2], q_new[3], q_new[0]]


def suggests_arrival(navigation) -> bool:
    """NavGPT VLA's implicit arrival signal.

    It never says STOP, but when its 1 m horizon collapses — the last waypoint
    is nearly where it already stands, with almost no heading change — it has run
    out of anywhere to go. VLNCE-EVAL requires TWO consecutive hits before
    stopping; that consecutiveness is tracked by the caller, since this is a
    single-shot read.
    """
    if not navigation:
        return False
    last = navigation[-1]
    if len(last) < 3:
        return False
    magnitude = math.sqrt(float(last[0]) ** 2 + float(last[1]) ** 2)
    return magnitude < LAST_WP_STOP_MAGNITUDE and abs(float(last[2])) < LAST_WP_STOP_THETA


class NavGPTVLAClient:
    """One NavGPT VLA service, one episode at a time.

    Calls are serialised behind a lock because the server keeps unsynchronised
    per-camera frame history; two overlapping ``/act`` calls interleave their
    appends and corrupt the model's temporal context.
    """

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._lock = threading.Lock()
        # consecutive arrival hits, per the two-hit rule
        self._arrival_streak = 0

    # ── lifecycle ──

    def info(self) -> dict[str, Any]:
        r = requests.get(self.base_url + "/info", timeout=30)
        r.raise_for_status()
        return r.json()

    def audit_state(self):
        with self._lock:
            r = requests.get(self.base_url + "/state", timeout=self.timeout)
            r.raise_for_status()
            return r.json()

    def observe_history(self, views):
        with self._lock:
            r = requests.post(self.base_url + '/observe',
                              json={'rgb_' + k: views[k] for k in ('front', 'right', 'back', 'left')},
                              timeout=self.timeout)
            r.raise_for_status()
            return r.json()

    def reset(self, success: bool | None = None) -> dict[str, Any]:
        """MANDATORY between episodes — this is what clears the frame history."""
        with self._lock:
            self._arrival_streak = 0
            r = requests.post(self.base_url + "/reset",
                              json={"success": success}, timeout=self.timeout)
            r.raise_for_status()
            return r.json()

    # ── inference ──

    def act(self, views: dict[str, str], instruction: str, episode_id: str,
            position, rotation_wxyz, is_stuck: bool = False) -> dict[str, Any]:
        """One VLA step.

        ``views`` maps ``front``/``right``/``back``/``left`` to base64 images —
        the keys are named so the wire order cannot be got wrong by accident.

        Returns a dict with ``navigation`` (8x3), the target ``position`` /
        ``rotation_xyzw`` composed from waypoint 1 (the 0.25 m point, matching
        the reference server's choice), and ``arrival`` /``arrival_streak``.
        """
        missing = [k for k in ("front", "right", "back", "left") if not views.get(k)]
        if missing:
            raise ValueError("views missing {} — all four are required".format(missing))

        payload = {
            # wire order F, R, B, L
            "rgb_front": views["front"],
            "rgb_right": views["right"],
            "rgb_back": views["back"],
            "rgb_left": views["left"],
            "text": instruction,
            "episode_id": str(episode_id),
            "global_gps": [float(v) for v in position],
            # WXYZ on the way in
            "global_compass": [float(v) for v in rotation_wxyz],
            "is_stuck": bool(is_stuck),
        }

        with self._lock:
            r = requests.post(self.base_url + "/act", json=payload, timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
            action, navigation = _parse_act(body)

            arrival = suggests_arrival(navigation)
            self._arrival_streak = self._arrival_streak + 1 if arrival else 0
            streak = self._arrival_streak

        target_pos, target_rot = None, None
        args = (action or {}).get("action_args") or {}
        # Heading change for the step in the server's sign (negative turns
        # left); teleport's blocked-step fallback turns by it.
        theta = args.get("theta")
        if theta is None and navigation and len(navigation) > 1:
            theta = -float(navigation[1][2])
        if args.get("position") is not None and args.get("rotation") is not None:
            # The server already composed waypoint 1 for us; XYZW on the way out.
            target_pos = [float(v) for v in args["position"]]
            target_rot = [float(v) for v in args["rotation"]]
        elif navigation and len(navigation) > 1:
            # Defensive: compose it ourselves from waypoint 1 the same way.
            target_pos, target_rot = compose_relative(
                position, rotation_wxyz, navigation[1][0], navigation[1][1], navigation[1][2])

        return {
            "navigation": navigation,
            "position": target_pos,
            "rotation_xyzw": target_rot,
            "theta": float(theta) if theta is not None else 0.0,
            "arrival": arrival,
            "arrival_streak": streak,
            "should_stop": streak >= 2,
            "raw_action": action,
        }


def _parse_act(body) -> tuple[dict, list]:
    """Normalise the /act reply.

    The server returns a 2-element ARRAY ``[action, navigation]``. Tolerate the
    object form and degenerate shapes too, the way VLNCE-EVAL's own client
    does.
    """
    if isinstance(body, list):
        action = body[0] if body else {}
        navigation = body[1] if len(body) > 1 else []
    elif isinstance(body, dict):
        if "error" in body:
            raise RuntimeError("NavGPT VLA error: {}".format(body["error"]))
        action = body.get("action") or {}
        navigation = body.get("navigation") or []
    else:
        raise RuntimeError("unexpected /act reply type: {}".format(type(body).__name__))
    return (action if isinstance(action, dict) else {},
            navigation if isinstance(navigation, list) else [])
