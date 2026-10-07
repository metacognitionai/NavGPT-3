"""HTTP API for NavGPT Environment — stdlib only.

Serves the protocol that NavGPT Planner and its MCP tool server use:

    GET  /health                        -> {"name": "r2rce", ...}
    POST /call/{fn}                     {"inputs": {...}} -> {"outputs": {...}}
    POST /env-panel/field/{name}        {"value": v}
    POST /env-panel/action/{name}       {"params": {}}

Verbs follow the ``<prefix>__<verb>`` convention used by the MCP tool server
from ``NAVGPT_VERB_PREFIX``:

    r2rce__reset               ensure-live; episode metadata
    r2rce__observe_egocentric  rgb (base64 PNG) / pose / instruction_text
    r2rce__step_discrete       0=STOP 1=FWD 2=LEFT 3=RIGHT

    r2rce__observe_pano        4 RGB views + 360-bin range scan + pose (0 steps)
    r2rce__agent_state         pose only (WXYZ quaternion)
    r2rce__step_hightolow      rotate then walk; reports how far it got
    r2rce__teleport            place at a world pose, with a discrete fallback

    r2rce__observed_map        top-down colour map of what the cameras saw
    r2rce__places              the place graph derived from the walked path
    r2rce__retrace_to          walk back to a place along ground already covered
    r2rce__retrace_path        walk back along the recorded path to a point passed
    r2rce__evaluate            NE/SR/SPL/OSR/nDTW/TL   [planner only]
    r2rce__trajectory          agent path vs reference  [planner only]

The middle group is tool-side: the MCP tool server may use it, because everything in it
is knowable from a depth camera plus odometry. The last group is planner-only —
it carries ground truth and must never back an agent tool.

Threaded service, single environment behind an RLock: habitat_sim is not reentrant,
and the planner runner and MCP tool server are independent clients.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .r2r import R2RCEEnv

log = logging.getLogger("navgpt.environment")

ENV = None  # set by serve()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ── plumbing ──

    def log_message(self, fmt, *args):  # quieter than the stdlib default
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _body(self):
        length = int(self.headers.get("content-length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return {}

    def _send(self, payload, status=200):
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    # ── routes ──

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            # The verb list rides along with /health, the endpoint preflight already
            # fetches, so preflight can catch a stale server with an older surface.
            out = dict(ENV.health())
            out["functions"] = sorted(VERBS)
            self._send(out)
            return
        if self.path.rstrip("/") == "/manifest":
            self._send({
                "name": "r2rce",
                "functions": sorted(VERBS),
                "env_panel": {"fields": ["split", "episode_index"],
                              "actions": ["play", "reset"]},
            })
            return
        self._send({"error": "not found: " + self.path}, 404)

    def do_POST(self):
        path = self.path.rstrip("/")
        body = self._body()
        try:
            if path.startswith("/call/"):
                fn = path[len("/call/"):]
                handler = VERBS.get(fn)
                if handler is None:
                    self._send({"error": "unknown function: " + fn}, 404)
                    return
                inputs = dict(body.get("inputs") or {})
                # Ablation runs also collect the panoramas passed during charged
                # motion, so the VLA's visual history matches what the robot saw.
                capture = inputs.pop('capture_history', False)
                if capture and fn.rsplit('__', 1)[-1] in ('step_hightolow', 'retrace_to', 'retrace_path'):
                    outputs = ENV.call_with_motion_history(handler, inputs)
                else:
                    outputs = handler(ENV, inputs)
                self._send({"outputs": outputs})
                return

            if path.startswith("/env-panel/field/"):
                name = path[len("/env-panel/field/"):]
                self._send(panel_field(ENV, name, body.get("value")))
                return

            if path.startswith("/env-panel/action/"):
                name = path[len("/env-panel/action/"):]
                self._send(panel_action(ENV, name, body.get("params") or {}))
                return
        except Exception as exc:  # noqa: BLE001 - report, never kill the server
            log.exception("request failed: %s", path)
            self._send({"error": "{}: {}".format(type(exc).__name__, exc)}, 500)
            return

        self._send({"error": "not found: " + path}, 404)


# ── env panel ──


def panel_field(env, name, value):
    if name == "split":
        return env.set_split(str(value))
    if name == "episode_index":
        # Stage the index; `play` seats it: a field change stages and the
        # action commits.
        env.stage_episode_index(int(value))
        return {"staged": {"episode_index": int(value)}}
    return {"error": "unknown field: " + name}


def panel_action(env, name, params):
    if name in ("play", "reset"):
        index = env.staged_episode_index()
        if index is None:
            return {"error": "no episode_index staged — set the field first"}
        result = env.set_episode_by_index(int(index))
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
    return {"error": "unknown action: " + name}


# ── verbs ──


def _reset(env, inputs):
    meta = env.ensure_live()
    if "error" in meta:
        raise RuntimeError(meta["error"])
    return {
        "instruction": meta["instruction"],
        "episode_id": meta["episode_id"],
        "scene_id": meta["scene_id"],
        "geodesic_distance": meta["geodesic_distance"],
        "dataset": meta.get("dataset"), "language": meta.get("language"),
        "instruction_id": meta.get("instruction_id"), "annotation_role": meta.get("annotation_role"),
    }


def _observe(env, inputs):
    # stamp defaults OFF: `planner_basic`'s pixels are part of a frozen baseline
    out = env.observe(bool(inputs.get("stamp")))
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _step(env, inputs):
    if "action" not in inputs:
        raise ValueError("step_discrete requires an 'action' input (0-3)")
    out = env.step(int(inputs["action"]))
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _evaluate(env, inputs):
    out = env.evaluate()
    if "error" in out:
        raise RuntimeError(out["error"])
    return {"metrics": out}


def _places(env, inputs):
    out = env.places()
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _retrace_to(env, inputs):
    out = env.retrace_to(inputs.get("place"), inputs.get("xy"))
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _retrace_path(env, inputs):
    if "position" not in inputs:
        raise ValueError("retrace_path requires 'position' (xyz)")
    out = env.retrace_path(inputs["position"], inputs.get("rotation_wxyz"))
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _observed_map(env, inputs):
    out = env.observed_map()
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _trajectory(env, inputs):
    out = env.trajectory()
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _observe_pano(env, inputs):
    out = env.observe_pano()
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _agent_state(env, inputs):
    out = env.agent_state()
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _step_hightolow(env, inputs):
    out = env.step_hightolow(
        float(inputs.get("angle_rad") or 0.0),
        float(inputs.get("distance_m") or 0.0),
        bool(inputs.get("route")),
    )
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def _teleport(env, inputs):
    if "position" not in inputs or "rotation" not in inputs:
        raise ValueError("teleport requires 'position' (xyz) and 'rotation' (xyzw)")
    out = env.teleport(inputs["position"], inputs["rotation"],
                       theta=float(inputs.get("theta") or 0.0),
                       is_stuck=bool(inputs.get("is_stuck", False)))
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


VERBS = {
    "r2rce__reset": _reset,
    "r2rce__observe_egocentric": _observe,
    "r2rce__step_discrete": _step,
    # Tool-side verbs. Reachable by the MCP tool server (so they can back agent tools),
    # but they expose only what a robot with a depth camera and odometry would
    # know — never the goal, the reference path, or the score.
    "r2rce__observe_pano": _observe_pano,
    "r2rce__agent_state": _agent_state,
    "r2rce__step_hightolow": _step_hightolow,
    "r2rce__teleport": _teleport,
    # The place graph. Tool-side, and for the same reason observe_pano is: it is
    # derived from the walked path alone — a place is somewhere the robot stood, an
    # edge is a stretch it walked — so it tells the agent nothing a robot with
    # wheel encoders would not already know. The names on those places are the
    # agent's own; nothing here reads the scene's semantics, which are not even
    # loaded (see R2RCEEnv._open_scene).
    "r2rce__places": _places,
    "r2rce__retrace_to": _retrace_to,
    "r2rce__retrace_path": _retrace_path,
    # Tool-side too: observed_map paints only cells the agent's own RGB-D actually
    # measured, so it discloses nothing a robot with a depth camera would not know.
    "r2rce__observed_map": _observed_map,
    # Planner-only verbs. The MCP tool server exposes none of these, so the agent cannot
    # reach its own score or the ground-truth route.
    "r2rce__evaluate": _evaluate,
    "r2rce__trajectory": _trajectory,
}


# ── entry point ──


def serve(*, dataset, split, data_root, scene_root, episodes_file, gt_split, gt_file,
          languages, roles, host, port, config, blind=False, verbose=False):
    """Load the episode set and serve it until interrupted. Arguments come from
    ``Settings.environment_options()``; see ``navgpt/environment/__main__.py``."""
    global ENV
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ENV = R2RCEEnv(
        dataset=dataset, languages=languages, roles=roles,
        data_root=data_root,
        scene_root=scene_root,
        split=split,
        episodes_file=episodes_file,
        gt_file=gt_file,
        gt_split=gt_split,
        blind=blind,
        config=config,
    )
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    log.info("serving r2rce on %s:%d — dataset=%s split=%s episodes=%d blind=%s",
             host, port, dataset, split, ENV.episode_count, blind)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        ENV.close()
