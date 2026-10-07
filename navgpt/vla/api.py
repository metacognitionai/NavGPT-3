"""NavGPT VLA adaptive multi-view inference service.

Uses NavGPTVLA for inference with multi-resolution image caching and adaptive
token allocation. Each /act call runs a fresh inference and returns a TELEPORT
action toward the second predicted waypoint, together with all eight waypoints.

This module is derived from VLNCE-EVAL by EPIC Lab, released under the MIT
license (see navgpt/vla/LICENSE).

Start it with ``python -m navgpt.vla --config <experiment.yaml>``.
"""
from __future__ import annotations

import sys
import os
from types import SimpleNamespace
import atexit
import logging
import signal

# ---------------------------------------------------------------------------
# NavGPT VLA service — inference and TELEPORT conversion
# ---------------------------------------------------------------------------

class NavGPTVLAService:
    """Wrap NavGPTVLA and convert its next waypoint into a TELEPORT action.

    The service deliberately performs one inference per ``/act`` request. It has
    no action queue; the only retained state is visual history inside the VLA."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self) -> None:
        self.agent.reset()

    def act(self, rgb_front, rgb_right, rgb_back, rgb_left, text,
            global_gps, global_compass, is_stuck):
        state = {"position": global_gps, "orientation": global_compass}

        # Run inference every step (matching NaVid behavior — no action buffering)
        images = [rgb_front, rgb_right, rgb_back, rgb_left]
        navigation = self.agent.act(images, text)

        # Use waypoint[1] as the action (matching NaVid)
        action = update_agent_state(
            current_global_state=state,
            relative_change={"x": navigation[1][0], "y": navigation[1][1], "theta": navigation[1][2]},
        )
        return add_is_stuck(action, is_stuck), navigation


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

# Service options; an experiment's `vla:` section overrides them (navgpt.settings).
DEFAULTS = {
    "model_path": None,
    "result_path": "results",         # holds the service log
    "host": "0.0.0.0",
    "port": 8000,
    "robot_type": "habitat_nav",
    "frame_cache": "precompute",      # or "on_demand"
    # adaptive visual-token allocation
    "max_nav_vis_tokens": 3072,
    "temporal_decay": 2.0,
    "camera_weights": None,
    "min_tokens_per_image": 4,
    "max_tokens_per_image": 196,
    "max_sample_size": 16,
    "gpu_memory_fraction": 0.0,       # >0 caps this process's share of the GPU
}


def main(options: dict):
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError("unknown VLA service option(s): {}".format(sorted(unknown)))
    args = SimpleNamespace(**{**DEFAULTS, **options})
    if not args.model_path:
        raise ValueError("model_path is required")

    # The CUDA/image service stack is imported only once options are valid.
    global np, quaternion, Flask, request, jsonify
    global add_is_stuck, update_agent_state, decode_base64_to_image
    import numpy as np
    import quaternion
    from flask import Flask, request, jsonify
    from .navigation import add_is_stuck, update_agent_state
    from .images import decode_base64_to_image

    app = Flask(__name__)

    # --- Logging ---
    os.makedirs(args.result_path, exist_ok=True)
    log_path = os.path.join(args.result_path, f"server_{args.port}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stderr)],
    )
    logger = logging.getLogger(__name__)
    logger.info("Server starting on port %d, log: %s", args.port, log_path)

    def _excepthook(exc_type, exc_value, exc_tb):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGABRT):
        signal.signal(sig, lambda s, f: (
            logger.error("Signal %s, shutting down", signal.Signals(s).name),
            sys.exit(1),
        ))
    atexit.register(lambda: logger.info("Server exited (port %d)", args.port))

    # Import lazily so `python -m navgpt.vla --help` remains a cheap check.
    from .agent import NavGPTVLA

    # --- Cap GPU memory so 2 models can share one GPU ---
    gpu_mem_frac = float(args.gpu_memory_fraction or 0)
    if gpu_mem_frac > 0:
        import torch
        torch.cuda.set_per_process_memory_fraction(gpu_mem_frac)
        logger.info("Set GPU memory fraction to %.2f", gpu_mem_frac)

    # --- Build agent + server ---
    try:
        agent = NavGPTVLA(
            model_path=args.model_path,
            view_num=4,
            camera_weights=args.camera_weights,
            max_nav_vis_tokens=args.max_nav_vis_tokens,
            temporal_decay=args.temporal_decay,
            min_tokens_per_image=args.min_tokens_per_image,
            max_tokens_per_image=args.max_tokens_per_image,
            max_sample_size=args.max_sample_size,
            frame_cache=args.frame_cache,
            robot_type=args.robot_type,
        )
        server = NavGPTVLAService(agent=agent)
    except Exception:
        logger.critical("Fatal error during initialisation", exc_info=True)
        sys.exit(1)

    # All history mutations share one lock, including reset and passive observations.
    import threading
    import functools
    history_lock = threading.RLock()
    def serialized(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with history_lock:
                return fn(*a, **kw)
        return wrapper

    # --- Routes ---
    @app.route('/info', methods=['GET'])
    def info():
        return jsonify({
            "model_name": "navgpt-vla",
            "history_observe": True,
            "seeded_reset": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_version": os.path.basename(args.model_path),
            "max_nav_vis_tokens": args.max_nav_vis_tokens,
            "frame_cache": args.frame_cache,
        })

    @app.route('/reset', methods=['POST'])
    @serialized
    def reset():
        try:
            data = request.json
            server.reset()
            server.agent.audit_history = bool(data and data.get('seed') is not None)
            if server.agent.audit_history:
                import random
                import torch
                seed = int(data['seed'])
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            return jsonify({"status": "reset complete"})
        except Exception:
            logger.exception("Error in /reset")
            return jsonify({"error": "internal server error"}), 500

    @app.route('/state', methods=['GET'])
    @serialized
    def audit_state():
        import hashlib
        import pickle
        import random
        import torch
        rng = hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state())))
        rng.update(torch.get_rng_state().cpu().numpy().tobytes())
        # Only the current GPU is owned by this VLA process.
        if torch.cuda.is_available():
            rng.update(torch.cuda.get_rng_state().cpu().numpy().tobytes())
        return jsonify({'history_sha256': server.agent.history_digest.hexdigest(),
                        'history_frames': len(server.agent.rgb_lists[0]),
                        'cache_steps': sorted(server.agent.multi_res_cache),
                        'rng_sha256': rng.hexdigest()})

    @app.route('/observe', methods=['POST'])
    @serialized
    def observe_history():
        data = request.json
        images = [decode_base64_to_image(data['rgb_' + name]) for name in ('front', 'right', 'back', 'left')]
        server.agent.observe_history(images)
        return jsonify({'status': 'observed', 'frames': len(server.agent.rgb_lists[0])})

    @app.route('/act', methods=['POST'])
    @serialized
    def act():
        try:
            data = request.json
            rgb_front = decode_base64_to_image(data['rgb_front'])
            rgb_right = decode_base64_to_image(data['rgb_right'])
            rgb_back = decode_base64_to_image(data['rgb_back'])
            rgb_left = decode_base64_to_image(data['rgb_left'])
            text = data['text']
            global_gps = data['global_gps']
            global_compass = quaternion.from_float_array(np.array(data['global_compass']))
            is_stuck = data['is_stuck']

            action, navigation = server.act(
                rgb_front, rgb_right, rgb_back, rgb_left, text,
                global_gps, global_compass, is_stuck)
            return jsonify([action, navigation])
        except Exception:
            logger.exception("Error in /act")
            return jsonify({"error": "internal server error"}), 500

    logger.info("Server ready, listening on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port)

