"""Run the NavGPT Environment service.

    python -m navgpt.environment --config configs/experiments/vln/navgpt3_astra_r2r.yaml

Run it with the Habitat environment's Python (habitat-sim 0.1.7). The experiment
config selects the dataset, episode set and rendering settings; ``--port`` and
``--gpu`` place one service of several for sharded runs.
"""

import argparse
import os
import sys


def _ensure_runtime_libraries():
    """habitat-sim needs its environment's libraries ahead of the system's and an
    explicit EGL vendor path. Both are read at process start, so set them and
    re-execute once."""
    if not sys.platform.startswith("linux"):
        return
    lib = os.path.join(sys.prefix, "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib in current.split(":"):
        return
    env = dict(os.environ, LD_LIBRARY_PATH=lib + (":" + current if current else ""))
    env.setdefault("__EGL_VENDOR_LIBRARY_DIRS",
                   "/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d")
    env.setdefault("MAGNUM_LOG", "quiet")
    env.setdefault("HABITAT_SIM_LOG", "quiet")
    os.execve(sys.executable, [sys.executable, "-m", "navgpt.environment", *sys.argv[1:]], env)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m navgpt.environment",
        description="NavGPT Environment service for R2R-CE / RxR-CE (habitat-sim 0.1.7)")
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--paths", help="machine paths file (default: configs/paths.yaml)")
    parser.add_argument("--port", type=int, help="override the port in environment.url")
    parser.add_argument("--gpu", type=int, default=0, help="rendering GPU (default 0)")
    parser.add_argument("--blind", action="store_true",
                        help="no rendering: real dynamics and metrics, synthetic pixels; "
                             "for plumbing checks only")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    from ..settings import load

    options = load(args.config, args.paths).environment_options()
    _ensure_runtime_libraries()
    if args.port:
        options["port"] = args.port
    # The renderer reads these when navgpt.environment.r2r is imported.
    os.environ["NAVGPT_INTERFACE_VARIANT"] = options.pop("interface")
    os.environ["NAVGPT_GPU_DEVICE_ID"] = str(args.gpu)

    from .api import serve

    serve(blind=args.blind, verbose=args.verbose, **options)


if __name__ == "__main__":
    main()
