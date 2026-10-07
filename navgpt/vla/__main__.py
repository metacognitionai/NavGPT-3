"""Run the NavGPT VLA service.

    python -m navgpt.vla --config configs/experiments/vln/navgpt3_astra_r2r.yaml

Run it with the VLA environment's Python. The experiment's ``vla:`` section picks
the checkpoint (a key from configs/paths.yaml) and inference settings;
``--port`` and ``--gpu`` place one service of several for sharded runs.
"""

import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m navgpt.vla",
                                     description="NavGPT VLA inference service")
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--paths", help="machine paths file (default: configs/paths.yaml)")
    parser.add_argument("--port", type=int, help="override the port in vla.url")
    parser.add_argument("--gpu", type=int, default=0, help="GPU for the model (default 0)")
    args = parser.parse_args(argv)

    from ..settings import load

    options = load(args.config, args.paths).vla_options()
    if args.port:
        options["port"] = args.port
    # Must be set before torch initialises CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.makedirs(options["result_path"], exist_ok=True)

    from .api import main as serve

    serve(options)


if __name__ == "__main__":
    main()
