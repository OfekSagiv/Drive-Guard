#!/usr/bin/env python3
"""
Distillation inference entrypoint.

This keeps folder structure aligned with vit_transformer, where `infer.py`
is the main inference script. Internally it delegates to `predict.py`.
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parent / "checkpoints" / "best_swin3d_driveguard.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distillation inference (wrapper around predict.py).")
    parser.add_argument("--data_root", type=str, default="ds_driveguard_16frames_roi.nosync")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    predict_script = Path(__file__).resolve().parent / "predict.py"
    cmd = [
        sys.executable,
        str(predict_script),
        "--data_root",
        args.data_root,
        "--checkpoint",
        args.checkpoint,
        "--image_size",
        str(args.image_size),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    print("[INFO] Launching predict.py through infer.py wrapper")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()