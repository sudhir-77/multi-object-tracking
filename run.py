"""Reproduce training, evaluation, and the C++ benchmark.

Usage:
    python run.py
    python run.py --quick
    python run.py --device cpu --out results
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tracker.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MOT17 tracking-by-detection pipeline")
    parser.add_argument("--quick", action="store_true", help="small subset, for tests and CI")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-coverage", action="store_true")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    run_pipeline(quick=args.quick, out_dir=out, device=args.device, skip_coverage=args.skip_coverage)


if __name__ == "__main__":
    main()
