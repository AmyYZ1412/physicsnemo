#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed five-backend GEMM configuration reported in the paper."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
REPO_ROOT = REPRO_ROOT.parent
PROFILER = HERE / "profile_gemm_backends.py"
sys.path.insert(0, str(REPRO_ROOT))

from config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path, default=REPRO_ROOT / "results/gemm"
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()["gemm"]
    command = [
        sys.executable,
        str(PROFILER),
        "--device",
        args.device,
        "--backends",
        ",".join(cfg["backends"]),
        "--num-moduli",
        str(cfg["num_moduli"]),
        "--warmup",
        str(cfg["warmup"]),
        "--repeat",
        str(cfg["repeat"]),
        "--seed",
        str(cfg["seed"]),
        "--scale",
        str(cfg["scale"]),
        "--rtol",
        str(cfg["rtol"]),
        "--atol",
        str(cfg["atol"]),
        "--output-json",
        str(args.output_dir / "paper_gemm.json"),
        "--output-csv",
        str(args.output_dir / "paper_gemm.csv"),
    ]
    for shape in cfg["shapes"]:
        command.extend(("--shape", shape))
    if cfg["fastmode"]:
        command.append("--fastmode")
    if args.strict:
        command.append("--strict")

    env = os.environ.copy()
    env["OZAKI_DIGIT_NUM_STREAMS"] = str(cfg["digit_streams"])
    env.setdefault(
        "OZAKI_LINEAR_BUILD_DIR", str(REPO_ROOT / ".cache/ozaki_linear_ext")
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
