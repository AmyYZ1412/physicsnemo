#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Shared runner for the fixed paper PINN stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
TRAINER = HERE / "train_convection.py"
sys.path.insert(0, str(REPRO_ROOT))

from config import VARIANTS, load_config  # noqa: E402


def run_stage(stage: str) -> int:
    parser = argparse.ArgumentParser(
        description=f"Run the fixed paper PINN {stage} stage"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path, default=REPRO_ROOT / "results/pinn"
    )
    parser.add_argument(
        "--variants",
        help="Comma-separated subset of the paper variants for this stage",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config()["pinn"]
    variants = list(config["variants"][stage])
    if args.variants:
        requested = [item.strip() for item in args.variants.split(",") if item.strip()]
        invalid = set(requested) - set(variants)
        if invalid:
            raise SystemExit(
                f"variant(s) not in the {stage} paper set: {sorted(invalid)}"
            )
        variants = requested
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variant(s): {sorted(unknown)}")

    target_steps = int(config["stages"][stage]["steps"])
    statuses = []
    for variant in variants:
        variant_dir = args.output_dir / variant
        # A completed convergence run is immutable. A later probe must use a
        # separate directory so its fresh 150-step checkpoint cannot replace
        # the 2000-step result or the checkpoint used for resumption.
        if stage == "probe" and (variant_dir / "checkpoint_2000.pt").exists():
            variant_dir = variant_dir / "probe_150"
        command = [
            sys.executable,
            str(TRAINER),
            "--variant",
            variant,
            "--target-steps",
            str(target_steps),
            "--device",
            args.device,
            "--output-dir",
            str(variant_dir),
        ]
        if stage == "convergence":
            command.append("--resume")
        print(" ".join(command), flush=True)
        if args.dry_run:
            status = 0
        else:
            status = subprocess.run(command, cwd=HERE, check=False).returncode
        statuses.append(
            {
                "variant": variant,
                "phase": stage,
                "target_steps": target_steps,
                "status": status,
            }
        )
        if status and args.fail_fast:
            break

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        status_path = args.output_dir / f"{stage}_status.json"
        status_path.write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    return 1 if any(row["status"] for row in statuses) else 0
