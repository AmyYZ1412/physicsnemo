# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "paper_config.toml"


@dataclass(frozen=True)
class BackendVariant:
    name: str
    backend: str
    dtype: str
    digit_streams: int


VARIANTS = {
    "native_fp32": BackendVariant("native_fp32", "native", "fp32", 1),
    "native_fp64": BackendVariant("native_fp64", "native", "fp64", 1),
    "int8": BackendVariant("int8", "int8", "fp64", 1),
    "fp8": BackendVariant("fp8", "fp8", "fp64", 1),
    "mxfp8_serial": BackendVariant("mxfp8_serial", "mxfp8", "fp64", 1),
    "mxfp8_streams4": BackendVariant("mxfp8_streams4", "mxfp8", "fp64", 4),
    "nvfp4_serial": BackendVariant("nvfp4_serial", "nvfp4", "fp64", 1),
    "nvfp4_streams4": BackendVariant("nvfp4_streams4", "nvfp4", "fp64", 4),
}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    gemm = config["gemm"]
    if gemm["backends"] != ["fp64", "int8", "fp8", "mxfp8", "nvfp4"]:
        raise ValueError("paper GEMM backends must remain the reported five-backend set")
    if len(gemm["shapes"]) != 4:
        raise ValueError("paper GEMM configuration must contain four PINN shapes")

    pinn = config["pinn"]
    if pinn["equation"] != "convection" or pinn["optimizer"] != "lbfgs":
        raise ValueError("paper PINN configuration must use Convection and L-BFGS")
    if pinn["stages"]["probe"]["steps"] != 150:
        raise ValueError("paper PINN probe must end at 150 L-BFGS iterations")
    if pinn["stages"]["convergence"]["steps"] != 2000:
        raise ValueError("paper PINN convergence run must end at 2000 iterations")
    expected_snapshots = {
        "probe": [0, 150],
        "convergence": [0, 150, 500, 1000, 1500, 2000],
    }
    for stage in ("probe", "convergence"):
        if pinn["stages"][stage]["snapshot_steps"] != expected_snapshots[stage]:
            raise ValueError(f"paper PINN {stage} snapshot schedule has changed")
        unknown = set(pinn["variants"][stage]) - set(VARIANTS)
        if unknown:
            raise ValueError(f"unknown {stage} variant(s): {sorted(unknown)}")
