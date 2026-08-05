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
        return tomllib.load(handle)
