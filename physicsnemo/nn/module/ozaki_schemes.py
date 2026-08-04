# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ozaki-II scheme definitions independent of PINN training code.

This module owns the algorithm-level choices for the selectable Ozaki schemes:
backend names, residue moduli, digit decompositions, and reconstruction formulas.
``ozaki_linear.py`` consumes these definitions to implement a generic linear
layer wrapper; PINN examples only choose which scheme to use.
"""

from __future__ import annotations

from typing import Literal


OzakiBackendName = Literal["int8", "fp8", "mxfp8", "nvfp4"]

BACKEND_ALIASES = {
    "int8": "int8",
    "int8_ozaki2": "int8",
    "fp8": "fp8",
    "fp8_ozaki2": "fp8",
    "mxfp8": "mxfp8",
    "mxfp8_ozaki2": "mxfp8",
    "nvfp4": "nvfp4",
    "nvfp4_ozaki2": "nvfp4",
}

EXECUTABLE_GEMMUL8_BACKENDS = {"int8", "fp8"}

NVFP4_DIGIT_RADIX = 9
NVFP4_DIGIT_MIN = -4
NVFP4_DIGIT_MAX = 4
NVFP4_DIGITS_PER_RESIDUE = 3
NVFP4_GEMM_SCHEDULE = (
    (2, 2, 4),
    (2, 1, 3),
    (2, 0, 2),
    (1, 2, 3),
    (1, 1, 2),
    (1, 0, 1),
    (0, 2, 2),
    (0, 1, 1),
    (0, 0, 0),
)
NVFP4_MODULI = (727, 719, 709, 701, 691, 683, 677, 673, 661, 659, 653, 647, 643, 641)

MXFP8_DIGIT_RADIX = 16
MXFP8_DIGIT_MIN = -16
MXFP8_DIGIT_MAX = 16
MXFP8_DIGITS_PER_RESIDUE = 2
MXFP8_KARATSUBA_GEMMS_PER_MODULUS = 3
MXFP8_MODULI = (511, 509, 503, 499, 491, 487, 481, 479, 467, 463, 461, 457, 449, 443)


def canonical_ozaki_backend(backend: str) -> OzakiBackendName:
    try:
        return BACKEND_ALIASES[backend.lower()]
    except KeyError as exc:
        raise ValueError(
            "unknown Ozaki backend. Expected one of: int8, fp8, mxfp8, nvfp4"
        ) from exc


def normalize_mod(value: int, modulus: int) -> int:
    out = int(value) % int(modulus)
    return out + modulus if out < 0 else out


def centered_residue(value: int, modulus: int) -> int:
    residue = normalize_mod(value, modulus)
    if residue != 0 and 2 * residue >= modulus:
        residue -= modulus
    return residue


def split_nvfp4_digits(value: int) -> tuple[int, int, int]:
    """Split a centered residue into d0 + 9*d1 + 81*d2, d_i in [-4, 4]."""

    max_abs = NVFP4_DIGIT_MAX * (
        NVFP4_DIGIT_RADIX * NVFP4_DIGIT_RADIX + NVFP4_DIGIT_RADIX + 1
    )
    if value < -max_abs or value > max_abs:
        raise ValueError(f"NVFP4 residue {value} outside three-digit coverage")
    d0 = ((value + NVFP4_DIGIT_MAX) % NVFP4_DIGIT_RADIX) - NVFP4_DIGIT_MAX
    q1 = (value - d0) // NVFP4_DIGIT_RADIX
    d1 = ((q1 + NVFP4_DIGIT_MAX) % NVFP4_DIGIT_RADIX) - NVFP4_DIGIT_MAX
    d2 = (q1 - d1) // NVFP4_DIGIT_RADIX
    digits = (int(d0), int(d1), int(d2))
    if any(d < NVFP4_DIGIT_MIN or d > NVFP4_DIGIT_MAX for d in digits):
        raise ValueError(f"NVFP4 digit outside [-4, 4]: {digits}")
    return digits


def split_mxfp8_digits(value: int) -> tuple[int, int]:
    """Split a centered residue into d0 + 16*d1 for the MXFP8 scheme.

    The MXFP8 residue backend uses Karatsuba, so the FP8 GEMMs must see exact
    integer inputs for d0, d1, and d0 + d1. The representation below keeps all
    three in [-16, 16].
    """

    if value < -272 or value > 272:
        raise ValueError(f"MXFP8 residue {value} outside two-digit coverage")
    d0 = ((value + 8) % MXFP8_DIGIT_RADIX) - 8
    d1 = (value - d0) // MXFP8_DIGIT_RADIX
    digit_sum = d0 + d1
    if digit_sum > MXFP8_DIGIT_MAX:
        d0 -= MXFP8_DIGIT_RADIX
        d1 += 1
    elif digit_sum < MXFP8_DIGIT_MIN:
        d0 += MXFP8_DIGIT_RADIX
        d1 -= 1
    digits = (int(d0), int(d1))
    if any(d < MXFP8_DIGIT_MIN or d > MXFP8_DIGIT_MAX for d in digits):
        raise ValueError(f"MXFP8 digit outside [-16, 16]: {digits}")
    if not MXFP8_DIGIT_MIN <= sum(digits) <= MXFP8_DIGIT_MAX:
        raise ValueError(f"MXFP8 Karatsuba digit sum outside [-16, 16]: {digits}")
    return digits


def reconstruct_nvfp4_digit_product(gemm_values: tuple[int, ...] | list[int]) -> int:
    """Reconstruct one NVFP4 residue product from the 9 scheduled digit GEMMs."""

    if len(gemm_values) != len(NVFP4_GEMM_SCHEDULE):
        raise ValueError("NVFP4 reconstruction requires 9 digit GEMM outputs")
    out = 0
    for value, (_, _, beta_power) in zip(gemm_values, NVFP4_GEMM_SCHEDULE):
        out += int(value) * (NVFP4_DIGIT_RADIX ** int(beta_power))
    return out


def reconstruct_mxfp8_karatsuba_product(
    hi_hi: int,
    lo_lo: int,
    sum_sum: int,
) -> int:
    """Reconstruct one MXFP8 radix-16 product from Karatsuba outputs."""

    cross = int(sum_sum) - int(hi_hi) - int(lo_lo)
    return (
        (MXFP8_DIGIT_RADIX * MXFP8_DIGIT_RADIX) * int(hi_hi)
        + MXFP8_DIGIT_RADIX * cross
        + int(lo_lo)
    )
