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

"""Ozaki Scheme II / GEMMul8 FP64-emulated linear layers.

This module provides a narrow, optional integration point for PINN MLPs that
need FP64 linear-layer GEMMs without changing the surrounding PyTorch or
PhysicsNeMo training loop. Inputs, parameters, and outputs are ``torch.float64``;
the internal GEMM path uses GEMMul8/Ozaki scaling with INT8 tensor-core GEMM and
FP64 reconstruction.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _extension_dir() -> Path:
    return _module_dir() / "ozaki_gemmul8"


@lru_cache(maxsize=1)
def _load_extension():
    if not torch.cuda.is_available():
        raise RuntimeError("OzakiLinear requires CUDA")

    ext_dir = _extension_dir()
    gemmul8_home = Path(os.environ.get("GEMMUL8_HOME", ext_dir / "GEMMul8"))
    header = gemmul8_home / "include" / "gemmul8.hpp"
    if not header.exists():
        raise RuntimeError(
            "GEMMul8 headers were not found. Set GEMMUL8_HOME or install the "
            "bundled ozaki_gemmul8/GEMMul8 files."
        )

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    os.environ.setdefault("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))

    sources = [
        ext_dir / "ozaki_linear_ext.cu",
        ext_dir / "gemmul8_dgemm" / "gemmul8_dgemm_int8_only.cu",
    ]
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise RuntimeError(f"OzakiLinear extension sources were not found: {missing}")

    build_dir = Path(
        os.environ.get(
            "OZAKI_LINEAR_BUILD_DIR",
            Path.home() / ".cache" / "physicsnemo" / "ozaki_linear_ext",
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name="physicsnemo_ozaki_linear_ext",
        sources=[str(path) for path in sources],
        extra_include_paths=[str(gemmul8_home / "include"), str(gemmul8_home / "src")],
        extra_cflags=["-std=c++20"],
        extra_cuda_cflags=["-std=c++20", "-O3", "-diag-suppress=177"],
        extra_ldflags=["-lcublas", "-lcublasLt", "-lcuda", "-ldl"],
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=os.environ.get("OZAKI_LINEAR_VERBOSE", "0") == "1",
    )


def _check_fp64_cuda(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float64:
        raise TypeError(f"{name} must be a torch.float64 tensor")
    if tensor.device.type != "cuda":
        raise TypeError(f"{name} must be a CUDA tensor")


def _check_triple_limb(
    name: str, tensor: torch.Tensor, ref: torch.Tensor | None = None
) -> None:
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be a torch.float32 tensor")
    if ref is not None:
        if tensor.device != ref.device:
            raise TypeError(f"{name} must be on the same device as the reference limb")
        if tensor.shape != ref.shape:
            raise ValueError(f"{name} shape must match the reference limb")


def split_float64_to_triple_float32(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split an FP64 tensor into three FP32 residual limbs for Ozaki backward."""

    if x.dtype != torch.float64:
        raise TypeError("x must be a torch.float64 tensor")
    hi = x.to(torch.float32)
    rem = x - hi.to(torch.float64)
    mid = rem.to(torch.float32)
    lo = (rem - mid.to(torch.float64)).to(torch.float32)
    return hi, mid, lo


class _ReconstructTripleFloat32ToFloat64(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hi, mid, lo):
        _check_triple_limb("hi", hi)
        _check_triple_limb("mid", mid, hi)
        _check_triple_limb("lo", lo, hi)
        return hi.to(torch.float64) + mid.to(torch.float64) + lo.to(torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None
        return split_float64_to_triple_float32(grad_output.contiguous())


def reconstruct_triple_float32_to_float64(
    hi: torch.Tensor, mid: torch.Tensor, lo: torch.Tensor
) -> torch.Tensor:
    return _ReconstructTripleFloat32ToFloat64.apply(hi, mid, lo)


def ozaki_linear_forward_triple(
    input_hi,
    input_mid,
    input_lo,
    weight_hi,
    weight_mid,
    weight_lo,
    bias_hi=None,
    bias_mid=None,
    bias_lo=None,
    num_moduli: int = 15,
    fastmode: bool = False,
):
    """Run Ozaki linear forward through the triple-limb extension API."""

    _check_triple_limb("input_hi", input_hi)
    _check_triple_limb("input_mid", input_mid, input_hi)
    _check_triple_limb("input_lo", input_lo, input_hi)
    _check_triple_limb("weight_hi", weight_hi)
    _check_triple_limb("weight_mid", weight_mid, weight_hi)
    _check_triple_limb("weight_lo", weight_lo, weight_hi)
    if input_hi.device != weight_hi.device:
        raise TypeError("input and weight limbs must be on the same CUDA device")
    if input_hi.shape[-1] != weight_hi.shape[1]:
        raise ValueError("input last dimension must match weight in_features")

    has_bias = bias_hi is not None or bias_mid is not None or bias_lo is not None
    if has_bias:
        if bias_hi is None or bias_mid is None or bias_lo is None:
            raise ValueError("bias limbs must be all provided or all omitted")
        _check_triple_limb("bias_hi", bias_hi)
        _check_triple_limb("bias_mid", bias_mid, bias_hi)
        _check_triple_limb("bias_lo", bias_lo, bias_hi)

    leading_shape = input_hi.shape[:-1]
    flat_hi = input_hi.reshape(-1, input_hi.shape[-1]).contiguous()
    flat_mid = input_mid.reshape(-1, input_mid.shape[-1]).contiguous()
    flat_lo = input_lo.reshape(-1, input_lo.shape[-1]).contiguous()

    ext = _load_extension()
    out_hi, out_mid, out_lo = ext.forward_triple(
        flat_hi,
        flat_mid,
        flat_lo,
        weight_hi.contiguous(),
        weight_mid.contiguous(),
        weight_lo.contiguous(),
        bias_hi.contiguous() if has_bias else None,
        bias_mid.contiguous() if has_bias else None,
        bias_lo.contiguous() if has_bias else None,
        int(num_moduli),
        bool(fastmode),
    )
    return (
        out_hi.reshape(*leading_shape, weight_hi.shape[0]),
        out_mid.reshape(*leading_shape, weight_hi.shape[0]),
        out_lo.reshape(*leading_shape, weight_hi.shape[0]),
    )


def _triple_matmul(
    a_hi, a_mid, a_lo, b_hi, b_mid, b_lo, *, num_moduli=15, fastmode=False
):
    return ozaki_linear_forward_triple(
        a_hi,
        a_mid,
        a_lo,
        b_hi.transpose(0, 1).contiguous(),
        b_mid.transpose(0, 1).contiguous(),
        b_lo.transpose(0, 1).contiguous(),
        num_moduli=num_moduli,
        fastmode=fastmode,
    )


def _ozaki_matmul_backward_from_float64(
    a,
    b,
    grad_output,
    *,
    needs_grad_a=True,
    needs_grad_b=True,
    num_moduli=15,
    fastmode=False,
):
    if not needs_grad_a and not needs_grad_b:
        return None, None

    _check_fp64_cuda("a", a)
    _check_fp64_cuda("b", b)
    _check_fp64_cuda("grad_output", grad_output)
    if a.dim() != 2 or b.dim() != 2 or grad_output.dim() != 2:
        raise ValueError("Ozaki matmul backward currently supports 2D tensors only")

    a_hi, a_mid, a_lo = split_float64_to_triple_float32(a)
    b_hi, b_mid, b_lo = split_float64_to_triple_float32(b)
    grad_hi, grad_mid, grad_lo = split_float64_to_triple_float32(grad_output)

    grad_a = None
    if needs_grad_a:
        grad_a_limbs = _triple_matmul(
            grad_hi,
            grad_mid,
            grad_lo,
            b_hi.transpose(0, 1).contiguous(),
            b_mid.transpose(0, 1).contiguous(),
            b_lo.transpose(0, 1).contiguous(),
            num_moduli=num_moduli,
            fastmode=fastmode,
        )
        grad_a = reconstruct_triple_float32_to_float64(*grad_a_limbs)

    grad_b = None
    if needs_grad_b:
        grad_b_limbs = _triple_matmul(
            a_hi.transpose(0, 1).contiguous(),
            a_mid.transpose(0, 1).contiguous(),
            a_lo.transpose(0, 1).contiguous(),
            grad_hi,
            grad_mid,
            grad_lo,
            num_moduli=num_moduli,
            fastmode=fastmode,
        )
        grad_b = reconstruct_triple_float32_to_float64(*grad_b_limbs)

    return grad_a, grad_b


def _grad_output_or_zero(grad: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
    if grad is None:
        return torch.zeros_like(like)
    return grad.to(torch.float32).contiguous()


def _zero_triple_like(
    like_hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = torch.zeros_like(like_hi)
    return zero, zero, zero


def _add_triple(
    accum: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    value: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if accum is None:
        return value
    return tuple(a + b for a, b in zip(accum, value, strict=True))


def _two_sum_float32(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    summation = a + b
    correction = (a - (summation - b)) + (b - (summation - a))
    return summation, correction


def _compress_components_float32(
    components: list[torch.Tensor], max_limbs: int
) -> list[torch.Tensor]:
    limbs: list[torch.Tensor] = []
    for component in components:
        residual = component
        next_limbs: list[torch.Tensor] = []
        for limb in limbs:
            residual, error = _two_sum_float32(residual, limb)
            next_limbs.append(error)
        next_limbs.append(residual)
        limbs = next_limbs

    while len(limbs) > max_limbs:
        residual = limbs[-1]
        next_limbs = limbs[:-2]
        summation, error = _two_sum_float32(limbs[-2], residual)
        next_limbs.extend([error, summation])
        limbs = next_limbs[-max_limbs:]
    while len(limbs) < max_limbs:
        limbs.insert(0, torch.zeros_like(limbs[0]))
    return limbs


def _pairwise_sum_dim0_components_float32(
    components: list[torch.Tensor], max_limbs: int
) -> list[torch.Tensor]:
    if not components:
        raise ValueError("components must be non-empty")
    limbs = _compress_components_float32(components, max_limbs=max_limbs)
    while limbs[0].shape[0] > 1:
        count = limbs[0].shape[0]
        even_count = (count // 2) * 2
        pair_components: list[torch.Tensor] = []
        for limb in limbs:
            pairs = limb[:even_count].reshape(even_count // 2, 2, *limb.shape[1:])
            pair_components.extend([pairs[:, 0], pairs[:, 1]])
        next_limbs = _compress_components_float32(pair_components, max_limbs=max_limbs)
        if even_count != count:
            next_limbs = [
                torch.cat([limb, carry[-1:].clone()], dim=0)
                for limb, carry in zip(next_limbs, limbs, strict=True)
            ]
        limbs = next_limbs
    return [limb.reshape(*limb.shape[1:]) for limb in limbs]


def ozaki_linear_backward_triple(
    input_hi: torch.Tensor,
    input_mid: torch.Tensor,
    input_lo: torch.Tensor,
    weight_hi: torch.Tensor,
    weight_mid: torch.Tensor,
    weight_lo: torch.Tensor,
    grad_output_hi: torch.Tensor,
    grad_output_mid: torch.Tensor,
    grad_output_lo: torch.Tensor,
    *,
    needs_grad_input: bool = True,
    needs_grad_weight: bool = True,
    needs_grad_bias: bool = True,
    num_moduli: int = 15,
    fastmode: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    """Compute Linear backward GEMMs through the triple-limb Ozaki path."""

    _check_triple_limb("input_hi", input_hi)
    _check_triple_limb("input_mid", input_mid, input_hi)
    _check_triple_limb("input_lo", input_lo, input_hi)
    _check_triple_limb("weight_hi", weight_hi)
    _check_triple_limb("weight_mid", weight_mid, weight_hi)
    _check_triple_limb("weight_lo", weight_lo, weight_hi)
    _check_triple_limb("grad_output_hi", grad_output_hi)
    _check_triple_limb("grad_output_mid", grad_output_mid, grad_output_hi)
    _check_triple_limb("grad_output_lo", grad_output_lo, grad_output_hi)
    if input_hi.device != weight_hi.device or input_hi.device != grad_output_hi.device:
        raise TypeError("input, weight, and grad_output limbs must share a CUDA device")
    if input_hi.shape[:-1] != grad_output_hi.shape[:-1]:
        raise ValueError("input and grad_output leading shapes must match")
    if input_hi.shape[-1] != weight_hi.shape[1]:
        raise ValueError("input last dimension must match weight in_features")
    if grad_output_hi.shape[-1] != weight_hi.shape[0]:
        raise ValueError("grad_output last dimension must match weight out_features")

    leading_shape = input_hi.shape[:-1]
    flat_input = tuple(
        limb.reshape(-1, limb.shape[-1]).contiguous()
        for limb in (input_hi, input_mid, input_lo)
    )
    flat_grad = tuple(
        limb.reshape(-1, limb.shape[-1]).contiguous()
        for limb in (grad_output_hi, grad_output_mid, grad_output_lo)
    )

    grad_input = None
    if needs_grad_input:
        grad_input_flat = _triple_matmul(
            *flat_grad,
            weight_hi.contiguous(),
            weight_mid.contiguous(),
            weight_lo.contiguous(),
            num_moduli=num_moduli,
            fastmode=fastmode,
        )
        grad_input = tuple(
            limb.reshape(*leading_shape, weight_hi.shape[1]) for limb in grad_input_flat
        )

    grad_weight = None
    if needs_grad_weight:
        grad_weight = _triple_matmul(
            flat_grad[0].transpose(0, 1).contiguous(),
            flat_grad[1].transpose(0, 1).contiguous(),
            flat_grad[2].transpose(0, 1).contiguous(),
            *flat_input,
            num_moduli=num_moduli,
            fastmode=fastmode,
        )

    grad_bias = None
    if needs_grad_bias:
        grad_bias = tuple(_load_extension().sum_dim0_triple(*flat_grad))

    return grad_input, grad_weight, grad_bias


class _OzakiLinearBackwardTripleFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_hi,
        input_mid,
        input_lo,
        weight_hi,
        weight_mid,
        weight_lo,
        grad_hi,
        grad_mid,
        grad_lo,
        needs_grad_input,
        needs_grad_weight,
        needs_grad_bias,
        num_moduli,
        fastmode,
    ):
        ctx.save_for_backward(
            input_hi,
            input_mid,
            input_lo,
            weight_hi,
            weight_mid,
            weight_lo,
            grad_hi,
            grad_mid,
            grad_lo,
        )
        ctx.needs_grad_input_value = bool(needs_grad_input)
        ctx.needs_grad_weight_value = bool(needs_grad_weight)
        ctx.needs_grad_bias_value = bool(needs_grad_bias)
        ctx.num_moduli = int(num_moduli)
        ctx.fastmode = bool(fastmode)

        grad_input, grad_weight, grad_bias = ozaki_linear_backward_triple(
            input_hi,
            input_mid,
            input_lo,
            weight_hi,
            weight_mid,
            weight_lo,
            grad_hi,
            grad_mid,
            grad_lo,
            needs_grad_input=ctx.needs_grad_input_value,
            needs_grad_weight=ctx.needs_grad_weight_value,
            needs_grad_bias=ctx.needs_grad_bias_value,
            num_moduli=ctx.num_moduli,
            fastmode=ctx.fastmode,
        )
        if grad_input is None:
            grad_input = _zero_triple_like(input_hi)
        if grad_weight is None:
            grad_weight = _zero_triple_like(weight_hi)
        if grad_bias is None:
            grad_bias = _zero_triple_like(weight_hi[:, 0])
        return (*grad_input, *grad_weight, *grad_bias)

    @staticmethod
    def backward(ctx, *grad_outputs):
        (
            input_hi,
            input_mid,
            input_lo,
            weight_hi,
            weight_mid,
            weight_lo,
            grad_hi,
            grad_mid,
            grad_lo,
        ) = ctx.saved_tensors
        (
            grad_grad_input_hi,
            grad_grad_input_mid,
            grad_grad_input_lo,
            grad_grad_weight_hi,
            grad_grad_weight_mid,
            grad_grad_weight_lo,
            grad_grad_bias_hi,
            grad_grad_bias_mid,
            grad_grad_bias_lo,
        ) = grad_outputs

        has_grad_grad_input = any(
            grad is not None
            for grad in (grad_grad_input_hi, grad_grad_input_mid, grad_grad_input_lo)
        )
        has_grad_grad_weight = any(
            grad is not None
            for grad in (grad_grad_weight_hi, grad_grad_weight_mid, grad_grad_weight_lo)
        )
        has_grad_grad_bias = any(
            grad is not None
            for grad in (grad_grad_bias_hi, grad_grad_bias_mid, grad_grad_bias_lo)
        )
        grad_grad_input = tuple(
            _grad_output_or_zero(grad, like)
            for grad, like in zip(
                (grad_grad_input_hi, grad_grad_input_mid, grad_grad_input_lo),
                (input_hi, input_mid, input_lo),
                strict=True,
            )
        )
        grad_grad_weight = tuple(
            _grad_output_or_zero(grad, like)
            for grad, like in zip(
                (grad_grad_weight_hi, grad_grad_weight_mid, grad_grad_weight_lo),
                (weight_hi, weight_mid, weight_lo),
                strict=True,
            )
        )
        grad_grad_bias = tuple(
            _grad_output_or_zero(grad, weight_hi[:, 0])
            for grad in (grad_grad_bias_hi, grad_grad_bias_mid, grad_grad_bias_lo)
        )

        leading_shape = input_hi.shape[:-1]
        flat_input = tuple(
            limb.reshape(-1, limb.shape[-1]).contiguous()
            for limb in (input_hi, input_mid, input_lo)
        )
        flat_weight = (
            weight_hi.contiguous(),
            weight_mid.contiguous(),
            weight_lo.contiguous(),
        )
        flat_grad = tuple(
            limb.reshape(-1, limb.shape[-1]).contiguous()
            for limb in (grad_hi, grad_mid, grad_lo)
        )
        flat_grad_grad_input = tuple(
            limb.reshape(-1, limb.shape[-1]).contiguous() for limb in grad_grad_input
        )
        flat_grad_grad_bias = tuple(
            limb.reshape(1, -1).contiguous() for limb in grad_grad_bias
        )

        d_input = d_weight = d_grad = None
        if ctx.needs_grad_weight_value and has_grad_grad_weight:
            d_input_flat = _triple_matmul(
                *flat_grad,
                *grad_grad_weight,
                num_moduli=ctx.num_moduli,
                fastmode=ctx.fastmode,
            )
            d_input = tuple(
                limb.reshape(*leading_shape, weight_hi.shape[1])
                for limb in d_input_flat
            )
            d_grad_from_weight = _triple_matmul(
                *flat_input,
                grad_grad_weight[0].transpose(0, 1).contiguous(),
                grad_grad_weight[1].transpose(0, 1).contiguous(),
                grad_grad_weight[2].transpose(0, 1).contiguous(),
                num_moduli=ctx.num_moduli,
                fastmode=ctx.fastmode,
            )
            d_grad = _add_triple(
                d_grad,
                tuple(limb.reshape_as(grad_hi) for limb in d_grad_from_weight),
            )
        if ctx.needs_grad_input_value and has_grad_grad_input:
            d_weight = _triple_matmul(
                flat_grad[0].transpose(0, 1).contiguous(),
                flat_grad[1].transpose(0, 1).contiguous(),
                flat_grad[2].transpose(0, 1).contiguous(),
                *flat_grad_grad_input,
                num_moduli=ctx.num_moduli,
                fastmode=ctx.fastmode,
            )
            d_grad_from_input = _triple_matmul(
                *flat_grad_grad_input,
                flat_weight[0].transpose(0, 1).contiguous(),
                flat_weight[1].transpose(0, 1).contiguous(),
                flat_weight[2].transpose(0, 1).contiguous(),
                num_moduli=ctx.num_moduli,
                fastmode=ctx.fastmode,
            )
            d_grad = _add_triple(
                d_grad,
                tuple(limb.reshape_as(grad_hi) for limb in d_grad_from_input),
            )
        if ctx.needs_grad_bias_value and has_grad_grad_bias:
            d_grad = _add_triple(
                d_grad,
                tuple(
                    limb.expand_as(flat_grad[0]).reshape_as(grad_hi)
                    for limb in flat_grad_grad_bias
                ),
            )

        if d_input is None:
            d_input = _zero_triple_like(input_hi)
        if d_weight is None:
            d_weight = _zero_triple_like(weight_hi)
        if d_grad is None:
            d_grad = _zero_triple_like(grad_hi)
        return (*d_input, *d_weight, *d_grad, None, None, None, None, None)


def ozaki_linear_backward_triple_autograd(
    input_hi: torch.Tensor,
    input_mid: torch.Tensor,
    input_lo: torch.Tensor,
    weight_hi: torch.Tensor,
    weight_mid: torch.Tensor,
    weight_lo: torch.Tensor,
    grad_hi: torch.Tensor,
    grad_mid: torch.Tensor,
    grad_lo: torch.Tensor,
    *,
    needs_grad_input: bool = True,
    needs_grad_weight: bool = True,
    needs_grad_bias: bool = True,
    num_moduli: int = 15,
    fastmode: bool = False,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    outputs = _OzakiLinearBackwardTripleFunction.apply(
        input_hi,
        input_mid,
        input_lo,
        weight_hi,
        weight_mid,
        weight_lo,
        grad_hi,
        grad_mid,
        grad_lo,
        bool(needs_grad_input),
        bool(needs_grad_weight),
        bool(needs_grad_bias),
        int(num_moduli),
        bool(fastmode),
    )
    return outputs[:3], outputs[3:6], outputs[6:9]


class _OzakiMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, num_moduli, fastmode):
        ext = _load_extension()
        ctx.save_for_backward(a, b)
        ctx.num_moduli = int(num_moduli)
        ctx.fastmode = bool(fastmode)
        return ext.matmul(a, b, ctx.num_moduli, ctx.fastmode)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a, grad_b = _ozaki_matmul_backward_from_float64(
            a,
            b,
            grad_output.contiguous(),
            needs_grad_a=ctx.needs_input_grad[0],
            needs_grad_b=ctx.needs_input_grad[1],
            num_moduli=ctx.num_moduli,
            fastmode=ctx.fastmode,
        )
        return grad_a, grad_b, None, None


def ozaki_matmul(
    a: torch.Tensor, b: torch.Tensor, num_moduli: int = 15, fastmode: bool = False
):
    """Compute ``a @ b`` with the Ozaki/GEMMul8 FP64-emulated GEMM path."""

    _check_fp64_cuda("a", a)
    _check_fp64_cuda("b", b)
    if a.device != b.device:
        raise TypeError("a and b must be on the same CUDA device")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Ozaki matmul currently supports 2D tensors only")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a and b inner dimensions must match")
    return _OzakiMatmul.apply(a, b, int(num_moduli), bool(fastmode))


class _OzakiLinear(torch.autograd.Function):
    """FP64 Linear boundary with Ozaki triple-limb forward and backward GEMMs."""

    @staticmethod
    def forward(ctx, input, weight, bias, num_moduli, fastmode):
        _check_fp64_cuda("input", input)
        _check_fp64_cuda("weight", weight)
        if bias is not None:
            _check_fp64_cuda("bias", bias)

        input_limbs = split_float64_to_triple_float32(input)
        weight_limbs = split_float64_to_triple_float32(weight)
        if bias is None:
            bias_limbs = (None, None, None)
        else:
            bias_limbs = split_float64_to_triple_float32(bias)

        out_limbs = ozaki_linear_forward_triple(
            *input_limbs,
            *weight_limbs,
            *bias_limbs,
            num_moduli=int(num_moduli),
            fastmode=bool(fastmode),
        )
        # Save the FP64 boundary tensors, rather than limbs produced inside
        # this no-grad custom-forward context. Backward re-splits them while
        # grad mode is active so PINN higher-order derivatives retain their
        # graph back to input coordinates and parameters.
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.bias_shape = tuple(bias.shape) if bias is not None else None
        ctx.num_moduli = int(num_moduli)
        ctx.fastmode = bool(fastmode)
        return reconstruct_triple_float32_to_float64(*out_limbs)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        input_hi, input_mid, input_lo = split_float64_to_triple_float32(input)
        weight_hi, weight_mid, weight_lo = split_float64_to_triple_float32(weight)
        grad_limbs = split_float64_to_triple_float32(grad_output.contiguous())

        # PINN residuals differentiate through grad_input. Computing parameter
        # gradients here would introduce unsupported third-order paths, so they
        # are deferred to the ordinary outer backward pass.
        create_graph = torch.is_grad_enabled()
        needs_grad_input = ctx.needs_input_grad[0]
        needs_grad_weight = ctx.needs_input_grad[1] and not create_graph
        needs_grad_bias = ctx.has_bias and ctx.needs_input_grad[2] and not create_graph
        grad_input_limbs, grad_weight_limbs, grad_bias_limbs = (
            ozaki_linear_backward_triple_autograd(
                input_hi,
                input_mid,
                input_lo,
                weight_hi,
                weight_mid,
                weight_lo,
                *grad_limbs,
                needs_grad_input=needs_grad_input,
                needs_grad_weight=needs_grad_weight,
                needs_grad_bias=needs_grad_bias,
                num_moduli=ctx.num_moduli,
                fastmode=ctx.fastmode,
            )
        )

        grad_input = (
            reconstruct_triple_float32_to_float64(*grad_input_limbs)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight = (
            reconstruct_triple_float32_to_float64(*grad_weight_limbs)
            if ctx.needs_input_grad[1]
            else None
        )
        grad_bias = (
            reconstruct_triple_float32_to_float64(*grad_bias_limbs).reshape(
                ctx.bias_shape
            )
            if ctx.has_bias and ctx.needs_input_grad[2]
            else None
        )
        return grad_input, grad_weight, grad_bias, None, None


def ozaki_linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    num_moduli: int = 15,
    fastmode: bool = False,
) -> torch.Tensor:
    """Linear forward where the GEMM is replaced by Ozaki/GEMMul8."""

    _check_fp64_cuda("input", input)
    _check_fp64_cuda("weight", weight)
    if input.device != weight.device:
        raise TypeError("input and weight must be on the same CUDA device")
    if bias is not None:
        _check_fp64_cuda("bias", bias)
        if bias.device != input.device:
            raise TypeError("bias must be on the same CUDA device as input")
    if input.shape[-1] != weight.shape[1]:
        raise ValueError("input last dimension must match weight in_features")

    return _OzakiLinear.apply(input, weight, bias, int(num_moduli), bool(fastmode))


class OzakiLinear(nn.Module):
    """Drop-in FP64 ``nn.Linear`` replacement using Ozaki/GEMMul8 GEMMs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_moduli: int = 15,
        fastmode: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_moduli = int(num_moduli)
        self.fastmode = bool(fastmode)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float64)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float64))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with the same scheme as ``torch.nn.Linear``."""

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return ozaki_linear_forward(
            input,
            self.weight,
            self.bias,
            num_moduli=self.num_moduli,
            fastmode=self.fastmode,
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, num_moduli: int = 15, fastmode: bool = False
    ) -> "OzakiLinear":
        """Create an Ozaki FP64 layer with parameters copied from ``linear``."""

        if not isinstance(linear, nn.Linear):
            raise TypeError("linear must be an instance of torch.nn.Linear")
        if linear.weight.dtype != torch.float64:
            raise TypeError(
                "OzakiLinear requires a torch.float64 source layer. "
                "Call model.double() before conversion."
            )

        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            num_moduli=num_moduli,
            fastmode=fastmode,
        )
        layer = layer.to(device=linear.weight.device)
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def extra_repr(self) -> str:
        """Return a concise layer representation."""

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, num_moduli={self.num_moduli}, "
            f"fastmode={self.fastmode}"
        )


def convert_linear_to_ozaki(
    module: nn.Module,
    *,
    num_moduli: int = 15,
    fastmode: bool = False,
    inplace: bool = True,
) -> nn.Module:
    """Recursively replace ``nn.Linear`` children with ``OzakiLinear``.

    The converter is intentionally thin: it does not change activation,
    residual, optimizer, loss, or PhysicsNeMo solver behavior.
    """

    if not inplace:
        import copy

        module = copy.deepcopy(module)

    if isinstance(module, nn.Linear):
        return OzakiLinear.from_linear(
            module,
            num_moduli=num_moduli,
            fastmode=fastmode,
        )

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                OzakiLinear.from_linear(
                    child, num_moduli=num_moduli, fastmode=fastmode
                ),
            )
        else:
            convert_linear_to_ozaki(
                child,
                num_moduli=num_moduli,
                fastmode=fastmode,
                inplace=True,
            )
    return module
