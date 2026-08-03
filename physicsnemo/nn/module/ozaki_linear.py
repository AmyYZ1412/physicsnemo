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
the internal GEMM path uses Ozaki scaling with selectable low-precision residue
backends and FP64 reconstruction.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from .ozaki_schemes import (
    EXECUTABLE_GEMMUL8_BACKENDS,
    MXFP8_MODULI,
    NVFP4_MODULI,
    OzakiBackendName,
    canonical_ozaki_backend,
)


_EXECUTABLE_GEMMUL8_BACKENDS = EXECUTABLE_GEMMUL8_BACKENDS

# Backends executed by the C++ digit-decomposition pipeline in
# ozaki_gemmul8/ozaki_digit_backends.cu (real MXFP8/NVFP4 hardware GEMMs via
# torch _scaled_mm_v2), implementing the Week 9 reference algorithm.
_EXECUTABLE_DIGIT_BACKENDS = ("mxfp8", "nvfp4")

# Naming convention:
# - "int8"/"fp8" use the GEMMul8 Backend::INT8 / Backend::FP8 paths.
# - "mxfp8" and "nvfp4" use the dedicated C++ digit-decomposition backend.


def _require_supported_runtime_backend(backend: str) -> None:
    canonical = canonical_ozaki_backend(backend)
    if canonical in _EXECUTABLE_GEMMUL8_BACKENDS:
        return
    if canonical in _EXECUTABLE_DIGIT_BACKENDS:
        return
    raise RuntimeError(f"Ozaki backend '{canonical}' is not executable in this build.")


def _extension_backend_name(backend: str) -> str:
    """Map executable public backend names onto extension backend names."""

    canonical = canonical_ozaki_backend(backend)
    if canonical not in _EXECUTABLE_GEMMUL8_BACKENDS:
        raise RuntimeError(
            f"Ozaki backend '{canonical}' is not a GEMMul8 extension backend."
        )
    return canonical


def _digit_backend_moduli(backend: str, num_moduli: int) -> tuple[int, ...]:
    canonical = canonical_ozaki_backend(backend)
    if canonical == "mxfp8":
        moduli = MXFP8_MODULI
    elif canonical == "nvfp4":
        moduli = NVFP4_MODULI
    else:
        raise RuntimeError(f"Ozaki backend '{canonical}' has no digit pipeline")
    count = min(int(num_moduli), len(moduli))
    if count <= 0:
        raise ValueError("num_moduli must be positive")
    return moduli[:count]


@lru_cache(maxsize=16)
def _digit_crt_basis(
    backend: str, num_moduli: int
) -> tuple[tuple[float, ...], tuple[float, ...], float, float]:
    """Centered signed-CRT basis and modulus product as hi/lo double pairs.

    Computed with arbitrary-precision integers, then each value is split into
    two float64 parts (hi + lo, exact) so the C++ side can reconstruct with
    double-double arithmetic at FP64-grade accuracy. Mathematically the same
    signed CRT as the Week 9 exact-integer reference.
    """

    moduli = _digit_backend_moduli(backend, num_moduli)
    product = math.prod(moduli)

    def split_exact(value: int) -> tuple[float, float]:
        hi = float(value)
        lo = float(value - int(hi))
        return hi, lo

    basis_hi: list[float] = []
    basis_lo: list[float] = []
    for modulus in moduli:
        partial = product // modulus
        value = (partial * pow(partial, -1, modulus)) % product
        if value > product // 2:
            value -= product
        hi, lo = split_exact(value)
        basis_hi.append(hi)
        basis_lo.append(lo)
    product_hi, product_lo = split_exact(product)
    return tuple(basis_hi), tuple(basis_lo), product_hi, product_lo


def _digit_matmul_ext(
    a: torch.Tensor,
    b: torch.Tensor,
    num_moduli: int,
    backend: str,
) -> torch.Tensor:
    """a @ b through the C++ MXFP8/NVFP4 digit-decomposition backend."""

    canonical = canonical_ozaki_backend(backend)
    basis_hi, basis_lo, product_hi, product_lo = _digit_crt_basis(
        canonical, int(num_moduli)
    )
    ext = _load_extension()
    return ext.matmul_digit(
        a,
        b,
        int(num_moduli),
        canonical,
        list(basis_hi),
        list(basis_lo),
        float(product_hi),
        float(product_lo),
    )


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
        ext_dir / "ozaki_digit_backends.cu",
        ext_dir / "gemmul8_dgemm" / "gemmul8_dgemm.cu",
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
        name="physicsnemo_ozaki_linear_ext_v2",
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


class _OzakiMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, num_moduli, fastmode, backend):
        backend = canonical_ozaki_backend(str(backend))
        _require_supported_runtime_backend(backend)
        ctx.save_for_backward(a, b)
        ctx.num_moduli = int(num_moduli)
        ctx.fastmode = bool(fastmode)
        ctx.backend = backend
        if backend in _EXECUTABLE_DIGIT_BACKENDS:
            return _digit_matmul_ext(a, b, ctx.num_moduli, backend)
        ext = _load_extension()
        return ext.matmul(
            a,
            b,
            ctx.num_moduli,
            ctx.fastmode,
            _extension_backend_name(backend),
        )

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = ozaki_matmul(
                grad_output,
                b.transpose(0, 1).contiguous(),
                num_moduli=ctx.num_moduli,
                fastmode=ctx.fastmode,
                backend=ctx.backend,
            )
        if ctx.needs_input_grad[1]:
            grad_b = ozaki_matmul(
                a.transpose(0, 1).contiguous(),
                grad_output,
                num_moduli=ctx.num_moduli,
                fastmode=ctx.fastmode,
                backend=ctx.backend,
            )
        return grad_a, grad_b, None, None, None


def ozaki_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    num_moduli: int = 15,
    fastmode: bool = False,
    backend: str = "int8",
):
    """Compute ``a @ b`` with the Ozaki/GEMMul8 FP64-emulated GEMM path."""

    backend = canonical_ozaki_backend(backend)
    _require_supported_runtime_backend(backend)
    _check_fp64_cuda("a", a)
    _check_fp64_cuda("b", b)
    if a.device != b.device:
        raise TypeError("a and b must be on the same CUDA device")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Ozaki matmul currently supports 2D tensors only")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a and b inner dimensions must match")
    return _OzakiMatmul.apply(a, b, int(num_moduli), bool(fastmode), backend)


class _OzakiLinear(torch.autograd.Function):
    """FP64 Linear boundary with Ozaki GEMMul8/digit forward and backward GEMMs."""

    @staticmethod
    def forward(ctx, input, weight, bias, num_moduli, fastmode, backend):
        _check_fp64_cuda("input", input)
        _check_fp64_cuda("weight", weight)
        if bias is not None:
            _check_fp64_cuda("bias", bias)
        backend = canonical_ozaki_backend(str(backend))
        _require_supported_runtime_backend(backend)

        if backend in _EXECUTABLE_GEMMUL8_BACKENDS:
            # int8 and fp8 share the GEMMul8 extension forward path.
            ext = _load_extension()
            output = ext.forward(
                input,
                weight,
                bias,
                int(num_moduli),
                bool(fastmode),
                _extension_backend_name(backend),
            )
        elif backend in _EXECUTABLE_DIGIT_BACKENDS:
            output = _digit_matmul_ext(
                input,
                weight.transpose(0, 1).contiguous(),
                int(num_moduli),
                backend,
            )
            if bias is not None:
                output = output + bias
        # Save the FP64 boundary tensors, rather than limbs produced inside
        # this no-grad custom-forward context. Backward re-splits them while
        # grad mode is active so PINN higher-order derivatives retain their
        # graph back to input coordinates and parameters.
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        ctx.bias_shape = tuple(bias.shape) if bias is not None else None
        ctx.num_moduli = int(num_moduli)
        ctx.fastmode = bool(fastmode)
        ctx.backend = backend
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        create_graph = torch.is_grad_enabled()
        if ctx.backend in _EXECUTABLE_GEMMUL8_BACKENDS or ctx.backend in _EXECUTABLE_DIGIT_BACKENDS:
            if create_graph:
                # PINN residuals differentiate through grad_input. Route it
                # through the autograd-visible ozaki_matmul so the graph stays
                # live back to inputs and parameters; parameter gradients are
                # deferred to the ordinary outer backward pass.
                grad_input = (
                    ozaki_matmul(
                        grad_output.contiguous(),
                        weight,
                        num_moduli=ctx.num_moduli,
                        fastmode=ctx.fastmode,
                        backend=ctx.backend,
                    )
                    if ctx.needs_input_grad[0]
                    else None
                )
                return grad_input, None, None, None, None, None
            grad_input = (
                ozaki_matmul(
                    grad_output.contiguous(),
                    weight,
                    num_moduli=ctx.num_moduli,
                    fastmode=ctx.fastmode,
                    backend=ctx.backend,
                )
                if ctx.needs_input_grad[0]
                else None
            )
            grad_weight = (
                ozaki_matmul(
                    grad_output.transpose(0, 1).contiguous(),
                    input,
                    num_moduli=ctx.num_moduli,
                    fastmode=ctx.fastmode,
                    backend=ctx.backend,
                )
                if ctx.needs_input_grad[1]
                else None
            )
            grad_bias = (
                grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)
                if ctx.has_bias and ctx.needs_input_grad[2]
                else None
            )
            return grad_input, grad_weight, grad_bias, None, None, None


def ozaki_linear_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    num_moduli: int = 15,
    fastmode: bool = False,
    backend: str = "int8",
) -> torch.Tensor:
    """Linear forward where the GEMM is replaced by Ozaki/GEMMul8."""

    backend = canonical_ozaki_backend(backend)
    _require_supported_runtime_backend(backend)
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

    return _OzakiLinear.apply(
        input, weight, bias, int(num_moduli), bool(fastmode), backend
    )


class OzakiLinear(nn.Module):
    """Drop-in FP64 ``nn.Linear`` replacement using Ozaki/GEMMul8 GEMMs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_moduli: int = 15,
        fastmode: bool = False,
        backend: str = "int8",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_moduli = int(num_moduli)
        self.fastmode = bool(fastmode)
        self.backend = canonical_ozaki_backend(backend)
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
            backend=self.backend,
        )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        num_moduli: int = 15,
        fastmode: bool = False,
        backend: str = "int8",
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
            backend=backend,
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
            f"fastmode={self.fastmode}, backend='{self.backend}'"
        )


def convert_linear_to_ozaki(
    module: nn.Module,
    *,
    num_moduli: int = 15,
    fastmode: bool = False,
    backend: str = "int8",
    inplace: bool = True,
) -> nn.Module:
    """Recursively replace ``nn.Linear`` children with ``OzakiLinear``.

    The converter is intentionally thin: it does not change activation,
    residual, optimizer, loss, or PhysicsNeMo solver behavior.
    """

    if not inplace:
        import copy

        module = copy.deepcopy(module)

    backend = canonical_ozaki_backend(backend)

    if isinstance(module, nn.Linear):
        return OzakiLinear.from_linear(
            module,
            num_moduli=num_moduli,
            fastmode=fastmode,
            backend=backend,
        )

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                OzakiLinear.from_linear(
                    child,
                    num_moduli=num_moduli,
                    fastmode=fastmode,
                    backend=backend,
                ),
            )
        else:
            convert_linear_to_ozaki(
                child,
                num_moduli=num_moduli,
                fastmode=fastmode,
                backend=backend,
                inplace=True,
            )
    return module
