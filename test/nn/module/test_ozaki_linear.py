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

import pytest
import torch
import torch.nn as nn

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.nn import OzakiLinear, convert_linear_to_ozaki
from physicsnemo.nn.module.ozaki_linear import _extension_backend_name
from physicsnemo.nn.module.ozaki_schemes import (
    MXFP8_KARATSUBA_GEMMS_PER_MODULUS,
    NVFP4_GEMM_SCHEDULE,
    canonical_ozaki_backend,
    reconstruct_mxfp8_karatsuba_product,
    reconstruct_nvfp4_digit_product,
    split_mxfp8_digits,
    split_nvfp4_digits,
)


def test_convert_linear_to_ozaki_replaces_nested_linear_layers():
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Tanh(),
        nn.Sequential(nn.Linear(8, 1)),
    ).double()

    converted = convert_linear_to_ozaki(model, inplace=False)

    assert isinstance(converted[0], OzakiLinear)
    assert isinstance(converted[2][0], OzakiLinear)
    assert isinstance(model[0], nn.Linear)
    assert converted[0].weight.dtype == torch.float64
    assert "num_moduli=15" in converted[0].extra_repr()


def test_convert_linear_to_ozaki_accepts_backend_selector():
    model = nn.Sequential(nn.Linear(2, 8), nn.Linear(8, 1)).double()

    converted = convert_linear_to_ozaki(model, backend="fp8", inplace=False)

    assert converted[0].backend == "fp8"
    assert converted[1].backend == "fp8"
    assert "backend='fp8'" in converted[0].extra_repr()


def test_ozaki_backend_aliases_and_nvfp4_digit_schedule():
    assert canonical_ozaki_backend("mxfp8_ozaki2") == "mxfp8"
    with pytest.raises(ValueError, match="unknown Ozaki backend"):
        canonical_ozaki_backend("ozaki_fp64")
    with pytest.raises(RuntimeError, match="not a GEMMul8 extension backend"):
        _extension_backend_name("mxfp8")
    with pytest.raises(RuntimeError, match="not a GEMMul8 extension backend"):
        _extension_backend_name("nvfp4")

    assert split_nvfp4_digits(0) == (0, 0, 0)
    assert split_nvfp4_digits(364) == (4, 4, 4)
    assert split_nvfp4_digits(-364) == (-4, -4, -4)
    assert len(NVFP4_GEMM_SCHEDULE) == 9

    gemm_values = [0] * 9
    gemm_values[-1] = 7
    assert reconstruct_nvfp4_digit_product(gemm_values) == 7

    assert MXFP8_KARATSUBA_GEMMS_PER_MODULUS == 3
    assert reconstruct_mxfp8_karatsuba_product(7, 11, 31) == (
        16 * 16 * 7 + 16 * (31 - 7 - 11) + 11
    )
    for value in range(-255, 256):
        lo, hi = split_mxfp8_digits(value)
        assert lo + 16 * hi == value
        assert -16 <= lo <= 16
        assert -16 <= hi <= 16
        assert -16 <= lo + hi <= 16


def test_convert_linear_to_ozaki_replaces_physicsnemo_fully_connected_layers():
    model = FullyConnected(
        in_features=2,
        layer_size=8,
        out_features=1,
        num_layers=2,
    ).double()

    converted = convert_linear_to_ozaki(model, inplace=False)

    assert isinstance(converted.layers[0].linear, OzakiLinear)
    assert isinstance(converted.layers[1].linear, OzakiLinear)
    assert isinstance(converted.final_layer.linear, OzakiLinear)
    assert isinstance(model.layers[0].linear, nn.Linear)


def test_convert_linear_to_ozaki_rejects_non_fp64_layers():
    with pytest.raises(TypeError, match="torch.float64"):
        convert_linear_to_ozaki(nn.Linear(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OzakiLinear requires CUDA")
def test_ozaki_linear_forward_backward_smoke():
    torch.manual_seed(0)
    native = nn.Linear(3, 4).cuda().double()
    ozaki = OzakiLinear.from_linear(native)
    native_x = torch.randn(5, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    ozaki_x = native_x.detach().clone().requires_grad_(True)

    native_out = native(native_x)
    native_out.square().mean().backward()
    ozaki_out = ozaki(ozaki_x)
    ozaki_out.square().mean().backward()

    assert ozaki_out.shape == (5, 4)
    assert ozaki_out.dtype == torch.float64
    assert torch.allclose(ozaki_out, native_out, rtol=1e-11, atol=1e-12)
    assert torch.allclose(ozaki_x.grad, native_x.grad, rtol=1e-11, atol=1e-12)
    assert torch.allclose(ozaki.weight.grad, native.weight.grad, rtol=1e-11, atol=1e-12)
    assert torch.allclose(ozaki.bias.grad, native.bias.grad, rtol=1e-11, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="OzakiLinear requires CUDA")
def test_ozaki_linear_supports_pinn_style_higher_order_autograd():
    torch.manual_seed(0)
    native = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).cuda().double()
    ozaki = convert_linear_to_ozaki(native, inplace=False)

    native_coords = torch.randn(
        6, 2, device="cuda", dtype=torch.float64, requires_grad=True
    )
    ozaki_coords = native_coords.detach().clone().requires_grad_(True)

    native_u = native(native_coords)
    native_u_x = torch.autograd.grad(
        native_u,
        native_coords,
        grad_outputs=torch.ones_like(native_u),
        create_graph=True,
    )[0][:, :1]
    native_loss = native_u_x.square().mean()
    native_loss.backward()

    ozaki_u = ozaki(ozaki_coords)
    ozaki_u_x = torch.autograd.grad(
        ozaki_u,
        ozaki_coords,
        grad_outputs=torch.ones_like(ozaki_u),
        create_graph=True,
    )[0][:, :1]
    ozaki_loss = ozaki_u_x.square().mean()
    ozaki_loss.backward()

    assert torch.allclose(ozaki_u, native_u, rtol=1e-11, atol=1e-12)
    assert torch.allclose(ozaki_u_x, native_u_x, rtol=1e-10, atol=1e-12)
    for native_parameter, ozaki_parameter in zip(
        native.parameters(), ozaki.parameters(), strict=True
    ):
        if native_parameter.grad is None:
            assert ozaki_parameter.grad is None
            continue
        assert ozaki_parameter.grad is not None
        assert torch.allclose(
            ozaki_parameter.grad,
            native_parameter.grad,
            rtol=1e-8,
            atol=1e-10,
        )
