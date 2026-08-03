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

"""Convection PINN comparison for native FP32, native FP64, and Ozaki FP64.

The Ozaki backend only replaces ``nn.Linear`` GEMMs in the PINN MLP. Activation,
PDE residual autograd, losses, and optimizer behavior remain native PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.nn import convert_linear_to_ozaki


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.net = FullyConnected(
            in_features=in_dim,
            layer_size=hidden_dim,
            out_features=1,
            num_layers=num_layers,
            activation_fn="tanh",
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, x], dim=-1))


def exact_solution(t: torch.Tensor, x: torch.Tensor, speed: float) -> torch.Tensor:
    return torch.sin(x - speed * t)


def make_points(num_points: int, device: torch.device, dtype: torch.dtype):
    t = torch.rand(num_points, 1, device=device, dtype=dtype)
    x = 2.0 * math.pi * torch.rand(num_points, 1, device=device, dtype=dtype)
    return t, x


def residual_loss(model: nn.Module, t: torch.Tensor, x: torch.Tensor, speed: float):
    t = t.detach().clone().requires_grad_(True)
    x = x.detach().clone().requires_grad_(True)
    u = model(t, x)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    return (u_t + speed * u_x).square().mean()


def initial_loss(model: nn.Module, x: torch.Tensor, speed: float):
    t0 = torch.zeros_like(x)
    return (model(t0, x) - exact_solution(t0, x, speed)).square().mean()


def relative_l2(
    model: nn.Module, device: torch.device, dtype: torch.dtype, speed: float
):
    n = 128
    t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype).reshape(-1, 1)
    x = torch.linspace(0.0, 2.0 * math.pi, n, device=device, dtype=dtype).reshape(-1, 1)
    tt, xx = torch.meshgrid(t.squeeze(-1), x.squeeze(-1), indexing="ij")
    tt = tt.reshape(-1, 1)
    xx = xx.reshape(-1, 1)
    with torch.no_grad():
        pred = model(tt, xx)
        truth = exact_solution(tt, xx, speed)
    return (pred - truth).norm() / truth.norm()


def build_model(args, device: torch.device, dtype: torch.dtype) -> nn.Module:
    model = MLP(hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(
        device=device, dtype=dtype
    )
    if args.backend == "ozaki_fp64":
        if dtype is not torch.float64:
            raise ValueError(
                "ozaki_fp64 backend requires --backend ozaki_fp64 with float64 tensors"
            )
        model = convert_linear_to_ozaki(
            model,
            num_moduli=args.num_moduli,
            fastmode=args.fastmode,
            backend=args.ozaki_backend,
            inplace=True,
        )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["native_fp32", "native_fp64", "ozaki_fp64"],
        default="native_fp64",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--num-moduli", type=int, default=15)
    parser.add_argument("--fastmode", action="store_true")
    parser.add_argument(
        "--ozaki-backend",
        choices=["int8", "fp8", "mxfp8", "nvfp4"],
        default="int8",
        help="Low-precision Ozaki residue backend used when --backend ozaki_fp64.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("ozaki_fp64_convection_metrics.json")
    )
    args = parser.parse_args()

    if args.backend == "ozaki_fp64" and not torch.cuda.is_available():
        raise RuntimeError("ozaki_fp64 backend requires CUDA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.backend == "native_fp32" else torch.float64
    torch.manual_seed(args.seed)

    model = build_model(args, device, dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        t_f, x_f = make_points(args.batch_size, device, dtype)
        _, x_i = make_points(args.batch_size, device, dtype)
        loss_pde = residual_loss(model, t_f, x_f, args.speed)
        loss_ic = initial_loss(model, x_i, args.speed)
        loss = loss_pde + loss_ic

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % max(1, args.steps // 20) == 0:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "loss_pde": float(loss_pde.detach().cpu()),
                    "loss_ic": float(loss_ic.detach().cpu()),
                }
            )

    elapsed = time.perf_counter() - start
    rel_l2 = relative_l2(model, device, dtype, args.speed)
    linear_gemm_backend = (
        args.ozaki_backend if args.backend == "ozaki_fp64" else args.backend
    )
    metrics = {
        "backend": args.backend,
        "linear_gemm_backend": linear_gemm_backend,
        "dtype": str(dtype).replace("torch.", ""),
        "steps": args.steps,
        "elapsed_sec": elapsed,
        "step_time_sec": elapsed / args.steps,
        "relative_l2": float(rel_l2.detach().cpu()),
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
