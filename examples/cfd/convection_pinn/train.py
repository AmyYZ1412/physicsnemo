#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Train a configurable 1D Convection PINN.

This is the general-purpose entry point for Convection experiments. It keeps
the PDE, network depth, optimizer, and Ozaki backend selectable from the
command line so users can run their own variants without going through the
paper reproducibility suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BACKEND_CHOICES = ("native", "int8", "fp8", "mxfp8", "nvfp4")


class VanillaPINN(nn.Module):
    """Simple tanh MLP for the Convection PINN."""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        layers: list[nn.Module] = []
        for layer_index in range(num_layers - 1):
            in_features = 2 if layer_index == 0 else hidden_dim
            layers.extend((nn.Linear(in_features, hidden_dim), nn.Tanh()))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, t), dim=-1))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def make_grid(
    x_range: tuple[float, float],
    t_range: tuple[float, float],
    x_num: int,
    t_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(*x_range, x_num)
    t = np.linspace(*t_range, t_num)
    x_mesh, t_mesh = np.meshgrid(x, t)
    points = np.concatenate((x_mesh[..., None], t_mesh[..., None]), axis=-1)
    return x, t, points


def split_points(
    points: np.ndarray, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.tensor(points, dtype=dtype, requires_grad=True, device=device)
    return tensor[..., 0:1], tensor[..., 1:2]


def build_training_tensors(
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    train_x: int,
    train_t: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    _, _, grid = make_grid((x_min, x_max), (t_min, t_max), train_x, train_t)
    return {
        "res": split_points(grid.reshape(-1, 2), dtype, device),
        "initial": split_points(grid[0, :, :], dtype, device),
        "upper": split_points(grid[:, -1, :], dtype, device),
        "lower": split_points(grid[:, 0, :], dtype, device),
    }


def loss_terms(
    model: nn.Module,
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
    beta: float,
) -> dict[str, torch.Tensor]:
    x_res, t_res = tensors["res"]
    x_initial, t_initial = tensors["initial"]
    x_upper, t_upper = tensors["upper"]
    x_lower, t_lower = tensors["lower"]

    pred_res = model(x_res, t_res)
    pred_initial = model(x_initial, t_initial)
    pred_upper = model(x_upper, t_upper)
    pred_lower = model(x_lower, t_lower)
    u_x, u_t = torch.autograd.grad(
        pred_res,
        (x_res, t_res),
        grad_outputs=torch.ones_like(pred_res),
        retain_graph=True,
        create_graph=True,
    )
    residual = torch.mean((u_t + beta * u_x) ** 2)
    boundary = torch.mean((pred_upper - pred_lower) ** 2)
    initial = torch.mean((pred_initial[:, 0] - torch.sin(x_initial[:, 0])) ** 2)
    return {
        "total": residual + boundary + initial,
        "residual": residual,
        "bc": boundary,
        "ic": initial,
    }


def evaluate(
    model: nn.Module,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    eval_x: int,
    eval_t: int,
    beta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    x_grid, t_grid, grid = make_grid((x_min, x_max), (t_min, t_max), eval_x, eval_t)
    points = torch.tensor(grid.reshape(-1, 2), dtype=dtype, device=device)
    model.eval()
    with torch.no_grad():
        prediction = model(points[:, 0:1], points[:, 1:2]).cpu().numpy()
    prediction = prediction.reshape(eval_t, eval_x)
    reference = np.sin(grid[..., 0] - beta * grid[..., 1])
    difference = reference - prediction
    return {
        "relative_l1": float(np.sum(np.abs(difference)) / np.sum(np.abs(reference))),
        "relative_l2": float(np.sqrt(np.sum(difference**2) / np.sum(reference**2))),
        "prediction": prediction,
        "reference": reference,
        "x_grid": x_grid,
        "t_grid": t_grid,
        "plot_extent": np.array([x_min, x_max, t_max, t_min]),
    }


def clear_coordinate_grads(
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for components in tensors.values():
        for component in components:
            component.grad = None


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "git_commit": commit,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
    }


def write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="native")
    parser.add_argument("--digit-streams", type=int, default=1)
    parser.add_argument("--num-moduli", type=int, default=15)
    parser.add_argument("--fastmode", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=2.0 * np.pi)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--train-x", type=int, default=201)
    parser.add_argument("--train-t", type=int, default=201)
    parser.add_argument("--eval-x", type=int, default=101)
    parser.add_argument("--eval-t", type=int, default=101)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-history-size", type=int, default=100)
    parser.add_argument("--tolerance-grad", type=float, default=1.0e-8)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/convection_pinn"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.num_layers < 2:
        raise SystemExit("--num-layers must be at least 2")
    if args.train_x < 2 or args.train_t < 2:
        raise SystemExit("--train-x and --train-t must be at least 2")
    if args.eval_x < 2 or args.eval_t < 2:
        raise SystemExit("--eval-x and --eval-t must be at least 2")

    if args.backend != "native" and args.digit_streams < 1:
        raise SystemExit("--digit-streams must be positive for Ozaki backends")

    if args.dry_run:
        print(
            "python",
            Path(__file__).name,
            "--backend",
            args.backend,
            "--steps",
            args.steps,
            "--hidden-dim",
            args.hidden_dim,
            "--num-layers",
            args.num_layers,
            "--output-dir",
            args.output_dir,
        )
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an NVIDIA CUDA GPU")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)

    set_seed(args.seed)
    dtype = torch.float64
    model = VanillaPINN(args.hidden_dim, args.num_layers).to(device=device, dtype=dtype)
    if args.backend != "native":
        from physicsnemo.nn import convert_linear_to_ozaki

        os.environ["OZAKI_DIGIT_NUM_STREAMS"] = str(args.digit_streams)
        os.environ.setdefault(
            "OZAKI_LINEAR_BUILD_DIR",
            str(REPO_ROOT / ".cache/ozaki_linear_ext"),
        )
        model = convert_linear_to_ozaki(
            model,
            backend=args.backend,
            num_moduli=args.num_moduli,
            fastmode=args.fastmode,
            inplace=True,
        )

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        line_search_fn="strong_wolfe",
        max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history_size,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
    )
    tensors = build_training_tensors(
        args.x_min,
        args.x_max,
        args.t_min,
        args.t_max,
        args.train_x,
        args.train_t,
        dtype,
        device,
    )

    output_dir = args.output_dir
    latest_path = output_dir / "checkpoint_latest.pt"
    stage_path = output_dir / f"checkpoint_{args.steps}.pt"
    log_path = output_dir / "training_log.npz"
    metrics_path = output_dir / f"metrics_{args.steps}.json"
    start_iteration = 0
    elapsed_before = 0.0
    loss_iters: list[int] = []
    loss_history: list[float] = []
    term_history: dict[str, list[float]] = {}
    eval_iters: list[int] = []
    l1_history: list[float] = []
    l2_history: list[float] = []
    snapshot_iters: list[int] = []
    snapshots: list[np.ndarray] = []

    if args.resume and latest_path.exists():
        checkpoint = load_checkpoint(latest_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = int(checkpoint["iteration"])
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        loss_iters = checkpoint.get("loss_iters", [])
        loss_history = checkpoint.get("loss_history", [])
        term_history = checkpoint.get("term_history", {})
        eval_iters = checkpoint.get("eval_iters", [])
        l1_history = checkpoint.get("l1_history", [])
        l2_history = checkpoint.get("l2_history", [])
        snapshot_iters = checkpoint.get("snapshot_iters", [])
        snapshots = checkpoint.get("snapshots", [])
        print(
            f"resumed from {latest_path.name} at iteration {start_iteration}",
            flush=True,
        )

    if start_iteration >= args.steps:
        print(f"already reached iteration {start_iteration}; nothing to do", flush=True)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    training_started_at = time.perf_counter()
    last_terms: dict[str, float] = {}

    def record_eval(iteration: int, keep_snapshot: bool) -> dict[str, Any]:
        result = evaluate(
            model,
            args.x_min,
            args.x_max,
            args.t_min,
            args.t_max,
            args.eval_x,
            args.eval_t,
            args.beta,
            dtype,
            device,
        )
        if iteration not in eval_iters:
            eval_iters.append(iteration)
            l1_history.append(result["relative_l1"])
            l2_history.append(result["relative_l2"])
        if keep_snapshot and iteration not in snapshot_iters:
            snapshot_iters.append(iteration)
            snapshots.append(result["prediction"])
        return result

    if not eval_iters:
        record_eval(0, keep_snapshot=True)

    def save_progress(iteration: int, result: dict[str, Any]) -> None:
        elapsed = elapsed_before + time.perf_counter() - training_started_at
        payload = {
            "experiment": {
                "backend": args.backend,
                "num_moduli": args.num_moduli,
                "digit_streams": args.digit_streams,
                "beta": args.beta,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "steps": args.steps,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "elapsed_seconds": elapsed,
            "loss_iters": loss_iters,
            "loss_history": loss_history,
            "term_history": term_history,
            "eval_iters": eval_iters,
            "l1_history": l1_history,
            "l2_history": l2_history,
            "snapshot_iters": snapshot_iters,
            "snapshots": snapshots,
        }
        atomic_torch_save(payload, latest_path)
        np.savez_compressed(
            log_path,
            beta=args.beta,
            backend=args.backend,
            dtype=str(dtype).replace("torch.", ""),
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            digit_streams=args.digit_streams,
            num_moduli=args.num_moduli,
            fastmode=args.fastmode,
            max_steps=args.steps,
            completed_iter=iteration,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
            train_x=args.train_x,
            train_t=args.train_t,
            train_loss=np.asarray(loss_history),
            loss_iters=np.asarray(loss_iters),
            eval_iters=np.asarray(eval_iters),
            relative_l1=np.asarray(l1_history),
            relative_l2=np.asarray(l2_history),
            snapshot_iters=np.asarray(snapshot_iters),
            snapshots=np.asarray(snapshots),
            reference=result["reference"],
            x_grid=result["x_grid"],
            t_grid=result["t_grid"],
            plot_extent=result["plot_extent"],
            elapsed_seconds=elapsed,
            seconds_per_iter=elapsed / iteration if iteration else np.nan,
            checkpoint_path=str(latest_path),
            **{
                f"term_{name}": np.asarray(values)
                for name, values in term_history.items()
            },
        )

    process_started_at = time.perf_counter()
    for iteration in range(start_iteration + 1, args.steps + 1):
        model.train()

        def closure() -> torch.Tensor:
            nonlocal last_terms
            optimizer.zero_grad()
            clear_coordinate_grads(tensors)
            terms = loss_terms(model, tensors, args.beta)
            terms["total"].backward()
            last_terms = {name: float(value.detach()) for name, value in terms.items()}
            return terms["total"]

        optimizer.step(closure)
        loss_iters.append(iteration)
        loss_history.append(last_terms["total"])
        for name, value in last_terms.items():
            term_history.setdefault(name, []).append(value)

        should_evaluate = (
            iteration % args.eval_every == 0
            or iteration == args.steps
            or iteration in {0, args.steps}
        )
        result = None
        if should_evaluate:
            result = record_eval(iteration, keep_snapshot=iteration in {0, args.steps})
            print(
                f"{args.backend}: {iteration}/{args.steps} "
                f"loss={last_terms['total']:.6e} "
                f"l2={result['relative_l2']:.6e}",
                flush=True,
            )
        if iteration % args.checkpoint_every == 0 and iteration < args.steps:
            if result is None:
                result = record_eval(iteration, keep_snapshot=False)
            save_progress(iteration, result)

    torch.cuda.synchronize(device)
    final_result = evaluate(
        model,
        args.x_min,
        args.x_max,
        args.t_min,
        args.t_max,
        args.eval_x,
        args.eval_t,
        args.beta,
        dtype,
        device,
    )
    save_progress(args.steps, final_result)
    shutil.copy2(latest_path, stage_path)
    elapsed = elapsed_before + time.perf_counter() - training_started_at
    process_wall = time.perf_counter() - process_started_at
    summary_row = {
        "backend": args.backend,
        "digit_streams": args.digit_streams,
        "completed_steps": args.steps,
        "final_relative_l1": final_result["relative_l1"],
        "final_relative_l2": final_result["relative_l2"],
        "final_train_loss": loss_history[-1],
        "elapsed_seconds": elapsed,
        "seconds_per_iteration": elapsed / args.steps,
        "process_wall_seconds": process_wall,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    metrics = {
        "schema_version": 1,
        "environment": environment_metadata(device),
        **summary_row,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_summary_csv(output_dir / "summary.csv", summary_row)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
