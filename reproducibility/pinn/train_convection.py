#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run one fixed Convection PINN backend with checkpoint/resume.

This is the in-repository training implementation for the reported
Convection1D problem, VanillaPINN model, and selectable linear backend. It
keeps the numerical and logging behavior needed by the reported 150- and
2000-iteration experiments without importing files outside this repository.
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

HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
REPO_ROOT = REPRO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPRO_ROOT))

import numpy as np  # noqa: E402
import physicsnemo  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from config import CONFIG_PATH, VARIANTS, load_config  # noqa: E402
from physicsnemo.nn import convert_linear_to_ozaki  # noqa: E402


class VanillaPINN(nn.Module):
    """Tanh MLP used by the current project Convection experiments."""

    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must include a hidden and output layer")
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
    cfg: dict[str, Any], dtype: torch.dtype, device: torch.device
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    _, _, grid = make_grid(
        (cfg["x_min"], cfg["x_max"]),
        (cfg["t_min"], cfg["t_max"]),
        cfg["train_x"],
        cfg["train_t"],
    )
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
    cfg: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    x_grid, t_grid, grid = make_grid(
        (cfg["x_min"], cfg["x_max"]),
        (cfg["t_min"], cfg["t_max"]),
        cfg["eval_x"],
        cfg["eval_t"],
    )
    points = torch.tensor(grid.reshape(-1, 2), dtype=dtype, device=device)
    model.eval()
    with torch.no_grad():
        prediction = model(points[:, 0:1], points[:, 1:2]).cpu().numpy()
    prediction = prediction.reshape(cfg["eval_t"], cfg["eval_x"])
    reference = np.sin(grid[..., 0] - cfg["beta"] * grid[..., 1])
    difference = reference - prediction
    return {
        "relative_l1": float(np.sum(np.abs(difference)) / np.sum(np.abs(reference))),
        "relative_l2": float(
            np.sqrt(np.sum(difference**2) / np.sum(reference**2))
        ),
        "prediction": prediction,
        "reference": reference,
        "x_grid": x_grid,
        "t_grid": t_grid,
        "plot_extent": np.array(
            [cfg["x_min"], cfg["x_max"], cfg["t_max"], cfg["t_min"]]
        ),
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
        "physicsnemo": physicsnemo.__version__,
        "git_commit": commit,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--target-steps", type=int, choices=(150, 2000), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    process_started_at = time.perf_counter()
    config = load_config()
    cfg = config["pinn"]
    stage_name = "probe" if args.target_steps == 150 else "convergence"
    stage_cfg = cfg["stages"][stage_name]
    if args.target_steps != stage_cfg["steps"]:
        raise RuntimeError("target does not match paper_config.toml")
    snapshot_steps = set(stage_cfg["snapshot_steps"])
    variant = VARIANTS[args.variant]
    os.environ["OZAKI_DIGIT_NUM_STREAMS"] = str(variant.digit_streams)
    os.environ.setdefault(
        "OZAKI_LINEAR_BUILD_DIR", str(REPO_ROOT / ".cache/ozaki_linear_ext")
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the fixed PINN suite requires an NVIDIA CUDA GPU")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    dtype = torch.float32 if variant.dtype == "fp32" else torch.float64
    if dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    set_seed(cfg["seed"])
    model = VanillaPINN(cfg["hidden_dim"], cfg["num_layers"]).to(
        device=device, dtype=dtype
    )
    if variant.backend != "native":
        model = convert_linear_to_ozaki(
            model,
            backend=variant.backend,
            num_moduli=cfg["num_moduli"],
            fastmode=cfg["fastmode"],
            inplace=True,
        )
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        line_search_fn="strong_wolfe",
        max_iter=cfg["lbfgs_max_iter"],
        history_size=cfg["lbfgs_history_size"],
        tolerance_grad=cfg["tolerance_grad"],
        tolerance_change=cfg["tolerance_change"],
    )
    tensors = build_training_tensors(cfg, dtype, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = args.output_dir / "checkpoint_latest.pt"
    stage_path = args.output_dir / f"checkpoint_{args.target_steps}.pt"
    log_path = args.output_dir / "training_log.npz"
    metrics_path = args.output_dir / f"metrics_{args.target_steps}.json"
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

    resume_path = latest_path
    if args.resume and not resume_path.exists() and args.target_steps == 2000:
        resume_path = args.output_dir / "checkpoint_150.pt"
    if args.resume and resume_path.exists():
        checkpoint = load_checkpoint(resume_path, device)
        expected = {"variant": args.variant, "pinn_config": cfg}
        if checkpoint.get("experiment") != expected:
            raise RuntimeError("checkpoint does not match paper_config.toml")
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
            f"resumed {args.variant} from {resume_path.name} "
            f"at iteration {start_iteration}",
            flush=True,
        )

    if start_iteration >= args.target_steps:
        print(
            f"{args.variant} already reached iteration {start_iteration}; nothing to do",
            flush=True,
        )
        return 0

    def record_eval(iteration: int, keep_snapshot: bool) -> dict[str, Any]:
        result = evaluate(model, cfg, dtype, device)
        if iteration not in eval_iters:
            eval_iters.append(iteration)
            l1_history.append(result["relative_l1"])
            l2_history.append(result["relative_l2"])
        if keep_snapshot and iteration not in snapshot_iters:
            snapshot_iters.append(iteration)
            snapshots.append(result["prediction"])
        return result

    if not eval_iters:
        record_eval(0, keep_snapshot=0 in snapshot_steps)

    training_started_at = time.perf_counter()
    last_terms: dict[str, float] = {}

    def save_progress(iteration: int, result: dict[str, Any]) -> None:
        elapsed = elapsed_before + time.perf_counter() - training_started_at
        payload = {
            "experiment": {"variant": args.variant, "pinn_config": cfg},
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
            pde=cfg["equation"],
            pde_config=json.dumps({"beta": cfg["beta"]}, sort_keys=True),
            dtype=variant.dtype,
            seed=cfg["seed"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            backend=variant.backend,
            variant=args.variant,
            digit_streams=variant.digit_streams,
            ozaki_num_moduli=cfg["num_moduli"],
            ozaki_fastmode=cfg["fastmode"],
            optimizer=cfg["optimizer"],
            max_iter=args.target_steps,
            completed_iter=iteration,
            eval_every=cfg["eval_every"],
            checkpoint_every=stage_cfg["checkpoint_every"],
            train_x=cfg["train_x"],
            train_t=cfg["train_t"],
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

    for iteration in range(start_iteration + 1, args.target_steps + 1):
        model.train()

        def closure() -> torch.Tensor:
            nonlocal last_terms
            optimizer.zero_grad()
            clear_coordinate_grads(tensors)
            terms = loss_terms(model, tensors, cfg["beta"])
            terms["total"].backward()
            last_terms = {name: float(value.detach()) for name, value in terms.items()}
            return terms["total"]

        optimizer.step(closure)
        loss_iters.append(iteration)
        loss_history.append(last_terms["total"])
        for name, value in last_terms.items():
            term_history.setdefault(name, []).append(value)

        should_evaluate = (
            iteration % cfg["eval_every"] == 0
            or iteration in snapshot_steps
            or iteration == args.target_steps
        )
        result = None
        if should_evaluate:
            result = record_eval(iteration, keep_snapshot=iteration in snapshot_steps)
            print(
                f"{args.variant}: {iteration}/{args.target_steps} "
                f"loss={last_terms['total']:.6e} "
                f"l2={result['relative_l2']:.6e}",
                flush=True,
            )
        if (
            iteration % stage_cfg["checkpoint_every"] == 0
            and iteration < args.target_steps
        ):
            if result is None:
                result = record_eval(iteration, keep_snapshot=False)
            save_progress(iteration, result)

    torch.cuda.synchronize(device)
    final_result = evaluate(model, cfg, dtype, device)
    save_progress(args.target_steps, final_result)
    shutil.copy2(latest_path, stage_path)
    elapsed = elapsed_before + time.perf_counter() - training_started_at
    process_wall = time.perf_counter() - process_started_at
    summary_row = {
        "variant": args.variant,
        "backend": variant.backend,
        "dtype": variant.dtype,
        "digit_streams": variant.digit_streams,
        "completed_steps": args.target_steps,
        "final_relative_l1": final_result["relative_l1"],
        "final_relative_l2": final_result["relative_l2"],
        "final_train_loss": loss_history[-1],
        "elapsed_seconds": elapsed,
        "seconds_per_iteration": elapsed / args.target_steps,
        "process_wall_seconds": process_wall,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    metrics = {
        "schema_version": 1,
        "paper_config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "environment": environment_metadata(device),
        **summary_row,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_summary_csv(args.output_dir / "summary.csv", summary_row)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
