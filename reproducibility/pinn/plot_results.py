#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the reported four-backend convergence figures."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/physicsnemo_ozaki_matplotlib")

HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
BACKENDS = (
    ("native_fp64", "FP64", "blue"),
    ("native_fp32", "FP32", "green"),
    ("int8", "INT8 Ozaki-II", "darkorange"),
    ("nvfp4_streams4", "NVFP4 Ozaki-II (4 streams)", "purple"),
)
SNAPSHOT_ITERS = (0, 500, 1000, 2000)


def load_log(root: Path, backend: str) -> np.lib.npyio.NpzFile:
    import numpy as np

    path = root / backend / "training_log.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; complete "
            "reproducibility/pinn/run_pinn_2000.py before plotting"
        )
    data = np.load(path, allow_pickle=False)
    completed = int(data["completed_iter"])
    if completed < 2000:
        data.close()
        raise RuntimeError(f"{backend} stopped at iteration {completed}, expected 2000")
    return data


def plot_curves(root: Path, logs: dict[str, np.lib.npyio.NpzFile]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for backend, label, color in BACKENDS:
        data = logs[backend]
        axes[0].semilogy(
            data["eval_iters"],
            data["relative_l2"],
            label=label,
            color=color,
            linewidth=1.7,
        )
        axes[1].semilogy(
            data["loss_iters"],
            data["train_loss"],
            label=label,
            color=color,
            linewidth=1.7,
        )
    axes[0].set(
        title="Relative L2 vs iteration", xlabel="iteration", ylabel="relative L2"
    )
    axes[1].set(
        title="Training loss vs iteration", xlabel="iteration", ylabel="train loss"
    )
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.savefig(root / "l2_and_loss_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_snapshots(root: Path, logs: dict[str, np.lib.npyio.NpzFile]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(
        len(BACKENDS),
        len(SNAPSHOT_ITERS),
        figsize=(14, 11),
        constrained_layout=True,
    )
    for row, (backend, label, _) in enumerate(BACKENDS):
        data = logs[backend]
        lookup = {
            int(iteration): index
            for index, iteration in enumerate(data["snapshot_iters"])
        }
        missing = set(SNAPSHOT_ITERS) - set(lookup)
        if missing:
            raise RuntimeError(f"{backend} is missing snapshots {sorted(missing)}")
        extent = np.asarray(data["plot_extent"], dtype=float)
        for column, iteration in enumerate(SNAPSHOT_ITERS):
            axis = axes[row, column]
            field = np.asarray(data["snapshots"][lookup[iteration]])
            image = axis.imshow(
                field,
                extent=extent,
                aspect="auto",
                origin="upper",
                cmap="viridis",
                vmin=-1.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            axis.set_title(f"{label}\niteration {iteration}")
            axis.set_xlabel("x")
            axis.set_ylabel("t")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(root / "dynamics_snapshots.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=REPRO_ROOT / "results/pinn"
    )
    args = parser.parse_args()
    root = args.input_dir.resolve()
    logs = {backend: load_log(root, backend) for backend, _, _ in BACKENDS}
    try:
        try:
            plot_curves(root, logs)
            plot_snapshots(root, logs)
        except ModuleNotFoundError as error:
            if error.name and error.name.startswith("matplotlib"):
                parser.error(
                    "plotting requires Matplotlib; install it with "
                    "`uv pip install matplotlib`"
                )
            raise
    finally:
        for data in logs.values():
            data.close()
    print(f"wrote figures under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
