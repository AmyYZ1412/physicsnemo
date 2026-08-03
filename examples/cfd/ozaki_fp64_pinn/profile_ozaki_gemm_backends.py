#!/usr/bin/env python3
"""Profile native FP64 GEMM and Ozaki-II GEMM backends.

The script records accuracy, runtime, throughput, and CUDA peak memory for:

- fp64: native ``torch.matmul`` on FP64 tensors.
- int8/fp8/mxfp8/nvfp4: ``ozaki_matmul(..., backend=...)`` on FP64 tensors.

Backend failures are reported as ``unavailable`` unless ``--strict`` is set.
INT8/FP8 use the GEMMul8 extension. MXFP8/NVFP4 require dedicated C++/CUDA
Ozaki-II backends; no Python or alternate backend path is used.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import statistics
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("fp64", "int8", "fp8", "mxfp8", "nvfp4")


def _load_ozaki_matmul() -> Callable[..., torch.Tensor]:
    module_dir = REPO_ROOT / "physicsnemo" / "nn" / "module"
    for name, path in (
        ("physicsnemo", REPO_ROOT / "physicsnemo"),
        ("physicsnemo.nn", REPO_ROOT / "physicsnemo" / "nn"),
        ("physicsnemo.nn.module", module_dir),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules.setdefault(name, package)

    spec = importlib.util.spec_from_file_location(
        "physicsnemo.nn.module.ozaki_linear",
        module_dir / "ozaki_linear.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ozaki_linear.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ozaki_matmul


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"shape must be MxKxN or M,K,N, got {value!r}"
        )
    try:
        m, k, n = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from error
    if m <= 0 or k <= 0 or n <= 0:
        raise argparse.ArgumentTypeError(f"shape values must be positive: {value!r}")
    return m, k, n


def _shape_suite(name: str) -> list[tuple[int, int, int]]:
    square = [(size, size, size) for size in (512, 1024, 2048, 4096)]
    square_large = square + [(size, size, size) for size in (5632, 8192, 12288)]
    paper_int8_accuracy = [
        (1024, size, 1024) for size in (1024, 2048, 4096, 8192, 16384)
    ]
    paper_int8_square = [
        (size, size, size) for size in (1024, 2048, 4096, 8192, 16384)
    ]
    pinn = [
        (40401, 2, 512),
        (40401, 512, 512),
        (40401, 512, 1),
        (512, 40401, 512),
    ]
    tiny = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    suites = {
        "tiny": tiny,
        "square": square,
        "square-large": square_large,
        "paper-int8-accuracy": paper_int8_accuracy,
        "paper-int8-square": paper_int8_square,
        "pinn": pinn,
        "all": tiny + square_large + paper_int8_accuracy + pinn,
    }
    return suites[name]


def _parse_backends(value: str) -> list[str]:
    backends = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [backend for backend in backends if backend not in BACKENDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown backend(s): {unknown}; choose from {', '.join(BACKENDS)}"
        )
    return backends


def _make_inputs(
    shape: tuple[int, int, int],
    *,
    device: torch.device,
    seed: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k, n = shape
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + m * 1000003 + k * 1009 + n)
    a = scale * torch.randn((m, k), dtype=torch.float64, device=device, generator=generator)
    b = scale * torch.randn((k, n), dtype=torch.float64, device=device, generator=generator)
    _sync(device)
    return a, b


def _run_backend(
    backend: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    num_moduli: int,
    fastmode: bool,
    ozaki_matmul: Callable[..., torch.Tensor],
) -> torch.Tensor:
    if backend == "fp64":
        return torch.matmul(a, b)
    return ozaki_matmul(
        a,
        b,
        num_moduli=num_moduli,
        fastmode=fastmode,
        backend=backend,
    )


def _measure(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> dict[str, float | None]:
    for _ in range(warmup):
        out = fn()
        del out
    _sync(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    times_ms: list[float] = []
    for _ in range(repeat):
        _sync(device)
        start = time.perf_counter()
        out = fn()
        _sync(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)
        del out

    stats: dict[str, float | None] = {
        "runtime_ms_mean": statistics.fmean(times_ms),
        "runtime_ms_median": statistics.median(times_ms),
        "runtime_ms_min": min(times_ms),
        "runtime_ms_max": max(times_ms),
    }
    if device.type == "cuda":
        stats["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        stats["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    else:
        stats["peak_allocated_mib"] = None
        stats["peak_reserved_mib"] = None
    return stats


def _accuracy(
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    diff = out - ref
    max_abs = torch.max(torch.abs(diff)).item()
    max_ref = torch.max(torch.abs(ref)).item()
    max_rel = max_abs / max(max_ref, torch.finfo(torch.float64).tiny)
    diff_norm = torch.linalg.vector_norm(diff).item()
    ref_norm = torch.linalg.vector_norm(ref).item()
    return {
        "max_abs_error": float(max_abs),
        "max_rel_error": float(max_rel),
        "relative_fro_error": float(diff_norm / max(ref_norm, torch.finfo(torch.float64).tiny)),
        "allclose": bool(torch.allclose(out, ref, rtol=rtol, atol=atol)),
        "rtol": float(rtol),
        "atol": float(atol),
    }


def _profile_backend(
    backend: str,
    shape: tuple[int, int, int],
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    device: torch.device,
    num_moduli: int,
    fastmode: bool,
    warmup: int,
    repeat: int,
    rtol: float,
    atol: float,
    strict: bool,
    ozaki_matmul: Callable[..., torch.Tensor],
) -> dict[str, Any]:
    m, k, n = shape
    flops = 2.0 * m * k * n
    row: dict[str, Any] = {
        "backend": backend,
        "shape": [m, k, n],
        "num_moduli": num_moduli if backend != "fp64" else None,
        "fastmode": fastmode if backend != "fp64" else None,
        "warmup": warmup,
        "repeat": repeat,
        "status": "ok",
    }

    def fn() -> torch.Tensor:
        return _run_backend(
            backend,
            a,
            b,
            num_moduli=num_moduli,
            fastmode=fastmode,
            ozaki_matmul=ozaki_matmul,
        )

    try:
        timing = _measure(fn, device=device, warmup=warmup, repeat=repeat)
        median_s = timing["runtime_ms_median"] * 1.0e-3
        timing["effective_tflops_median"] = (
            flops / median_s / 1.0e12 if median_s > 0 else None
        )

        ref = torch.matmul(a, b)
        out = fn()
        _sync(device)
        accuracy = _accuracy(out, ref, rtol=rtol, atol=atol)
        del out, ref
        if device.type == "cuda":
            torch.cuda.empty_cache()

        row["timing"] = timing
        row["accuracy"] = accuracy
    except Exception as error:
        if strict:
            raise
        row["status"] = "unavailable"
        row["error_type"] = type(error).__name__
        row["error"] = str(error)
        row["timing"] = None
        row["accuracy"] = None
    return row


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        metadata.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_device_index": index,
                "cuda_capability": [props.major, props.minor],
                "total_memory_mib": total_bytes / 2**20,
                "free_memory_mib_at_start": free_bytes / 2**20,
                "multi_processor_count": props.multi_processor_count,
            }
        )
    return metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "status",
        "shape",
        "num_moduli",
        "fastmode",
        "runtime_ms_median",
        "runtime_ms_mean",
        "effective_tflops_median",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "max_abs_error",
        "max_rel_error",
        "relative_fro_error",
        "allclose",
        "error_type",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            timing = row.get("timing") or {}
            accuracy = row.get("accuracy") or {}
            writer.writerow(
                {
                    "backend": row["backend"],
                    "status": row["status"],
                    "shape": "x".join(str(value) for value in row["shape"]),
                    "num_moduli": row.get("num_moduli"),
                    "fastmode": row.get("fastmode"),
                    "runtime_ms_median": timing.get("runtime_ms_median"),
                    "runtime_ms_mean": timing.get("runtime_ms_mean"),
                    "effective_tflops_median": timing.get("effective_tflops_median"),
                    "peak_allocated_mib": timing.get("peak_allocated_mib"),
                    "peak_reserved_mib": timing.get("peak_reserved_mib"),
                    "max_abs_error": accuracy.get("max_abs_error"),
                    "max_rel_error": accuracy.get("max_rel_error"),
                    "relative_fro_error": accuracy.get("relative_fro_error"),
                    "allclose": accuracy.get("allclose"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backends",
        type=_parse_backends,
        default=list(BACKENDS),
        help="Comma-separated backends. Default: fp64,int8,fp8,mxfp8,nvfp4.",
    )
    parser.add_argument(
        "--suite",
        choices=(
            "tiny",
            "square",
            "square-large",
            "paper-int8-accuracy",
            "paper-int8-square",
            "pinn",
            "all",
        ),
        default="square",
        help="Built-in matrix validation set.",
    )
    parser.add_argument(
        "--shape",
        type=_parse_shape,
        action="append",
        default=[],
        help="Custom shape as MxKxN. May be repeated; overrides --suite.",
    )
    parser.add_argument("--num-moduli", type=int, default=15)
    parser.add_argument("--fastmode", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise on the first backend failure instead of recording it.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/ozaki_gemm_backend_profile.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/ozaki_gemm_backend_profile.csv"),
    )
    args = parser.parse_args()

    if args.repeat <= 0 or args.warmup < 0:
        raise SystemExit("--repeat must be positive and --warmup must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this benchmark on a GPU machine.")

    ozaki_matmul = _load_ozaki_matmul()
    shapes = args.shape if args.shape else _shape_suite(args.suite)
    rows: list[dict[str, Any]] = []

    for shape in shapes:
        a, b = _make_inputs(shape, device=device, seed=args.seed, scale=args.scale)
        for backend in args.backends:
            row = _profile_backend(
                backend,
                shape,
                a=a,
                b=b,
                device=device,
                num_moduli=args.num_moduli,
                fastmode=args.fastmode,
                warmup=args.warmup,
                repeat=args.repeat,
                rtol=args.rtol,
                atol=args.atol,
                strict=args.strict,
                ozaki_matmul=ozaki_matmul,
            )
            rows.append(row)
            timing = row.get("timing") or {}
            accuracy = row.get("accuracy") or {}
            print(
                "gemm_profile,"
                f"backend={backend},"
                f"shape={'x'.join(str(value) for value in shape)},"
                f"status={row['status']},"
                f"median_ms={timing.get('runtime_ms_median')},"
                f"peak_allocated_mib={timing.get('peak_allocated_mib')},"
                f"rel_fro={accuracy.get('relative_fro_error')},"
                f"allclose={accuracy.get('allclose')}"
            )
        del a, b
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "ozaki_gemm_backends",
        "metadata": _device_metadata(device),
        "config": {
            "backends": args.backends,
            "suite": args.suite,
            "custom_shapes": [list(shape) for shape in args.shape],
            "num_moduli": args.num_moduli,
            "fastmode": args.fastmode,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "scale": args.scale,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "rows": rows,
        "notes": [
            "FP64 uses native torch.matmul on FP64 tensors.",
            "INT8, FP8, MXFP8, and NVFP4 use the Ozaki-II interface.",
            "INT8/FP8 use the GEMMul8 extension; MXFP8/NVFP4 require dedicated C++/CUDA backends and do not use a Python fallback.",
            "Runtime excludes first-use extension build time after warmup completes.",
            "CUDA peak memory is torch.cuda.max_memory_allocated/reserved during repeated calls.",
            "Unavailable rows indicate a runtime, build, or backend execution failure.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
