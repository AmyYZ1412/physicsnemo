#!/usr/bin/env python3
"""Probe Ozaki GEMM backend availability on the current CUDA machine."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import types
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ("int8", "fp8", "mxfp8", "nvfp4")


def _load_ozaki_module():
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
    return module


def _device_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        metadata.update(
            {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return metadata


def _probe_backend(module, backend: str) -> dict[str, Any]:
    row: dict[str, Any] = {"backend": backend}
    try:
        canonical = module.canonical_ozaki_backend(backend)
        row["canonical"] = canonical
        if canonical in getattr(module, "_EXECUTABLE_GEMMUL8_BACKENDS", set()):
            row["execution_path"] = "gemmul8_extension"
            row["extension_backend"] = module._extension_backend_name(backend)
        else:
            row["execution_path"] = "unavailable"
        row["selector_gate"] = "ok"
    except Exception as exc:
        row["selector_gate"] = "unavailable"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
        return row

    if not torch.cuda.is_available():
        row["tiny_gemm"] = "skipped"
        row["error_type"] = "RuntimeError"
        row["error"] = "CUDA is not available"
        return row

    try:
        device = torch.device("cuda:0")
        a = torch.randn((16, 16), device=device, dtype=torch.float64)
        b = torch.randn((16, 16), device=device, dtype=torch.float64)
        out = module.ozaki_matmul(a, b, backend=backend, num_moduli=15)
        torch.cuda.synchronize(device)
        ref = torch.matmul(a, b)
        row["tiny_gemm"] = "ok"
        row["max_abs_error"] = float((out - ref).abs().max().item())
        row["allclose"] = bool(torch.allclose(out, ref, rtol=1.0e-10, atol=1.0e-10))
    except Exception as exc:
        row["tiny_gemm"] = "unavailable"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
    return row


def main() -> int:
    module = _load_ozaki_module()
    payload = {
        "metadata": _device_metadata(),
        "rows": [_probe_backend(module, backend) for backend in BACKENDS],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
