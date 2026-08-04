#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CUDA_EXTRA="${1:-cu12}"
case "$CUDA_EXTRA" in
    cu12|cu13) ;;
    *)
        echo "usage: bash reproducibility/setup_environment.sh [cu12|cu13]" >&2
        exit 2
        ;;
esac

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$REPRO_DIR/.." && pwd)"
cd "$REPO_ROOT"

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools uv
uv sync --active --frozen --extra "$CUDA_EXTRA"

python - <<'PY'
import torch
import physicsnemo

print("PhysicsNeMo:", physicsnemo.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
