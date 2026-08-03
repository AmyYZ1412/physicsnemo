# Ozaki GEMM Backend Profile Runbook

This runbook is for a fresh GPU machine used to benchmark native FP64 GEMM
against Ozaki-II GEMM backends.

## Backends

- `fp64`: native `torch.matmul` on FP64 tensors.
- `int8`: Ozaki-II through the current GEMMul8 INT8 path.
- `fp8`: Ozaki-II through the current GEMMul8 FP8 path.
- `mxfp8`: Ozaki-II MXFP8 interface; requires a dedicated CUDA backend.
- `nvfp4`: Ozaki-II NVFP4 interface; requires a dedicated CUDA backend.

If the dedicated MXFP8 or NVFP4 backend is absent, runtime failures are recorded
as `unavailable` by default; use `--strict` when every selected backend is
expected to run successfully.

## Fresh Machine Setup

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install uv
uv sync --extra cu13
```

If `uv` is not available on the machine, use pip directly:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -e ".[cu13]"
```

The first Ozaki backend run builds the bundled CUDA extension. These cache
paths keep build artifacts local to the checkout:

```bash
export XDG_CACHE_HOME="$PWD/.cache"
export WARP_CACHE_DIR="$PWD/.cache/warp"
export OZAKI_LINEAR_BUILD_DIR="$PWD/.cache/ozaki_linear_ext"
```

`TORCH_CUDA_ARCH_LIST` is detected from the active CUDA device by default.

## Smoke Test

Use a tiny shape first to verify the environment and extension build:

```bash
python examples/cfd/ozaki_fp64_pinn/probe_ozaki_backend_capabilities.py

python examples/cfd/ozaki_fp64_pinn/profile_ozaki_gemm_backends.py \
  --device cuda:0 \
  --backends fp64,int8,fp8,mxfp8,nvfp4 \
  --shape 64x64x64 \
  --warmup 1 \
  --repeat 2 \
  --output-json results/ozaki_gemm_backend_smoke.json \
  --output-csv results/ozaki_gemm_backend_smoke.csv
```

## Square GEMM Profile

This is the main GEMM benchmark suite:

```bash
python examples/cfd/ozaki_fp64_pinn/profile_ozaki_gemm_backends.py \
  --device cuda:0 \
  --backends fp64,int8,fp8,mxfp8,nvfp4 \
  --suite square-large \
  --warmup 3 \
  --repeat 10 \
  --output-json results/ozaki_gemm_backend_profile.json \
  --output-csv results/ozaki_gemm_backend_profile.csv
```

For INT8 Ozaki-II paper-aligned shapes, use:

```bash
python examples/cfd/ozaki_fp64_pinn/profile_ozaki_gemm_backends.py \
  --device cuda:0 \
  --backends fp64,int8,fp8,mxfp8,nvfp4 \
  --suite paper-int8-square \
  --warmup 30 \
  --repeat 30 \
  --output-json results/ozaki_gemm_backend_profile_paper_int8_square.json \
  --output-csv results/ozaki_gemm_backend_profile_paper_int8_square.csv
```

The paper accuracy-shape sweep is available as `--suite paper-int8-accuracy`.

## PINN-Like GEMM Shapes

Run this after the square suite if the machine has enough time:

```bash
python examples/cfd/ozaki_fp64_pinn/profile_ozaki_gemm_backends.py \
  --device cuda:0 \
  --backends fp64,int8,fp8,mxfp8,nvfp4 \
  --suite pinn \
  --warmup 3 \
  --repeat 10 \
  --output-json results/ozaki_gemm_backend_profile_pinn.json \
  --output-csv results/ozaki_gemm_backend_profile_pinn.csv
```

## Recorded Metrics

Each `backend + shape` row records:

- Accuracy: `max_abs_error`, `max_rel_error`, `relative_fro_error`, `allclose`.
- Runtime: mean, median, min, and max milliseconds.
- Speed: median effective TFLOP/s computed from `2*M*K*N`.
- Memory: CUDA peak allocated and reserved MiB.
- Environment: GPU name, CUDA capability, PyTorch version, CUDA version, and
  benchmark config.
