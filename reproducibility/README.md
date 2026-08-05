# GEMM and PINN Reproducibility

This top-level directory fixes the configurations used by the reported GEMM
and Convection PINN experiments. It is a paper reproduction suite, not a
generic PDE trainer. Every path and command in this document is relative to
the root of this PhysicsNeMo Git repository.

The fixed values live in
`reproducibility/paper_config.toml`. GEMM programs are under
`reproducibility/gemm/`, PINN programs are under `reproducibility/pinn/`, and
generated outputs go under `reproducibility/results/`. The PINN trainer is
self-contained in this repository and uses L-BFGS. See
`reproducibility/PROVENANCE.md` for provenance and scope.

## PINN entry points and trainer

`reproducibility/pinn/run_pinn_150.py` and
`reproducibility/pinn/run_pinn_2000.py` are the user-facing experiment entries.
They call `reproducibility/pinn/run_pinn_suite.py`, which selects the configured
backends, target iteration count, and checkpoint/resume behavior. For each
backend, the suite launches `reproducibility/pinn/train_convection.py`.

`train_convection.py` performs one concrete training run: it builds the fixed
Convection problem and network, creates L-BFGS, computes the losses, advances
the optimizer, evaluates the solution, and writes checkpoints and logs. It is
therefore the training implementation used by both PINN entry points, while
the runner files provide experiment orchestration.

## Fixed configurations

The GEMM comparison contains five FP64 or FP64-emulating backends:

- native FP64
- Ozaki-II INT8
- Ozaki-II FP8
- Ozaki-II MXFP8
- Ozaki-II NVFP4

It runs the four PINN layer shapes `40401x2x512`, `40401x512x512`,
`40401x512x1`, and `512x40401x512` with 15 moduli, seed 0, input scale 0.25,
3 warmups, and 10 measured repetitions. MXFP8/NVFP4 use four CUDA streams,
matching the values in the report table.

The PINN experiment solves

```text
u_t + 50 u_x = 0,  x in [0, 2*pi],  t in [0, 1],
u(x, 0) = sin(x),  with periodic spatial boundary conditions.
```

It uses a `201x201` training grid, `101x101` evaluation grid, hidden width 512,
4 linear layers, `tanh`, seed 0, and L-BFGS with strong-Wolfe line search,
history size 100, and at most 20 inner iterations.

The 150-step timing/memory probe covers six numerical backends. MXFP8 and
NVFP4 are each measured with serial and four-stream schedules, producing eight
rows. The 2000-step convergence validation covers the four configurations
reported in the convergence study: native FP32, native FP64, INT8, and
four-stream NVFP4.

## Environment

Requirements:

- Python 3.11 or newer supported by this checkout
- NVIDIA CUDA GPU
- a CUDA toolkit/compiler compatible with the selected PyTorch build
- enough memory for the chosen backend; the reported 150-step peak reached
  about 13.4 GiB allocated for four-stream NVFP4

From the repository root, create the locked source environment for CUDA 12 or
13:

```bash
bash reproducibility/setup_environment.sh cu12
# or: bash reproducibility/setup_environment.sh cu13
source .venv/bin/activate
```

CUDA 12 is the software line used by the reported experiment: the lock file
selects PyTorch `2.11.0+cu128`, matching the reported Blackwell run. CUDA 13 remains available
for functional reproduction on newer systems, but its timings should not be
compared directly with the reported CUDA 12 results.

The first Ozaki run compiles the bundled CUDA extension. Build products are
placed under the repository's ignored `.cache/` directory.

## Run GEMM

```bash
python reproducibility/gemm/run_gemm.py --device cuda:0
```

Use `--strict` when every backend is required to succeed. Without it, an
unsupported backend is recorded as `unavailable` while the remaining rows
continue. Results are written under
`reproducibility/results/gemm/`.

## Run the 150-step PINN probe

```bash
python reproducibility/pinn/run_pinn_150.py --device cuda:0
```

To run only selected reported variants:

```bash
python reproducibility/pinn/run_pinn_150.py \
  --variants native_fp64,int8,fp8
```

## Run or resume the 2000-step PINN validation

```bash
python reproducibility/pinn/run_pinn_2000.py --device cuda:0
```

The 2000-step entry enables resume mode and launches exactly one
training process with a target of 2000 iterations. If a compatible latest or
150-iteration checkpoint exists, training continues from it, including the
L-BFGS optimizer history. If neither checkpoint exists, the same process trains
continuously from iteration 0 through iteration 2000; it does not stop or
restart at iteration 150.

The 150-step entry always starts a fresh probe. If a completed 2000-step
result already exists, it uses a separate `probe_150` directory so it cannot
overwrite the convergence result.

Each variant writes stage checkpoints, JSON/CSV summaries, and a cumulative
`training_log.npz` under
`reproducibility/results/pinn/<variant>/`. The NPZ contains
per-step losses, 50-step evaluations, PDE loss terms, reference data, and the
reported snapshot iterations. A pre-existing `checkpoint_150.pt` is preserved
when a run is continued to `checkpoint_2000.pt`; `checkpoint_latest.pt`
supports recovery after an interruption. If the latest checkpoint is absent,
the 2000-step trainer can fall back to the preserved 150-step checkpoint.
Generated results are ignored by Git except for the directory's `.gitignore`
file. If a 2000-step result already exists and the 150-step probe is run
again, the probe is written under
`reproducibility/results/pinn/<variant>/probe_150/` so the convergence result
is preserved.

## Plot the 2000-step results

The plotting entry expects all four convergence variants to have completed:

```bash
python reproducibility/pinn/plot_results.py
```

It writes the comparison curves and the four-backend dynamics/snapshot figure
under `reproducibility/results/pinn/`. Matplotlib is installed by
`reproducibility/setup_environment.sh` together with the training
dependencies.

## Inspect commands without running

All suite runners support `--dry-run`:

```bash
python reproducibility/gemm/run_gemm.py --dry-run
python reproducibility/pinn/run_pinn_150.py --dry-run
python reproducibility/pinn/run_pinn_2000.py --dry-run
```

These commands read
`reproducibility/paper_config.toml` and print the exact subprocesses without
allocating GPU memory or starting training.
