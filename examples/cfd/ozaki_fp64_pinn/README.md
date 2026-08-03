# Ozaki FP64 PINN

This example trains a one-dimensional convection PINN with a
``physicsnemo.models.mlp.FullyConnected`` model and compares selectable
linear-layer GEMM backends:

- ``native_fp32``
- ``native_fp64``
- ``ozaki_fp64`` / ``int8``
- ``fp8``
- ``mxfp8``
- ``nvfp4``

The Ozaki backend converts only the ``torch.nn.Linear`` modules inside the
PhysicsNeMo MLP. It replaces these FP64 matrix multiplications:

- forward: ``X @ W.T``
- input gradient: ``grad_Y @ W``
- weight gradient: ``grad_Y.T @ X``

Activations, PDE residual autodiff, loss reductions, optimizer updates, and
other PhysicsNeMo/PyTorch operations remain unchanged.

The PINN training code only selects the linear GEMM backend. Ozaki-II scheme
definitions are kept separately in
``physicsnemo/nn/module/ozaki_schemes.py``. The generic linear-layer adapter and
autograd boundary live in ``physicsnemo/nn/module/ozaki_linear.py``.

Run the six variants with:

```bash
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend native_fp32
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend native_fp64
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend ozaki_fp64 --ozaki-backend int8
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend ozaki_fp64 --ozaki-backend fp8
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend ozaki_fp64 --ozaki-backend mxfp8
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend ozaki_fp64 --ozaki-backend nvfp4
```

``ozaki_fp64`` requires an NVIDIA CUDA GPU. On first use, PyTorch builds the
bundled CUDA extension. INT8 and FP8 use GEMMul8; MXFP8 and NVFP4 use the
dedicated digit-decomposition CUDA backend. FP128 and FP256 are not part of
this integration. Output metrics record the selected implementation in
``linear_gemm_backend``.

## GEMM backend profiling

Before running full PINN training, profile the GEMM kernels directly:

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

For PINN-like matrix shapes:

```bash
python examples/cfd/ozaki_fp64_pinn/profile_ozaki_gemm_backends.py \
  --device cuda:0 \
  --suite pinn
```

Each row records accuracy versus native FP64 GEMM, median/mean/min/max runtime,
effective TFLOP/s, and CUDA peak allocated/reserved memory. INT8 and FP8 use
the GEMMul8 extension. MXFP8 and NVFP4 require dedicated C++/CUDA Ozaki-II
backends; the integration does not use a Python fallback.
