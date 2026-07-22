# Ozaki FP64 PINN

This example trains a one-dimensional convection PINN with a
``physicsnemo.models.mlp.FullyConnected`` model and compares three linear-layer
backends:

- ``native_fp32``
- ``native_fp64``
- ``ozaki_fp64``

The Ozaki backend converts only the ``torch.nn.Linear`` modules inside the
PhysicsNeMo MLP. It replaces these FP64 matrix multiplications:

- forward: ``X @ W.T``
- input gradient: ``grad_Y @ W``
- weight gradient: ``grad_Y.T @ X``

Activations, PDE residual autodiff, loss reductions, optimizer updates, and
other PhysicsNeMo/PyTorch operations remain unchanged.

Run the three variants with:

```bash
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend native_fp32
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend native_fp64
python examples/cfd/ozaki_fp64_pinn/convection_ozaki_fp64.py --backend ozaki_fp64
```

``ozaki_fp64`` requires an NVIDIA CUDA GPU. On first use, PyTorch builds the
bundled CUDA extension. The implementation targets only FP64 Ozaki Scheme II
with GEMMul8; FP128, FP256, and precision-unit decomposition are not part of
this integration.
