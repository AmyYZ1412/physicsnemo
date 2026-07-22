# GEMMul8 Sources

This directory contains the CUDA and header sources required by
``physicsnemo.nn.OzakiLinear``.

The GEMMul8 headers are vendored from the RIKEN R-CCS GEMMul8 project and are
distributed under the MIT license in ``LICENSE``. The CUDA binding files in
this directory implement the PhysicsNeMo Ozaki Scheme II FP64 linear-layer
integration.

The extension is built lazily by PyTorch on the first CUDA use. Set
``GEMMUL8_HOME`` to an alternate GEMMul8 source directory only when replacing
the bundled sources intentionally.
