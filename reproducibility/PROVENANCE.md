# Reproduction Provenance

All paths in this document are relative to the root of this PhysicsNeMo Git
repository. The complete reproduction suite is under
`reproducibility/`; it does not depend on files outside the
repository or on archived code.

## Scope

The suite reproduces the reported experiments only:

- five-backend GEMM measurements using native FP64 and the INT8, FP8, MXFP8,
  and NVFP4 Ozaki-II implementations;
- the fixed one-dimensional Convection PINN at 150 and 2000 outer iterations;
- checkpoint/resume, metrics, logs, and comparison plots for those runs.

`reproducibility/paper_config.toml` fixes the paper parameters.
`reproducibility/pinn/train_convection.py` contains
the equation-specific residual, initial condition, periodic boundary loss,
network, L-BFGS closure, and checkpoint handling. The two `run_pinn_*.py`
entries in the same directory select the reported backend variants, while
`reproducibility/gemm/run_gemm.py` invokes the colocated
`reproducibility/gemm/profile_gemm_backends.py` with the four fixed PINN matrix
shapes.

## Artifact-verified configuration

The retained experiment checkpoints and cumulative logs were used to verify:

- equation `convection` with `beta=50`;
- a 201x201 training grid and 101x101 evaluation grid;
- hidden width 512, four linear layers, tanh activations, and seed 0;
- L-BFGS with strong-Wolfe line search, history size 100, at most 20 inner
  iterations, gradient tolerance `1e-8`, and change tolerance `1e-10`;
- evaluation every 50 outer iterations;
- a 150-iteration checkpoint with snapshots at 0 and 150;
- continuation to 2000 iterations with snapshots at 0, 150, 500, 1000, 1500,
  and 2000, including restored L-BFGS optimizer state;
- 15 Ozaki moduli with fast mode disabled.

The GEMM artifacts record the four shapes in
`reproducibility/paper_config.toml`, five backends, 3
warmups, 10 measured repetitions, seed 0, input scale 0.25, and `1e-10`
relative and absolute tolerances.

## Convection origin

Convection is required by the reported PINN experiment, but it was not a concrete PDE
shipped in the original NVIDIA PhysicsNeMo v2.0.0 checkout. This project first
added a project-specific example directory and its hard-coded Convection/Adam
demo in commit `62854e3`. The demo did not match the reported grid, network,
boundary loss, or optimizer and has been removed. The fixed implementation in
`reproducibility/pinn/train_convection.py` replaces it and uses
L-BFGS.

The Ozaki linear conversion is not Convection-specific. INT8, FP8, MXFP8, and
NVFP4 identify distinct backends explicitly in
`reproducibility/paper_config.toml` and the result metadata;
the directory name deliberately does not collapse them into an ambiguous
"Ozaki FP64" label.

## Official PDE inventory

PhysicsNeMo's `physicsnemo/sym/eq/` package supplies the `PDE` base class,
gradient utilities, and `PhysicsInformer`; it does not supply a registry of
ready-made equations. Concrete PDE definitions are embedded in separate
upstream examples. In this checkout they include:

- incompressible Navier-Stokes in `examples/cfd/ldc_pinns/`,
  `examples/cfd/inverse_pinns/`, and `examples/cfd/datacenter/`;
- advection-diffusion in `examples/cfd/inverse_pinns/`;
- diffusion/Darcy in `examples/cfd/darcy_physics_informed/`;
- Stokes in `examples/cfd/stokes_mgn/`;
- nonlinear shallow-water equations in `examples/cfd/swe_nonlinear_pino/`;
- magnetohydrodynamics equations in `examples/cfd/mhd_pino/`;
- additional incompressible-flow residuals in the external-aerodynamics
  training examples.

The core package documentation also uses Poisson as a minimal API
illustration, but that is not a selectable equation catalog. The original
checkout does not contain concrete Convection, Reaction, Helmholtz, or
Allen-Cahn trainers. PhysicsNeMo examples define their own geometry, residuals,
constraints, data, and optimization loop rather than selecting arbitrary PDEs
through one common command.

## Optimizer boundary

Every project-owned PINN entry in `reproducibility/pinn/` uses
L-BFGS. Upstream PhysicsNeMo examples that intentionally use Adam are outside
this suite and are unchanged.
