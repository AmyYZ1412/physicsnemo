# Convection PINN Example

This is the general-purpose Convection PINN entry point for custom runs.
It is separate from `reproducibility/`, which keeps the fixed paper
configuration only.

Run from the repository root:

```bash
python examples/cfd/convection_pinn/train.py \
  --backend nvfp4 \
  --digit-streams 4 \
  --steps 2000 \
  --device cuda:0 \
  --output-dir outputs/convection_nvfp4
```

Common customizations:

```bash
python examples/cfd/convection_pinn/train.py --help
```

The main knobs are `--hidden-dim`, `--num-layers`, `--steps`, `--beta`, and
`--backend`. Use `--resume` to continue from `checkpoint_latest.pt` in the
chosen output directory.
