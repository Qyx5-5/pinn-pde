# PINN PDE

A compact PyTorch toolkit for building physics-informed neural network experiments for partial differential equations.

## Features

- Small generic core for neural models, autograd derivatives, PDE residual losses, and training loops
- Equation-specific code kept in examples instead of the package API
- Lightweight examples for 1D Schrodinger and viscous Burgers equations
- Minimal test suite for import, differentiation, training, and example smoke checks

## Install

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from pinn_pde import MLP, PDEProblem, ResidualLoss, CompositeLoss, Trainer, laplacian
from pinn_pde.training import sample_grid_1d

def poisson(points, model):
    u = model(points)
    return laplacian(u, points) + torch.sin(torch.pi * points)

model = MLP(input_dim=1, hidden_layers=(32, 32))
problem = PDEProblem(residual=poisson)
loss = CompositeLoss({"residual": ResidualLoss(problem)})

trainer = Trainer(
    model=model,
    loss_fn=loss,
    batch_fn=lambda: {"residual": sample_grid_1d((0.0, 1.0), 64)},
)
trainer.fit(steps=100)
```

## Layout

```text
src/pinn_pde/     generic PINN models, operators, losses, and training
examples/         concrete PDE cases such as Schrodinger and Burgers
tests/            lightweight correctness and smoke tests
```

## Canonical Figures

These examples show how a PINN uses neural-network derivatives to satisfy a PDE residual plus boundary/initial constraints.

### Poisson: residual learning with a known answer

![Poisson PINN case](docs/figures/poisson.png)

The left panel compares the PINN solution with the exact solution. The right panel shows pointwise error. This is the simplest check that the residual and boundary losses are wired correctly.

### Heat: time-dependent diffusion

![Heat PINN case](docs/figures/heat.png)

The solution should smooth/decay over time. The right panel shows the optimization trace, so new runs should be judged by both the curve shape and whether the loss actually converges.

### Schrodinger: eigenstate with physical reference

![Schrodinger PINN case](docs/figures/schrodinger_harmonic.png)

The learned wavefunction is compared with a finite-difference reference. The energy and probability-density panels are the main physical checks; a visually smooth wavefunction is not enough.

## Run Examples

```bash
python examples/schrodinger_1d.py
python examples/burgers_1d.py
python examples/canonical_cases.py poisson --steps 500
python examples/canonical_cases.py heat --steps 500
python examples/canonical_cases.py advection_diffusion --steps 500
python examples/canonical_cases.py schrodinger --potential double_well --steps 500
```

Generated figures are saved under `outputs/`, which is ignored by Git. Use `--formats png pdf svg` to choose formats. For report use, increase `--steps`, keep `--seed` fixed, and verify error/loss/reference diagnostics.

## Tests

```bash
pytest
```

## Adding A PDE

Define the residual in an example file, construct a `PDEProblem`, attach residual, boundary, or initial losses, and train with `Trainer`. Keep reusable mechanics in `src/pinn_pde`; keep equation constants, potentials, initial data, and plots in `examples`.
