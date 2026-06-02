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

## Examples

```bash
python examples/schrodinger_1d.py
python examples/burgers_1d.py
python examples/canonical_cases.py poisson --steps 500
python examples/canonical_cases.py heat --steps 500
python examples/canonical_cases.py advection_diffusion --steps 500
python examples/canonical_cases.py schrodinger --potential double_well --steps 500
```

The canonical cases save publication-style PNG/PDF figures under `outputs/`, which is ignored by Git. Use `--formats png pdf svg` to choose output formats. For PINN figures intended for reports, increase `--steps`, keep `--seed` fixed, and verify the loss/error panels before using the figure.

Canonical figures are designed as field-entry case studies:

- Poisson: elliptic residual learning with an exact solution and pointwise error.
- Heat: parabolic time evolution with exact sine-mode decay and optimization trace.
- Advection-diffusion: transport plus smoothing from a localized pulse.
- Schrodinger: stationary eigenproblem with normalization and a finite-difference reference for energy, wavefunction scale, and probability density.

## Tests

```bash
pytest
```

## Adding A PDE

Define the residual in an example file, construct a `PDEProblem`, attach residual, boundary, or initial losses, and train with `Trainer`. Keep reusable mechanics in `src/pinn_pde`; keep equation constants, potentials, initial data, and plots in `examples`.
