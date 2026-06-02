from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


TensorFn = Callable[[torch.Tensor, torch.nn.Module], torch.Tensor]


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return dy/dx for batched scalar outputs."""
    if not y.requires_grad:
        return torch.zeros_like(x)
    grad = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    return torch.zeros_like(x) if grad is None else grad


def derivative(y: torch.Tensor, x: torch.Tensor, component: int = 0, order: int = 1) -> torch.Tensor:
    """Return a selected partial derivative of a scalar field."""
    value = y
    for _ in range(order):
        if not value.requires_grad:
            return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        value = gradient(value, x)[:, component : component + 1]
    return value


def jacobian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return gradients for each output component as (batch, output, input)."""
    parts = []
    for idx in range(y.shape[1]):
        parts.append(gradient(y[:, idx : idx + 1], x))
    return torch.stack(parts, dim=1)


def laplacian(y: torch.Tensor, x: torch.Tensor, components: tuple[int, ...] | None = None) -> torch.Tensor:
    """Return sum_i d2y/dx_i2 for selected coordinate components."""
    if components is None:
        components = tuple(range(x.shape[1]))
    terms = [derivative(y, x, component=component, order=2) for component in components]
    return torch.stack(terms, dim=0).sum(dim=0)


@dataclass(frozen=True)
class PDEProblem:
    """Minimal PDE definition consumed by generic PINN losses."""

    residual: TensorFn
    initial: TensorFn | None = None
    boundary: TensorFn | None = None
