from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .problems import PDEProblem


TensorTarget = torch.Tensor | Callable[[torch.Tensor], torch.Tensor]


def _target_value(target: TensorTarget, points: torch.Tensor) -> torch.Tensor:
    return target(points) if callable(target) else target


@dataclass
class ResidualLoss:
    problem: PDEProblem
    weight: float = 1.0

    def __call__(self, model: nn.Module, points: torch.Tensor) -> torch.Tensor:
        residual = self.problem.residual(points, model)
        return self.weight * torch.mean(residual.square())


@dataclass
class InitialLoss:
    target: TensorTarget
    weight: float = 1.0

    def __call__(self, model: nn.Module, points: torch.Tensor) -> torch.Tensor:
        pred = model(points)
        target = _target_value(self.target, points).to(device=pred.device, dtype=pred.dtype)
        return self.weight * torch.mean((pred - target).square())


@dataclass
class BoundaryLoss:
    target: TensorTarget | None = None
    weight: float = 1.0

    def __call__(self, model: nn.Module, points: torch.Tensor) -> torch.Tensor:
        pred = model(points)
        target = torch.zeros_like(pred) if self.target is None else _target_value(self.target, points)
        target = target.to(device=pred.device, dtype=pred.dtype)
        return self.weight * torch.mean((pred - target).square())


class CompositeLoss:
    """Weighted sum of named loss terms."""

    def __init__(self, terms: dict[str, Callable[[nn.Module, torch.Tensor], torch.Tensor]]) -> None:
        self.terms = terms

    def __call__(self, model: nn.Module, batches: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total: torch.Tensor | None = None
        values: dict[str, float] = {}
        for name, term in self.terms.items():
            value = term(model, batches[name])
            total = value if total is None else total + value
            values[name] = float(value.detach().cpu())
        if total is None:
            raise ValueError("CompositeLoss needs at least one term")
        return total, values
