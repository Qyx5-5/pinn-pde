from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn


BatchFn = Callable[[], dict[str, torch.Tensor]]
LossFn = Callable[[nn.Module, dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]


def sample_interior(bounds: list[tuple[float, float]], n: int, device: str = "cpu") -> torch.Tensor:
    lows = torch.tensor([b[0] for b in bounds], device=device, dtype=torch.float32)
    highs = torch.tensor([b[1] for b in bounds], device=device, dtype=torch.float32)
    points = lows + (highs - lows) * torch.rand(n, len(bounds), device=device)
    points.requires_grad_(True)
    return points


def sample_boundary_1d(x_bounds: tuple[float, float], n: int, device: str = "cpu") -> torch.Tensor:
    n_left = n // 2
    n_right = n - n_left
    left = torch.full((n_left, 1), float(x_bounds[0]), device=device)
    right = torch.full((n_right, 1), float(x_bounds[1]), device=device)
    points = torch.cat([left, right], dim=0)
    points.requires_grad_(True)
    return points


def sample_grid_1d(x_bounds: tuple[float, float], n: int, device: str = "cpu") -> torch.Tensor:
    points = torch.linspace(float(x_bounds[0]), float(x_bounds[1]), n, device=device).reshape(-1, 1)
    points.requires_grad_(True)
    return points


@dataclass
class Trainer:
    model: nn.Module
    loss_fn: LossFn
    batch_fn: BatchFn
    lr: float = 1e-3
    gradient_clip: float | None = 1.0
    optimizer: torch.optim.Optimizer = field(init=False)
    history: list[dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def step(self) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        loss, parts = self.loss_fn(self.model, self.batch_fn())
        loss.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        record = {"loss": float(loss.detach().cpu()), **parts}
        self.history.append(record)
        return record

    def fit(self, steps: int, log_every: int = 0) -> list[dict[str, float]]:
        for step in range(1, steps + 1):
            record = self.step()
            if log_every and step % log_every == 0:
                print(f"step {step}: " + ", ".join(f"{k}={v:.4e}" for k, v in record.items()))
        return self.history
