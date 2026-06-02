"""1D viscous Burgers PINN example."""

from __future__ import annotations

import math

import torch

from pinn_pde import CompositeLoss, InitialLoss, MLP, PDEProblem, ResidualLoss, Trainer, derivative
from pinn_pde.losses import BoundaryLoss
from pinn_pde.training import sample_interior


def burgers_residual(points: torch.Tensor, model: MLP, viscosity: float = 0.01 / math.pi) -> torch.Tensor:
    u = model(points)
    u_x = derivative(u, points, component=0, order=1)
    u_t = derivative(u, points, component=1, order=1)
    u_xx = derivative(u, points, component=0, order=2)
    return u_t + u * u_x - viscosity * u_xx


def initial_condition(points: torch.Tensor) -> torch.Tensor:
    x = points[:, 0:1]
    return -torch.sin(math.pi * x)


def make_trainer(device: str = "cpu") -> Trainer:
    model = MLP(2, hidden_layers=(32, 32, 32)).to(device)
    problem = PDEProblem(residual=burgers_residual)
    loss = CompositeLoss(
        {
            "residual": ResidualLoss(problem),
            "initial": InitialLoss(initial_condition),
            "boundary": BoundaryLoss(),
        }
    )

    def batches() -> dict[str, torch.Tensor]:
        interior = sample_interior([(-1.0, 1.0), (0.0, 1.0)], 256, device)
        initial = torch.cat(
            [sample_interior([(-1.0, 1.0)], 128, device), torch.zeros(128, 1, device=device)],
            dim=1,
        )
        t = torch.rand(128, 1, device=device)
        x_boundary = torch.cat(
            [
                torch.full((64, 1), -1.0, device=device),
                torch.full((64, 1), 1.0, device=device),
            ],
            dim=0,
        )
        boundary = torch.cat(
            [x_boundary, t],
            dim=1,
        )
        boundary.requires_grad_(True)
        return {"residual": interior, "initial": initial, "boundary": boundary}

    return Trainer(model=model, loss_fn=loss, batch_fn=batches, lr=1e-3)


if __name__ == "__main__":
    trainer = make_trainer()
    trainer.fit(steps=1000, log_every=100)
