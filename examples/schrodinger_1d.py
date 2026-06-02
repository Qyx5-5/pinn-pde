"""1D time-independent Schrodinger example with a harmonic potential."""

from __future__ import annotations

import torch

from pinn_pde import BoundaryLoss, CompositeLoss, MLP, PDEProblem, ResidualLoss, Trainer, laplacian
from pinn_pde.training import sample_boundary_1d, sample_grid_1d


def harmonic_schrodinger(points: torch.Tensor, model: MLP) -> torch.Tensor:
    psi = model(points)
    energy = model.scalars["energy"]
    potential = 0.5 * points.square()
    return -0.5 * laplacian(psi, points) + potential * psi - energy * psi


def make_trainer(device: str = "cpu") -> Trainer:
    model = MLP(1, hidden_layers=(32, 32), learnable_scalars=("energy",)).to(device)
    problem = PDEProblem(residual=harmonic_schrodinger)
    loss = CompositeLoss(
        {
            "residual": ResidualLoss(problem),
            "boundary": BoundaryLoss(weight=2.0),
        }
    )

    def batches() -> dict[str, torch.Tensor]:
        return {
            "residual": sample_grid_1d((-5.0, 5.0), 128, device),
            "boundary": sample_boundary_1d((-5.0, 5.0), 32, device),
        }

    return Trainer(model=model, loss_fn=loss, batch_fn=batches, lr=1e-3)


if __name__ == "__main__":
    trainer = make_trainer()
    trainer.fit(steps=1000, log_every=100)
    print(f"learned energy: {trainer.model.scalars['energy'].item():.6f}")
