import torch
from torch import nn

from examples.burgers_1d import burgers_residual
from examples.burgers_1d import make_trainer as make_burgers_trainer
from examples.schrodinger_1d import harmonic_schrodinger
from examples.schrodinger_1d import make_trainer as make_schrodinger_trainer


class ConstantModel(nn.Module):
    def forward(self, points):
        return points[:, 0:1] * 0.0 + 1.0


class HarmonicGroundState(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalars = {"energy": torch.tensor(0.5)}

    def forward(self, points):
        return torch.exp(-0.5 * points.square())


def test_burgers_residual_is_zero_for_constant_solution():
    points = torch.rand(64, 2, requires_grad=True)
    residual = burgers_residual(points, ConstantModel())

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_schrodinger_residual_is_small_for_harmonic_ground_state():
    points = torch.linspace(-3.0, 3.0, 64).reshape(-1, 1)
    points.requires_grad_(True)
    residual = harmonic_schrodinger(points, HarmonicGroundState())

    assert torch.max(torch.abs(residual)) < 1e-5


def test_burgers_example_trainer_has_all_loss_terms():
    trainer = make_burgers_trainer()
    record = trainer.step()
    assert record["loss"] >= 0.0
    assert set(record) == {"loss", "residual", "initial", "boundary"}


def test_schrodinger_example_trainer_has_all_loss_terms():
    trainer = make_schrodinger_trainer()
    record = trainer.step()
    assert record["loss"] >= 0.0
    assert set(record) == {"loss", "residual", "boundary"}
