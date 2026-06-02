from pathlib import Path

import torch
from torch import nn

from examples.canonical_cases import (
    advection_diffusion_residual,
    heat_residual,
    make_poisson_trainer,
    make_schrodinger_trainer,
    plot_case,
    poisson_residual,
    finite_difference_schrodinger_reference,
    potential,
    schrodinger_residual,
)


class SineModel(nn.Module):
    def forward(self, points):
        return torch.sin(torch.pi * points[:, 0:1])


class HeatExactModel(nn.Module):
    def forward(self, points):
        x = points[:, 0:1]
        t = points[:, 1:2]
        return torch.exp(-0.05 * torch.pi**2 * t) * torch.sin(torch.pi * x)


class ConstantTimeModel(nn.Module):
    def forward(self, points):
        return points[:, 0:1] * 0.0 + 1.0


class HarmonicGroundState(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalars = {"energy": torch.tensor(0.5)}

    def forward(self, points):
        return torch.exp(-0.5 * points.square())


def test_poisson_residual_matches_sine_solution():
    points = torch.linspace(0.0, 1.0, 64).reshape(-1, 1)
    points.requires_grad_(True)
    residual = poisson_residual(points, SineModel())

    assert torch.max(torch.abs(residual)) < 1e-5


def test_heat_residual_matches_exact_sine_decay():
    x = torch.linspace(0.0, 1.0, 32).reshape(-1, 1)
    t = torch.linspace(0.0, 0.5, 32).reshape(-1, 1)
    points = torch.cat([x, t], dim=1)
    points.requires_grad_(True)
    residual = heat_residual(points, HeatExactModel())

    assert torch.max(torch.abs(residual)) < 1e-5


def test_advection_diffusion_residual_is_zero_for_constant_solution():
    points = torch.rand(32, 2, requires_grad=True)
    residual = advection_diffusion_residual(points, ConstantTimeModel())

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_schrodinger_harmonic_ground_state_residual():
    points = torch.linspace(-3.0, 3.0, 64).reshape(-1, 1)
    points.requires_grad_(True)
    residual = schrodinger_residual(points, HarmonicGroundState(), potential("harmonic"))

    assert torch.max(torch.abs(residual)) < 1e-5


def test_schrodinger_potential_catalog_shapes():
    points = torch.linspace(-2.0, 2.0, 17).reshape(-1, 1)
    for name in ["harmonic", "finite_well", "double_well", "periodic", "morse"]:
        assert potential(name)(points).shape == points.shape


def test_harmonic_fd_reference_has_reasonable_ground_energy():
    _x, psi, energy = finite_difference_schrodinger_reference(potential("harmonic"), points=160)

    assert psi.shape == (160, 1)
    assert abs(energy - 0.5) < 0.05


def test_plot_case_writes_png(tmp_path: Path):
    trainer = make_poisson_trainer()
    trainer.fit(2)
    outputs = plot_case("poisson", trainer, tmp_path / "poisson", formats=("png", "pdf"))

    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_schrodinger_trainer_accepts_all_potentials():
    for name in ["harmonic", "finite_well", "double_well", "periodic", "morse"]:
        trainer = make_schrodinger_trainer(name)
        record = trainer.step()
        assert record["loss"] >= 0.0
        assert "normalization" in record
