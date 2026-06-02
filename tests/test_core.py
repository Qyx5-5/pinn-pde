import torch
from torch import nn

from pinn_pde import (
    BoundaryLoss,
    CompositeLoss,
    InitialLoss,
    MLP,
    PDEProblem,
    ResidualLoss,
    Trainer,
    derivative,
    jacobian,
    laplacian,
)
from pinn_pde.training import sample_grid_1d, sample_interior


def test_mlp_output_shape():
    model = MLP(2, 1, hidden_layers=(8, 8))
    points = torch.zeros(5, 2)
    assert model(points).shape == (5, 1)


def test_derivative_and_laplacian_shapes():
    x = sample_grid_1d((-1.0, 1.0), 10)
    y = x.square()
    assert derivative(y, x).shape == (10, 1)
    assert laplacian(y, x).shape == (10, 1)


def test_autograd_derivatives_match_exact_polynomial_values():
    x = sample_grid_1d((-1.0, 1.0), 21)
    y = x**3

    assert torch.allclose(derivative(y, x), 3.0 * x.square(), atol=1e-5)
    assert torch.allclose(laplacian(y, x), 6.0 * x, atol=1e-5)


def test_laplacian_2d_matches_exact_quadratic_values():
    points = sample_interior([(-1.0, 1.0), (-1.0, 1.0)], 32)
    y = points[:, 0:1].square() + points[:, 1:2].square()

    assert torch.allclose(laplacian(y, points), torch.full((32, 1), 4.0), atol=1e-5)


def test_jacobian_matches_exact_vector_function_values():
    points = sample_interior([(-1.0, 1.0), (-1.0, 1.0)], 16)
    y = torch.cat([points[:, 0:1].square(), points[:, 0:1] * points[:, 1:2]], dim=1)
    j = jacobian(y, points)

    assert j.shape == (16, 2, 2)
    assert torch.allclose(j[:, 0, 0:1], 2.0 * points[:, 0:1], atol=1e-5)
    assert torch.allclose(j[:, 0, 1:2], torch.zeros(16, 1), atol=1e-5)
    assert torch.allclose(j[:, 1, 0:1], points[:, 1:2], atol=1e-5)
    assert torch.allclose(j[:, 1, 1:2], points[:, 0:1], atol=1e-5)


def test_boundary_transform_enforces_known_zero_boundary():
    def transform(points, raw):
        return points * (1.0 - points) * raw

    model = MLP(1, 1, hidden_layers=(8,), boundary_transform=transform)
    boundary = torch.tensor([[0.0], [1.0]])

    assert torch.allclose(model(boundary), torch.zeros(2, 1), atol=1e-7)


def test_composite_loss_applies_named_terms_and_weights():
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    points = torch.ones(4, 1, requires_grad=True)
    loss = CompositeLoss(
        {
            "initial": InitialLoss(torch.ones(4, 1), weight=2.0),
            "boundary": BoundaryLoss(weight=3.0),
        }
    )

    total, parts = loss(model, {"initial": points, "boundary": points})

    assert torch.isclose(total, torch.tensor(2.0))
    assert parts == {"initial": 2.0, "boundary": 0.0}


def test_trainer_reduces_supervised_loss_for_linear_target():
    torch.manual_seed(1)
    model = MLP(1, 1, hidden_layers=(16,))
    points = torch.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    target = 2.0 * points - 0.5
    loss = CompositeLoss({"initial": InitialLoss(target)})

    trainer = Trainer(model, loss, lambda: {"initial": points}, lr=5e-3, gradient_clip=None)
    first = trainer.step()["loss"]
    for _ in range(200):
        last = trainer.step()["loss"]

    assert last < 0.05
    assert last < first * 0.2


def test_trainer_records_loss_parts():
    def residual(points, model):
        return model(points)

    model = MLP(1, 1, hidden_layers=(8,))
    problem = PDEProblem(residual=residual)
    loss = CompositeLoss({"residual": ResidualLoss(problem), "boundary": BoundaryLoss()})

    def batches():
        points = sample_grid_1d((-1.0, 1.0), 8)
        return {"residual": points, "boundary": points}

    record = Trainer(model, loss, batches).step()
    assert record["loss"] >= 0.0
    assert set(record) == {"loss", "residual", "boundary"}
