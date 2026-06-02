import torch

from pinn_pde import BoundaryLoss, CompositeLoss, MLP, PDEProblem, ResidualLoss, Trainer, derivative, laplacian
from pinn_pde.training import sample_grid_1d


def test_mlp_output_shape():
    model = MLP(2, 1, hidden_layers=(8, 8))
    points = torch.zeros(5, 2)
    assert model(points).shape == (5, 1)


def test_derivative_and_laplacian_shapes():
    x = sample_grid_1d((-1.0, 1.0), 10)
    y = x.square()
    assert derivative(y, x).shape == (10, 1)
    assert laplacian(y, x).shape == (10, 1)


def test_trainer_smoke_step():
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
