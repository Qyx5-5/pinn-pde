"""Small PINN toolkit for PDE experiments."""

from .losses import BoundaryLoss, CompositeLoss, InitialLoss, ResidualLoss
from .models import MLP
from .problems import PDEProblem, derivative, gradient, jacobian, laplacian
from .training import Trainer, sample_boundary_1d, sample_grid_1d, sample_interior

__all__ = [
    "BoundaryLoss",
    "CompositeLoss",
    "InitialLoss",
    "MLP",
    "PDEProblem",
    "ResidualLoss",
    "Trainer",
    "derivative",
    "gradient",
    "jacobian",
    "laplacian",
    "sample_boundary_1d",
    "sample_grid_1d",
    "sample_interior",
]
