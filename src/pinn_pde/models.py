from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "sin": torch.sin,
    "gelu": torch.nn.functional.gelu,
}


class MLP(nn.Module):
    """Fully connected network used by PINN examples."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_layers: Sequence[int] = (64, 64, 64),
        activation: str = "tanh",
        boundary_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        learnable_scalars: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation '{activation}'")

        dims = [input_dim, *hidden_layers, output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.activation = _ACTIVATIONS[activation]
        self.boundary_transform = boundary_transform
        self.scalars = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for name in learnable_scalars}
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers[:-1]:
            y = self.activation(layer(y))
        y = self.layers[-1](y)
        if self.boundary_transform is not None:
            y = self.boundary_transform(x, y)
        return y
