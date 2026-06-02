"""Canonical PINN PDE cases with optional visualizations."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pinn_pde import BoundaryLoss, CompositeLoss, InitialLoss, MLP, PDEProblem, ResidualLoss, Trainer, derivative, laplacian
from pinn_pde.training import sample_boundary_1d, sample_grid_1d, sample_interior


Potential = Callable[[torch.Tensor], torch.Tensor]
FIGSIZE = (7.2, 4.4)
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#d62728",
    "accent": "#2ca02c",
    "muted": "#6b7280",
}


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, output: Path, formats: tuple[str, ...]) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = output.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", transparent=False)
        written.append(path)
    plt.close(fig)
    return written


def caption(ax: plt.Axes, text: str) -> None:
    ax.figure.text(
        0.08,
        0.045,
        text,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="#374151",
        wrap=True,
    )


def two_panel_figure() -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.24, wspace=0.42)
    return fig, axes


def finite_difference_schrodinger_reference(
    potential_fn: Potential,
    bounds: tuple[float, float] = (-4.0, 4.0),
    points: int = 400,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    x = torch.linspace(*bounds, points).reshape(-1, 1)
    dx = float(x[1] - x[0])
    v = potential_fn(x).reshape(-1)
    diagonal = torch.full((points,), 1.0 / dx**2) + v
    off = torch.full((points - 1,), -0.5 / dx**2)
    hamiltonian = torch.diag(diagonal) + torch.diag(off, 1) + torch.diag(off, -1)
    eigvals, eigvecs = torch.linalg.eigh(hamiltonian)
    psi = eigvecs[:, 0:1]
    psi = psi / torch.sqrt(torch.sum(psi.square()) * dx)
    return x, psi, float(eigvals[0])


def potential(name: str) -> Potential:
    if name == "harmonic":
        return lambda x: 0.5 * x.square()
    if name == "finite_well":
        return lambda x: torch.where(torch.abs(x) < 1.0, torch.zeros_like(x), torch.full_like(x, 8.0))
    if name == "double_well":
        return lambda x: 0.25 * x**4 - x.square()
    if name == "periodic":
        return lambda x: 1.0 - torch.cos(2.0 * torch.pi * x)
    if name == "morse":
        return lambda x: 4.0 * (1.0 - torch.exp(-0.8 * (x + 1.0))).square()
    raise ValueError(f"unknown potential '{name}'")


def poisson_residual(points: torch.Tensor, model: MLP) -> torch.Tensor:
    u = model(points)
    return laplacian(u, points) + torch.pi**2 * torch.sin(torch.pi * points)


def heat_residual(points: torch.Tensor, model: MLP, diffusivity: float = 0.05) -> torch.Tensor:
    u = model(points)
    return derivative(u, points, component=1) - diffusivity * derivative(u, points, component=0, order=2)


def advection_diffusion_residual(
    points: torch.Tensor,
    model: MLP,
    velocity: float = 1.0,
    diffusivity: float = 0.02,
) -> torch.Tensor:
    u = model(points)
    return (
        derivative(u, points, component=1)
        + velocity * derivative(u, points, component=0)
        - diffusivity * derivative(u, points, component=0, order=2)
    )


def schrodinger_residual(points: torch.Tensor, model: MLP, potential_fn: Potential) -> torch.Tensor:
    psi = model(points)
    energy = model.scalars["energy"]
    return -0.5 * laplacian(psi, points) + potential_fn(points) * psi - energy * psi


class SchrodingerNormalizationLoss:
    def __init__(self, domain: tuple[float, float], weight: float = 10.0) -> None:
        self.domain = domain
        self.weight = weight

    def __call__(self, model: MLP, points: torch.Tensor) -> torch.Tensor:
        psi = model(points)
        length = self.domain[1] - self.domain[0]
        norm = torch.mean(psi.square()) * length
        return self.weight * (norm - 1.0).square()


def make_poisson_trainer(device: str = "cpu") -> Trainer:
    model = MLP(1, hidden_layers=(32, 32)).to(device)
    problem = PDEProblem(residual=poisson_residual)
    loss = CompositeLoss({"residual": ResidualLoss(problem), "boundary": BoundaryLoss(weight=5.0)})

    def batches() -> dict[str, torch.Tensor]:
        return {
            "residual": sample_grid_1d((0.0, 1.0), 96, device),
            "boundary": sample_boundary_1d((0.0, 1.0), 16, device),
        }

    return Trainer(model, loss, batches, lr=2e-3)


def make_heat_trainer(device: str = "cpu") -> Trainer:
    model = MLP(2, hidden_layers=(32, 32)).to(device)
    problem = PDEProblem(residual=heat_residual)
    loss = CompositeLoss(
        {
            "residual": ResidualLoss(problem),
            "initial": InitialLoss(lambda p: torch.sin(torch.pi * p[:, 0:1]), weight=3.0),
            "boundary": BoundaryLoss(weight=3.0),
        }
    )

    def batches() -> dict[str, torch.Tensor]:
        initial_x = sample_grid_1d((0.0, 1.0), 64, device)
        initial = torch.cat([initial_x, torch.zeros_like(initial_x)], dim=1)
        t = torch.rand(64, 1, device=device)
        boundary_x = torch.cat([torch.zeros(32, 1, device=device), torch.ones(32, 1, device=device)])
        boundary = torch.cat([boundary_x, t], dim=1)
        boundary.requires_grad_(True)
        return {
            "residual": sample_interior([(0.0, 1.0), (0.0, 1.0)], 192, device),
            "initial": initial,
            "boundary": boundary,
        }

    return Trainer(model, loss, batches, lr=2e-3)


def make_advection_diffusion_trainer(device: str = "cpu") -> Trainer:
    model = MLP(2, hidden_layers=(32, 32)).to(device)
    problem = PDEProblem(residual=advection_diffusion_residual)
    loss = CompositeLoss(
        {
            "residual": ResidualLoss(problem),
            "initial": InitialLoss(lambda p: torch.exp(-60.0 * (p[:, 0:1] - 0.25).square()), weight=3.0),
            "boundary": BoundaryLoss(weight=2.0),
        }
    )

    def batches() -> dict[str, torch.Tensor]:
        initial_x = sample_grid_1d((0.0, 1.0), 64, device)
        initial = torch.cat([initial_x, torch.zeros_like(initial_x)], dim=1)
        t = torch.rand(64, 1, device=device)
        boundary_x = torch.cat([torch.zeros(32, 1, device=device), torch.ones(32, 1, device=device)])
        boundary = torch.cat([boundary_x, t], dim=1)
        boundary.requires_grad_(True)
        return {
            "residual": sample_interior([(0.0, 1.0), (0.0, 0.5)], 192, device),
            "initial": initial,
            "boundary": boundary,
        }

    return Trainer(model, loss, batches, lr=2e-3)


def make_schrodinger_trainer(potential_name: str = "harmonic", device: str = "cpu") -> Trainer:
    model = MLP(1, hidden_layers=(32, 32), learnable_scalars=("energy",)).to(device)
    potential_fn = potential(potential_name)
    domain = (-4.0, 4.0)
    problem = PDEProblem(residual=lambda points, net: schrodinger_residual(points, net, potential_fn))
    loss = CompositeLoss(
        {
            "residual": ResidualLoss(problem),
            "boundary": BoundaryLoss(weight=3.0),
            "normalization": SchrodingerNormalizationLoss(domain, weight=5.0),
        }
    )

    def batches() -> dict[str, torch.Tensor]:
        residual = sample_grid_1d(domain, 128, device)
        return {
            "residual": residual,
            "boundary": sample_boundary_1d(domain, 16, device),
            "normalization": residual,
        }

    return Trainer(model, loss, batches, lr=1e-3)


def train_case(name: str, steps: int, device: str = "cpu", potential_name: str = "harmonic", seed: int = 7) -> Trainer:
    torch.manual_seed(seed)
    makers = {
        "poisson": make_poisson_trainer,
        "heat": make_heat_trainer,
        "advection_diffusion": make_advection_diffusion_trainer,
    }
    trainer = make_schrodinger_trainer(potential_name, device) if name == "schrodinger" else makers[name](device)
    trainer.fit(steps)
    return trainer


def plot_case(
    name: str,
    trainer: Trainer,
    output: Path,
    potential_name: str = "harmonic",
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    set_publication_style()
    trainer.model.eval()
    with torch.no_grad():
        if name in {"heat", "advection_diffusion"}:
            x = torch.linspace(0.0, 1.0, 200).reshape(-1, 1)
            t0 = torch.zeros_like(x)
            t1 = torch.full_like(x, 1.0 if name == "heat" else 0.5)
            y0 = trainer.model(torch.cat([x, t0], dim=1)).cpu()
            y1 = trainer.model(torch.cat([x, t1], dim=1)).cpu()
            fig, (ax, loss_ax) = two_panel_figure()
            ax.plot(x.cpu(), y0, color=COLORS["muted"], label="initial")
            ax.plot(x.cpu(), y1, color=COLORS["primary"], label=f"PINN, t={float(t1[0]):.1f}")
            if name == "heat":
                exact = torch.exp(-0.05 * torch.pi**2 * t1) * torch.sin(torch.pi * x)
                ax.plot(x.cpu(), exact.cpu(), "--", color=COLORS["secondary"], label="exact")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$u(x,t)$")
            ax.legend(frameon=False)
            history = [item["loss"] for item in trainer.history]
            loss_ax.semilogy(range(1, len(history) + 1), history, color=COLORS["accent"])
            loss_ax.set_xlabel("training step")
            loss_ax.set_ylabel("total loss")
            loss_ax.set_title("Optimization Trace")
            if name == "heat":
                caption(ax, r"Heat equation: $u_t=0.05\,u_{xx}$, $x\in[0,1]$, Dirichlet $u=0$, initial state $\sin(\pi x)$.")
            else:
                caption(ax, r"Advection-diffusion: $u_t+u_x=0.02\,u_{xx}$, $x\in[0,1]$, localized Gaussian initial pulse.")
        else:
            bounds = (0.0, 1.0) if name == "poisson" else (-4.0, 4.0)
            x = torch.linspace(*bounds, 240).reshape(-1, 1)
            y = trainer.model(x).cpu()
            fig, axes = two_panel_figure()
            if name == "poisson":
                axes[0].plot(x.cpu(), y, color=COLORS["primary"], label="PINN")
                exact = torch.sin(torch.pi * x).cpu()
                axes[0].plot(x.cpu(), exact, "--", color=COLORS["secondary"], label="exact")
                axes[1].plot(x.cpu(), torch.abs(y - exact), color=COLORS["accent"])
                axes[1].set_ylabel(r"$|u_{\mathrm{PINN}}-u_{\mathrm{exact}}|$")
                axes[1].set_title("Pointwise Error")
                caption(axes[0], r"Poisson equation: $u_{xx}=-\pi^2\sin(\pi x)$, $x\in[0,1]$, $u(0)=u(1)=0$; exact solution is $\sin(\pi x)$.")
            if name == "schrodinger":
                ref_x, ref_psi, ref_energy = finite_difference_schrodinger_reference(potential(potential_name), bounds, points=len(x))
                v = potential(potential_name)(x).cpu()
                y_norm = y / torch.sqrt(torch.trapezoid(y.squeeze().square(), x.squeeze()).clamp_min(1e-8))
                if torch.sum(y_norm.squeeze() * ref_psi.squeeze()) < 0:
                    ref_psi = -ref_psi
                axes[0].plot(x.cpu(), y_norm, color=COLORS["primary"], label="PINN")
                axes[0].plot(ref_x.cpu(), ref_psi.cpu(), "--", color=COLORS["secondary"], label="FD reference")
                twin = axes[0].twinx()
                twin.spines["right"].set_visible(True)
                twin.plot(x.cpu(), v, ":", color=COLORS["muted"], label=r"$V(x)$")
                twin.set_ylabel(r"$V(x)$")
                axes[1].plot(x.cpu(), y_norm.square(), color=COLORS["primary"], label="PINN")
                axes[1].plot(ref_x.cpu(), ref_psi.square().cpu(), "--", color=COLORS["secondary"], label="FD reference")
                axes[1].set_ylabel(r"$|\psi(x)|^2$")
                axes[1].set_title("Probability Density")
                energy = trainer.model.scalars["energy"].detach().cpu().item()
                energy_error = abs(energy - ref_energy)
                axes[0].text(
                    0.08,
                    0.86,
                    f"PINN: $E={energy:.3f}$\nFD: $E={ref_energy:.3f}$\n$|\\Delta E|={energy_error:.3f}$",
                    transform=axes[0].transAxes,
                    fontsize=8.5,
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
                )
                axes[1].legend(frameon=False)
                caption(axes[0], r"Stationary Schrodinger equation: $-\frac{1}{2}\psi_{xx}+V(x)\psi=E\psi$; FD reference gives physical scale and convergence check.")
            axes[0].set_xlabel(r"$x$")
            axes[0].set_ylabel(r"$u(x)$" if name == "poisson" else r"$\psi(x)$")
            axes[0].legend(frameon=False)
            axes[1].set_xlabel(r"$x$")
    title = f"Schrodinger: {potential_name.replace('_', ' ').title()}" if name == "schrodinger" else name.replace("_", " ").title()
    fig.suptitle(title, y=0.95)
    return save_figure(fig, output, formats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["poisson", "heat", "advection_diffusion", "schrodinger"])
    parser.add_argument("--potential", default="harmonic", choices=["harmonic", "finite_well", "double_well", "periodic", "morse"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    trainer = train_case(args.case, args.steps, potential_name=args.potential, seed=args.seed)
    stem = args.case if args.case != "schrodinger" else f"schrodinger_{args.potential}"
    paths = plot_case(args.case, trainer, args.output_dir / stem, potential_name=args.potential, formats=tuple(args.formats))
    print("saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
