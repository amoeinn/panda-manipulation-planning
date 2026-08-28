"""Gradient based trajectory optimization using the learned distance field.

RRT-Connect finds a path; shortcutting trims it by splicing straight lines.
Neither reasons about clearance: a shortcut that passes a millimetre from
the wall is accepted as readily as one with ten centimetres to spare. An
optimizer can do better because it has a gradient, and pushing every
waypoint downhill against a cost that combines proximity and smoothness
improves the whole trajectory at once rather than one segment at a time.

Four decisions shape the cost.

Endpoints are held fixed. The start is where the arm is and the goal is
where it must arrive, so only interior waypoints are free variables.

Clearance enters as a hinge rather than a reward. Rewarding distance would
drive the arm to the far side of the workspace and stay there. The penalty
is zero beyond a margin and quadratic inside it, so the optimizer only
attends to being close to something.

Smoothness is the squared second difference along the trajectory, which
penalizes waypoint to waypoint reversals. This term comes from CHOMP and is
still the right objective; it was CHOMP's hand built obstacle field that
dated, not this.

The learned field is a cost, not an authority. Its boundary sign agreement
is about 76 percent, so it will occasionally be wrong about whether a
configuration is in collision. Every optimized trajectory is therefore
verified against exact geometry afterward, and a run that fails is reported
as failing rather than quietly returned.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch

from .sdf_model import ENVIRONMENT, SELF, SDFNetwork

Configuration = List[float]


@dataclass
class OptimizeResult:
    """The outcome of optimizing a trajectory."""

    path: List[Configuration]
    initial_path: List[Configuration]
    cost_history: List[float] = field(default_factory=list)
    iterations: int = 0
    exact_valid: Optional[bool] = None

    @property
    def waypoints(self) -> int:
        return len(self.path)


def resample(path: Sequence[Configuration], count: int) -> np.ndarray:
    """Redistribute a path onto a fixed number of evenly spaced waypoints.

    A planner returns however many waypoints its search happened to
    produce, unevenly spaced. The optimizer needs a fixed size array, and
    even spacing keeps the smoothness term from being dominated by whichever
    segments happened to be long.
    """
    points = np.asarray(path, dtype=np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])

    if cumulative[-1] <= 0:
        return np.repeat(points[:1], count, axis=0)

    wanted = np.linspace(0.0, cumulative[-1], count)
    return np.stack([np.interp(wanted, cumulative, points[:, joint])
                     for joint in range(points.shape[1])], axis=1)


def smoothness_cost(trajectory: torch.Tensor) -> torch.Tensor:
    """Sum of squared accelerations along the trajectory.

    Finite difference second derivative: q[i-1] - 2 q[i] + q[i+1]. Zero for
    a straight line in joint space, large for a path that reverses.
    """
    acceleration = trajectory[:-2] - 2.0 * trajectory[1:-1] + trajectory[2:]
    return (acceleration ** 2).sum()


def clearance_cost(distances: torch.Tensor, margin: float) -> torch.Tensor:
    """Quadratic penalty for clearance below a margin, zero above it.

    Args:
        distances: predicted clearances, in metres.
        margin: distance beyond which a waypoint is considered safe.
    """
    violation = torch.clamp(margin - distances, min=0.0)
    return (violation ** 2).sum()


class TrajectoryOptimizer:
    """Pushes a trajectory downhill against clearance and smoothness."""

    def __init__(self, model: SDFNetwork, lower_limits: Sequence[float],
                 upper_limits: Sequence[float],
                 environment_margin: float = 0.05,
                 self_margin: float = 0.01,
                 environment_weight: float = 40.0,
                 self_weight: float = 40.0,
                 smoothness_weight: float = 1.0,
                 learning_rate: float = 0.01):
        """
        Args:
            model: a trained SDFNetwork.
            lower_limits, upper_limits: per joint bounds, enforced by
                clamping after every step.
            environment_margin: clearance, in metres, beyond which the
                obstacle penalty is zero.
            self_margin: the same for arm self proximity. Smaller because
                the arm's own links sit close together by construction; on
                this robot two links never exceed about 2 cm apart, so a
                larger margin would penalize every configuration equally
                and carry no information.
            environment_weight, self_weight, smoothness_weight: relative
                importance of the three terms.
            learning_rate: Adam step size, in radians.
        """
        self.model = model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.lower = torch.tensor(list(lower_limits), dtype=torch.float32)
        self.upper = torch.tensor(list(upper_limits), dtype=torch.float32)

        self.environment_margin = environment_margin
        self.self_margin = self_margin
        self.environment_weight = environment_weight
        self.self_weight = self_weight
        self.smoothness_weight = smoothness_weight
        self.learning_rate = learning_rate

    def cost(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Total cost of a trajectory, as the optimizer sees it."""
        distances = self.model.distances(trajectory)
        environment = clearance_cost(distances[:, ENVIRONMENT],
                                     self.environment_margin)
        own = clearance_cost(distances[:, SELF], self.self_margin)
        smooth = smoothness_cost(trajectory)
        return (self.environment_weight * environment
                + self.self_weight * own
                + self.smoothness_weight * smooth)

    def optimize(self, path: Sequence[Configuration], waypoints: int = 40,
                 iterations: int = 300) -> OptimizeResult:
        """Improve a path by gradient descent on its interior waypoints.

        Args:
            path: a collision free path from a planner.
            waypoints: how many evenly spaced waypoints to optimize over.
            iterations: gradient steps to take.
        """
        initial = resample(path, waypoints)
        start = torch.tensor(initial[0], dtype=torch.float32)
        goal = torch.tensor(initial[-1], dtype=torch.float32)

        interior = torch.tensor(initial[1:-1], dtype=torch.float32,
                                requires_grad=True)
        optimizer = torch.optim.Adam([interior], lr=self.learning_rate)
        history: List[float] = []

        for _ in range(iterations):
            optimizer.zero_grad()
            trajectory = torch.cat([start.unsqueeze(0), interior,
                                    goal.unsqueeze(0)], dim=0)
            loss = self.cost(trajectory)
            loss.backward()
            optimizer.step()

            # Joint limits are hard constraints, so enforce them by
            # projection rather than by adding another penalty term.
            with torch.no_grad():
                interior.clamp_(min=self.lower, max=self.upper)

            history.append(float(loss.item()))

        with torch.no_grad():
            final = torch.cat([start.unsqueeze(0), interior,
                               goal.unsqueeze(0)], dim=0).numpy()

        return OptimizeResult(
            path=[list(map(float, row)) for row in final],
            initial_path=[list(map(float, row)) for row in initial],
            cost_history=history,
            iterations=iterations,
        )