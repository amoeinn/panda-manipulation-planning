"""Sample training data for a learned signed distance field.

A collision checker answers yes or no. An optimizer needs to know which way
to move, which means a smooth signed distance: positive clearance when the
arm is free, negative penetration depth when it is not.

PyBullet computes that number already. getClosestPoints with a large query
distance returns the separation between two bodies even when they are not
touching, and a negative distance when they overlap. Labels are free; the
work is in deciding what to measure.

Distance to the environment and distance to the arm's own links are kept
separate, and this is the important design decision here. Collapsing them
into a single minimum does not work on this robot: links 4 and 6 are joined
through a joint whose range never lets them straighten, so they sit about
two centimetres apart for the whole of configuration space and supply the
minimum in roughly seventy percent of configurations. A combined field
therefore reports the arm's fixed geometry rather than the obstacles, and
saturates at two centimetres no matter how much open space surrounds the
robot. Measured separately, the environment term spans a useful range while
the self term stays small, and a consumer can weight the two independently.

Sampling matters too. Most random configurations sit well clear of the
obstacles, so uniform sampling leaves the collision boundary thinly
covered, and the boundary is exactly where a gradient has to be right.
Half the samples are drawn near it by bisection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p

Configuration = List[float]

# Query distance for getClosestPoints. Separations beyond this are not
# reported, so both fields saturate here.
MAX_QUERY_DISTANCE = 0.5


@dataclass
class SDFDataset:
    """Configurations with their environment and self clearances."""

    configurations: np.ndarray        # (n, 7) joint angles
    environment_distance: np.ndarray  # (n,) metres to nearest obstacle
    self_distance: np.ndarray         # (n,) metres between arm links

    def __len__(self) -> int:
        return len(self.configurations)

    @property
    def targets(self) -> np.ndarray:
        """Both distances stacked, as the network predicts them."""
        return np.stack([self.environment_distance, self.self_distance],
                        axis=1)

    def split(self, train_fraction: float = 0.8, seed: int = 0):
        """Split into train and test sets."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self))
        cut = int(len(self) * train_fraction)
        train, test = order[:cut], order[cut:]
        return (
            SDFDataset(self.configurations[train],
                       self.environment_distance[train],
                       self.self_distance[train]),
            SDFDataset(self.configurations[test],
                       self.environment_distance[test],
                       self.self_distance[test]),
        )


class SignedDistanceOracle:
    """Ground truth clearances for a configuration.

    Returns two numbers rather than one: distance to the nearest
    environment obstacle, and the smallest separation between any two arm
    links that are capable of touching. See the module docstring for why
    combining them destroys the environment signal.
    """

    def __init__(self, panda, checker):
        """
        Args:
            panda: a Panda instance.
            checker: a CollisionChecker, for its obstacle list, its set of
                testable link pairs, and its permanent contact exclusions.
        """
        self.panda = panda
        self.checker = checker
        self.pairs = list(checker._pairs)

    def _self_distance(self) -> float:
        """Smallest separation between testable link pairs, current pose."""
        nearest = MAX_QUERY_DISTANCE
        for a, b in self.pairs:
            contacts = p.getClosestPoints(self.panda.robot, self.panda.robot,
                                          MAX_QUERY_DISTANCE,
                                          linkIndexA=a, linkIndexB=b)
            if contacts:
                nearest = min(nearest, min(c[8] for c in contacts))
        return nearest

    def _environment_distance(self) -> float:
        """Distance to the nearest obstacle, current pose.

        Structural contacts are skipped. The robot base rests on the ground
        plane permanently, so including it would report zero for every
        configuration.
        """
        nearest = MAX_QUERY_DISTANCE
        for body in self.checker.obstacles:
            for contact in p.getClosestPoints(self.panda.robot, body,
                                              MAX_QUERY_DISTANCE):
                if (body, contact[3]) in self.checker.ignored_environment:
                    continue
                nearest = min(nearest, contact[8])
        return nearest

    def __call__(self, configuration: Configuration) -> Tuple[float, float]:
        """Return (environment distance, self distance) in metres.

        Positive is clearance, negative is penetration depth.
        """
        saved = self.panda.get_configuration()
        self.panda.set_configuration(configuration)
        environment = self._environment_distance()
        own = self._self_distance()
        self.panda.set_configuration(saved)
        return float(environment), float(own)

    def worst(self, configuration: Configuration) -> float:
        """The more binding of the two clearances.

        Useful for deciding validity, where either kind of collision is
        disqualifying. Not useful as a training target.
        """
        environment, own = self(configuration)
        return min(environment, own)


def _uniform(panda, rng) -> Configuration:
    """A uniform random configuration respecting each joint's own range."""
    return [float(rng.uniform(lo, hi))
            for lo, hi in zip(panda.lower_limits, panda.upper_limits)]


def _near_boundary(panda, oracle, rng, tries: int = 12,
                   band: float = 0.05) -> Optional[Configuration]:
    """Find a configuration close to the collision boundary.

    Bisects between a free configuration and a colliding one on the
    environment distance, since that is the term a planner has to reason
    about. Returns None if no colliding configuration turns up.
    """
    free = _uniform(panda, rng)
    if oracle(free)[0] <= 0:
        return free

    hit = None
    for _ in range(tries):
        candidate = _uniform(panda, rng)
        if oracle(candidate)[0] < 0:
            hit = candidate
            break
    if hit is None:
        return None

    low, high = np.asarray(free), np.asarray(hit)
    for _ in range(tries):
        middle = list((low + high) / 2)
        distance = oracle(middle)[0]
        if abs(distance) < band:
            return middle
        if distance > 0:
            low = np.asarray(middle)
        else:
            high = np.asarray(middle)
    return list((low + high) / 2)


def build_dataset(panda, oracle, samples: int = 20000,
                  boundary_fraction: float = 0.5,
                  seed: int = 0) -> SDFDataset:
    """Sample configurations and label them with both clearances.

    Args:
        panda: a Panda instance.
        oracle: a SignedDistanceOracle.
        samples: how many configurations to generate.
        boundary_fraction: share drawn near the collision boundary rather
            than uniformly.
        seed: makes the dataset reproducible.
    """
    rng = np.random.default_rng(seed)
    configurations, environment, own = [], [], []

    boundary_target = int(samples * boundary_fraction)
    boundary_found = 0

    while len(configurations) < samples:
        if boundary_found < boundary_target:
            candidate = _near_boundary(panda, oracle, rng)
            if candidate is None:
                candidate = _uniform(panda, rng)
            else:
                boundary_found += 1
        else:
            candidate = _uniform(panda, rng)

        environment_distance, self_distance = oracle(candidate)
        configurations.append(candidate)
        environment.append(environment_distance)
        own.append(self_distance)

    return SDFDataset(
        configurations=np.asarray(configurations, dtype=np.float32),
        environment_distance=np.asarray(environment, dtype=np.float32),
        self_distance=np.asarray(own, dtype=np.float32),
    )