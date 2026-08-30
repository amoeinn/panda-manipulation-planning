"""Collision checking for the Panda.

Inverse kinematics answers "what joint angles put the gripper there" and
nothing more. It will happily return a configuration with the arm buried in
the floor or folded through itself. A planner needs a separate question
answered: is this configuration physically valid?

Some link pairs touch no matter what the arm does. Links separated only by
a fixed joint are rigidly attached, and the robot base rests on the ground
plane. Rather than hand listing those pairs, which means guessing at the
model's internal structure, they are found by measurement: sample the arm
across many configurations, and any pair in contact in every single one is
structural and gets disabled. This is the approach MoveIt's setup assistant
takes, and it adapts automatically if the model changes.

The two fingers are excluded outright rather than by calibration. Their
separation is a commanded width, not a consequence of the arm's pose, so a
calibration run with the gripper open records them as separable and a later
closed gripper then reports a permanent self collision. They are also
mechanically mirrored and cannot be driven into each other, so the pair
carries no information at any width.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

Configuration = List[float]
LinkPair = Tuple[int, int]

# Link index of the robot base. PyBullet numbers it separately from joints.
BASE_LINK = -1

# The two prismatic finger links. Their separation is commanded, not a
# property of the arm's configuration, so it is never a collision.
FINGER_PAIR: LinkPair = (9, 10)


@dataclass
class CollisionReport:
    """What, if anything, a configuration collides with."""

    self_pairs: List[LinkPair] = field(default_factory=list)
    environment: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def free(self) -> bool:
        return not self.self_pairs and not self.environment

    def __repr__(self) -> str:
        if self.free:
            return "CollisionReport(free)"
        parts = []
        if self.self_pairs:
            parts.append(f"self={self.self_pairs}")
        if self.environment:
            links = sorted({l for ls in self.environment.values() for l in ls})
            parts.append(f"environment links={links}")
        return f"CollisionReport({', '.join(parts)})"


class CollisionChecker:
    """Answers whether a configuration or a motion between two is valid."""

    def __init__(self, panda, obstacles: Optional[Iterable[int]] = None,
                 margin: float = 0.0, calibration_samples: int = 200,
                 seed: int = 0):
        """
        Args:
            panda: a Panda instance.
            obstacles: body ids to treat as environment. The ground plane is
                included automatically.
            margin: treat links closer than this as colliding. A small
                positive value gives a safety buffer.
            calibration_samples: how many random configurations to use when
                identifying structurally permanent contacts.
            seed: makes the calibration reproducible.
        """
        self.panda = panda
        self.robot = panda.robot
        self.margin = margin
        self.obstacles = [panda.plane] + list(obstacles or [])
        self.num_links = p.getNumJoints(self.robot)

        # Every link pair, base included, that is worth testing at all.
        self._all_pairs = [
            (a, b)
            for a in range(BASE_LINK, self.num_links)
            for b in range(a + 1, self.num_links)
            if (a, b) != FINGER_PAIR
        ]

        self.ignored_self_pairs: Set[LinkPair] = set()
        self.ignored_environment: Set[Tuple[int, int]] = set()
        self._calibrate(calibration_samples, seed)

        self._pairs = [pair for pair in self._all_pairs
                       if pair not in self.ignored_self_pairs]

    # ----------------------------------------------------------- calibration

    def _calibrate(self, samples: int, seed: int) -> None:
        """Find contacts that persist across every sampled configuration.

        A pair touching in one pose is a collision. A pair touching in all of
        them is how the robot is built.
        """
        rng = np.random.default_rng(seed)
        saved = self.panda.get_configuration()

        candidate_self: Optional[Set[LinkPair]] = None
        candidate_env: Optional[Set[Tuple[int, int]]] = None

        configurations = [list(saved)]
        for _ in range(samples):
            configurations.append([
                float(rng.uniform(lo, hi))
                for lo, hi in zip(self.panda.lower_limits,
                                  self.panda.upper_limits)
            ])

        for configuration in configurations:
            self.panda.set_configuration(configuration)

            touching_self = {
                (a, b) for a, b in self._all_pairs
                if p.getClosestPoints(self.robot, self.robot, self.margin,
                                      linkIndexA=a, linkIndexB=b)
            }
            touching_env = {
                (body, contact[3])
                for body in self.obstacles
                for contact in p.getClosestPoints(self.robot, body, self.margin)
            }

            # Intersect: only pairs touching in every configuration survive.
            candidate_self = (touching_self if candidate_self is None
                              else candidate_self & touching_self)
            candidate_env = (touching_env if candidate_env is None
                             else candidate_env & touching_env)

            if not candidate_self and not candidate_env:
                break

        self.ignored_self_pairs = candidate_self or set()
        self.ignored_environment = candidate_env or set()
        self.panda.set_configuration(saved)

    # --------------------------------------------------------------- queries

    def _with_configuration(self, configuration: Optional[Configuration]):
        """Set a configuration if given, returning the one to restore."""
        if configuration is None:
            return None
        saved = self.panda.get_configuration()
        self.panda.set_configuration(configuration)
        return saved

    def check(self, configuration: Optional[Configuration] = None) -> CollisionReport:
        """Report every collision for a configuration.

        Slower than in_collision because it does not stop at the first hit.
        Use it for diagnostics, not inside a planning loop.
        """
        saved = self._with_configuration(configuration)

        self_pairs = [
            (a, b) for a, b in self._pairs
            if p.getClosestPoints(self.robot, self.robot, self.margin,
                                  linkIndexA=a, linkIndexB=b)
        ]

        environment: Dict[int, List[int]] = {}
        for body in self.obstacles:
            links = sorted({
                contact[3]
                for contact in p.getClosestPoints(self.robot, body, self.margin)
                if (body, contact[3]) not in self.ignored_environment
            })
            if links:
                environment[body] = links

        if saved is not None:
            self.panda.set_configuration(saved)
        return CollisionReport(self_pairs, environment)

    def in_collision(self, configuration: Optional[Configuration] = None) -> bool:
        """Is this configuration in collision? Returns on the first hit.

        This is the function a planner calls thousands of times, so it exits
        as early as it can. Environment checks come first because a single
        query covers the whole robot, while self collision needs one query
        per link pair.
        """
        saved = self._with_configuration(configuration)
        hit = False

        for body in self.obstacles:
            for contact in p.getClosestPoints(self.robot, body, self.margin):
                if (body, contact[3]) not in self.ignored_environment:
                    hit = True
                    break
            if hit:
                break

        if not hit:
            for a, b in self._pairs:
                if p.getClosestPoints(self.robot, self.robot, self.margin,
                                      linkIndexA=a, linkIndexB=b):
                    hit = True
                    break

        if saved is not None:
            self.panda.set_configuration(saved)
        return hit

    def is_valid(self, configuration: Configuration) -> bool:
        """Within joint limits and collision free."""
        return (self.panda.within_limits(configuration)
                and not self.in_collision(configuration))

    # ------------------------------------------------------------------ edges

    def edge_is_valid(self, start: Configuration, end: Configuration,
                      resolution: float = 0.05) -> bool:
        """Is the straight line between two configurations collision free?

        Checking only the endpoints is not enough: an arm can start and end
        clear while sweeping through an obstacle in between. The segment is
        sampled at a fixed angular resolution, which is what makes collision
        checking the expensive part of any sampling based planner.

        Args:
            resolution: maximum joint angle change, in radians, between
                consecutive samples.
        """
        start_array = np.asarray(start, dtype=float)
        end_array = np.asarray(end, dtype=float)
        largest_step = float(np.max(np.abs(end_array - start_array)))
        steps = max(2, int(np.ceil(largest_step / resolution)) + 1)

        for t in np.linspace(0.0, 1.0, steps):
            if self.in_collision(start_array + t * (end_array - start_array)):
                return False
        return True