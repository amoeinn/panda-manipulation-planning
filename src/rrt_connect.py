"""RRT-Connect for the Panda arm.

Plain RRT grows one tree from the start and takes a single step per
iteration. RRT-Connect grows two trees, one from the start and one from the
goal, and after each extension greedily drives the other tree toward the
new node until it connects or hits something. Growing from the goal matters
for manipulators: a goal wedged between obstacles is much easier to escape
from than to stumble into.

Configuration space here is the seven arm joint angles. Distance is measured
in radians across all seven, which treats every joint as equally expensive
to move. That is a simplification: rotating the base sweeps the whole arm
through space while rotating the wrist barely moves anything.
"""

from dataclasses import dataclass, field
from enum import Enum
from random import Random
from typing import List, Optional, Sequence, Tuple

import numpy as np

Configuration = List[float]


class ExtendResult(Enum):
    """Outcome of trying to extend a tree toward a target."""

    REACHED = "reached"      # arrived at the target
    ADVANCED = "advanced"    # moved closer but did not arrive
    TRAPPED = "trapped"      # blocked immediately, nothing added


@dataclass
class PlanResult:
    """The outcome of a planning query."""

    path: Optional[List[Configuration]]
    start_tree: List[Configuration] = field(default_factory=list)
    goal_tree: List[Configuration] = field(default_factory=list)
    iterations: int = 0
    collision_checks: int = 0

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def nodes(self) -> int:
        return len(self.start_tree) + len(self.goal_tree)

    def path_length(self) -> Optional[float]:
        """Total joint space distance travelled, in radians."""
        if not self.path:
            return None
        return float(sum(
            np.linalg.norm(np.asarray(b) - np.asarray(a))
            for a, b in zip(self.path, self.path[1:])
        ))


class Tree:
    """A search tree over configurations, with parent pointers."""

    def __init__(self, root: Configuration):
        self.nodes: List[Configuration] = [list(root)]
        self.parents: List[int] = [-1]

    def add(self, configuration: Configuration, parent: int) -> int:
        self.nodes.append(list(configuration))
        self.parents.append(parent)
        return len(self.nodes) - 1

    def nearest(self, target: Configuration) -> int:
        """Index of the closest node. Brute force, fine at these sizes."""
        target_array = np.asarray(target)
        distances = np.linalg.norm(np.asarray(self.nodes) - target_array, axis=1)
        return int(np.argmin(distances))

    def path_to_root(self, index: int) -> List[Configuration]:
        """Walk parent pointers from a node back to the root."""
        path = []
        while index != -1:
            path.append(self.nodes[index])
            index = self.parents[index]
        return path


class RRTConnect:
    """Bidirectional sampling based planner over the arm's joint space."""

    def __init__(self, panda, checker, step_size: float = 0.3,
                 max_iterations: int = 3000, seed: Optional[int] = None):
        """
        Args:
            panda: a Panda instance, for its joint limits.
            checker: a CollisionChecker.
            step_size: how far to move per extension, in radians of joint
                space distance. Larger is faster but more likely to tunnel
                through thin obstacles between collision checks.
            max_iterations: give up after this many samples.
            seed: makes a run reproducible.
        """
        self.panda = panda
        self.checker = checker
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.rng = Random(seed)
        self._checks = 0

    # ---------------------------------------------------------------- pieces

    def sample(self) -> Configuration:
        """A uniform random configuration, respecting each joint's own range.

        Joint 3 runs from -pi to 0 and joint 5 from -0.087 to 3.822, so a
        shared range would put most samples outside the limits.
        """
        return [self.rng.uniform(lo, hi)
                for lo, hi in zip(self.panda.lower_limits,
                                  self.panda.upper_limits)]

    def _valid(self, configuration: Configuration) -> bool:
        self._checks += 1
        return not self.checker.in_collision(configuration)

    def steer(self, origin: Configuration, target: Configuration) -> Configuration:
        """Move from origin toward target by at most step_size."""
        origin_array = np.asarray(origin)
        delta = np.asarray(target) - origin_array
        distance = float(np.linalg.norm(delta))
        if distance <= self.step_size:
            return list(target)
        return list(origin_array + delta * (self.step_size / distance))

    def extend(self, tree: Tree, target: Configuration) -> Tuple[ExtendResult, int]:
        """Take one step from the tree's nearest node toward a target."""
        nearest_index = tree.nearest(target)
        candidate = self.steer(tree.nodes[nearest_index], target)

        if not self._valid(candidate):
            return ExtendResult.TRAPPED, nearest_index

        new_index = tree.add(candidate, nearest_index)
        arrived = np.allclose(candidate, target, atol=1e-6)
        return (ExtendResult.REACHED if arrived else ExtendResult.ADVANCED), new_index

    def connect(self, tree: Tree, target: Configuration) -> Tuple[ExtendResult, int]:
        """Extend repeatedly toward a target until reached or blocked.

        This greedy loop is what separates RRT-Connect from plain RRT. One
        iteration can cover a lot of ground rather than a single step.
        """
        result, index = ExtendResult.ADVANCED, -1
        while result == ExtendResult.ADVANCED:
            result, index = self.extend(tree, target)
        return result, index

    # ----------------------------------------------------------------- query

    def plan(self, start: Configuration, goal: Configuration) -> PlanResult:
        """Find a collision free path between two configurations."""
        self._checks = 0

        if not self.checker.is_valid(start):
            raise ValueError("start configuration is invalid")
        if not self.checker.is_valid(goal):
            raise ValueError("goal configuration is invalid")

        start_tree = Tree(start)
        goal_tree = Tree(goal)
        swapped = False

        for iteration in range(1, self.max_iterations + 1):
            sample = self.sample()

            result, new_index = self.extend(start_tree, sample)
            if result != ExtendResult.TRAPPED:
                # Drive the other tree at whatever we just added.
                connect_result, connect_index = self.connect(
                    goal_tree, start_tree.nodes[new_index])

                if connect_result == ExtendResult.REACHED:
                    a, b = (goal_tree, start_tree) if swapped else (start_tree, goal_tree)
                    ai = connect_index if swapped else new_index
                    bi = new_index if swapped else connect_index

                    # Each half runs node to root, so reverse the start side
                    # and drop the duplicated meeting configuration.
                    path = a.path_to_root(ai)[::-1] + b.path_to_root(bi)[1:]
                    return PlanResult(
                        path=path,
                        start_tree=a.nodes,
                        goal_tree=b.nodes,
                        iterations=iteration,
                        collision_checks=self._checks,
                    )

            # Alternate which tree leads, so neither runs far ahead.
            start_tree, goal_tree = goal_tree, start_tree
            swapped = not swapped

        a, b = (goal_tree, start_tree) if swapped else (start_tree, goal_tree)
        return PlanResult(
            path=None,
            start_tree=a.nodes,
            goal_tree=b.nodes,
            iterations=self.max_iterations,
            collision_checks=self._checks,
        )