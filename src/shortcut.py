"""Path shortcutting for sampling based planners.

RRT-Connect returns a path that reaches the goal with no notion of doing so
efficiently. The route wanders because it is assembled from random samples,
not because the detours are necessary. Shortcutting is the standard fix:
repeatedly pick two points on the path, and if the straight line between
them is collision free, replace everything in between with that line.

Every accepted shortcut strictly reduces length, so the process only
improves the path and can be stopped at any point. It cannot escape the
homotopy class the planner found: if the arm went over the wall, no amount
of shortcutting will route it under.
"""

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Sequence, Tuple

import numpy as np

Configuration = List[float]


@dataclass
class ShortcutResult:
    """A shortened path and what it took to get there."""

    path: List[Configuration]
    original_length: float
    final_length: float
    attempts: int
    accepted: int
    collision_checks: int

    @property
    def reduction(self) -> float:
        """Fraction of the original length removed."""
        if self.original_length == 0:
            return 0.0
        return 1.0 - self.final_length / self.original_length


def path_length(path: Sequence[Configuration]) -> float:
    """Total joint space distance along a path, in radians."""
    return float(sum(
        np.linalg.norm(np.asarray(b) - np.asarray(a))
        for a, b in zip(path, path[1:])
    ))


def _cumulative(path: Sequence[Configuration]) -> np.ndarray:
    """Distance from the start to each waypoint."""
    steps = [0.0]
    for a, b in zip(path, path[1:]):
        steps.append(steps[-1] + float(np.linalg.norm(np.asarray(b) - np.asarray(a))))
    return np.asarray(steps)


def _interpolate(path: Sequence[Configuration], cumulative: np.ndarray,
                 distance: float) -> Tuple[int, Configuration]:
    """The configuration at a given distance along the path.

    Returns the index of the segment it falls in, along with the point
    itself, so the caller knows which waypoints to splice around.
    """
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = max(0, min(index, len(path) - 2))

    span = cumulative[index + 1] - cumulative[index]
    if span <= 0:
        return index, list(path[index])

    t = (distance - cumulative[index]) / span
    a = np.asarray(path[index])
    b = np.asarray(path[index + 1])
    return index, list(a + t * (b - a))


def shortcut(path: Sequence[Configuration], checker,
             max_attempts: int = 200, resolution: float = 0.02,
             seed: Optional[int] = None) -> ShortcutResult:
    """Shorten a path by splicing in collision free straight lines.

    Args:
        path: the planner's output, start to goal.
        checker: a CollisionChecker.
        max_attempts: how many shortcuts to try.
        resolution: collision checking step, in radians. Must be at least as
            fine as the planner used, or a shortcut can tunnel through an
            obstacle the planner correctly avoided.
        seed: makes a run reproducible.

    Returns:
        A ShortcutResult. The path is never longer than the input.
    """
    rng = Random(seed)
    working = [list(configuration) for configuration in path]
    original = path_length(working)
    checks = 0
    accepted = 0

    for _ in range(max_attempts):
        if len(working) < 3:
            break  # nothing between the endpoints to cut out

        cumulative = _cumulative(working)
        total = cumulative[-1]
        if total <= 0:
            break

        # Two points anywhere along the path, not just at waypoints.
        first = rng.uniform(0.0, total)
        second = rng.uniform(0.0, total)
        if first > second:
            first, second = second, first
        if second - first < 1e-6:
            continue

        start_index, start_point = _interpolate(working, cumulative, first)
        end_index, end_point = _interpolate(working, cumulative, second)
        if end_index <= start_index:
            continue  # both inside the same segment, nothing to remove

        candidate = (working[: start_index + 1] + [start_point, end_point]
                     + working[end_index + 1:])

        # Splicing only removes length if it actually drops waypoints. When
        # both points land in adjacent segments there is nothing between
        # them to remove, and the splice just inserts two more.
        if len(candidate) >= len(working):
            continue
        if path_length(candidate) >= path_length(working):
            continue

        # Check every segment the splice creates, not only the new middle.
        # The joining segments are sub-portions of segments in the current
        # working path, which is itself the product of earlier splices, so
        # they cannot be assumed free.
        checks += 1
        spliced = [
            (working[start_index], start_point),
            (start_point, end_point),
            (end_point, working[end_index + 1]),
        ]
        if not all(checker.edge_is_valid(a, b, resolution=resolution)
                   for a, b in spliced):
            continue

        working = candidate
        accepted += 1

    return ShortcutResult(
        path=working,
        original_length=original,
        final_length=path_length(working),
        attempts=max_attempts,
        accepted=accepted,
        collision_checks=checks,
    )