"""Plan a collision free motion for the Panda with RRT-Connect.

Every returned path is verified rather than trusted: it must start at the
start, end at the goal, and every segment between waypoints must be
collision free at a finer resolution than the planner used.

Usage:
    python examples/run_planner_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.panda import HOME_CONFIGURATION, Panda
from src.rrt_connect import RRTConnect


def verify(checker, result, start, goal) -> list:
    """Return a list of problems with a path. Empty means it is sound."""
    problems = []
    path = result.path
    if not np.allclose(path[0], start, atol=1e-6):
        problems.append("path does not begin at the start configuration")
    if not np.allclose(path[-1], goal, atol=1e-6):
        problems.append("path does not end at the goal configuration")
    for i, (a, b) in enumerate(zip(path, path[1:])):
        if not checker.edge_is_valid(a, b, resolution=0.02):
            problems.append(f"segment {i} to {i + 1} passes through a collision")
    return problems


def main() -> None:
    with Panda(gui=False) as panda:
        checker = CollisionChecker(panda)

        # A goal chosen by sampling rather than by hand, so it is known to
        # be valid and reasonably far from the start.
        rng = np.random.default_rng(7)
        goal = None
        while goal is None:
            candidate = [float(rng.uniform(lo, hi))
                         for lo, hi in zip(panda.lower_limits,
                                           panda.upper_limits)]
            if (checker.is_valid(candidate)
                    and np.linalg.norm(np.array(candidate)
                                       - np.array(HOME_CONFIGURATION)) > 3.0):
                goal = candidate

        print("start:", "  ".join(f"{a:+.2f}" for a in HOME_CONFIGURATION))
        print("goal: ", "  ".join(f"{a:+.2f}" for a in goal))
        print(f"joint space separation: "
              f"{np.linalg.norm(np.array(goal) - np.array(HOME_CONFIGURATION)):.2f} rad")

        print("\nplanning across five seeds")
        header = (f"{'seed':>5} {'found':>6} {'iters':>7} {'nodes':>7} "
                  f"{'checks':>8} {'path (rad)':>11} {'time (s)':>9}  verified")
        print(header)
        print("-" * len(header))

        for seed in range(5):
            planner = RRTConnect(panda, checker, seed=seed)
            began = time.perf_counter()
            result = planner.plan(HOME_CONFIGURATION, goal)
            elapsed = time.perf_counter() - began

            if not result.found:
                print(f"{seed:>5} {'no':>6} {result.iterations:>7} "
                      f"{result.nodes:>7} {result.collision_checks:>8}")
                continue

            problems = verify(checker, result, HOME_CONFIGURATION, goal)
            verdict = "ok" if not problems else f"FAILED: {problems[0]}"
            print(f"{seed:>5} {'yes':>6} {result.iterations:>7} "
                  f"{result.nodes:>7} {result.collision_checks:>8} "
                  f"{result.path_length():>11.2f} {elapsed:>9.2f}  {verdict}")

        print("\nstraight line from start to goal, ignoring obstacles:")
        direct = float(np.linalg.norm(np.array(goal) - np.array(HOME_CONFIGURATION)))
        clear = checker.edge_is_valid(HOME_CONFIGURATION, goal, resolution=0.02)
        print(f"  distance {direct:.2f} rad, collision free: {clear}")


if __name__ == "__main__":
    main()