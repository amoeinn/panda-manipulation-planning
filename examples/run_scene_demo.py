"""Measure how much each scene constrains the arm, then plan across a wall.

The planner demo in an empty scene solved every query in one iteration,
because the straight line was already free. This compares scenes by the
fraction of random configurations that survive collision checking, then
runs a query whose direct route is blocked.

Usage:
    python examples/run_scene_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.panda import HOME_CONFIGURATION, Panda, Pose
from src.rrt_connect import RRTConnect
from src.scene import build


def free_fraction(panda, checker, trials: int = 400, seed: int = 3) -> float:
    """What share of uniformly sampled configurations are collision free?"""
    rng = np.random.default_rng(seed)
    free = 0
    for _ in range(trials):
        candidate = [float(rng.uniform(lo, hi))
                     for lo, hi in zip(panda.lower_limits, panda.upper_limits)]
        if checker.is_valid(candidate):
            free += 1
    return free / trials


def main() -> None:
    print("how much does each scene constrain the arm?")
    print(f"{'scene':<10} {'bodies':>7} {'free configurations':>21}")
    print("-" * 40)

    for name in ("empty", "table", "divided"):
        with Panda(gui=False) as panda:
            scene = build(name)
            checker = CollisionChecker(panda, obstacles=scene.bodies)
            fraction = free_fraction(panda, checker)
            print(f"{name:<10} {len(scene):>7} {fraction:>20.0%}")

    print("\nreaching across the wall on the divided table")
    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)

        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        configurations = {}
        for label in ("pick", "place"):
            x, y, z = scene.landmarks[label]
            target = Pose(position=(x, y, z + 0.10),
                          orientation=home_pose.orientation)
            result = panda.inverse_kinematics(target)
            valid = checker.is_valid(result.configuration)
            print(f"  IK for {label:<6} error {result.position_error:.4f}  "
                  f"solved {result.solved}  collision free {valid}")
            configurations[label] = result.configuration

        start, goal = configurations["pick"], configurations["place"]
        separation = float(np.linalg.norm(np.array(goal) - np.array(start)))
        direct_clear = checker.edge_is_valid(start, goal, resolution=0.02)
        print(f"\n  joint space separation: {separation:.2f} rad")
        print(f"  straight line collision free: {direct_clear}")

        if not (checker.is_valid(start) and checker.is_valid(goal)):
            print("\n  one endpoint is invalid, cannot plan this query")
            return

        print("\n  planning across five seeds")
        header = (f"  {'seed':>5} {'found':>6} {'iters':>7} {'nodes':>7} "
                  f"{'checks':>8} {'path':>8} {'time (s)':>9}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for seed in range(5):
            planner = RRTConnect(panda, checker, seed=seed)
            began = time.perf_counter()
            result = planner.plan(start, goal)
            elapsed = time.perf_counter() - began
            if result.found:
                print(f"  {seed:>5} {'yes':>6} {result.iterations:>7} "
                      f"{result.nodes:>7} {result.collision_checks:>8} "
                      f"{result.path_length():>8.2f} {elapsed:>9.2f}")
            else:
                print(f"  {seed:>5} {'no':>6} {result.iterations:>7} "
                      f"{result.nodes:>7} {result.collision_checks:>8}")


if __name__ == "__main__":
    main()