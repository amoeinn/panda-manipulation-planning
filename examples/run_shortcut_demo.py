"""Measure what shortcutting removes from an RRT-Connect path.

The planner's path reaches the goal without any notion of efficiency.
This runs the same query across several seeds, shortcuts each result, and
verifies the shortened path is still collision free at a finer resolution
than the shortcutter used.

Usage:
    python examples/run_shortcut_demo.py
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
from src.shortcut import path_length, shortcut


def endpoints(panda, checker, scene):
    """IK solutions for the pick and place landmarks, lifted clear."""
    home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
    out = {}
    for label in ("pick", "place"):
        x, y, z = scene.landmarks[label]
        target = Pose(position=(x, y, z + 0.10),
                      orientation=home_pose.orientation)
        out[label] = panda.inverse_kinematics(target).configuration
    return out["pick"], out["place"]


def sound(checker, path, start, goal) -> bool:
    """Does the path start and end where it should, with every segment free?"""
    if not np.allclose(path[0], start, atol=1e-6):
        return False
    if not np.allclose(path[-1], goal, atol=1e-6):
        return False
    return all(checker.edge_is_valid(a, b, resolution=0.01)
               for a, b in zip(path, path[1:]))


def main() -> None:
    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)
        start, goal = endpoints(panda, checker, scene)

        direct = float(np.linalg.norm(np.array(goal) - np.array(start)))
        print(f"straight line distance: {direct:.2f} rad (blocked by the wall)")

        header = (f"{'seed':>5} {'planned':>9} {'shortcut':>9} {'saved':>7} "
                  f"{'waypoints':>11} {'accepted':>9} {'time (s)':>9}  verified")
        print("\n" + header)
        print("-" * len(header))

        planned_total = 0.0
        shortcut_total = 0.0

        for seed in range(5):
            planner = RRTConnect(panda, checker, seed=seed)
            result = planner.plan(start, goal)
            if not result.found:
                print(f"{seed:>5}  planning failed")
                continue

            began = time.perf_counter()
            trimmed = shortcut(result.path, checker, max_attempts=200, seed=seed)
            elapsed = time.perf_counter() - began

            ok = sound(checker, trimmed.path, start, goal)
            planned_total += trimmed.original_length
            shortcut_total += trimmed.final_length

            print(f"{seed:>5} {trimmed.original_length:>9.2f} "
                  f"{trimmed.final_length:>9.2f} {trimmed.reduction:>6.0%} "
                  f"{len(result.path):>5} to {len(trimmed.path):<3} "
                  f"{trimmed.accepted:>9} {elapsed:>9.2f}  "
                  f"{'ok' if ok else 'FAILED'}")

        print(f"\naverage planned:   {planned_total / 5:.2f} rad")
        print(f"average shortcut:  {shortcut_total / 5:.2f} rad")
        print(f"average reduction: {1 - shortcut_total / planned_total:.0%}")
        print(f"lower bound (straight line, if it were free): {direct:.2f} rad")

        print("\nhow reduction varies with attempt budget, seed 2")
        planner = RRTConnect(panda, checker, seed=2)
        result = planner.plan(start, goal)
        print(f"  planned {path_length(result.path):.2f} rad "
              f"over {len(result.path)} waypoints")
        for attempts in (10, 50, 200, 1000):
            trimmed = shortcut(result.path, checker,
                               max_attempts=attempts, seed=0)
            print(f"  {attempts:>5} attempts -> {trimmed.final_length:.2f} rad "
                  f"({trimmed.reduction:.0%} shorter, "
                  f"{trimmed.accepted} accepted)")


if __name__ == "__main__":
    main()