"""Find out why shortcut paths fail verification.

Two candidate causes:
  1. the splice keeps waypoints it should drop, so the path grows
  2. an endpoint drifts, so the path no longer starts or ends where it must
  3. a spliced segment genuinely passes through the wall

This reports which one, rather than assuming.

Usage:
    python examples/debug_shortcut.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.panda import HOME_CONFIGURATION, Panda, Pose
from src.rrt_connect import RRTConnect
from src.scene import build
from src.shortcut import shortcut


def main() -> None:
    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)

        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        ends = {}
        for label in ("pick", "place"):
            x, y, z = scene.landmarks[label]
            target = Pose(position=(x, y, z + 0.10),
                          orientation=home_pose.orientation)
            ends[label] = panda.inverse_kinematics(target).configuration
        start, goal = ends["pick"], ends["place"]

        planner = RRTConnect(panda, checker, seed=0)
        planned = planner.plan(start, goal)
        trimmed = shortcut(planned.path, checker, max_attempts=200, seed=0)
        path = trimmed.path

        print(f"planned {len(planned.path)} waypoints, "
              f"shortcut {len(path)} waypoints")

        start_drift = float(np.linalg.norm(np.array(path[0]) - np.array(start)))
        goal_drift = float(np.linalg.norm(np.array(path[-1]) - np.array(goal)))
        print(f"\nendpoint drift: start {start_drift:.2e}, goal {goal_drift:.2e}")
        print(f"  start within 1e-6: {start_drift < 1e-6}")
        print(f"  goal  within 1e-6: {goal_drift < 1e-6}")

        print("\nper segment collision check at resolution 0.01")
        bad = []
        for i, (a, b) in enumerate(zip(path, path[1:])):
            span = float(np.linalg.norm(np.array(b) - np.array(a)))
            free = checker.edge_is_valid(a, b, resolution=0.01)
            if not free:
                bad.append(i)
            flag = "" if free else "   <-- blocked"
            print(f"  {i:>3} -> {i + 1:<3} span {span:>6.3f} rad  "
                  f"free {str(free):<5}{flag}")

        print(f"\nblocked segments: {bad if bad else 'none'}")

        if bad:
            print("\nre-checking a blocked segment at the resolution "
                  "shortcut used (0.02)")
            i = bad[0]
            coarse = checker.edge_is_valid(path[i], path[i + 1], resolution=0.02)
            print(f"  segment {i}: free at 0.02 = {coarse}, "
                  f"free at 0.01 = False")
            if coarse:
                print("  the coarse check missed a collision the fine check finds")

        print("\nduplicate or near duplicate waypoints")
        duplicates = 0
        for i, (a, b) in enumerate(zip(path, path[1:])):
            if float(np.linalg.norm(np.array(b) - np.array(a))) < 1e-9:
                duplicates += 1
        print(f"  {duplicates} zero length segments")


if __name__ == "__main__":
    main()