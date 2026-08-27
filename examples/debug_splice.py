"""Compare the segments shortcut validates against the segments it produces.

One accepted shortcut is enough to produce a blocked path, so the fault is
in a single splice rather than in accumulated error. This instruments one
splice and prints both sets side by side.

Usage:
    python examples/debug_splice.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from random import Random

from src.collision import CollisionChecker
from src.panda import HOME_CONFIGURATION, Panda, Pose
from src.rrt_connect import RRTConnect
from src.scene import build
from src.shortcut import _cumulative, _interpolate, path_length


def main() -> None:
    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)

        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        ends = {}
        for label in ("pick", "place"):
            x, y, z = scene.landmarks[label]
            ends[label] = panda.inverse_kinematics(
                Pose(position=(x, y, z + 0.10),
                     orientation=home_pose.orientation)).configuration

        planned = RRTConnect(panda, checker, seed=0).plan(ends["pick"],
                                                          ends["place"])
        working = [list(c) for c in planned.path]
        print(f"planned path: {len(working)} waypoints")
        print("every planned segment free: "
              f"{all(checker.edge_is_valid(a, b, resolution=0.01) for a, b in zip(working, working[1:]))}")

        rng = Random(0)
        for attempt in range(1, 400):
            cumulative = _cumulative(working)
            total = cumulative[-1]
            first = rng.uniform(0.0, total)
            second = rng.uniform(0.0, total)
            if first > second:
                first, second = second, first
            if second - first < 1e-6:
                continue

            si, sp = _interpolate(working, cumulative, first)
            ei, ep = _interpolate(working, cumulative, second)
            if ei <= si:
                continue

            candidate = working[: si + 1] + [sp, ep] + working[ei + 1:]
            if len(candidate) >= len(working):
                continue
            if path_length(candidate) >= path_length(working):
                continue

            checked = [
                ("working[si] -> sp", working[si], sp),
                ("sp -> ep", sp, ep),
                ("ep -> working[ei+1]", ep, working[ei + 1]),
            ]
            if not all(checker.edge_is_valid(a, b, resolution=0.02)
                       for _, a, b in checked):
                continue

            print(f"\naccepted on attempt {attempt}")
            print(f"  si={si}  ei={ei}  waypoints {len(working)} -> {len(candidate)}")

            print("\n  segments shortcut checked:")
            for name, a, b in checked:
                free = checker.edge_is_valid(a, b, resolution=0.01)
                print(f"    {name:<24} span {np.linalg.norm(np.array(b)-np.array(a)):.3f}  free {free}")

            print("\n  segments actually in the candidate path:")
            for i, (a, b) in enumerate(zip(candidate, candidate[1:])):
                free = checker.edge_is_valid(a, b, resolution=0.01)
                flag = "" if free else "   <-- blocked"
                print(f"    {i:>2} -> {i+1:<2} span {np.linalg.norm(np.array(b)-np.array(a)):.3f}  free {free}{flag}")

            print("\n  where the checked segments sit in the candidate:")
            for label, point in (("sp", sp), ("ep", ep)):
                idx = [i for i, c in enumerate(candidate)
                       if np.allclose(c, point, atol=1e-12)]
                print(f"    {label} appears at index {idx}")
            print(f"    working[si] at candidate index {si}")
            print(f"    working[ei+1] value equals candidate index "
                  f"{[i for i, c in enumerate(candidate) if np.allclose(c, working[ei+1], atol=1e-12)]}")
            break


if __name__ == "__main__":
    main()