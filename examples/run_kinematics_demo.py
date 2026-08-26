"""Check forward and inverse kinematics on the Panda.

Forward kinematics is exact: joint angles fully determine the gripper pose.
Inverse kinematics is approximate, may fail, and for a redundant arm has
many answers, so every result is verified rather than trusted.

Usage:
    python examples/run_kinematics_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.panda import HOME_CONFIGURATION, Panda, Pose


def main() -> None:
    with Panda(gui=False) as panda:
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        print(f"home configuration -> {home_pose}")

        print("\ninverse kinematics, verified by forward kinematics:")
        header = f"{'target':<24} {'error (m)':>11}  {'limits':>7}   result"
        print(header)
        print("-" * (len(header) + 6))

        targets = [
            ("directly reachable", (0.5, 0.0, 0.5)),
            ("low and forward", (0.55, 0.0, 0.25)),
            ("off to the side", (0.3, 0.4, 0.5)),
            ("close to the base", (0.2, 0.0, 0.6)),
            ("behind the robot", (-0.4, 0.0, 0.4)),
            ("far beyond reach", (1.5, 0.0, 0.5)),
            ("below the floor", (0.5, 0.0, -0.3)),
        ]

        for label, position in targets:
            target = Pose(position=position, orientation=home_pose.orientation)
            result = panda.inverse_kinematics(target)
            limits = "ok" if result.within_limits else "VIOLATED"
            verdict = "solved" if result.solved else "no solution"
            print(f"{label:<24} {result.position_error:>11.5f}  "
                  f"{limits:>7}   {verdict}")

        print("\nredundancy: one target, three seeds, three arm postures")
        target = Pose(position=(0.5, 0.0, 0.5), orientation=home_pose.orientation)
        seeds = {
            "home": HOME_CONFIGURATION,
            "elbow tucked": [0.5, -0.6, 0.3, -2.0, 0.2, 1.4, 0.9],
            "twisted base": [-0.8, -1.0, -0.4, -2.5, -0.3, 1.8, 0.2],
        }
        for name, seed in seeds.items():
            result = panda.inverse_kinematics(target, seed_configuration=seed)
            angles = "  ".join(f"{a:+.3f}" for a in result.configuration)
            print(f"  {name:<14} err {result.position_error:.5f}   {angles}")


if __name__ == "__main__":
    main()