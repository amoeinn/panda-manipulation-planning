"""Verify the collision checker against configurations with known answers.

Hand picked joint angles are a poor test, because the expected answer is
itself a guess. Every case here is either derived (the below floor IK
solution must collide) or found by search (sample until a self collision
turns up, then confirm it).

Usage:
    python examples/run_collision_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.panda import HOME_CONFIGURATION, Panda, Pose


def main() -> None:
    with Panda(gui=False) as panda:
        checker = CollisionChecker(panda)

        print("calibration: contacts present in every sampled configuration")
        print(f"  structural link pairs: {sorted(checker.ignored_self_pairs)}")
        print(f"  structural environment contacts: "
              f"{sorted(checker.ignored_environment)}")

        print("\nhome configuration")
        print(f"  {checker.check(HOME_CONFIGURATION)}")

        print("\nIK solution that reached below the floor")
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        below = Pose(position=(0.5, 0.0, -0.3),
                     orientation=home_pose.orientation)
        result = panda.inverse_kinematics(below)
        report = checker.check(result.configuration)
        print(f"  IK reports solved={result.solved}, "
              f"error={result.position_error:.5f}")
        print(f"  collision checker reports {report}")

        print("\nsearching random configurations for a self collision")
        rng = np.random.default_rng(1)
        found = None
        for attempt in range(1, 2001):
            candidate = [float(rng.uniform(lo, hi))
                         for lo, hi in zip(panda.lower_limits,
                                           panda.upper_limits)]
            report = checker.check(candidate)
            if report.self_pairs:
                found = (attempt, candidate, report)
                break

        if found:
            attempt, candidate, report = found
            angles = "  ".join(f"{a:+.2f}" for a in candidate)
            print(f"  found after {attempt} samples: {report}")
            print(f"  configuration: {angles}")
        else:
            print("  none found in 2000 samples")

        print("\nhow often is a random configuration valid?")
        rng = np.random.default_rng(2)
        valid = 0
        trials = 500
        for _ in range(trials):
            candidate = [float(rng.uniform(lo, hi))
                         for lo, hi in zip(panda.lower_limits,
                                           panda.upper_limits)]
            if checker.is_valid(candidate):
                valid += 1
        print(f"  {valid} of {trials} ({100 * valid / trials:.0f} percent)")

        print("\nedge validity")
        target = [0.0, 0.6, 0.0, -1.2, 0.0, 1.8, 0.785]
        print(f"  home valid: {checker.is_valid(HOME_CONFIGURATION)}")
        print(f"  target valid: {checker.is_valid(target)}")
        print(f"  straight line between them free: "
              f"{checker.edge_is_valid(HOME_CONFIGURATION, target)}")


if __name__ == "__main__":
    main()