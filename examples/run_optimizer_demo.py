"""Compare the planner, shortcutting, and gradient based optimization.

Four trajectories for the same query, measured the same way: what
RRT-Connect returns, that path shortcutted, that path optimized, and
shortcutting followed by optimization.

Clearance is reported twice. The whole path figure includes the start and
goal, which are fixed IK solutions no method is allowed to move, so it is
bounded below by whatever clearance those endpoints happen to have and is
identical across every method and seed. The interior figure excludes the
first and last tenth of the trajectory and is the number that reflects what
optimization actually did.

Exact validity is a gate rather than a score. The learned field is wrong
about roughly one in four configurations near contact, so an optimized
trajectory that geometry rejects has to be reported as rejected.

Usage:
    python examples/run_optimizer_demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.optimizer import TrajectoryOptimizer, resample
from src.panda import HOME_CONFIGURATION, Panda, Pose
from src.rrt_connect import RRTConnect
from src.scene import build
from src.sdf_data import SignedDistanceOracle
from src.sdf_model import load
from src.shortcut import path_length, shortcut

MODEL = Path(__file__).resolve().parent.parent / "data" / "sdf_divided.pt"


def endpoints(panda, scene):
    """IK solutions for the pick and place landmarks, lifted clear."""
    home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
    out = {}
    for label in ("pick", "place"):
        x, y, z = scene.landmarks[label]
        out[label] = panda.inverse_kinematics(
            Pose(position=(x, y, z + 0.10),
                 orientation=home_pose.orientation)).configuration
    return out["pick"], out["place"]


def roughness(path) -> float:
    """Sum of squared accelerations, on a common 40 waypoint resampling.

    Resampling first matters: a path with six waypoints and one with forty
    are not otherwise comparable, since the second difference depends on
    spacing.
    """
    points = resample(path, 40)
    acceleration = points[:-2] - 2.0 * points[1:-1] + points[2:]
    return float((acceleration ** 2).sum())


def clearance_profile(oracle, path, waypoints: int = 40):
    """True environment clearance at evenly spaced points along a path."""
    points = resample(path, waypoints)
    return np.array([oracle(list(row))[0] for row in points])


def exact_valid(checker, path, resolution: float = 0.01) -> bool:
    """Is every segment collision free under exact geometry?"""
    return all(checker.edge_is_valid(a, b, resolution=resolution)
               for a, b in zip(path, path[1:]))


def main() -> None:
    if not MODEL.exists():
        print(f"missing {MODEL}. Run examples/train_sdf.py first.")
        return

    model = load(MODEL)

    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)
        oracle = SignedDistanceOracle(panda, checker)
        start, goal = endpoints(panda, scene)

        start_clearance = oracle(start)[0]
        goal_clearance = oracle(goal)[0]
        print(f"fixed endpoints: start clearance {start_clearance:.4f} m, "
              f"goal {goal_clearance:.4f} m")
        print("no method can improve on these, so whole path minimum "
              "clearance is bounded by them\n")

        optimizer = TrajectoryOptimizer(model, panda.lower_limits,
                                        panda.upper_limits)

        header = (f"{'seed':>4} {'method':<22} {'length':>7} {'rough':>8} "
                  f"{'min all':>9} {'min mid':>9} {'time':>7}  exact")
        print(header)
        print("-" * len(header))

        totals = {}

        for seed in range(5):
            planned = RRTConnect(panda, checker, seed=seed).plan(start, goal)
            if not planned.found:
                print(f"{seed:>4}  planning failed")
                continue

            variants = [("RRT-Connect", planned.path, 0.0)]

            began = time.perf_counter()
            trimmed = shortcut(planned.path, checker, max_attempts=200,
                               seed=seed)
            variants.append(("shortcut", trimmed.path,
                             time.perf_counter() - began))

            began = time.perf_counter()
            optimized = optimizer.optimize(planned.path)
            variants.append(("optimized", optimized.path,
                             time.perf_counter() - began))

            began = time.perf_counter()
            both = optimizer.optimize(trimmed.path)
            variants.append(("shortcut + optimized", both.path,
                             time.perf_counter() - began))

            for name, path, elapsed in variants:
                profile = clearance_profile(oracle, path)
                interior = profile[4:-4]  # drop the fixed endpoint region

                length = path_length(path)
                rough = roughness(path)
                valid = exact_valid(checker, path)

                totals.setdefault(name, []).append(
                    (length, rough, profile.min(), interior.min(), valid))

                print(f"{seed:>4} {name:<22} {length:>7.2f} {rough:>8.4f} "
                      f"{profile.min():>8.4f}m {interior.min():>8.4f}m "
                      f"{elapsed:>6.2f}s  {'ok' if valid else 'REJECTED'}")
            print()

        print(f"{'method':<22} {'length':>8} {'rough':>9} {'min all':>10} "
              f"{'min mid':>10} {'valid':>7}")
        print("-" * 70)
        for name, rows in totals.items():
            lengths, roughs, alls, mids, valids = zip(*rows)
            print(f"{name:<22} {np.mean(lengths):>8.2f} "
                  f"{np.mean(roughs):>9.4f} {np.mean(alls):>9.4f}m "
                  f"{np.mean(mids):>9.4f}m {sum(valids)}/{len(valids):>5}")


if __name__ == "__main__":
    main()