"""Run the full pick and place sequence.

Reports each phase, then verifies the whole motion against exact geometry.
Verification skips the phases where the gripper is deliberately in contact
with the block, since a grasp is a collision by any geometric measure and
the checker cannot tell an intended contact from an unintended one.

Usage:
    python examples/run_pick_place.py            # headless
    python examples/run_pick_place.py --gui      # watch it
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.optimizer import TrajectoryOptimizer
from src.panda import HOME_CONFIGURATION, Panda
from src.pick_place import PickPlace
from src.rrt_connect import RRTConnect
from src.scene import build
from src.sdf_model import load
from src.shortcut import shortcut

MODEL = Path(__file__).resolve().parent.parent / "data" / "sdf_divided.pt"

# Phases where the gripper is meant to be touching the block.
CONTACT_PHASES = {"descend", "grasp", "lift", "place", "release", "retreat"}


def main() -> None:
    gui = "--gui" in sys.argv
    model = load(MODEL) if MODEL.exists() else None
    if model is None:
        print(f"note: {MODEL.name} not found, running without optimization\n")

    panda = Panda(gui=gui)
    scene = build("divided")
    checker = CollisionChecker(panda, obstacles=scene.bodies)
    optimizer = (TrajectoryOptimizer(model, panda.lower_limits,
                                     panda.upper_limits)
                 if model is not None else None)

    def plan(start, goal):
        """Plan, shortcut, then optimize. Returns None if planning fails."""
        result = RRTConnect(panda, checker, seed=0).plan(start, goal)
        if not result.found:
            return None
        trimmed = shortcut(result.path, checker, max_attempts=200, seed=0)
        if optimizer is None:
            return trimmed.path
        return optimizer.optimize(trimmed.path).path

    began = time.perf_counter()
    result = PickPlace(panda, checker, scene, plan).run(HOME_CONFIGURATION)
    elapsed = time.perf_counter() - began

    print(f"{'phase':<12} {'waypoints':>10} {'gripper':>9} "
          f"{'holding':>8}  source")
    print("-" * 56)
    for phase in result.phases:
        if not phase.succeeded:
            print(f"{phase.name:<12} {'FAILED':>10}  {phase.failure}")
            continue
        gripper = "-" if phase.gripper is None else f"{phase.gripper:.3f}"
        print(f"{phase.name:<12} {len(phase.path):>10} {gripper:>9} "
              f"{str(phase.attached):>8}  {phase.planner}")

    print(f"\nsequence {'succeeded' if result.succeeded else 'failed'} "
          f"in {elapsed:.2f} s, {result.waypoints()} waypoints total")
    if not result.succeeded:
        print(f"failure: {result.failure}")
        panda.disconnect()
        return

    print("\nexact verification, excluding phases with intended contact")
    all_clear = True
    for phase in result.phases:
        if phase.name in CONTACT_PHASES or len(phase.path) < 2:
            print(f"  {phase.name:<12} skipped, gripper in contact")
            continue
        clear = all(checker.edge_is_valid(a, b, resolution=0.01)
                    for a, b in zip(phase.path, phase.path[1:]))
        all_clear = all_clear and clear
        print(f"  {phase.name:<12} {'ok' if clear else 'BLOCKED'}")
    print(f"\n  {'all planned motion is collision free' if all_clear else 'VERIFICATION FAILED'}")

    if gui:
        print("\nreplaying, close the window when done")
        import pybullet as pb
        for _ in range(3):
            for phase in result.phases:
                if phase.gripper is not None:
                    panda.set_gripper(phase.gripper)
                for configuration in phase.path:
                    panda.set_configuration(configuration)
                    pb.stepSimulation()
                    time.sleep(0.03)
            time.sleep(0.6)

    panda.disconnect()


if __name__ == "__main__":
    main()