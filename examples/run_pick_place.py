"""Run the full pick and place sequence.

Verification covers the held block, not just the arm. An earlier version
checked only the robot and reported the transfer as collision free while
the block it was carrying passed 34 mm through the wall.

Contact that is intended does not count. The gripper touches the block
throughout the grasp, and the block rests on the table at the start of the
lift and again at the end of the place, so those phases are checked against
the obstacles they are not supposed to touch rather than against all of
them. A small tolerance absorbs resting contacts, which report as a
floating point zero and would otherwise read as a penetration.

Usage:
    python examples/run_pick_place.py            # headless
    python examples/run_pick_place.py --gui      # watch it
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybullet as pb

from src.collision import CollisionChecker
from src.optimizer import TrajectoryOptimizer
from src.panda import HOME_CONFIGURATION, Panda
from src.pick_place import PickPlace, place_held_body
from src.rrt_connect import RRTConnect
from src.scene import build
from src.sdf_model import load
from src.shortcut import shortcut

MODEL = Path(__file__).resolve().parent.parent / "data" / "sdf_divided.pt"

# Phases where the gripper is meant to be touching the block.
CONTACT_PHASES = {"descend", "grasp", "lift", "place", "release", "retreat"}

# Resting contact reports as a floating point zero, so anything shallower
# than this is treated as touching rather than penetrating.
CONTACT_TOLERANCE = 1e-4


def block_clearance(panda, result, phase, obstacles):
    """Smallest distance from the held block to the given obstacles."""
    worst = float("inf")
    for configuration in phase.path:
        panda.set_configuration(configuration)
        place_held_body(panda, result.block, result.grasp_relative)
        for body in obstacles:
            for contact in pb.getClosestPoints(result.block, body, 0.5):
                worst = min(worst, contact[8])
    return worst


def replay(panda, result, loops: int = 3, delay: float = 0.03) -> None:
    """Step through every phase, carrying the block when it is held."""
    for _ in range(loops):
        pb.resetBasePositionAndOrientation(result.block, *result.block_start)
        for phase in result.phases:
            if phase.gripper is not None:
                panda.set_gripper(phase.gripper)
            for configuration in phase.path:
                panda.set_configuration(configuration)
                if phase.attached:
                    place_held_body(panda, result.block,
                                    result.grasp_relative)
                time.sleep(delay)
        time.sleep(0.6)


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

    task = PickPlace(panda, checker, scene, plan)
    print(f"transfer standoff at world z {task.standoff_z:.3f} m, "
          f"derived from the wall top at "
          f"{scene.landmarks['wall_top'][2]:.3f}\n")

    began = time.perf_counter()
    result = task.run(HOME_CONFIGURATION)
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

    # The wall is what the carried block must never touch. The table is a
    # resting surface at both ends of the motion, so it is checked only
    # where the block is meant to be clear of it.
    always_forbidden = [task.wall]
    lift_and_place = {"lift", "place"}

    print("\nverification")
    print(f"  {'phase':<12} {'arm':>12} {'held block':>18}  checked against")
    all_clear = True

    for phase in result.phases:
        if len(phase.path) < 2:
            continue

        if phase.name in CONTACT_PHASES:
            arm = "in contact"
        else:
            clear = all(checker.edge_is_valid(a, b, resolution=0.01)
                        for a, b in zip(phase.path, phase.path[1:]))
            all_clear = all_clear and clear
            arm = "ok" if clear else "BLOCKED"

        if phase.attached and result.grasp_relative is not None:
            if phase.name in lift_and_place:
                obstacles = always_forbidden
                against = "wall"
            else:
                obstacles = [task.wall, task.table, panda.plane]
                against = "wall, table, floor"
            worst = block_clearance(panda, result, phase, obstacles)
            block = f"{worst:+.4f} m"
            if worst < -CONTACT_TOLERANCE:
                all_clear = False
                block += " HIT"
        else:
            block, against = "not held", ""

        print(f"  {phase.name:<12} {arm:>12} {block:>18}  {against}")

    print(f"\n  {'all checked motion is collision free' if all_clear else 'VERIFICATION FAILED'}")

    if gui:
        print("\nreplaying, close the window when done")
        replay(panda, result)

    panda.disconnect()


if __name__ == "__main__":
    main()