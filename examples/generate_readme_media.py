"""Generate the images embedded in the README.

Two artefacts. An animated pick and place, rendered offscreen so the camera
is chosen rather than being whatever a window happened to show, and a
figure comparing what each stage of the pipeline does to a trajectory.

The camera sits on the far side of the table from the robot, high and
looking down across the wall. Chosen by rendering the sequence from a sweep
of angles and comparing the results rather than by reasoning about which
way yaw points, which is how the first attempt ended up with the wall
between the viewer and the grasp.

Usage:
    python examples/generate_readme_media.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
from PIL import Image

from src.collision import CollisionChecker
from src.optimizer import TrajectoryOptimizer, resample
from src.panda import FINGER_OPEN, HOME_CONFIGURATION, Panda
from src.pick_place import PickPlace, place_held_body
from src.rrt_connect import RRTConnect
from src.scene import build
from src.sdf_data import SignedDistanceOracle
from src.sdf_model import load
from src.shortcut import path_length, shortcut

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / "sdf_divided.pt"
DOCS = ROOT / "docs"

WIDTH, HEIGHT = 640, 480
FOV = 45

CAMERA = dict(
    cameraTargetPosition=(0.55, 0.0, 0.38),
    distance=1.35,
    yaw=105,
    pitch=-30,
    roll=0,
    upAxisIndex=2,
)


def render_frame(panda) -> Image.Image:
    """One offscreen frame from the fixed camera."""
    view = pb.computeViewMatrixFromYawPitchRoll(**CAMERA)
    projection = pb.computeProjectionMatrixFOV(
        fov=FOV, aspect=WIDTH / HEIGHT, nearVal=0.1, farVal=4.0)
    _, _, pixels, _, _ = pb.getCameraImage(
        WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=projection,
        renderer=pb.ER_TINY_RENDERER)
    return Image.fromarray(
        np.reshape(np.asarray(pixels, dtype=np.uint8),
                   (HEIGHT, WIDTH, 4))[:, :, :3])


def make_gif(panda, result, every: int = 2, hold: int = 8) -> Path:
    """Render the sequence to an animated GIF.

    Args:
        every: keep one frame in this many, to hold the file size down.
        hold: extra frames at the end before the loop restarts.
    """
    frames = []
    pb.resetBasePositionAndOrientation(result.block, *result.block_start)

    for phase in result.phases:
        if phase.gripper is not None:
            panda.set_gripper(phase.gripper)
        for index, configuration in enumerate(phase.path):
            panda.set_configuration(configuration)
            if phase.attached:
                place_held_body(panda, result.block, result.grasp_relative)
            # Always keep the first and last frame of a phase, so short
            # phases are not dropped entirely by the sampling.
            if index % every == 0 or index == len(phase.path) - 1:
                frames.append(render_frame(panda))

    frames.extend([frames[-1]] * hold)

    output = DOCS / "pick_place.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=60, loop=0, optimize=True)
    return output


def make_comparison(panda, checker, oracle, optimizer, scene) -> Path:
    """Figure comparing the planner, shortcutting and optimization.

    The clearance panel shows the mean across seeds with a band for the
    spread. Plotting every seed as its own line was unreadable: fifteen
    overlapping traces, and the methods indistinguishable.
    """
    from src.panda import Pose

    home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
    ends = {}
    for label in ("pick", "place"):
        x, y, z = scene.landmarks[label]
        ends[label] = panda.inverse_kinematics(
            Pose(position=(x, y, z + 0.10),
                 orientation=home_pose.orientation)).configuration
    start, goal = ends["pick"], ends["place"]

    variants = {}
    for seed in range(5):
        planned = RRTConnect(panda, checker, seed=seed).plan(start, goal)
        trimmed = shortcut(planned.path, checker, max_attempts=200, seed=seed)
        optimized = optimizer.optimize(trimmed.path)
        for name, path in (("RRT-Connect", planned.path),
                           ("shortcut", trimmed.path),
                           ("shortcut + optimized", optimized.path)):
            variants.setdefault(name, []).append(path)

    colors = {"RRT-Connect": "#6a6a6a",
              "shortcut": "#3a76c4",
              "shortcut + optimized": "#c4523a"}

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))

    samples = 60
    fraction = np.linspace(0, 1, samples)
    profiles = {}

    for name, paths in variants.items():
        traces = []
        for path in paths:
            points = resample(path, samples)
            traces.append([oracle(list(row))[0] * 1000 for row in points])
        traces = np.asarray(traces)
        profiles[name] = traces

        left.fill_between(fraction, traces.min(axis=0), traces.max(axis=0),
                          color=colors[name], alpha=0.15, linewidth=0)
        left.plot(fraction, traces.mean(axis=0), color=colors[name],
                  linewidth=2.0, label=name)

    left.axhline(50, color="#1a1a1a", linewidth=0.8, linestyle=(0, (4, 3)))
    left.text(0.015, 53, "optimizer margin", fontsize=8, color="#4a4a4a")
    left.set_xlabel("fraction along the trajectory")
    left.set_ylabel("clearance to nearest obstacle (mm)")
    left.set_title("Clearance along the transfer\n"
                   "mean and range over five seeds", fontsize=11)
    left.legend(fontsize=9, loc="lower right")
    left.grid(alpha=0.2, linewidth=0.5)
    left.set_xlim(0, 1)
    left.set_ylim(bottom=0)

    names = list(variants)
    lengths = [np.mean([path_length(p) for p in variants[n]]) for n in names]
    clearances = [profiles[n][:, 6:-6].min(axis=1).mean() for n in names]

    positions = np.arange(len(names))
    right.barh(positions - 0.19, lengths, height=0.36, color="#3a76c4")
    twin = right.twiny()
    twin.barh(positions + 0.19, clearances, height=0.36, color="#c4523a")

    for index, (length, clearance) in enumerate(zip(lengths, clearances)):
        right.text(length + 0.04, index - 0.19, f"{length:.2f}",
                   va="center", fontsize=8, color="#2a5490")
        twin.text(clearance + 0.6, index + 0.19, f"{clearance:.0f}",
                  va="center", fontsize=8, color="#8f3a29")

    right.set_yticks(positions)
    right.set_yticklabels(names, fontsize=9)
    right.set_xlabel("path length (rad)", color="#3a76c4")
    twin.set_xlabel("worst clearance away from the endpoints (mm)",
                    color="#c4523a")
    right.set_title("Shortcutting removes clearance to save length;\n"
                    "optimization restores it for almost none back",
                    fontsize=11)
    right.grid(axis="x", alpha=0.2, linewidth=0.5)
    right.invert_yaxis()
    right.set_xlim(0, max(lengths) * 1.18)
    twin.set_xlim(0, max(clearances) * 1.18)

    plt.tight_layout()
    output = DOCS / "comparison.png"
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    if not MODEL.exists():
        print(f"missing {MODEL}. Run examples/train_sdf.py first.")
        return

    DOCS.mkdir(exist_ok=True)
    model = load(MODEL)

    panda = Panda(gui=False)
    scene = build("divided")
    checker = CollisionChecker(panda, obstacles=scene.bodies)
    optimizer = TrajectoryOptimizer(model, panda.lower_limits,
                                    panda.upper_limits)

    def plan(start, goal):
        result = RRTConnect(panda, checker, seed=0).plan(start, goal)
        if not result.found:
            return None
        trimmed = shortcut(result.path, checker, max_attempts=200, seed=0)
        return optimizer.optimize(trimmed.path).path

    print("running the pick and place sequence")
    result = PickPlace(panda, checker, scene, plan).run(HOME_CONFIGURATION)
    if not result.succeeded:
        print(f"sequence failed: {result.failure}")
        panda.disconnect()
        return

    print("rendering the animation")
    gif = make_gif(panda, result)
    print(f"  {gif} ({gif.stat().st_size / 1e6:.1f} MB)")

    # A fresh checker: the pick and place sequence removes the block from
    # the obstacle list when it grasps and does not restore it, so reusing
    # that checker would plan the comparison in a world with no block in it.
    # The sequence leaves the world changed: the block ends at the place
    # location and is no longer in the obstacle list. Both have to be undone
    # before measuring, or the comparison plans around a block that has
    # already been moved and reports longer paths than the same query does
    # from a clean start.
    print("building the comparison figure")
    pb.resetBasePositionAndOrientation(result.block, *result.block_start)
    panda.set_configuration(HOME_CONFIGURATION)
    panda.set_gripper(FINGER_OPEN)
    comparison_checker = CollisionChecker(panda, obstacles=scene.bodies)
    oracle = SignedDistanceOracle(panda, comparison_checker)
    figure = make_comparison(panda, comparison_checker, oracle, optimizer,
                             scene)
    print(f"  {figure} ({figure.stat().st_size / 1e3:.0f} KB)")

    panda.disconnect()


if __name__ == "__main__":
    main()