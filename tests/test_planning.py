"""Tests for the Panda planning stack.

The properties asserted here are the ones that were actually violated at
some point while building this, plus the invariants that make the rest
trustworthy. Every regression test below corresponds to a bug that shipped
briefly and was caught by measurement rather than by reading the code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybullet as pb
import pytest
import torch

from src.collision import CollisionChecker
from src.optimizer import resample, smoothness_cost
from src.panda import HOME_CONFIGURATION, Panda, Pose
from src.pick_place import PickPlace, place_held_body, straight_line
from src.rrt_connect import RRTConnect
from src.scene import build
from src.sdf_data import SignedDistanceOracle
from src.shortcut import path_length, shortcut


@pytest.fixture(scope="module")
def world():
    """One simulator for the whole module. Loading is the slow part."""
    panda = Panda(gui=False)
    scene = build("divided")
    checker = CollisionChecker(panda, obstacles=scene.bodies)
    oracle = SignedDistanceOracle(panda, checker)
    yield panda, scene, checker, oracle
    panda.disconnect()


@pytest.fixture(scope="module")
def query(world):
    """Pick and place configurations, lifted clear of the table."""
    panda, scene, checker, _ = world
    home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
    out = []
    for label in ("pick", "place"):
        x, y, z = scene.landmarks[label]
        result = panda.inverse_kinematics(
            Pose(position=(x, y, z + 0.10),
                 orientation=home_pose.orientation))
        assert result.solved, f"IK failed for {label}"
        assert checker.is_valid(result.configuration)
        out.append(result.configuration)
    return tuple(out)


class TestPanda:
    def test_home_is_collision_free(self, world):
        _, _, checker, _ = world
        assert checker.is_valid(HOME_CONFIGURATION)

    def test_forward_kinematics_is_deterministic(self, world):
        panda, _, _, _ = world
        first = panda.forward_kinematics(HOME_CONFIGURATION)
        second = panda.forward_kinematics(HOME_CONFIGURATION)
        assert first.position == second.position

    def test_forward_kinematics_restores_configuration(self, world):
        """Querying a pose must not leave the robot somewhere else."""
        panda, _, _, _ = world
        panda.set_configuration(HOME_CONFIGURATION)
        panda.forward_kinematics([0.3] * 7)
        assert np.allclose(panda.get_configuration(), HOME_CONFIGURATION,
                           atol=1e-9)

    def test_ik_reports_failure_for_unreachable_targets(self, world):
        panda, _, _, _ = world
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        far = Pose(position=(1.5, 0.0, 0.5),
                   orientation=home_pose.orientation)
        assert not panda.inverse_kinematics(far).solved

    def test_ik_solution_is_verified_not_trusted(self, world):
        """A solved result must actually reach its target."""
        panda, _, _, _ = world
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        target = Pose(position=(0.5, 0.0, 0.5),
                      orientation=home_pose.orientation)
        result = panda.inverse_kinematics(target)
        assert result.solved
        achieved = panda.forward_kinematics(result.configuration)
        assert np.linalg.norm(np.array(achieved.position)
                              - np.array(target.position)) < 0.01

    def test_redundancy_gives_different_postures(self, world):
        """A 7 DOF arm reaching a 6 DOF pose has many solutions.

        Regression: passing null space arguments whose length did not match
        the nine movable joints made PyBullet silently ignore them, so every
        seed returned an identical solution.
        """
        panda, _, _, _ = world
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        target = Pose(position=(0.5, 0.0, 0.5),
                      orientation=home_pose.orientation)
        seeds = [HOME_CONFIGURATION,
                 [0.5, -0.6, 0.3, -2.0, 0.2, 1.4, 0.9],
                 [-0.8, -1.0, -0.4, -2.5, -0.3, 1.8, 0.2]]
        solutions = [panda.inverse_kinematics(target, seed_configuration=s)
                     for s in seeds]
        assert all(s.solved for s in solutions)
        spread = max(np.abs(np.array(a.configuration)
                            - np.array(b.configuration)).max()
                     for a in solutions for b in solutions)
        assert spread > 0.1


class TestCollisionChecker:
    def test_structural_contacts_are_excluded(self, world):
        """The base rests on the plane permanently.

        Regression: without calibration, every configuration reported a
        collision because that contact was counted.
        """
        _, _, checker, _ = world
        assert checker.ignored_environment

    def test_closed_gripper_is_not_a_self_collision(self, world):
        """Regression: calibration runs with the gripper open, so the finger
        pair is recorded as separable. Closing the gripper put both fingers
        at the same position and every validity check failed afterward."""
        panda, _, checker, _ = world
        panda.set_gripper(0.0)
        valid = checker.is_valid(HOME_CONFIGURATION)
        panda.set_gripper(0.04)
        assert valid

    def test_edge_checking_catches_what_endpoints_miss(self, world):
        """Two valid configurations can have an invalid motion between them.

        Regression: RRT-Connect validated only the configuration each step
        ended at, so the arm could sweep through the wall between two poses
        that were each individually clear.
        """
        panda, _, checker, _ = world
        rng = np.random.default_rng(0)
        found = False
        for _ in range(400):
            a = [float(rng.uniform(lo, hi))
                 for lo, hi in zip(panda.lower_limits, panda.upper_limits)]
            b = [float(rng.uniform(lo, hi))
                 for lo, hi in zip(panda.lower_limits, panda.upper_limits)]
            if (checker.is_valid(a) and checker.is_valid(b)
                    and not checker.edge_is_valid(a, b, resolution=0.02)):
                found = True
                break
        assert found, "expected some valid pair with a blocked motion"


class TestAttachedObject:
    """Regression tests for the bug that let a carried block pass through
    the wall while the motion verified as collision free."""

    def test_attached_object_is_checked_against_the_world(self, world):
        """An attached body must be able to invalidate a configuration.

        This is the property that was missing entirely: grasping removed
        the block from the obstacle list and put nothing in its place, so
        nothing looked at it again.
        """
        panda, scene, checker, _ = world
        _, wall, block, _ = scene.bodies

        # A pose that puts the gripper low, beside the wall, where a held
        # block would intersect it.
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        x, y, z = scene.landmarks["pick"]
        low = panda.inverse_kinematics(
            Pose(position=(x, y, z + 0.055),
                 orientation=home_pose.orientation))
        assert low.solved

        checker.obstacles.remove(block)
        panda.set_configuration(low.configuration)
        checker.attach(block)

        # Sweep toward the wall until the held block hits something.
        blocked = False
        for offset in np.linspace(0.0, 0.44, 25):
            target = panda.inverse_kinematics(
                Pose(position=(x, y + offset, z + 0.055),
                     orientation=home_pose.orientation),
                seed_configuration=low.configuration)
            if not target.solved:
                continue
            if checker.in_collision(target.configuration):
                blocked = True
                break

        checker.detach(as_obstacle=True)
        assert blocked, "a held block never invalidated any configuration"

    def test_detach_restores_the_obstacle(self, world):
        panda, scene, checker, _ = world
        _, _, block, _ = scene.bodies
        panda.set_configuration(HOME_CONFIGURATION)
        checker.attach(block)
        assert block not in checker.obstacles
        checker.detach(as_obstacle=True)
        assert block in checker.obstacles
        assert checker.attached is None

    def test_allowed_bodies_permit_intended_contact(self, world):
        """Setting a block down means touching what it lands on."""
        panda, scene, checker, _ = world
        table, _, block, _ = scene.bodies
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        x, y, z = scene.landmarks["place"]

        down = panda.inverse_kinematics(
            Pose(position=(x, y, z + 0.055),
                 orientation=home_pose.orientation))
        assert down.solved

        checker.obstacles.remove(block)
        panda.set_configuration(down.configuration)
        checker.attach(block)
        checker.allow_contact_with(table, *scene.bodies[3:])
        permitted = checker.is_valid(down.configuration)
        checker.detach(as_obstacle=True)
        assert permitted

    def test_held_block_pose_follows_the_gripper(self, world):
        """The block's pose relative to the gripper must not change.

        Not that it translates the same distance as the gripper: the block
        is held at an offset, so any rotation of the end effector swings it
        through an arc and it travels further than the gripper origin does.
        Rigid attachment is about the relative transform, not the distance.

        Tolerance is 1e-6 rather than tighter. Recovering the relative pose
        means a quaternion inversion and two multiplications, and PyBullet
        works in single precision, so agreement to about eight significant
        figures is the most that survives the round trip.
        """
        panda, scene, checker, _ = world
        _, _, block, _ = scene.bodies
        panda.set_configuration(HOME_CONFIGURATION)
        checker.attach(block)
        relative = checker.attached.relative

        def relative_pose():
            place_held_body(panda, block, relative)
            state = pb.getLinkState(panda.robot, 11,
                                    computeForwardKinematics=True)
            inverse = pb.invertTransform(state[4], state[5])
            block_pose = pb.getBasePositionAndOrientation(block)
            return pb.multiplyTransforms(inverse[0], inverse[1], *block_pose)

        before_position, before_orientation = relative_pose()
        first = panda.forward_kinematics()

        panda.set_configuration([0.3, -0.5, 0.1, -2.0, 0.0, 1.6, 0.7])
        after_position, after_orientation = relative_pose()
        second = panda.forward_kinematics()

        checker.detach(as_obstacle=True)

        moved = np.linalg.norm(np.array(second.position)
                               - np.array(first.position))
        assert moved > 0.05, "the gripper needs to actually move"
        assert np.allclose(before_position, after_position, atol=1e-6)
        assert np.allclose(before_orientation, after_orientation, atol=1e-6)


class TestPickPlaceGeometry:
    def test_standoff_clears_the_wall_for_a_carried_block(self, world):
        """Regression: a standoff 0.14 m above the landmark put a held block
        below the top of a 0.25 m wall, so no plan could get it across."""
        panda, scene, checker, _ = world
        task = PickPlace(panda, checker, scene, plan=lambda a, b: None)
        wall_top = scene.landmarks["wall_top"][2]
        block_bottom = task.standoff_z - task.grasp_height - 0.03
        assert block_bottom > wall_top

    def test_standoff_is_reachable(self, world):
        panda, scene, checker, _ = world
        task = PickPlace(panda, checker, scene, plan=lambda a, b: None)
        home_pose = panda.forward_kinematics(HOME_CONFIGURATION)
        for label in ("pick", "place"):
            x, y, _ = scene.landmarks[label]
            result = panda.inverse_kinematics(
                Pose(position=(x, y, task.standoff_z),
                     orientation=home_pose.orientation))
            assert result.solved, f"standoff unreachable above {label}"


class TestSignedDistanceOracle:
    def test_environment_distance_has_usable_range(self, world):
        """Regression: a combined minimum saturated at 0.021 m.

        Two arm links sit permanently about two centimetres apart, so a
        single scalar minimum reported the robot's own geometry rather than
        the obstacles. Separating the terms is what fixed it.
        """
        panda, _, _, oracle = world
        rng = np.random.default_rng(1)
        values = []
        for _ in range(150):
            c = [float(rng.uniform(lo, hi))
                 for lo, hi in zip(panda.lower_limits, panda.upper_limits)]
            values.append(oracle(c)[0])
        values = np.asarray(values)
        assert values.max() > 0.05
        assert values.min() < 0.0

    def test_agrees_with_the_collision_checker(self, world):
        """A negative clearance must mean the checker also sees a collision."""
        panda, _, checker, oracle = world
        rng = np.random.default_rng(2)
        for _ in range(60):
            c = [float(rng.uniform(lo, hi))
                 for lo, hi in zip(panda.lower_limits, panda.upper_limits)]
            worst = min(oracle(c))
            if abs(worst) < 0.002:
                continue  # too close to the boundary to compare reliably
            assert (worst < 0) == checker.in_collision(c)


class TestRRTConnect:
    def test_finds_a_path(self, world, query):
        panda, _, checker, _ = world
        start, goal = query
        result = RRTConnect(panda, checker, seed=0).plan(start, goal)
        assert result.found

    def test_path_endpoints_are_the_query(self, world, query):
        panda, _, checker, _ = world
        start, goal = query
        result = RRTConnect(panda, checker, seed=0).plan(start, goal)
        assert np.allclose(result.path[0], start, atol=1e-6)
        assert np.allclose(result.path[-1], goal, atol=1e-6)

    def test_every_segment_is_collision_free(self, world, query):
        """Regression: the tunnelling bug produced paths through the wall.

        Verified at a finer resolution than the planner used, so a path that
        merely stepped over an obstacle between checks still fails.
        """
        panda, _, checker, _ = world
        start, goal = query
        for seed in range(3):
            result = RRTConnect(panda, checker, seed=seed).plan(start, goal)
            assert result.found
            for a, b in zip(result.path, result.path[1:]):
                assert checker.edge_is_valid(a, b, resolution=0.005)

    def test_is_reproducible(self, world, query):
        panda, _, checker, _ = world
        start, goal = query
        first = RRTConnect(panda, checker, seed=3).plan(start, goal)
        second = RRTConnect(panda, checker, seed=3).plan(start, goal)
        assert np.allclose(first.path, second.path)

    def test_rejects_an_invalid_start(self, world, query):
        panda, _, checker, _ = world
        _, goal = query
        buried = [0.0, 1.5, 0.0, -0.2, 0.0, 0.2, 0.0]
        assert not checker.is_valid(buried)
        with pytest.raises(ValueError):
            RRTConnect(panda, checker, seed=0).plan(buried, goal)


class TestShortcut:
    def test_never_lengthens_a_path(self, world, query):
        panda, _, checker, _ = world
        start, goal = query
        for seed in range(3):
            planned = RRTConnect(panda, checker, seed=seed).plan(start, goal)
            trimmed = shortcut(planned.path, checker, max_attempts=100,
                               seed=seed)
            assert trimmed.final_length <= trimmed.original_length + 1e-9

    def test_result_stays_collision_free(self, world, query):
        """Regression: the splice checked the new middle segment but not the
        two joining segments it also created."""
        panda, _, checker, _ = world
        start, goal = query
        for seed in range(3):
            planned = RRTConnect(panda, checker, seed=seed).plan(start, goal)
            trimmed = shortcut(planned.path, checker, max_attempts=100,
                               seed=seed)
            for a, b in zip(trimmed.path, trimmed.path[1:]):
                assert checker.edge_is_valid(a, b, resolution=0.005)

    def test_preserves_endpoints(self, world, query):
        panda, _, checker, _ = world
        start, goal = query
        planned = RRTConnect(panda, checker, seed=0).plan(start, goal)
        trimmed = shortcut(planned.path, checker, max_attempts=100, seed=0)
        assert np.allclose(trimmed.path[0], start, atol=1e-6)
        assert np.allclose(trimmed.path[-1], goal, atol=1e-6)


class TestOptimizerPieces:
    def test_resample_keeps_endpoints(self):
        path = [[0.0] * 7, [0.5] * 7, [1.0] * 7]
        points = resample(path, 20)
        assert len(points) == 20
        assert np.allclose(points[0], path[0])
        assert np.allclose(points[-1], path[-1])

    def test_resample_spaces_evenly(self):
        """Uneven input spacing must not survive resampling."""
        path = [[0.0] * 7, [0.05] * 7, [1.0] * 7]
        points = resample(path, 30)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        assert steps.std() < 1e-6

    def test_smoothness_is_zero_for_a_straight_line(self):
        line = torch.tensor(
            np.linspace(0.0, 1.0, 20)[:, None].repeat(7, axis=1),
            dtype=torch.float32)
        assert float(smoothness_cost(line)) < 1e-9

    def test_smoothness_penalizes_a_reversal(self):
        line = torch.tensor(
            np.linspace(0.0, 1.0, 20)[:, None].repeat(7, axis=1),
            dtype=torch.float32)
        jagged = line.clone()
        jagged[10] += 0.5
        assert float(smoothness_cost(jagged)) > float(smoothness_cost(line))

    def test_straight_line_hits_both_ends(self):
        a = [0.0] * 7
        b = [0.4] * 7
        path = straight_line(a, b, waypoints=9)
        assert len(path) == 9
        assert np.allclose(path[0], a)
        assert np.allclose(path[-1], b)