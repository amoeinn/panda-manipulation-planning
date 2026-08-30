"""A pick and place sequence for the Panda.

Planning a single motion is one query. A manipulation task is a sequence of
them, each with its own start and goal, and the sequence has structure that
a single query does not: phases must run in order, a failure in one must
stop the rest rather than produce a partial motion, and the world changes
partway through.

That last point is the substantive one. Once the gripper closes on the
block, the block travels with the arm. Until then it is an obstacle to
avoid; afterward it is part of the moving body and the table below it is
what must be avoided. A checker that keeps treating the grasped block as a
fixed obstacle will refuse to plan the transfer, because the arm is holding
the thing it is trying to stay away from.

The phases:

  approach   move above the block, gripper open
  descend    straight down to the grasp pose
  grasp      close the fingers, attach the block
  lift       straight up, clear of the table
  transfer   plan over the wall to above the place location
  place      straight down to the release pose
  release    open the fingers, detach the block
  retreat    straight up and away
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from .panda import FINGER_CLOSED, FINGER_OPEN, Panda, Pose

Configuration = List[float]


@dataclass
class Phase:
    """One step of the sequence."""

    name: str
    path: List[Configuration] = field(default_factory=list)
    gripper: Optional[float] = None      # commanded width, if any
    attached: bool = False               # is the block held during this phase
    planner: str = ""                    # how the path was produced
    failure: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


@dataclass
class PickPlaceResult:
    """The outcome of the whole sequence."""

    phases: List[Phase] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return bool(self.phases) and all(p.succeeded for p in self.phases)

    @property
    def failure(self) -> Optional[str]:
        for phase in self.phases:
            if phase.failure:
                return f"{phase.name}: {phase.failure}"
        return None

    def waypoints(self) -> int:
        return sum(len(p.path) for p in self.phases)


class GraspedObject:
    """Tracks a block that is rigidly held by the gripper.

    PyBullet has constraints for this, but for planning we only need the
    block to move with the arm and to stop being an obstacle, which a fixed
    constraint plus a change to the checker's obstacle list achieves.
    """

    def __init__(self, panda: Panda, body: int, end_effector_link: int = 11):
        self.panda = panda
        self.body = body
        self.link = end_effector_link
        self.constraint: Optional[int] = None

    def attach(self) -> None:
        """Rigidly fix the block to the gripper at its current relative pose."""
        if self.constraint is not None:
            return
        link_state = p.getLinkState(self.panda.robot, self.link,
                                    computeForwardKinematics=True)
        block_position, block_orientation = p.getBasePositionAndOrientation(
            self.body)

        inverse_position, inverse_orientation = p.invertTransform(
            link_state[4], link_state[5])
        relative_position, relative_orientation = p.multiplyTransforms(
            inverse_position, inverse_orientation,
            block_position, block_orientation)

        self.constraint = p.createConstraint(
            parentBodyUniqueId=self.panda.robot,
            parentLinkIndex=self.link,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=relative_position,
            childFramePosition=(0, 0, 0),
            parentFrameOrientation=relative_orientation,
        )

    def detach(self) -> None:
        if self.constraint is not None:
            p.removeConstraint(self.constraint)
            self.constraint = None


def straight_line(start: Configuration, end: Configuration,
                  waypoints: int = 12) -> List[Configuration]:
    """Interpolate between two configurations.

    Used for the short approach and retreat motions, where a planner would
    be overkill and a straight line in joint space is what is wanted.
    """
    a, b = np.asarray(start), np.asarray(end)
    return [list(a + t * (b - a)) for t in np.linspace(0.0, 1.0, waypoints)]


class PickPlace:
    """Runs the pick and place sequence, planning each phase in turn."""

    def __init__(self, panda: Panda, checker, scene,
                 plan: Callable[[Configuration, Configuration],
                                Optional[List[Configuration]]],
                 approach_height: float = 0.14,
                 grasp_height: float = 0.055):
        """
        Args:
            panda: a Panda instance.
            checker: a CollisionChecker whose obstacle list can be changed
                when the block is grasped.
            scene: a Scene with pick and place landmarks and a block body.
            plan: a function taking two configurations and returning a path
                or None. Lets the caller choose planner and post processing.
            approach_height: metres above the landmark for the standoff
                pose, chosen to clear the wall.
            grasp_height: metres above the landmark for the grasp pose.
        """
        self.panda = panda
        self.checker = checker
        self.scene = scene
        self.plan = plan
        self.approach_height = approach_height
        self.grasp_height = grasp_height

        # The block is the third body in the divided table scene: table,
        # wall, block, place marker.
        self.block = scene.bodies[2]
        self.grasped = GraspedObject(panda, self.block)

    def _pose_above(self, landmark: str, height: float) -> Pose:
        x, y, z = self.scene.landmarks[landmark]
        home = self.panda.forward_kinematics()
        return Pose(position=(x, y, z + height), orientation=home.orientation)

    def _solve(self, landmark: str, height: float,
               seed: Optional[Configuration] = None
               ) -> Tuple[Optional[Configuration], str]:
        """IK for a pose above a landmark, verified before it is returned."""
        result = self.panda.inverse_kinematics(
            self._pose_above(landmark, height), seed_configuration=seed)
        if not result.solved:
            return None, f"no IK solution, error {result.position_error:.4f} m"
        if not self.checker.is_valid(result.configuration):
            return None, "IK solution is in collision"
        return result.configuration, ""

    def _set_block_as_obstacle(self, is_obstacle: bool) -> None:
        """Add or remove the block from what the checker avoids."""
        if is_obstacle and self.block not in self.checker.obstacles:
            self.checker.obstacles.append(self.block)
        elif not is_obstacle and self.block in self.checker.obstacles:
            self.checker.obstacles.remove(self.block)

    def run(self, home: Configuration) -> PickPlaceResult:
        """Execute the full sequence from a home configuration."""
        result = PickPlaceResult()

        def fail(name: str, reason: str) -> PickPlaceResult:
            result.phases.append(Phase(name=name, failure=reason))
            return result

        # The block is an obstacle until it is grasped.
        self._set_block_as_obstacle(True)
        self.panda.set_configuration(home)
        self.panda.set_gripper(FINGER_OPEN)

        pick_above, error = self._solve("pick", self.approach_height)
        if pick_above is None:
            return fail("approach", error)

        path = self.plan(home, pick_above)
        if path is None:
            return fail("approach", "no path from home to the pick standoff")
        result.phases.append(Phase("approach", path, gripper=FINGER_OPEN,
                                   planner="planned"))

        # Descending onto the block means touching it, so it stops being an
        # obstacle for this phase.
        self._set_block_as_obstacle(False)
        pick_at, error = self._solve("pick", self.grasp_height,
                                     seed=pick_above)
        if pick_at is None:
            return fail("descend", error)
        result.phases.append(Phase("descend", straight_line(pick_above, pick_at),
                                   gripper=FINGER_OPEN, planner="straight line"))

        self.panda.set_configuration(pick_at)
        self.panda.set_gripper(FINGER_CLOSED)
        self.grasped.attach()
        result.phases.append(Phase("grasp", [pick_at], gripper=FINGER_CLOSED,
                                   attached=True, planner="gripper only"))

        result.phases.append(Phase("lift", straight_line(pick_at, pick_above),
                                   gripper=FINGER_CLOSED, attached=True,
                                   planner="straight line"))

        place_above, error = self._solve("place", self.approach_height,
                                         seed=pick_above)
        if place_above is None:
            return fail("transfer", error)

        transfer = self.plan(pick_above, place_above)
        if transfer is None:
            return fail("transfer", "no path over the wall")
        result.phases.append(Phase("transfer", transfer,
                                   gripper=FINGER_CLOSED, attached=True,
                                   planner="planned"))

        place_at, error = self._solve("place", self.grasp_height,
                                      seed=place_above)
        if place_at is None:
            return fail("place", error)
        result.phases.append(Phase("place",
                                   straight_line(place_above, place_at),
                                   gripper=FINGER_CLOSED, attached=True,
                                   planner="straight line"))

        self.panda.set_configuration(place_at)
        self.grasped.detach()
        self.panda.set_gripper(FINGER_OPEN)
        result.phases.append(Phase("release", [place_at], gripper=FINGER_OPEN,
                                   planner="gripper only"))

        result.phases.append(Phase("retreat",
                                   straight_line(place_at, place_above),
                                   gripper=FINGER_OPEN,
                                   planner="straight line"))

        self._set_block_as_obstacle(True)
        return result