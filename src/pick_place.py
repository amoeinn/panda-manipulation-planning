"""A pick and place sequence for the Panda.

Planning a single motion is one query. A manipulation task is a sequence of
them, each with its own start and goal, and the sequence has structure that
a single query does not: phases must run in order, a failure in one must
stop the rest rather than produce a partial motion, and the world changes
partway through.

That last point is the substantive one, and getting it half right is worse
than not doing it at all. An early version removed the grasped block from
the obstacle list and stopped there, so during the transfer the planner
neither avoided the block nor accounted for it. The arm cleared the wall
with 27 mm to spare while the block it was carrying passed 34 mm through
the wall, and nothing in the verification noticed, because nothing was
looking at the block. The checker now attaches the block instead: it stops
being an obstacle and starts being a moving body whose swept volume is
checked against the world at every configuration.

What counts as a fault changes by phase. Descending to set the block down
means touching the surface it lands on, which would be a collision during
the transfer and is the purpose of the placement, so that contact is
allowed explicitly for the phases where it is intended.

The standoff height is computed in world coordinates. A held block hangs
about a grasp height below the gripper, so for its underside to clear the
wall the gripper must reach the wall top plus that grasp height plus a
margin. Working in offsets relative to landmarks instead invites mixing
datums: an earlier version measured the wall from the table surface and the
standoff from the landmark 30 mm above it, and produced a target beyond the
arm's usable reach.

The phases:

  approach   move above the block, gripper open
  descend    straight down to the grasp pose
  grasp      close the fingers, attach the block to the gripper
  lift       straight up, clear of the table
  transfer   plan over the wall to above the place location
  place      straight down to the release pose
  release    open the fingers, detach the block
  retreat    straight up and away
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pybullet as p

from .panda import FINGER_CLOSED, FINGER_OPEN, Panda, Pose

Configuration = List[float]
Transform = Tuple[Tuple[float, float, float],
                  Tuple[float, float, float, float]]


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
    block: Optional[int] = None
    block_start: Optional[Transform] = None
    grasp_relative: Optional[Transform] = None

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


def place_held_body(panda: Panda, body: int, relative: Transform,
                    link: int = 11) -> None:
    """Move a held body to match the arm's current configuration.

    Replay teleports joints with resetJointState, which bypasses the physics
    constraint solver, so a held object has to be positioned explicitly.
    """
    state = p.getLinkState(panda.robot, link, computeForwardKinematics=True)
    position, orientation = p.multiplyTransforms(state[4], state[5], *relative)
    p.resetBasePositionAndOrientation(body, position, orientation)


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
                 standoff_z: Optional[float] = None,
                 grasp_height: float = 0.055,
                 wall_margin: float = 0.055):
        """
        Args:
            panda: a Panda instance.
            checker: a CollisionChecker supporting attach and detach.
            scene: a Scene with pick and place landmarks and a block body.
            plan: a function taking two configurations and returning a path
                or None. Lets the caller choose planner and post processing.
            standoff_z: world height, in metres, for the gripper during the
                transfer. Derived from the wall when not given, since a
                standoff below the wall makes the transfer impossible for a
                carried block no matter how well it is planned.
            grasp_height: metres above the landmark for the grasp pose. Also
                how far the block hangs below the gripper once held.
            wall_margin: extra clearance above the wall top, in metres.
        """
        self.panda = panda
        self.checker = checker
        self.scene = scene
        self.plan = plan
        self.grasp_height = grasp_height

        # The divided table scene lists table, wall, block, place marker.
        self.table, self.wall, self.block, self.marker = scene.bodies

        if standoff_z is None:
            wall_top = scene.landmarks["wall_top"][2]
            standoff_z = wall_top + grasp_height + wall_margin
        self.standoff_z = standoff_z

    def _pose_at(self, landmark: str, world_z: float) -> Pose:
        """A gripper pose above a landmark, at an absolute world height."""
        x, y, _ = self.scene.landmarks[landmark]
        home = self.panda.forward_kinematics()
        return Pose(position=(x, y, world_z), orientation=home.orientation)

    def _grasp_z(self, landmark: str) -> float:
        return self.scene.landmarks[landmark][2] + self.grasp_height

    def _solve(self, landmark: str, world_z: float,
               seed: Optional[Configuration] = None
               ) -> Tuple[Optional[Configuration], str]:
        """IK for a pose above a landmark, verified before it is returned."""
        result = self.panda.inverse_kinematics(
            self._pose_at(landmark, world_z), seed_configuration=seed)
        if not result.solved:
            return None, (f"no IK solution at z={world_z:.3f}, "
                          f"error {result.position_error:.4f} m")
        if not self.checker.is_valid(result.configuration):
            return None, (f"IK solution at z={world_z:.3f} invalid: "
                          f"{self.checker.check(result.configuration)}")
        return result.configuration, ""

    def run(self, home: Configuration) -> PickPlaceResult:
        """Execute the full sequence from a home configuration."""
        result = PickPlaceResult(
            block=self.block,
            block_start=p.getBasePositionAndOrientation(self.block),
        )

        def fail(name: str, reason: str) -> PickPlaceResult:
            result.phases.append(Phase(name=name, failure=reason))
            return result

        self.checker.detach(as_obstacle=True)
        self.panda.set_configuration(home)
        self.panda.set_gripper(FINGER_OPEN)

        pick_above, error = self._solve("pick", self.standoff_z)
        if pick_above is None:
            return fail("approach", error)

        path = self.plan(home, pick_above)
        if path is None:
            return fail("approach", "no path from home to the pick standoff")
        result.phases.append(Phase("approach", path, gripper=FINGER_OPEN,
                                   planner="planned"))

        # Descending onto the block means touching it, so it is neither an
        # obstacle nor attached during this phase.
        if self.block in self.checker.obstacles:
            self.checker.obstacles.remove(self.block)
        pick_at, error = self._solve("pick", self._grasp_z("pick"),
                                     seed=pick_above)
        if pick_at is None:
            return fail("descend", error)
        result.phases.append(Phase("descend", straight_line(pick_above, pick_at),
                                   gripper=FINGER_OPEN, planner="straight line"))

        self.panda.set_configuration(pick_at)
        self.panda.set_gripper(FINGER_CLOSED)
        self.checker.attach(self.block)
        result.grasp_relative = self.checker.attached.relative
        result.phases.append(Phase("grasp", [pick_at], gripper=FINGER_CLOSED,
                                   attached=True, planner="gripper only"))

        result.phases.append(Phase("lift", straight_line(pick_at, pick_above),
                                   gripper=FINGER_CLOSED, attached=True,
                                   planner="straight line"))

        place_above, error = self._solve("place", self.standoff_z,
                                         seed=pick_above)
        if place_above is None:
            return fail("transfer", error)

        transfer = self.plan(pick_above, place_above)
        if transfer is None:
            return fail("transfer", "no path over the wall carrying the block")
        result.phases.append(Phase("transfer", transfer,
                                   gripper=FINGER_CLOSED, attached=True,
                                   planner="planned"))

        # Setting the block down means touching what it lands on. The place
        # marker and the table are intended contacts from here on.
        self.checker.allow_contact_with(self.marker, self.table)

        place_at, error = self._solve("place", self._grasp_z("place"),
                                      seed=place_above)
        if place_at is None:
            return fail("place", error)
        result.phases.append(Phase("place",
                                   straight_line(place_above, place_at),
                                   gripper=FINGER_CLOSED, attached=True,
                                   planner="straight line"))

        self.panda.set_configuration(place_at)
        place_held_body(self.panda, self.block, self.checker.attached.relative)
        self.checker.detach(as_obstacle=False)
        self.panda.set_gripper(FINGER_OPEN)
        result.phases.append(Phase("release", [place_at], gripper=FINGER_OPEN,
                                   planner="gripper only"))

        result.phases.append(Phase("retreat",
                                   straight_line(place_at, place_above),
                                   gripper=FINGER_OPEN,
                                   planner="straight line"))

        return result