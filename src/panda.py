"""A thin wrapper around the PyBullet Franka Panda model.

PyBullet's API is index based and verbose: joint states come back as tuples,
link indices and joint indices share a numbering that includes fixed joints,
and every query needs the body id passed in. This class holds that detail in
one place so planning code can work in terms of configurations.

A configuration here always means the seven arm joint angles, in radians,
ordered by joint index. The gripper is not part of a configuration; it is
commanded separately.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

# Seven revolute arm joints. Indices 7, 8 and 11 are fixed; 9 and 10 are
# the fingers. See examples/inspect_panda.py for the full listing.
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]
FINGER_JOINTS = [9, 10]

# Frame between the fingertips, which is what a grasp pose refers to.
END_EFFECTOR_LINK = 11

FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0

# A comfortable arm pose, away from joint limits and self collision.
HOME_CONFIGURATION = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# A solve is accepted when the end effector lands this close to the target.
IK_TOLERANCE = 0.005

Configuration = Sequence[float]


@dataclass(frozen=True)
class Pose:
    """A position and orientation in world coordinates."""

    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion, xyzw

    def __repr__(self) -> str:
        x, y, z = self.position
        return f"Pose(xyz=({x:.3f}, {y:.3f}, {z:.3f}))"


@dataclass
class IKResult:
    """The outcome of an inverse kinematics solve.

    PyBullet's solver always returns joint angles, even for targets that are
    unreachable, so the caller has to check rather than trust. This carries
    the verification alongside the answer.
    """

    configuration: List[float]
    position_error: float
    within_limits: bool

    @property
    def solved(self) -> bool:
        return self.position_error <= IK_TOLERANCE and self.within_limits


class Panda:
    """A Franka Panda loaded into a PyBullet simulation."""

    def __init__(self, gui: bool = False, base_position=(0.0, 0.0, 0.0)):
        """Connect to PyBullet and load the robot on a ground plane.

        Args:
            gui: open a viewer window. False runs headless, which is much
                faster and is what planning and tests should use.
            base_position: where to put the robot base.
        """
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=base_position,
            useFixedBase=True,
        )

        self.lower_limits, self.upper_limits = self._read_limits()
        self.set_configuration(HOME_CONFIGURATION)
        self.set_gripper(FINGER_OPEN)

    # ---------------------------------------------------------------- state

    def _read_limits(self) -> Tuple[List[float], List[float]]:
        """Read per joint limits from the URDF rather than hard coding them."""
        lower, upper = [], []
        for joint in ARM_JOINTS:
            info = p.getJointInfo(self.robot, joint)
            lower.append(info[8])
            upper.append(info[9])
        return lower, upper

    def get_configuration(self) -> List[float]:
        """Return the current arm joint angles."""
        return [s[0] for s in p.getJointStates(self.robot, ARM_JOINTS)]

    def set_configuration(self, configuration: Configuration) -> None:
        """Teleport the arm to a configuration.

        This bypasses physics entirely, which is what a planner wants: it is
        asking "would this pose be in collision", not "can the motors get
        there".
        """
        if len(configuration) != len(ARM_JOINTS):
            raise ValueError(
                f"expected {len(ARM_JOINTS)} joint angles, got {len(configuration)}"
            )
        for joint, angle in zip(ARM_JOINTS, configuration):
            p.resetJointState(self.robot, joint, angle)

    def set_gripper(self, width: float) -> None:
        """Open or close both fingers together. Width is per finger."""
        width = float(np.clip(width, FINGER_CLOSED, FINGER_OPEN))
        for joint in FINGER_JOINTS:
            p.resetJointState(self.robot, joint, width)

    # ------------------------------------------------------------ kinematics

    def forward_kinematics(self, configuration: Optional[Configuration] = None) -> Pose:
        """Where does the end effector sit for a given configuration?

        Forward kinematics is the easy direction: joint angles determine the
        gripper pose uniquely. If no configuration is given, the current one
        is used.
        """
        if configuration is None:
            return self._end_effector_pose()

        saved = self.get_configuration()
        self.set_configuration(configuration)
        pose = self._end_effector_pose()
        self.set_configuration(saved)
        return pose

    def _end_effector_pose(self) -> Pose:
        state = p.getLinkState(self.robot, END_EFFECTOR_LINK,
                               computeForwardKinematics=True)
        # Index 4 and 5 are the link frame in world coordinates, which is
        # what we want. Index 0 and 1 are the centre of mass frame.
        return Pose(position=state[4], orientation=state[5])

    def inverse_kinematics(
        self,
        target: Pose,
        seed_configuration: Optional[Configuration] = None,
        iterations: int = 200,
        residual_threshold: float = 1e-4,
    ) -> IKResult:
        """Find joint angles putting the end effector at a target pose.

        This is the hard direction. A 7 DOF arm reaching a 6 DOF pose is
        redundant, so solutions form a continuum rather than a discrete set,
        and some targets have no solution at all.

        The solver is seeded from the robot's current joint state, so the
        seed is set by moving the arm before solving rather than by passing
        restPoses. PyBullet's null space arguments were measured to be both
        silently ignored when their length does not match the number of
        movable joints, and roughly twenty times less accurate when it does.
        See examples/debug_ik.py for that experiment.

        Because the solver returns angles for unreachable targets rather than
        failing, the answer is verified with forward kinematics and against
        the joint limits before being handed back.
        """
        saved = self.get_configuration()
        if seed_configuration is not None:
            self.set_configuration(seed_configuration)

        solution = p.calculateInverseKinematics(
            self.robot,
            END_EFFECTOR_LINK,
            target.position,
            target.orientation,
            maxNumIterations=iterations,
            residualThreshold=residual_threshold,
        )
        # PyBullet returns a value for every movable joint, fingers included.
        configuration = list(solution[: len(ARM_JOINTS)])

        achieved = self.forward_kinematics(configuration)
        error = float(np.linalg.norm(
            np.array(achieved.position) - np.array(target.position)
        ))

        self.set_configuration(saved)
        return IKResult(
            configuration=configuration,
            position_error=error,
            within_limits=self.within_limits(configuration),
        )

    # -------------------------------------------------------------- validity

    def within_limits(self, configuration: Configuration) -> bool:
        """Is every joint inside its own range?"""
        return all(
            lower <= angle <= upper
            for angle, lower, upper in zip(
                configuration, self.lower_limits, self.upper_limits
            )
        )

    def disconnect(self) -> None:
        p.disconnect(self.client)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.disconnect()