"""Collision checking for the Panda.

Inverse kinematics answers "what joint angles put the gripper there" and
nothing more. It will happily return a configuration with the arm buried in
the floor or folded through itself. A planner needs a separate question
answered: is this configuration physically valid?

Some link pairs touch no matter what the arm does. Links separated only by
a fixed joint are rigidly attached, and the robot base rests on the ground
plane. Rather than hand listing those pairs, which means guessing at the
model's internal structure, they are found by measurement: sample the arm
across many configurations, and any pair in contact in every single one is
structural and gets disabled. This is the approach MoveIt's setup assistant
takes, and it adapts automatically if the model changes.

The two fingers are excluded outright rather than by calibration. Their
separation is a commanded width, not a consequence of the arm's pose, so a
calibration run with the gripper open records them as separable and a later
closed gripper then reports a permanent self collision. They are also
mechanically mirrored and cannot be driven into each other, so the pair
carries no information at any width.

A grasped object is neither an obstacle nor irrelevant. Once the gripper
closes on a block, the block travels with the arm: it must stop being
something to avoid and start being something whose swept volume is checked
against the world. Attaching it here rather than merely removing it from
the obstacle list is what makes the planner route a carried object over a
wall instead of dragging it through one.

Some contact is intended. A grasp means the fingers touch the block, and
setting the block down means the block touches the surface it lands on.
Both are expressed as allowances on the attached object rather than by
skipping the check, so anything else touching it is still caught.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

Configuration = List[float]
LinkPair = Tuple[int, int]

# Link index of the robot base. PyBullet numbers it separately from joints.
BASE_LINK = -1

# The two prismatic finger links. Their separation is commanded, not a
# property of the arm's configuration, so it is never a collision.
FINGER_PAIR: LinkPair = (9, 10)

# Frame the gripper holds objects relative to.
END_EFFECTOR_LINK = 11

# Hand and fingers: the links a grasped object is expected to touch.
GRIPPER_LINKS = (8, 9, 10, 11)

Transform = Tuple[Tuple[float, float, float],
                  Tuple[float, float, float, float]]


@dataclass
class AttachedObject:
    """A body rigidly held by the gripper.

    Stores the pose of the body relative to the end effector at the moment
    it was grasped, so its world pose can be recomputed for any arm
    configuration without stepping physics.
    """

    body: int
    relative: Transform
    allowed_links: Set[int] = field(default_factory=set)
    allowed_bodies: Set[int] = field(default_factory=set)


@dataclass
class CollisionReport:
    """What, if anything, a configuration collides with."""

    self_pairs: List[LinkPair] = field(default_factory=list)
    environment: Dict[int, List[int]] = field(default_factory=dict)
    attached: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def free(self) -> bool:
        return not self.self_pairs and not self.environment and not self.attached

    def __repr__(self) -> str:
        if self.free:
            return "CollisionReport(free)"
        parts = []
        if self.self_pairs:
            parts.append(f"self={self.self_pairs}")
        if self.environment:
            links = sorted({l for ls in self.environment.values() for l in ls})
            parts.append(f"environment links={links}")
        if self.attached:
            parts.append(f"held object hits bodies={sorted(self.attached)}")
        return f"CollisionReport({', '.join(parts)})"


class CollisionChecker:
    """Answers whether a configuration or a motion between two is valid."""

    def __init__(self, panda, obstacles: Optional[Iterable[int]] = None,
                 margin: float = 0.0, calibration_samples: int = 200,
                 seed: int = 0):
        """
        Args:
            panda: a Panda instance.
            obstacles: body ids to treat as environment. The ground plane is
                included automatically.
            margin: treat links closer than this as colliding. A small
                positive value gives a safety buffer.
            calibration_samples: how many random configurations to use when
                identifying structurally permanent contacts.
            seed: makes the calibration reproducible.
        """
        self.panda = panda
        self.robot = panda.robot
        self.margin = margin
        self.obstacles = [panda.plane] + list(obstacles or [])
        self.num_links = p.getNumJoints(self.robot)
        self.attached: Optional[AttachedObject] = None

        # Every link pair, base included, that is worth testing at all.
        self._all_pairs = [
            (a, b)
            for a in range(BASE_LINK, self.num_links)
            for b in range(a + 1, self.num_links)
            if (a, b) != FINGER_PAIR
        ]

        self.ignored_self_pairs: Set[LinkPair] = set()
        self.ignored_environment: Set[Tuple[int, int]] = set()
        self._calibrate(calibration_samples, seed)

        self._pairs = [pair for pair in self._all_pairs
                       if pair not in self.ignored_self_pairs]

    # ------------------------------------------------------------ attachment

    def attach(self, body: int, allowed_links: Optional[Iterable[int]] = None,
               allowed_bodies: Optional[Iterable[int]] = None) -> None:
        """Treat a body as held by the gripper from now on.

        The body stops being an obstacle and starts being checked against
        the rest of the world at every configuration. Its pose relative to
        the end effector is recorded now and reapplied later.

        Args:
            body: the body id to attach.
            allowed_links: robot links permitted to touch it, the hand and
                fingers by default, since a grasp is a contact.
            allowed_bodies: world bodies permitted to touch it, for
                intended contacts such as the surface it will be set on.
        """
        link_state = p.getLinkState(self.robot, END_EFFECTOR_LINK,
                                    computeForwardKinematics=True)
        body_position, body_orientation = p.getBasePositionAndOrientation(body)
        inverse = p.invertTransform(link_state[4], link_state[5])
        relative = p.multiplyTransforms(inverse[0], inverse[1],
                                        body_position, body_orientation)

        self.attached = AttachedObject(
            body=body,
            relative=relative,
            allowed_links=set(allowed_links if allowed_links is not None
                              else GRIPPER_LINKS),
            allowed_bodies=set(allowed_bodies or ()),
        )
        if body in self.obstacles:
            self.obstacles.remove(body)

    def allow_contact_with(self, *bodies: int) -> None:
        """Permit the held object to touch these bodies from now on.

        Used when a phase changes what counts as intended: descending to set
        a block down means touching the surface below it, which is a fault
        during the transfer and the point of the placement.
        """
        if self.attached is not None:
            self.attached.allowed_bodies.update(bodies)

    def detach(self, as_obstacle: bool = True) -> None:
        """Release the held body, optionally restoring it as an obstacle."""
        if self.attached is None:
            return
        body = self.attached.body
        self.attached = None
        if as_obstacle and body not in self.obstacles:
            self.obstacles.append(body)

    def _place_attached(self) -> None:
        """Move the held body to match the current arm configuration."""
        if self.attached is None:
            return
        link_state = p.getLinkState(self.robot, END_EFFECTOR_LINK,
                                    computeForwardKinematics=True)
        position, orientation = p.multiplyTransforms(
            link_state[4], link_state[5], *self.attached.relative)
        p.resetBasePositionAndOrientation(self.attached.body,
                                          position, orientation)

    def _attached_hits(self) -> Dict[int, List[int]]:
        """Bodies the held object touches, at the current configuration."""
        if self.attached is None:
            return {}
        self._place_attached()

        hits: Dict[int, List[int]] = {}
        for body in self.obstacles:
            if body in self.attached.allowed_bodies:
                continue
            if p.getClosestPoints(self.attached.body, body, self.margin):
                hits[body] = [-1]

        touching_robot = [
            contact[3]
            for contact in p.getClosestPoints(self.attached.body, self.robot,
                                              self.margin)
            if contact[3] not in self.attached.allowed_links
        ]
        if touching_robot:
            hits[self.robot] = sorted(set(touching_robot))
        return hits

    # ----------------------------------------------------------- calibration

    def _calibrate(self, samples: int, seed: int) -> None:
        """Find contacts that persist across every sampled configuration.

        A pair touching in one pose is a collision. A pair touching in all of
        them is how the robot is built.
        """
        rng = np.random.default_rng(seed)
        saved = self.panda.get_configuration()

        candidate_self: Optional[Set[LinkPair]] = None
        candidate_env: Optional[Set[Tuple[int, int]]] = None

        configurations = [list(saved)]
        for _ in range(samples):
            configurations.append([
                float(rng.uniform(lo, hi))
                for lo, hi in zip(self.panda.lower_limits,
                                  self.panda.upper_limits)
            ])

        for configuration in configurations:
            self.panda.set_configuration(configuration)

            touching_self = {
                (a, b) for a, b in self._all_pairs
                if p.getClosestPoints(self.robot, self.robot, self.margin,
                                      linkIndexA=a, linkIndexB=b)
            }
            touching_env = {
                (body, contact[3])
                for body in self.obstacles
                for contact in p.getClosestPoints(self.robot, body, self.margin)
            }

            candidate_self = (touching_self if candidate_self is None
                              else candidate_self & touching_self)
            candidate_env = (touching_env if candidate_env is None
                             else candidate_env & touching_env)

            if not candidate_self and not candidate_env:
                break

        self.ignored_self_pairs = candidate_self or set()
        self.ignored_environment = candidate_env or set()
        self.panda.set_configuration(saved)

    # --------------------------------------------------------------- queries

    def _with_configuration(self, configuration: Optional[Configuration]):
        """Set a configuration if given, returning the one to restore."""
        if configuration is None:
            return None
        saved = self.panda.get_configuration()
        self.panda.set_configuration(configuration)
        return saved

    def check(self, configuration: Optional[Configuration] = None) -> CollisionReport:
        """Report every collision for a configuration.

        Slower than in_collision because it does not stop at the first hit.
        Use it for diagnostics, not inside a planning loop.
        """
        saved = self._with_configuration(configuration)

        self_pairs = [
            (a, b) for a, b in self._pairs
            if p.getClosestPoints(self.robot, self.robot, self.margin,
                                  linkIndexA=a, linkIndexB=b)
        ]

        environment: Dict[int, List[int]] = {}
        for body in self.obstacles:
            links = sorted({
                contact[3]
                for contact in p.getClosestPoints(self.robot, body, self.margin)
                if (body, contact[3]) not in self.ignored_environment
            })
            if links:
                environment[body] = links

        attached = self._attached_hits()

        if saved is not None:
            self.panda.set_configuration(saved)
        return CollisionReport(self_pairs, environment, attached)

    def in_collision(self, configuration: Optional[Configuration] = None) -> bool:
        """Is this configuration in collision? Returns on the first hit.

        This is the function a planner calls thousands of times, so it exits
        as early as it can. Environment checks come first because a single
        query covers the whole robot, while self collision needs one query
        per link pair.
        """
        saved = self._with_configuration(configuration)
        hit = False

        for body in self.obstacles:
            for contact in p.getClosestPoints(self.robot, body, self.margin):
                if (body, contact[3]) not in self.ignored_environment:
                    hit = True
                    break
            if hit:
                break

        if not hit:
            for a, b in self._pairs:
                if p.getClosestPoints(self.robot, self.robot, self.margin,
                                      linkIndexA=a, linkIndexB=b):
                    hit = True
                    break

        if not hit and self.attached is not None:
            hit = bool(self._attached_hits())

        if saved is not None:
            self.panda.set_configuration(saved)
        return hit

    def is_valid(self, configuration: Configuration) -> bool:
        """Within joint limits and collision free."""
        return (self.panda.within_limits(configuration)
                and not self.in_collision(configuration))

    # ------------------------------------------------------------------ edges

    def edge_is_valid(self, start: Configuration, end: Configuration,
                      resolution: float = 0.05) -> bool:
        """Is the straight line between two configurations collision free?

        Checking only the endpoints is not enough: an arm can start and end
        clear while sweeping through an obstacle in between. The segment is
        sampled at a fixed angular resolution, which is what makes collision
        checking the expensive part of any sampling based planner.

        Args:
            resolution: maximum joint angle change, in radians, between
                consecutive samples.
        """
        start_array = np.asarray(start, dtype=float)
        end_array = np.asarray(end, dtype=float)
        largest_step = float(np.max(np.abs(end_array - start_array)))
        steps = max(2, int(np.ceil(largest_step / resolution)) + 1)

        for t in np.linspace(0.0, 1.0, steps):
            if self.in_collision(start_array + t * (end_array - start_array)):
                return False
        return True