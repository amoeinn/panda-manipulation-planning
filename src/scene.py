"""Workspace scenes for the Panda.

An empty scene barely constrains a fixed base arm, so a planner has nothing
to do. These scenes add the furniture a pick and place task actually has: a
table the arm must stay above, objects on it, and a divider that blocks the
direct route between two sides.

Bodies are created from primitive shapes rather than loaded from URDFs, so
a scene is defined in one place and its dimensions are visible in the code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pybullet as p

Vector3 = Tuple[float, float, float]

# Muted colors, distinguishable in the viewer without being garish.
TABLE_COLOR = (0.72, 0.68, 0.62, 1.0)
WALL_COLOR = (0.60, 0.62, 0.66, 1.0)
OBJECT_COLOR = (0.80, 0.42, 0.30, 1.0)
TARGET_COLOR = (0.30, 0.55, 0.45, 1.0)


@dataclass
class Scene:
    """Bodies making up a workspace, and the landmarks a task refers to."""

    bodies: List[int] = field(default_factory=list)
    landmarks: Dict[str, Vector3] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.bodies)


def _box(half_extents: Vector3, position: Vector3,
         color: Tuple[float, float, float, float],
         mass: float = 0.0) -> int:
    """Create a box. Mass zero makes it static, which is right for furniture."""
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents,
                                 rgbaColor=color)
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


def empty_scene() -> Scene:
    """Nothing but the ground plane the Panda already loads."""
    return Scene()


def table_scene(height: float = 0.20) -> Scene:
    """A table in front of the robot.

    The arm must now work above a surface rather than in open space, which
    rules out the whole lower half of its reachable volume.
    """
    top = _box(half_extents=(0.35, 0.45, 0.02),
               position=(0.55, 0.0, height),
               color=TABLE_COLOR)
    return Scene(
        bodies=[top],
        landmarks={"table_surface": (0.55, 0.0, height + 0.02)},
    )


def divided_table_scene(height: float = 0.20,
                        wall_height: float = 0.25) -> Scene:
    """A table split by a vertical wall, with a pick and a place location.

    The wall runs along x and is thin in y, so it separates the near side
    of the table from the far side. Pick and place sit at y = -0.22 and
    y = +0.22, on opposite sides, which means the direct route between them
    is blocked and the arm has to lift over. This is the geometry that makes
    a planner earn its keep, and it is what a bin to bin transfer looks like.
    """
    surface = height + 0.02
    top = _box(half_extents=(0.35, 0.45, 0.02),
               position=(0.55, 0.0, height),
               color=TABLE_COLOR)
    wall = _box(half_extents=(0.30, 0.02, wall_height / 2),
                position=(0.55, 0.0, surface + wall_height / 2),
                color=WALL_COLOR)

    pick = (0.55, -0.22, surface + 0.03)
    place = (0.55, 0.22, surface + 0.03)

    pick_marker = _box(half_extents=(0.03, 0.03, 0.03),
                       position=pick, color=OBJECT_COLOR)
    place_marker = _box(half_extents=(0.04, 0.04, 0.002),
                        position=(place[0], place[1], surface),
                        color=TARGET_COLOR)

    return Scene(
        bodies=[top, wall, pick_marker, place_marker],
        landmarks={
            "table_surface": (0.55, 0.0, surface),
            "pick": pick,
            "place": place,
            "wall_top": (0.55, 0.0, surface + wall_height),
        },
    )

SCENES = {
    "empty": empty_scene,
    "table": table_scene,
    "divided": divided_table_scene,
}


def build(name: str) -> Scene:
    """Create a scene by name."""
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}, choose from {sorted(SCENES)}")
    return SCENES[name]()