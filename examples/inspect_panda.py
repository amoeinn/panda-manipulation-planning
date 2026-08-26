"""Print the Panda's joint structure.

Before planning anything, we need to know what the robot is: which joints
actually move, what their limits are, and which link is the end effector.
PyBullet ships the Panda URDF, so nothing needs downloading.

Note that PyBullet indexes every joint including fixed ones, and link
indices match joint indices. The seven arm joints are 0 through 6; the two
finger joints at 9 and 10 are a gripper, not planning degrees of freedom.

Usage:
    python examples/inspect_panda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybullet as p
import pybullet_data

JOINT_TYPES = {
    p.JOINT_REVOLUTE: "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR: "planar",
    p.JOINT_FIXED: "fixed",
}


def main() -> None:
    # DIRECT is a headless connection: physics without a window.
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    num_joints = p.getNumJoints(robot)
    print(f"\nfranka_panda/panda.urdf: {num_joints} joints\n")

    header = (f"{'idx':>3}  {'name':<26} {'type':<10} "
              f"{'lower':>8} {'upper':>8} {'range':>8}")
    print(header)
    print("-" * len(header))

    revolute, prismatic = [], []
    for i in range(num_joints):
        info = p.getJointInfo(robot, i)
        name = info[1].decode()
        jtype_code = info[2]
        jtype = JOINT_TYPES.get(jtype_code, "unknown")
        lower, upper = info[8], info[9]

        if jtype_code == p.JOINT_FIXED:
            limits = f"{'':>8} {'':>8} {'':>8}"
        else:
            limits = f"{lower:>8.3f} {upper:>8.3f} {upper - lower:>8.3f}"
            if jtype_code == p.JOINT_REVOLUTE:
                revolute.append(i)
            elif jtype_code == p.JOINT_PRISMATIC:
                prismatic.append(i)

        print(f"{i:>3}  {name:<26} {jtype:<10} {limits}")

    print(f"\narm joints (plan over these): {revolute}")
    print(f"arm degrees of freedom: {len(revolute)}")
    print(f"finger joints (open/close together): {prismatic}")

    # Frame a grasp pose is measured from, between the fingertips rather
    # than at the wrist.
    ee_index = 11
    print(f"end effector link {ee_index}: "
          f"{p.getJointInfo(robot, ee_index)[12].decode()}")

    # Asymmetric limits matter: uniform sampling over a shared range would
    # generate invalid configurations for these joints.
    print("\njoints whose range is not centred on zero:")
    for i in revolute:
        lower, upper = p.getJointInfo(robot, i)[8:10]
        if abs(lower + upper) > 0.2:
            print(f"  joint {i}: [{lower:.3f}, {upper:.3f}]")

    p.disconnect()


if __name__ == "__main__":
    main()