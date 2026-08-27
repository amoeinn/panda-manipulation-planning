"""Find what is capping the signed distance at about 2 cm.

Two dataset revisions have failed to move the maximum clearance above
0.0208 m, so the cap is not the pairs removed so far. This reports, for a
set of configurations, which specific body or link pair supplies the
minimum, and how that minimum is distributed across sources.

Usage:
    python examples/debug_sdf.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybullet as p

from src.collision import CollisionChecker
from src.panda import HOME_CONFIGURATION, Panda
from src.scene import build
from src.sdf_data import MAX_QUERY_DISTANCE, SignedDistanceOracle


def main() -> None:
    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)
        oracle = SignedDistanceOracle(panda, checker)

        print(f"obstacle body ids: {checker.obstacles}")
        print(f"plane id {panda.plane}, robot id {panda.robot}")
        print(f"scene bodies: {scene.bodies}")
        print(f"ignored environment contacts: "
              f"{sorted(checker.ignored_environment)}")

        print("\nwhat supplies the minimum, at the home configuration")
        panda.set_configuration(HOME_CONFIGURATION)

        rows = []
        for body in checker.obstacles:
            for contact in p.getClosestPoints(panda.robot, body,
                                              MAX_QUERY_DISTANCE):
                ignored = (body, contact[3]) in checker.ignored_environment
                rows.append((contact[8], f"body {body} link {contact[3]}",
                             ignored))
        for a, b in oracle.pairs:
            contacts = p.getClosestPoints(panda.robot, panda.robot,
                                          MAX_QUERY_DISTANCE,
                                          linkIndexA=a, linkIndexB=b)
            if contacts:
                rows.append((min(c[8] for c in contacts),
                             f"self {a} to {b}", False))

        rows.sort()
        print(f"  {'distance':>10}  {'source':<22} ignored")
        for distance, source, ignored in rows[:15]:
            print(f"  {distance:>10.4f}  {source:<22} {ignored}")

        print("\nover 300 random configurations, which source wins")
        rng = np.random.default_rng(0)
        winners = Counter()
        minima = []

        for _ in range(300):
            configuration = [float(rng.uniform(lo, hi))
                             for lo, hi in zip(panda.lower_limits,
                                               panda.upper_limits)]
            panda.set_configuration(configuration)

            best, best_source = MAX_QUERY_DISTANCE, "nothing within query range"
            for body in checker.obstacles:
                for contact in p.getClosestPoints(panda.robot, body,
                                                  MAX_QUERY_DISTANCE):
                    if (body, contact[3]) in checker.ignored_environment:
                        continue
                    if contact[8] < best:
                        best, best_source = contact[8], f"body {body}"
            for a, b in oracle.pairs:
                contacts = p.getClosestPoints(panda.robot, panda.robot,
                                              MAX_QUERY_DISTANCE,
                                              linkIndexA=a, linkIndexB=b)
                for contact in contacts:
                    if contact[8] < best:
                        best, best_source = contact[8], f"self {a} to {b}"

            winners[best_source] += 1
            minima.append(best)

        for source, count in winners.most_common(12):
            print(f"  {count:>4}  {source}")

        minima = np.asarray(minima)
        print(f"\nminimum over those 300: {minima.min():.4f}")
        print(f"maximum over those 300: {minima.max():.4f}")

        print("\nsanity: distance from the arm to the table alone, "
              "arm pointing up")
        upright = [0.0, -1.5, 0.0, -0.2, 0.0, 1.5, 0.0]
        panda.set_configuration(upright)
        for body in checker.obstacles:
            contacts = p.getClosestPoints(panda.robot, body,
                                          MAX_QUERY_DISTANCE)
            if contacts:
                print(f"  body {body}: {min(c[8] for c in contacts):.4f} m "
                      f"({len(contacts)} contact records)")
            else:
                print(f"  body {body}: beyond {MAX_QUERY_DISTANCE} m")


if __name__ == "__main__":
    main()