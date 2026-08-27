"""Generate and inspect the signed distance dataset.

Worth reading the distributions before training anything on them. Three
failure modes turned up while building this: every sample reporting a
collision, because the base resting on the ground plane dominated the
minimum; every sample capped at two centimetres, because two arm links that
can never straighten apart dominated it instead; and the realisation that
no amount of pair exclusion fixes the second, because those links genuinely
can touch. Hence two separate targets rather than one.

Usage:
    python examples/build_sdf_dataset.py [samples]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.panda import HOME_CONFIGURATION, Panda
from src.scene import build
from src.sdf_data import MAX_QUERY_DISTANCE, SignedDistanceOracle, build_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def describe(name: str, values: np.ndarray) -> None:
    print(f"\n{name}")
    print(f"  range {values.min():.4f} to {values.max():.4f} m")
    print(f"  negative (in collision): {(values < 0).mean():.1%}")
    print(f"  within 5 cm of contact:  {(np.abs(values) < 0.05).mean():.1%}")
    print(f"  saturated at query limit: "
          f"{(values >= MAX_QUERY_DISTANCE - 1e-6).mean():.1%}")

    edges = [-1.0, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    counts, _ = np.histogram(values, bins=edges)
    for lo, hi, count in zip(edges, edges[1:], counts):
        bar = "#" * int(40 * count / max(counts.max(), 1))
        print(f"  {lo:>6.2f} to {hi:>5.2f}  {count:>7}  {bar}")


def main() -> None:
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    output = DATA_DIR / f"sdf_divided_{samples}.npz"

    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)
        oracle = SignedDistanceOracle(panda, checker)

        print(f"testable link pairs: {len(oracle.pairs)}")
        environment, own = oracle(HOME_CONFIGURATION)
        print(f"home configuration: environment {environment:.4f} m, "
              f"self {own:.4f} m")

        began = time.perf_counter()
        dataset = build_dataset(panda, oracle, samples=samples, seed=0)
        elapsed = time.perf_counter() - began
        print(f"\ngenerated {len(dataset)} samples in {elapsed:.1f} s "
              f"({len(dataset) / elapsed:.0f} per second)")

        describe("environment distance", dataset.environment_distance)
        describe("self distance", dataset.self_distance)

        binding = (dataset.self_distance < dataset.environment_distance)
        print(f"\nself is the binding term in {binding.mean():.1%} of samples")

        DATA_DIR.mkdir(exist_ok=True)
        np.savez_compressed(
            output,
            configurations=dataset.configurations,
            environment_distance=dataset.environment_distance,
            self_distance=dataset.self_distance,
        )
        print(f"\nsaved {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()