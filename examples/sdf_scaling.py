"""Does the environment head improve with more data, or is it saturated?

At 20000 samples the model drove training loss to zero while test error
stayed flat, which is overfitting rather than the underfitting first
suspected. That points at data as the constraint, but pointing is not
measuring. This trains the same architecture on several subset sizes and
reports where each metric lands.

If boundary error falls steadily with data, more data is the fix and the
curve says roughly how much. If it flattens well short of useful, the
limitation is the representation: predicting a scalar from seven joint
angles asks the network to internally solve forward kinematics and then
compute distances to four separate bodies. cuRobo avoids that by computing
link positions analytically and querying a field indexed by 3D position,
which is a far easier function to learn.

Usage:
    python examples/sdf_scaling.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.sdf_data import SDFDataset
from src.sdf_train import train

DATA = Path(__file__).resolve().parent.parent / "data" / "sdf_divided_200000.npz"
SIZES = [10000, 25000, 50000, 100000, 200000]


def subset(dataset: SDFDataset, n: int, seed: int = 0) -> SDFDataset:
    """Take the first n samples, after a fixed shuffle."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset))[:n]
    return SDFDataset(
        configurations=dataset.configurations[order],
        environment_distance=dataset.environment_distance[order],
        self_distance=dataset.self_distance[order],
    )


def main() -> None:
    stored = np.load(DATA)
    full = SDFDataset(
        configurations=stored["configurations"],
        environment_distance=stored["environment_distance"],
        self_distance=stored["self_distance"],
    )
    print(f"loaded {len(full)} samples from {DATA.name}\n")

    header = (f"{'samples':>8} {'train':>8} {'test':>8} "
              f"{'env mae':>9} {'env bnd':>9} {'env sign':>9} "
              f"{'self bnd':>9} {'time':>7}")
    print(header)
    print("-" * len(header))

    for n in SIZES:
        part = subset(full, n)
        began = time.perf_counter()
        _, history = train(part, epochs=150, verbose=False)
        elapsed = time.perf_counter() - began

        m = history.metrics
        print(f"{n:>8} "
              f"{history.train_loss[-1]:>8.5f} "
              f"{history.test_loss[-1]:>8.5f} "
              f"{m['environment_mae'] * 1000:>8.1f}m "
              f"{m['environment_boundary_mae'] * 1000:>8.1f}m "
              f"{m['environment_boundary_sign']:>8.1%} "
              f"{m['self_boundary_mae'] * 1000:>8.1f}m "
              f"{elapsed:>6.0f}s")

    print("\nboundary mae for a constant predictor was 15.9 mm on the "
          "20000 sample split")
    print("the model has to beat that to be worth anything near contact")


if __name__ == "__main__":
    main()