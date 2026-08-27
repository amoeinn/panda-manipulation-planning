"""Train the learned signed distance field and check it is usable.

Accuracy on its own does not tell us whether the model is any good for
planning. Three further checks here: how it compares against a predictor
that ignores its input entirely, whether the predicted sign agrees with the
truth near the collision boundary, and whether the gradient points in a
direction that actually increases clearance when followed.

Usage:
    python examples/train_sdf.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.collision import CollisionChecker
from src.panda import Panda
from src.scene import build
from src.sdf_data import SDFDataset, SignedDistanceOracle
from src.sdf_model import ENVIRONMENT, LearnedSDF, save
from src.sdf_train import train

DATA = Path(__file__).resolve().parent.parent / "data" / "sdf_divided_200000.npz"
MODEL = Path(__file__).resolve().parent.parent / "data" / "sdf_divided.pt"


def load_dataset() -> SDFDataset:
    stored = np.load(DATA)
    return SDFDataset(
        configurations=stored["configurations"],
        environment_distance=stored["environment_distance"],
        self_distance=stored["self_distance"],
    )


def report(model_metrics, baseline_metrics) -> None:
    """Print each metric beside what a constant predictor would score."""
    for name in ("environment", "self"):
        constant = baseline_metrics[f"{name}_constant"]
        print(f"\n  {name}  (constant baseline predicts "
              f"{constant * 1000:.1f} mm)")
        rows = [
            ("mean absolute error, mm", f"{name}_mae", 1000.0, "{:.1f}"),
            ("sign agreement, all", f"{name}_sign_agreement", 100.0, "{:.1f}%"),
        ]
        if f"{name}_boundary_mae" in model_metrics:
            count = int(model_metrics[f"{name}_boundary_count"])
            rows += [
                (f"boundary mae, mm ({count} samples)",
                 f"{name}_boundary_mae", 1000.0, "{:.1f}"),
                ("boundary sign agreement",
                 f"{name}_boundary_sign", 100.0, "{:.1f}%"),
            ]
        print(f"    {'metric':<34} {'model':>10} {'baseline':>10}")
        for label, key, scale, fmt in rows:
            model_value = fmt.format(model_metrics[key] * scale)
            base_value = fmt.format(baseline_metrics[key] * scale)
            print(f"    {label:<34} {model_value:>10} {base_value:>10}")


def main() -> None:
    dataset = load_dataset()
    print(f"dataset: {len(dataset)} samples from {DATA.name}")

    began = time.perf_counter()
    model, history = train(dataset, epochs=150)
    print(f"trained in {time.perf_counter() - began:.1f} s")

    print(f"final train loss {history.train_loss[-1]:.5f}, "
          f"test loss {history.test_loss[-1]:.5f}")

    print("\nheld out accuracy, against a predictor that ignores its input")
    report(history.metrics, history.baseline)

    save(model, MODEL)
    print(f"\nsaved {MODEL} ({MODEL.stat().st_size / 1e6:.1f} MB)")

    print("\ndoes the gradient point somewhere useful?")
    learned = LearnedSDF(model)

    with Panda(gui=False) as panda:
        scene = build("divided")
        checker = CollisionChecker(panda, obstacles=scene.bodies)
        oracle = SignedDistanceOracle(panda, checker)

        rng = np.random.default_rng(11)
        improved = 0
        trials = 0
        step = 0.05

        while trials < 200:
            candidate = [float(rng.uniform(lo, hi))
                         for lo, hi in zip(panda.lower_limits,
                                           panda.upper_limits)]
            before, _ = oracle(candidate)
            if abs(before) > 0.05:
                continue  # only interesting near the boundary

            direction = learned.gradient(candidate, target=ENVIRONMENT)
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            moved = np.asarray(candidate) + step * direction / norm
            moved = np.clip(moved, panda.lower_limits, panda.upper_limits)

            after, _ = oracle(list(moved))
            improved += int(after > before)
            trials += 1

        print(f"  followed the environment gradient one 0.05 rad step from "
              f"{trials} configurations near the boundary")
        print(f"  clearance increased in {improved / trials:.1%} of them")

        random_improved = 0
        rng = np.random.default_rng(11)
        trials = 0
        while trials < 200:
            candidate = [float(rng.uniform(lo, hi))
                         for lo, hi in zip(panda.lower_limits,
                                           panda.upper_limits)]
            before, _ = oracle(candidate)
            if abs(before) > 0.05:
                continue
            direction = rng.normal(size=7)
            moved = np.asarray(candidate) + step * direction / np.linalg.norm(direction)
            moved = np.clip(moved, panda.lower_limits, panda.upper_limits)
            after, _ = oracle(list(moved))
            random_improved += int(after > before)
            trials += 1

        print(f"  a random direction increased it in "
              f"{random_improved / trials:.1%}")


if __name__ == "__main__":
    main()