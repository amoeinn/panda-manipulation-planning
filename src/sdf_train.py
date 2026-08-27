"""Train the learned signed distance field.

Reports more than a loss value. A model can score a low mean absolute error
while being wrong about the sign of the distance near zero, and the sign
near zero is what a planner needs. Every metric is therefore reported both
overall and restricted to samples near contact, and beside a predictor that
ignores its input and returns the training median. A model that fails to
beat that baseline has learned the distribution rather than the geometry.

Samples near the boundary are weighted more heavily in the loss. Precision
at fourteen centimetres of clearance is worth little; precision at two
millimetres decides whether a trajectory is usable.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .sdf_model import ENVIRONMENT, SELF, SDFNetwork


@dataclass
class TrainingHistory:
    """Loss and metrics recorded during training."""

    train_loss: List[float] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    baseline: Dict[str, float] = field(default_factory=dict)


def _score(guess: np.ndarray, truth: np.ndarray, name: str,
           boundary_band: float) -> Dict[str, float]:
    """Error and sign agreement for one target, overall and near contact."""
    results = {
        f"{name}_mae": float(np.mean(np.abs(guess - truth))),
        f"{name}_sign_agreement": float(np.mean((guess < 0) == (truth < 0))),
    }
    near = np.abs(truth) < boundary_band
    if near.any():
        results[f"{name}_boundary_mae"] = float(
            np.mean(np.abs(guess[near] - truth[near])))
        results[f"{name}_boundary_sign"] = float(
            np.mean((guess[near] < 0) == (truth[near] < 0)))
        results[f"{name}_boundary_count"] = float(near.sum())
    return results


def evaluate(model: SDFNetwork, configurations: np.ndarray,
             targets: np.ndarray,
             boundary_band: float = 0.02) -> Dict[str, float]:
    """Measure error, and whether the sign is right where it matters."""
    model.eval()
    with torch.no_grad():
        predicted = model.distances(
            torch.tensor(configurations, dtype=torch.float32)).numpy()

    results: Dict[str, float] = {}
    for index, name in ((ENVIRONMENT, "environment"), (SELF, "self")):
        results.update(_score(predicted[:, index], targets[:, index],
                              name, boundary_band))
    return results


def constant_baseline(train_targets: np.ndarray, test_targets: np.ndarray,
                      boundary_band: float = 0.02) -> Dict[str, float]:
    """Score a predictor that ignores its input and returns the median."""
    results: Dict[str, float] = {}
    for index, name in ((ENVIRONMENT, "environment"), (SELF, "self")):
        constant = float(np.median(train_targets[:, index]))
        guess = np.full(len(test_targets), constant)
        results.update(_score(guess, test_targets[:, index], name,
                              boundary_band))
        results[f"{name}_constant"] = constant
    return results


def boundary_weights(targets: torch.Tensor, band: float = 0.05,
                     peak: float = 4.0) -> torch.Tensor:
    """Weight each sample by how close it is to contact.

    A sample sitting on the boundary counts `peak` times as much as one far
    away, decaying smoothly with distance so the weighting does not create
    a discontinuity in the loss.
    """
    closeness = torch.exp(-(targets.abs() / band) ** 2)
    return 1.0 + (peak - 1.0) * closeness


def train(dataset, epochs: int = 300, batch_size: int = 512,
          learning_rate: float = 2e-3, width: int = 512, depth: int = 6,
          harmonics: int = 4, seed: int = 0, verbose: bool = True):
    """Fit an SDFNetwork to a dataset.

    Targets are normalized by their own mean and standard deviation before
    the loss is taken, so the environment head does not dominate the self
    head purely because it spans a wider range. The normalization is stored
    in the model, so predictions come back in metres.

    Returns:
        (model, history)
    """
    torch.manual_seed(seed)

    train_set, test_set = dataset.split(train_fraction=0.8, seed=seed)

    train_x = torch.tensor(train_set.configurations, dtype=torch.float32)
    train_y = torch.tensor(train_set.targets, dtype=torch.float32)
    test_x = torch.tensor(test_set.configurations, dtype=torch.float32)
    test_y = torch.tensor(test_set.targets, dtype=torch.float32)

    mean = train_y.mean(dim=0)
    std = train_y.std(dim=0).clamp(min=1e-6)
    weights = boundary_weights(train_y)

    model = SDFNetwork(joints=train_x.shape[1], width=width, depth=depth,
                       harmonics=harmonics)
    model.target_mean.copy_(mean)
    model.target_std.copy_(std)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=epochs)
    element_loss = nn.SmoothL1Loss(reduction="none", beta=0.1)

    loader = DataLoader(
        TensorDataset(train_x, (train_y - mean) / std, weights),
        batch_size=batch_size, shuffle=True)
    history = TrainingHistory()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad()
            loss = (element_loss(model(batch_x), batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch_x)
        schedule.step()

        model.eval()
        with torch.no_grad():
            test_loss = element_loss(model(test_x),
                                     (test_y - mean) / std).mean().item()

        history.train_loss.append(running / len(train_x))
        history.test_loss.append(test_loss)

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"  epoch {epoch:>4}  train {history.train_loss[-1]:.5f}  "
                  f"test {test_loss:.5f}")

    history.metrics = evaluate(model, test_set.configurations,
                               test_set.targets)
    history.baseline = constant_baseline(train_set.targets, test_set.targets)
    return model, history