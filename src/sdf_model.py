"""A learned signed distance field over the arm's configuration space.

Predicts two clearances from a joint configuration: distance to the nearest
environment obstacle, and the smallest separation between arm links that
can touch. Both in metres, positive for clearance and negative for
penetration.

Input encoding matters more than it first appears. Raw joint angles ask the
network to learn that a revolute joint is periodic, so the first encoding
here is sine and cosine of each angle. That alone was not enough: a plain
MLP on those fourteen features underfits, reaching about 21 mm error on the
environment distance whether measured on training or held out data, which
is far too coarse when the quantity of interest near contact is only a few
millimetres. MLPs are biased toward smooth low frequency functions, and a
distance field changes sharply near a boundary. Adding higher harmonics,
sin(k*theta) and cos(k*theta) for several k, supplies a basis that can
represent that variation. This is the standard Fourier feature treatment
used in neural implicit surface work.

The point of the model is not accuracy in the abstract. It is a usable
gradient: an optimizer asks which way to move to increase clearance. Note
that a model can have a correct gradient and a wrong value, so this is not
a collision checker and must not be used as one. Geometry decides validity.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

Configuration = Sequence[float]

ENVIRONMENT = 0
SELF = 1


def encode(configurations: torch.Tensor, harmonics: int = 4) -> torch.Tensor:
    """Fourier features of the joint angles.

    Args:
        configurations: (batch, joints) joint angles in radians.
        harmonics: how many frequencies to include. k = 1 gives the plain
            sine and cosine pair; higher k adds the sharper variation the
            network needs near a collision boundary.

    Returns:
        (batch, joints * 2 * harmonics) tensor.
    """
    features = []
    for k in range(1, harmonics + 1):
        features.append(torch.sin(k * configurations))
        features.append(torch.cos(k * configurations))
    return torch.cat(features, dim=-1)


class SDFNetwork(nn.Module):
    """MLP mapping a configuration to two clearances."""

    def __init__(self, joints: int = 7, width: int = 512, depth: int = 6,
                 harmonics: int = 4):
        super().__init__()
        self.joints = joints
        self.harmonics = harmonics

        inputs = joints * 2 * harmonics
        layers: List[nn.Module] = [nn.Linear(inputs, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

        # Filled in at training time so the model carries its own
        # normalization and can be used without the training script.
        self.register_buffer("target_mean", torch.zeros(2))
        self.register_buffer("target_std", torch.ones(2))

    def forward(self, configurations: torch.Tensor) -> torch.Tensor:
        """Predict normalized clearances for a batch of configurations."""
        return self.net(encode(configurations, self.harmonics))

    def distances(self, configurations: torch.Tensor) -> torch.Tensor:
        """Predict clearances in metres, undoing the normalization."""
        return self(configurations) * self.target_std + self.target_mean


@dataclass
class LearnedSDF:
    """Convenience wrapper for using a trained network on one configuration.

    Keeps the model in evaluation mode and handles tensor conversion, so
    planning code can work with plain lists of joint angles.
    """

    model: SDFNetwork

    def __post_init__(self) -> None:
        self.model.eval()

    def __call__(self, configuration: Configuration) -> Tuple[float, float]:
        """Return predicted (environment, self) clearances in metres."""
        with torch.no_grad():
            tensor = torch.tensor([list(configuration)], dtype=torch.float32)
            values = self.model.distances(tensor)[0]
        return float(values[ENVIRONMENT]), float(values[SELF])

    def gradient(self, configuration: Configuration,
                 target: int = ENVIRONMENT) -> np.ndarray:
        """Direction in joint space that most increases a clearance.

        This is the reason for learning the field rather than querying the
        simulator: the derivative comes for free.
        """
        tensor = torch.tensor([list(configuration)], dtype=torch.float32,
                              requires_grad=True)
        value = self.model.distances(tensor)[0, target]
        value.backward()
        return tensor.grad[0].detach().numpy()


def save(model: SDFNetwork, path) -> None:
    """Write the model and its shape to disk."""
    torch.save({
        "state_dict": model.state_dict(),
        "joints": model.joints,
        "harmonics": model.harmonics,
        "width": model.net[0].out_features,
        "depth": sum(1 for m in model.net if isinstance(m, nn.Linear)) - 1,
    }, path)


def load(path) -> SDFNetwork:
    """Read a model back, reconstructing its shape from the checkpoint."""
    checkpoint = torch.load(path, weights_only=True)
    model = SDFNetwork(joints=checkpoint["joints"],
                       width=checkpoint["width"],
                       depth=checkpoint["depth"],
                       harmonics=checkpoint["harmonics"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model