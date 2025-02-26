r"""Models of the LinODE-Net package."""

__all__ = [
    # Sub-Packages
    "filters",
    "encoders",
    "embeddings",
    "system",
    # Type Hint
    "Model",
    # Constants
    "MODELS",
    # Classes
    "SpectralNorm",
    "LinearContraction",
    "iResNetBlock",
    "iResNet",
    "ResNet",
    "ResNetBlock",
    "LinODE",
    "LinODEnet",
    "LinODECell",
    # Functions
    "spectral_norm",
    # Filters
]

from typing import Final

from models.linodenet.models import embeddings, encoders, filters, system
from models.linodenet.models._linodenet import LinODE, LinODEnet
from models.linodenet.models.encoders import (
    LinearContraction,
    ResNet,
    ResNetBlock,
    SpectralNorm,
    iResNet,
    iResNetBlock,
    spectral_norm,
)
from models.linodenet.models.system import LinODECell
from torch import nn

Model = nn.Module
r"""Type hint for models."""

MODELS: Final[dict[str, type[nn.Module]]] = {
    "LinearContraction": LinearContraction,
    "iResNetBlock": iResNetBlock,
    "iResNet": iResNet,
    "LinODECell": LinODECell,
    "LinODE": LinODE,
    "LinODEnet": LinODEnet,
}
r"""Dictionary containing all available models."""
