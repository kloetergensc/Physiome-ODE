r"""Models for the latent dynamical system."""

__all__ = [
    # Types
    "System",
    # Meta-Objects
    "SYSTEMS",
    # Classes
    "LinODECell",
]


from typing import Final, TypeAlias

from models.linodenet.models.system._system import LinODECell
from torch import nn

System: TypeAlias = nn.Module
r"""Type hint for the system model."""


SYSTEMS: Final[dict[str, type[System]]] = {"LinODECell": LinODECell}
r"""Dictionary of all available system models."""
