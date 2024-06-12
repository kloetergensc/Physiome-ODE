r"""Embedding Models."""

__all__ = [
    # Types
    "Embedding",
    # Meta-Objects
    "EMBEDDINGS",
    # Classes
    "ConcatEmbedding",
    "ConcatProjection",
]

from typing import Final, TypeAlias

from models.linodenet.models.embeddings._embeddings import (
    ConcatEmbedding,
    ConcatProjection,
)
from torch import nn

Embedding: TypeAlias = nn.Module
r"""Type hint for Embeddings."""


EMBEDDINGS: Final[dict[str, type[nn.Module]]] = {
    "ConcatEmbedding": ConcatEmbedding,
    "ConcatProjection": ConcatProjection,
}
r"""Dictionary of available embeddings."""
