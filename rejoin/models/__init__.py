"""Model-facing interfaces and framework-independent reference adapters."""

from rejoin.models.base import ReferenceTargetModel
from rejoin.models.protocols import DraftModel, TargetModel

__all__ = ["DraftModel", "ReferenceTargetModel", "TargetModel"]
