"""Paper output generation module."""

from .figures import FigureGenerator
from .tables import TableGenerator
from .managerial import ManagerialInsights
from .animation import TrainingAnimator

__all__ = ["FigureGenerator", "TableGenerator", "ManagerialInsights", "TrainingAnimator"]
