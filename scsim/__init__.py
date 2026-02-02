"""Single-cell RNA-seq simulator using the Splatter statistical framework."""

from .config import SimulationConfig
from .core import ScSim

__all__ = ["ScSim", "SimulationConfig"]
