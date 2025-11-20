"""
CFD (Computational Fluid Dynamics) Module

This module provides comprehensive CFD capabilities including:
- Mesh generation using Gmsh
- OpenFOAM simulation setup and execution
- Result parsing and visualization
- Cloud CFD integration (SimScale)
"""

from .gmsh_mesher import GmshMesher
from .pyfoam_wrapper import PyFoamWrapper
from .result_parser import ResultParser
from .simscale_api import SimScaleConnector

__version__ = "0.1.0"
__all__ = [
    "GmshMesher",
    "PyFoamWrapper",
    "ResultParser",
    "SimScaleConnector",
]
