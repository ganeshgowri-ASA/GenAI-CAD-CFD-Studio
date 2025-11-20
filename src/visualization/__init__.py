"""
Advanced 3D Visualization Module using PyVista

This module provides professional-grade 3D visualization capabilities for CAD models,
CFD simulation results, and mesh quality analysis using PyVista and Plotly.

Main Components:
- PyVistaViewer: Core 3D rendering engine for CAD and CFD data
- StreamlitPyVista: Streamlit integration wrapper for interactive visualization
- PlotlyCharts: 2D plotting utilities for convergence, residuals, and statistics
- ExportRenderer: High-quality image and animation export functionality
- Utils: Mesh conversion, optimization, and caching utilities
"""

from .pyvista_viewer import PyVistaViewer
from .streamlit_pyvista import StreamlitPyVista
from .plotly_charts import PlotlyCharts
from .export_renderer import ExportRenderer
from .utils import (
    MeshConverter,
    MeshOptimizer,
    MeshCache,
    ColorMapUtils,
    convert_mesh,
    optimize_for_display
)

__version__ = "1.0.0"
__all__ = [
    "PyVistaViewer",
    "StreamlitPyVista",
    "PlotlyCharts",
    "ExportRenderer",
    "MeshConverter",
    "MeshOptimizer",
    "MeshCache",
    "ColorMapUtils",
    "convert_mesh",
    "optimize_for_display",
]
