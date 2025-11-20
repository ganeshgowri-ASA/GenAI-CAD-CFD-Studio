"""
Advanced 3D Visualization Module using PyVista

This module provides professional-grade 3D visualization capabilities for CAD models,
CFD simulation results, and mesh quality analysis using PyVista and Plotly.

Main Components:
- PyVistaViewer: Core 3D rendering engine for CAD and CFD data (optional, requires pyvista)
- StreamlitPyVista: Streamlit integration wrapper for interactive visualization (optional, requires pyvista)
- PlotlyCharts: 2D plotting utilities for convergence, residuals, and statistics (always available)
- ExportRenderer: High-quality image and animation export functionality (optional, requires pyvista)
- Utils: Mesh conversion, optimization, and caching utilities (optional, requires pyvista)
- preview_basic: Basic 3D preview using Plotly (always available)
"""

# Always available - Plotly-based visualization (no heavy dependencies)
from .plotly_charts import PlotlyCharts
from .preview_basic import plot_geometry_data, plot_mesh_3d, create_camera_controls

__version__ = "1.0.0"
__all__ = [
    "PlotlyCharts",
    "plot_geometry_data",
    "plot_mesh_3d",
    "create_camera_controls",
]

# Optional PyVista-based components (heavy 3D libraries)
# These are only available if pyvista is installed
try:
    from .pyvista_viewer import PyVistaViewer
    __all__.append("PyVistaViewer")
    PYVISTA_VIEWER_AVAILABLE = True
except ImportError:
    PyVistaViewer = None
    PYVISTA_VIEWER_AVAILABLE = False

try:
    from .streamlit_pyvista import StreamlitPyVista
    __all__.append("StreamlitPyVista")
    STREAMLIT_PYVISTA_AVAILABLE = True
except ImportError:
    StreamlitPyVista = None
    STREAMLIT_PYVISTA_AVAILABLE = False

try:
    from .export_renderer import ExportRenderer
    __all__.append("ExportRenderer")
    EXPORT_RENDERER_AVAILABLE = True
except ImportError:
    ExportRenderer = None
    EXPORT_RENDERER_AVAILABLE = False

try:
    from .utils import (
        MeshConverter,
        MeshOptimizer,
        MeshCache,
        ColorMapUtils,
        convert_mesh,
        optimize_for_display
    )
    __all__.extend([
        "MeshConverter",
        "MeshOptimizer",
        "MeshCache",
        "ColorMapUtils",
        "convert_mesh",
        "optimize_for_display",
    ])
    UTILS_AVAILABLE = True
except ImportError:
    MeshConverter = None
    MeshOptimizer = None
    MeshCache = None
    ColorMapUtils = None
    convert_mesh = None
    optimize_for_display = None
    UTILS_AVAILABLE = False
