"""
GenAI CAD-CFD Studio - File I/O Module

Comprehensive CAD and mesh file import/export system supporting multiple formats.

Main Components:
- DXFParser: Parse AutoCAD DXF files (2D/3D drawings)
- STEPHandler: Handle STEP CAD files (3D solid models)
- STLHandler: Load and manipulate STL mesh files
- MeshConverter: Convert between 20+ mesh formats
- UniversalImporter: Unified interface for all supported formats

Supported Formats:
- CAD: DXF, STEP/STP, IGES/IGS, BREP
- Mesh: STL, OBJ, PLY, OFF
- FEA/CFD: VTK, VTU, MSH, ANSYS, Abaqus, CGNS, Exodus, and more

Example Usage:
    from src.io import UniversalImporter

    # Import any supported file
    importer = UniversalImporter()
    geometry = importer.import_file('model.step')

    # Access unified geometry data
    vertices = geometry['vertices']
    volume = geometry['volume']
    bounds = geometry['bounds']

    # Or use specific handlers
    from src.io import STLHandler

    stl = STLHandler()
    mesh = stl.load_mesh('part.stl')
    properties = stl.calculate_properties()
"""

from .dxf_parser import DXFParser
from .step_handler import STEPHandler
from .stl_handler import STLHandler
from .mesh_converter import MeshConverter
from .universal_importer import GeometryData, parse

__all__ = [
    'DXFParser',
    'STEPHandler',
    'STLHandler',
    'MeshConverter',
    'GeometryData',
    'parse'
]

__version__ = '1.0.0'
__author__ = 'GenAI CAD-CFD Studio'

# Module-level convenience function
def import_file(filepath: str, **kwargs):
    """
    Convenience function to import any supported file format.

    Args:
        filepath: Path to the CAD or mesh file
        **kwargs: Additional arguments passed to the specific parser

    Returns:
        GeometryData object with unified geometry data

    Example:
        >>> from src.io import import_file
        >>> geometry = import_file('model.step')
        >>> print(f"Volume: {geometry.volume}")
    """
    return parse(filepath, **kwargs)


def get_supported_formats():
    """
    Get list of all supported file formats.

    Returns:
        List of supported file extensions

    Example:
        >>> from src.io import get_supported_formats
        >>> formats = get_supported_formats()
        >>> print(formats)  # Output: ['stl', 'obj', 'ply', 'dxf', ...]
    """
    return ['stl', 'obj', 'ply', 'dxf', 'dwg', 'step', 'stp', 'iges', 'igs', 'brep']
