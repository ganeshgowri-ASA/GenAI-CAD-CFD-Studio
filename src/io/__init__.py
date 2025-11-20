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
from .universal_importer import UniversalImporter

__all__ = [
    'DXFParser',
    'STEPHandler',
    'STLHandler',
    'MeshConverter',
    'UniversalImporter'
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
        Dictionary with unified geometry data

    Example:
        >>> from src.io import import_file
        >>> geometry = import_file('model.step')
        >>> print(f"Volume: {geometry['volume']}")
    """
    importer = UniversalImporter()
    return importer.import_file(filepath, **kwargs)


def get_supported_formats():
    """
    Get list of all supported file formats.

    Returns:
        Dictionary mapping file extension to format category

    Example:
        >>> from src.io import get_supported_formats
        >>> formats = get_supported_formats()
        >>> print(formats['stl'])  # Output: 'stl'
    """
    return UniversalImporter.get_supported_formats()
