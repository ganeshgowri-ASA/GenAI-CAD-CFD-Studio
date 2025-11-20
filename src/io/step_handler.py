"""
STEP Handler Module

Handles import/export of STEP (Standard for the Exchange of Product model data) files
using pythonocc-core and aocxchange libraries.
"""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib_Add
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE,
                                 TopAbs_SOLID, TopAbs_SHELL, TopAbs_WIRE)
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    PYTHONOCC_AVAILABLE = True
except ImportError:
    PYTHONOCC_AVAILABLE = False
    TopoDS_Shape = None


class STEPHandler:
    """
    Handler for STEP file import/export operations.

    Provides functionality to:
    - Import STEP files
    - Export to different formats
    - Calculate geometric properties (volume, surface area, bounding box)
    - Extract shape topology information
    """

    def __init__(self):
        """Initialize STEP handler."""
        if not PYTHONOCC_AVAILABLE:
            raise ImportError(
                "pythonocc-core is not installed. "
                "Install it with: conda install -c conda-forge pythonocc-core"
            )
        self.shape = None
        self.reader = None

    def import_step(self, filepath: str) -> Any:
        """
        Import a STEP file and return the TopoDS_Shape object.

        Args:
            filepath: Path to the STEP file

        Returns:
            TopoDS_Shape object representing the imported geometry

        Raises:
            FileNotFoundError: If the file doesn't exist
            RuntimeError: If the STEP file cannot be read
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"STEP file not found: {filepath}")

        # Create STEP reader
        self.reader = STEPControl_Reader()

        # Read the STEP file
        status = self.reader.ReadFile(str(filepath))

        if status != IFSelect_RetDone:
            raise RuntimeError(f"Error reading STEP file: {filepath}")

        # Transfer roots to the document
        self.reader.TransferRoots()

        # Get the shape
        self.shape = self.reader.OneShape()

        if self.shape.IsNull():
            raise RuntimeError(f"No shape could be extracted from STEP file: {filepath}")

        return self.shape

    def export_step(self, shape: Any, filepath: str) -> bool:
        """
        Export a TopoDS_Shape to a STEP file.

        Args:
            shape: TopoDS_Shape object to export
            filepath: Output file path

        Returns:
            True if export successful, False otherwise
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        status = writer.Write(str(filepath))

        return status == IFSelect_RetDone

    def export_to_format(self, shape: Any, filepath: str, format_type: str = 'step') -> bool:
        """
        Export shape to different CAD formats.

        Args:
            shape: TopoDS_Shape object to export
            filepath: Output file path
            format_type: Format type ('step', 'iges', 'stl', 'brep')

        Returns:
            True if export successful, False otherwise

        Raises:
            ValueError: If format type is not supported
        """
        format_type = format_type.lower()

        if format_type == 'step' or format_type == 'stp':
            return self.export_step(shape, filepath)

        elif format_type == 'iges' or format_type == 'igs':
            from OCC.Core.IGESControl import IGESControl_Writer
            writer = IGESControl_Writer()
            writer.AddShape(shape)
            writer.Write(str(filepath))
            return True

        elif format_type == 'stl':
            from OCC.Core.StlAPI import StlAPI_Writer
            # Mesh the shape first
            mesh = BRepMesh_IncrementalMesh(shape, 0.1, False, 0.1, True)
            mesh.Perform()
            writer = StlAPI_Writer()
            return writer.Write(shape, str(filepath))

        elif format_type == 'brep':
            from OCC.Core.BRepTools import breptools_Write
            return breptools_Write(shape, str(filepath))

        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def get_volume(self, shape: Optional[Any] = None) -> float:
        """
        Calculate the volume of a shape.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            Volume in cubic units
        """
        if shape is None:
            shape = self.shape

        if shape is None or shape.IsNull():
            return 0.0

        props = GProp_GProps()
        brepgprop_VolumeProperties(shape, props)
        return props.Mass()

    def get_surface_area(self, shape: Optional[Any] = None) -> float:
        """
        Calculate the surface area of a shape.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            Surface area in square units
        """
        if shape is None:
            shape = self.shape

        if shape is None or shape.IsNull():
            return 0.0

        props = GProp_GProps()
        brepgprop_SurfaceProperties(shape, props)
        return props.Mass()

    def get_bounding_box(self, shape: Optional[Any] = None) -> Tuple[float, float, float, float, float, float]:
        """
        Get the bounding box of a shape.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        if shape is None:
            shape = self.shape

        if shape is None or shape.IsNull():
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox)

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def get_center_of_mass(self, shape: Optional[Any] = None) -> Tuple[float, float, float]:
        """
        Get the center of mass of a shape.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            Tuple of (x, y, z) coordinates
        """
        if shape is None:
            shape = self.shape

        if shape is None or shape.IsNull():
            return (0.0, 0.0, 0.0)

        props = GProp_GProps()
        brepgprop_VolumeProperties(shape, props)
        com = props.CentreOfMass()

        return (com.X(), com.Y(), com.Z())

    def get_topology_info(self, shape: Optional[Any] = None) -> Dict[str, int]:
        """
        Get topology information about the shape.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            Dictionary with counts of vertices, edges, faces, shells, and solids
        """
        if shape is None:
            shape = self.shape

        if shape is None or shape.IsNull():
            return {
                'vertices': 0,
                'edges': 0,
                'faces': 0,
                'wires': 0,
                'shells': 0,
                'solids': 0
            }

        topology = {
            'vertices': 0,
            'edges': 0,
            'faces': 0,
            'wires': 0,
            'shells': 0,
            'solids': 0
        }

        # Count vertices
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
        while explorer.More():
            topology['vertices'] += 1
            explorer.Next()

        # Count edges
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        while explorer.More():
            topology['edges'] += 1
            explorer.Next()

        # Count faces
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            topology['faces'] += 1
            explorer.Next()

        # Count wires
        explorer = TopExp_Explorer(shape, TopAbs_WIRE)
        while explorer.More():
            topology['wires'] += 1
            explorer.Next()

        # Count shells
        explorer = TopExp_Explorer(shape, TopAbs_SHELL)
        while explorer.More():
            topology['shells'] += 1
            explorer.Next()

        # Count solids
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        while explorer.More():
            topology['solids'] += 1
            explorer.Next()

        return topology

    def get_properties(self, shape: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get comprehensive geometric properties of a shape.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            Dictionary containing volume, surface area, bounding box,
            center of mass, and topology information
        """
        if shape is None:
            shape = self.shape

        bbox = self.get_bounding_box(shape)

        properties = {
            'volume': self.get_volume(shape),
            'surface_area': self.get_surface_area(shape),
            'bounds': bbox,
            'center_of_mass': self.get_center_of_mass(shape),
            'topology': self.get_topology_info(shape),
            'dimensions': {
                'length': bbox[1] - bbox[0],
                'width': bbox[3] - bbox[2],
                'height': bbox[5] - bbox[4]
            }
        }

        return properties

    def extract_vertices(self, shape: Optional[Any] = None) -> np.ndarray:
        """
        Extract all vertices from the shape.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            NumPy array of vertex coordinates (N x 3)
        """
        if shape is None:
            shape = self.shape

        if shape is None or shape.IsNull():
            return np.array([])

        vertices = []
        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)

        while explorer.More():
            vertex = explorer.Current()
            pnt = BRep_Tool.Pnt(vertex)
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
            explorer.Next()

        return np.array(vertices)

    def is_valid(self, shape: Optional[Any] = None) -> bool:
        """
        Check if a shape is valid.

        Args:
            shape: TopoDS_Shape object (uses self.shape if None)

        Returns:
            True if shape is valid, False otherwise
        """
        if shape is None:
            shape = self.shape

        if shape is None or shape.IsNull():
            return False

        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        analyzer = BRepCheck_Analyzer(shape)
        return analyzer.IsValid()
