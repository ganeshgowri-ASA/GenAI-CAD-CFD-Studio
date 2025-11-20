"""
Gmsh Mesher Module
Handles mesh generation using Gmsh for CFD simulations.
"""

import gmsh
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np


class GmshMesher:
    """
    Wrapper for Gmsh mesh generation.

    Features:
    - STEP file import
    - Mesh size control
    - Refinement zones
    - Boundary layer generation
    - Export to various formats
    """

    def __init__(self):
        """Initialize Gmsh mesher."""
        self.initialized = False
        self.geometry_loaded = False
        self.mesh_generated = False

    def initialize(self, verbose: bool = False):
        """
        Initialize Gmsh.

        Args:
            verbose: Enable verbose output
        """
        gmsh.initialize()
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        self.initialized = True

    def finalize(self):
        """Finalize and cleanup Gmsh."""
        if self.initialized:
            gmsh.finalize()
            self.initialized = False

    def load_step_file(self, step_file: Path) -> bool:
        """
        Load geometry from STEP file.

        Args:
            step_file: Path to STEP file

        Returns:
            Success status
        """
        if not self.initialized:
            self.initialize()

        try:
            gmsh.model.add("cfd_model")
            gmsh.merge(str(step_file))
            self.geometry_loaded = True
            return True
        except Exception as e:
            print(f"Error loading STEP file: {e}")
            return False

    def set_mesh_size(self, global_size: float, min_size: Optional[float] = None,
                     max_size: Optional[float] = None):
        """
        Set global mesh size parameters.

        Args:
            global_size: Target element size
            min_size: Minimum element size
            max_size: Maximum element size
        """
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size or global_size * 0.1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size or global_size * 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", global_size)

    def add_refinement_box(self, center: Tuple[float, float, float],
                          dimensions: Tuple[float, float, float],
                          element_size: float):
        """
        Add a box refinement zone.

        Args:
            center: Box center coordinates
            dimensions: Box dimensions (width, height, depth)
            element_size: Element size in the box
        """
        x, y, z = center
        dx, dy, dz = [d/2 for d in dimensions]

        # Create a field for refinement
        field_id = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(field_id, "VIn", element_size)
        gmsh.model.mesh.field.setNumber(field_id, "VOut", element_size * 2)
        gmsh.model.mesh.field.setNumber(field_id, "XMin", x - dx)
        gmsh.model.mesh.field.setNumber(field_id, "XMax", x + dx)
        gmsh.model.mesh.field.setNumber(field_id, "YMin", y - dy)
        gmsh.model.mesh.field.setNumber(field_id, "YMax", y + dy)
        gmsh.model.mesh.field.setNumber(field_id, "ZMin", z - dz)
        gmsh.model.mesh.field.setNumber(field_id, "ZMax", z + dz)

        return field_id

    def add_refinement_sphere(self, center: Tuple[float, float, float],
                             radius: float, element_size: float):
        """
        Add a spherical refinement zone.

        Args:
            center: Sphere center coordinates
            radius: Sphere radius
            element_size: Element size in the sphere
        """
        field_id = gmsh.model.mesh.field.add("Ball")
        gmsh.model.mesh.field.setNumber(field_id, "VIn", element_size)
        gmsh.model.mesh.field.setNumber(field_id, "VOut", element_size * 2)
        gmsh.model.mesh.field.setNumber(field_id, "XCenter", center[0])
        gmsh.model.mesh.field.setNumber(field_id, "YCenter", center[1])
        gmsh.model.mesh.field.setNumber(field_id, "ZCenter", center[2])
        gmsh.model.mesh.field.setNumber(field_id, "Radius", radius)

        return field_id

    def add_refinement_zones(self, zones: List[Dict[str, Any]]):
        """
        Add multiple refinement zones.

        Args:
            zones: List of refinement zone specifications
        """
        field_ids = []

        for zone in zones:
            zone_type = zone.get("type")

            if zone_type == "Box":
                field_id = self.add_refinement_box(
                    zone["center"],
                    zone["dims"],
                    zone["size"]
                )
                field_ids.append(field_id)

            elif zone_type == "Sphere":
                field_id = self.add_refinement_sphere(
                    zone["center"],
                    zone["radius"],
                    zone["size"]
                )
                field_ids.append(field_id)

        # Combine all refinement fields
        if field_ids:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    def generate_mesh(self, dimension: int = 3, algorithm: int = 1) -> bool:
        """
        Generate mesh.

        Args:
            dimension: Mesh dimension (2 or 3)
            algorithm: Meshing algorithm (1=Delaunay, 5=Frontal, etc.)

        Returns:
            Success status
        """
        if not self.geometry_loaded:
            print("Error: No geometry loaded")
            return False

        try:
            gmsh.model.mesh.setAlgorithm(dimension, algorithm)
            gmsh.model.mesh.generate(dimension)
            self.mesh_generated = True
            return True
        except Exception as e:
            print(f"Error generating mesh: {e}")
            return False

    def get_mesh_statistics(self) -> Dict[str, int]:
        """
        Get mesh statistics.

        Returns:
            Dictionary with node and element counts
        """
        if not self.mesh_generated:
            return {"nodes": 0, "elements": 0, "triangles": 0, "tetrahedra": 0}

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        num_nodes = len(node_tags)

        # Get element counts
        elem_types, _, _ = gmsh.model.mesh.getElements()

        num_triangles = 0
        num_tetrahedra = 0

        for elem_type in elem_types:
            if elem_type == 2:  # Triangle
                _, _, elem_tags = gmsh.model.mesh.getElements(2, -1)
                num_triangles = len(elem_tags[0]) if elem_tags else 0
            elif elem_type == 4:  # Tetrahedron
                _, _, elem_tags = gmsh.model.mesh.getElements(3, -1)
                num_tetrahedra = len(elem_tags[0]) if elem_tags else 0

        return {
            "nodes": num_nodes,
            "elements": num_triangles + num_tetrahedra,
            "triangles": num_triangles,
            "tetrahedra": num_tetrahedra
        }

    def export_mesh(self, output_file: Path, format: str = "msh"):
        """
        Export mesh to file.

        Args:
            output_file: Output file path
            format: Export format (msh, vtk, stl, etc.)
        """
        if not self.mesh_generated:
            print("Error: No mesh to export")
            return False

        try:
            # Ensure correct extension
            if format == "msh":
                output_file = output_file.with_suffix(".msh")
            elif format == "vtk":
                output_file = output_file.with_suffix(".vtk")
            elif format == "stl":
                output_file = output_file.with_suffix(".stl")

            gmsh.write(str(output_file))
            return True
        except Exception as e:
            print(f"Error exporting mesh: {e}")
            return False

    def visualize(self):
        """Launch Gmsh GUI for visualization (if available)."""
        if self.initialized:
            gmsh.fltk.run()


def create_mesh_from_config(step_file: Path, config: Dict[str, Any],
                           output_file: Path) -> Tuple[bool, Dict[str, int]]:
    """
    Create mesh from configuration.

    Args:
        step_file: Path to STEP file
        config: Mesh configuration dictionary
        output_file: Output mesh file path

    Returns:
        Tuple of (success, statistics)
    """
    mesher = GmshMesher()

    try:
        # Initialize
        mesher.initialize(verbose=False)

        # Load geometry
        if not mesher.load_step_file(step_file):
            return False, {}

        # Set mesh size
        mesher.set_mesh_size(
            config.get("global_size", 0.1),
            config.get("min_size"),
            config.get("max_size")
        )

        # Add refinement zones
        refinement_zones = config.get("refinement_zones", [])
        if refinement_zones:
            mesher.add_refinement_zones(refinement_zones)

        # Generate mesh
        if not mesher.generate_mesh(dimension=3):
            return False, {}

        # Get statistics
        stats = mesher.get_mesh_statistics()

        # Export
        mesher.export_mesh(output_file, format="msh")

        return True, stats

    except Exception as e:
        print(f"Error in mesh generation: {e}")
        return False, {}

    finally:
        mesher.finalize()
