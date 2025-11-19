"""
Gmsh Mesher Module

Provides mesh generation capabilities using Gmsh Python API for CFD simulations.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import gmsh
except ImportError:
    gmsh = None

try:
    import meshio
except ImportError:
    meshio = None

try:
    import pyvista as pv
except ImportError:
    pv = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GmshMesher:
    """
    Mesh generator using Gmsh for CFD applications.

    This class provides methods to:
    - Load STEP geometry files
    - Generate 3D tetrahedral meshes
    - Define refinement zones
    - Export meshes in various formats
    - Visualize meshes
    - Compute mesh statistics
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize GmshMesher.

        Args:
            verbose: Enable verbose output from Gmsh
        """
        if gmsh is None:
            raise ImportError(
                "gmsh is not installed. Install it with: pip install gmsh"
            )

        self.verbose = verbose
        self.initialized = False

    def _initialize_gmsh(self):
        """Initialize Gmsh if not already initialized."""
        if not self.initialized:
            gmsh.initialize()
            if not self.verbose:
                gmsh.option.setNumber("General.Terminal", 0)
            self.initialized = True

    def _finalize_gmsh(self):
        """Finalize Gmsh session."""
        if self.initialized:
            gmsh.finalize()
            self.initialized = False

    def generate_mesh(
        self,
        step_file: str,
        mesh_size: float = 0.1,
        refinement_zones: Optional[List[Dict[str, Any]]] = None,
        output_file: Optional[str] = None,
        algorithm: str = "delaunay",
        optimize: bool = True,
        mesh_order: int = 1
    ) -> str:
        """
        Generate 3D tetrahedral mesh from STEP file.

        Args:
            step_file: Path to STEP geometry file
            mesh_size: Global mesh element size (default: 0.1)
            refinement_zones: List of refinement zone specifications
                Example: [{"type": "box", "x_min": 0, "x_max": 1,
                          "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1,
                          "size": 0.05}]
            output_file: Output mesh file path (.msh format)
            algorithm: Meshing algorithm ('delaunay', 'frontal', 'mmg3d')
            optimize: Apply mesh optimization
            mesh_order: Element order (1=linear, 2=quadratic)

        Returns:
            Path to generated mesh file

        Raises:
            FileNotFoundError: If STEP file doesn't exist
            RuntimeError: If mesh generation fails
        """
        if not os.path.exists(step_file):
            raise FileNotFoundError(f"STEP file not found: {step_file}")

        if output_file is None:
            base_name = os.path.splitext(step_file)[0]
            output_file = f"{base_name}.msh"

        try:
            self._initialize_gmsh()
            gmsh.clear()
            gmsh.model.add("cfd_mesh")

            # Load STEP geometry
            logger.info(f"Loading STEP file: {step_file}")
            gmsh.model.occ.importShapes(step_file)
            gmsh.model.occ.synchronize()

            # Set global mesh size
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.1)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size * 2.0)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)

            # Set meshing algorithm
            algorithm_map = {
                "delaunay": 1,
                "frontal": 6,
                "mmg3d": 7,
                "hxt": 10
            }
            gmsh.option.setNumber(
                "Mesh.Algorithm3D",
                algorithm_map.get(algorithm.lower(), 1)
            )

            # Set mesh element order
            gmsh.option.setNumber("Mesh.ElementOrder", mesh_order)

            # Apply refinement zones
            if refinement_zones:
                self._apply_refinement_zones(refinement_zones)

            # Set characteristic length on all points
            points = gmsh.model.getEntities(0)
            for point in points:
                gmsh.model.mesh.setSize([point], mesh_size)

            # Generate mesh
            logger.info("Generating 3D mesh...")
            gmsh.model.mesh.generate(3)

            # Optimize mesh
            if optimize:
                logger.info("Optimizing mesh...")
                gmsh.model.mesh.optimize("Netgen")

            # Save mesh
            gmsh.write(output_file)
            logger.info(f"Mesh saved to: {output_file}")

            return output_file

        except Exception as e:
            logger.error(f"Mesh generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate mesh: {str(e)}")
        finally:
            self._finalize_gmsh()

    def _apply_refinement_zones(self, refinement_zones: List[Dict[str, Any]]):
        """
        Apply mesh refinement zones.

        Args:
            refinement_zones: List of refinement specifications
        """
        for zone in refinement_zones:
            zone_type = zone.get("type", "box")
            size = zone.get("size", 0.05)

            if zone_type == "box":
                # Create refinement box
                x_min = zone.get("x_min", 0)
                x_max = zone.get("x_max", 1)
                y_min = zone.get("y_min", 0)
                y_max = zone.get("y_max", 1)
                z_min = zone.get("z_min", 0)
                z_max = zone.get("z_max", 1)

                # Create box for refinement field
                field_id = gmsh.model.mesh.field.add("Box")
                gmsh.model.mesh.field.setNumber(field_id, "VIn", size)
                gmsh.model.mesh.field.setNumber(field_id, "VOut", size * 2)
                gmsh.model.mesh.field.setNumber(field_id, "XMin", x_min)
                gmsh.model.mesh.field.setNumber(field_id, "XMax", x_max)
                gmsh.model.mesh.field.setNumber(field_id, "YMin", y_min)
                gmsh.model.mesh.field.setNumber(field_id, "YMax", y_max)
                gmsh.model.mesh.field.setNumber(field_id, "ZMin", z_min)
                gmsh.model.mesh.field.setNumber(field_id, "ZMax", z_max)

                gmsh.model.mesh.field.setAsBackgroundMesh(field_id)

            elif zone_type == "distance":
                # Refine near surfaces
                surfaces = zone.get("surfaces", [])
                if surfaces:
                    field_id = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(field_id, "SurfacesList", surfaces)

                    threshold_id = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(threshold_id, "InField", field_id)
                    gmsh.model.mesh.field.setNumber(threshold_id, "SizeMin", size)
                    gmsh.model.mesh.field.setNumber(threshold_id, "SizeMax", size * 2)
                    gmsh.model.mesh.field.setNumber(threshold_id, "DistMin", 0.1)
                    gmsh.model.mesh.field.setNumber(threshold_id, "DistMax", 0.5)

                    gmsh.model.mesh.field.setAsBackgroundMesh(threshold_id)

    def visualize_mesh(
        self,
        mesh_file: str,
        show_edges: bool = True,
        show_quality: bool = False,
        screenshot: Optional[str] = None
    ) -> Any:
        """
        Visualize mesh using PyVista.

        Args:
            mesh_file: Path to mesh file
            show_edges: Show mesh edges
            show_quality: Color by mesh quality
            screenshot: Save screenshot to file

        Returns:
            PyVista plotter object

        Raises:
            ImportError: If PyVista or meshio not installed
            FileNotFoundError: If mesh file doesn't exist
        """
        if pv is None:
            raise ImportError(
                "pyvista is not installed. Install it with: pip install pyvista"
            )
        if meshio is None:
            raise ImportError(
                "meshio is not installed. Install it with: pip install meshio"
            )

        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        # Read mesh
        logger.info(f"Loading mesh: {mesh_file}")
        mesh = meshio.read(mesh_file)

        # Convert to PyVista format
        points = mesh.points
        cells = []
        cell_types = []

        for cell_block in mesh.cells:
            if cell_block.type == "tetra":
                cells.extend(cell_block.data)
                cell_types.extend([pv.CellType.TETRA] * len(cell_block.data))
            elif cell_block.type == "triangle":
                cells.extend(cell_block.data)
                cell_types.extend([pv.CellType.TRIANGLE] * len(cell_block.data))

        if not cells:
            logger.warning("No cells found in mesh")
            return None

        # Create unstructured grid
        cells_array = np.array(cells)
        if cells_array.ndim == 2:
            # Convert to PyVista format: [n_points, p0, p1, ..., pn]
            n_points_per_cell = cells_array.shape[1]
            cells_pyvista = np.hstack([
                np.full((cells_array.shape[0], 1), n_points_per_cell),
                cells_array
            ]).flatten()
        else:
            cells_pyvista = cells_array

        grid = pv.UnstructuredGrid(cells_pyvista, cell_types, points)

        # Create plotter
        plotter = pv.Plotter()

        if show_quality:
            # Compute and display mesh quality
            grid = grid.compute_cell_quality(quality_measure='scaled_jacobian')
            plotter.add_mesh(
                grid,
                scalars='CellQuality',
                show_edges=show_edges,
                cmap='RdYlGn',
                clim=[-1, 1]
            )
        else:
            plotter.add_mesh(grid, show_edges=show_edges, color='lightblue')

        plotter.add_axes()
        plotter.view_isometric()

        if screenshot:
            plotter.screenshot(screenshot)
            logger.info(f"Screenshot saved to: {screenshot}")

        return plotter

    def get_mesh_stats(self, mesh_file: str) -> Dict[str, Any]:
        """
        Get mesh statistics.

        Args:
            mesh_file: Path to mesh file

        Returns:
            Dictionary containing mesh statistics:
                - n_nodes: Number of nodes
                - n_elements: Number of elements
                - n_tetra: Number of tetrahedral elements
                - n_triangle: Number of triangle elements
                - volume: Total mesh volume (if applicable)
                - quality_min: Minimum element quality
                - quality_mean: Mean element quality
                - quality_std: Standard deviation of element quality

        Raises:
            FileNotFoundError: If mesh file doesn't exist
        """
        if meshio is None:
            raise ImportError(
                "meshio is not installed. Install it with: pip install meshio"
            )

        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        # Read mesh
        mesh = meshio.read(mesh_file)

        stats = {
            "n_nodes": len(mesh.points),
            "n_elements": 0,
            "n_tetra": 0,
            "n_triangle": 0,
            "n_hexahedron": 0,
            "n_quad": 0,
        }

        # Count elements by type
        for cell_block in mesh.cells:
            n_cells = len(cell_block.data)
            stats["n_elements"] += n_cells

            if cell_block.type == "tetra":
                stats["n_tetra"] += n_cells
            elif cell_block.type == "triangle":
                stats["n_triangle"] += n_cells
            elif cell_block.type == "hexahedron":
                stats["n_hexahedron"] += n_cells
            elif cell_block.type == "quad":
                stats["n_quad"] += n_cells

        # Compute mesh quality if PyVista is available
        if pv is not None:
            try:
                # Convert to PyVista
                points = mesh.points
                cells = []
                cell_types = []

                for cell_block in mesh.cells:
                    if cell_block.type == "tetra":
                        for cell in cell_block.data:
                            cells.extend([4] + cell.tolist())
                            cell_types.append(pv.CellType.TETRA)

                if cells:
                    grid = pv.UnstructuredGrid(cells, cell_types, points)
                    grid = grid.compute_cell_quality(quality_measure='scaled_jacobian')

                    quality = grid['CellQuality']
                    stats["quality_min"] = float(np.min(quality))
                    stats["quality_max"] = float(np.max(quality))
                    stats["quality_mean"] = float(np.mean(quality))
                    stats["quality_std"] = float(np.std(quality))

                    # Count poor quality elements
                    stats["n_poor_quality"] = int(np.sum(quality < 0.2))

            except Exception as e:
                logger.warning(f"Failed to compute mesh quality: {str(e)}")

        return stats

    def export_mesh(
        self,
        mesh_file: str,
        output_format: str = "vtk",
        output_file: Optional[str] = None
    ) -> str:
        """
        Export mesh to different format.

        Args:
            mesh_file: Input mesh file path
            output_format: Output format ('vtk', 'vtu', 'stl', 'obj')
            output_file: Output file path (auto-generated if None)

        Returns:
            Path to exported mesh file
        """
        if meshio is None:
            raise ImportError(
                "meshio is not installed. Install it with: pip install meshio"
            )

        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        # Read mesh
        mesh = meshio.read(mesh_file)

        # Generate output filename
        if output_file is None:
            base_name = os.path.splitext(mesh_file)[0]
            output_file = f"{base_name}.{output_format}"

        # Export mesh
        meshio.write(output_file, mesh)
        logger.info(f"Mesh exported to: {output_file}")

        return output_file

    def __del__(self):
        """Cleanup Gmsh on deletion."""
        try:
            self._finalize_gmsh()
        except:
            pass
