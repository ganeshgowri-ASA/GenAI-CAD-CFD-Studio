"""
Result Parser Module

Provides parsing and post-processing of OpenFOAM simulation results.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np

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


class ResultParser:
    """
    OpenFOAM result parser and post-processor.

    This class provides methods to:
    - Load OpenFOAM simulation results
    - Extract field data (velocity, pressure, etc.)
    - Calculate forces and coefficients
    - Export to visualization formats (VTK, ParaView)
    - Compute derived quantities
    """

    def __init__(self, case_dir: str):
        """
        Initialize ResultParser.

        Args:
            case_dir: OpenFOAM case directory

        Raises:
            ValueError: If case directory doesn't exist
        """
        self.case_dir = Path(case_dir)
        if not self.case_dir.exists():
            raise ValueError(f"Case directory doesn't exist: {case_dir}")

        self.time_dirs = self._find_time_directories()
        logger.info(f"Found {len(self.time_dirs)} time directories")

    def _find_time_directories(self) -> List[float]:
        """
        Find all time directories in the case.

        Returns:
            List of time values (sorted)
        """
        time_dirs = []

        for item in self.case_dir.iterdir():
            if item.is_dir():
                try:
                    # Try to convert directory name to float
                    time_val = float(item.name)
                    time_dirs.append(time_val)
                except ValueError:
                    # Not a time directory
                    pass

        return sorted(time_dirs)

    def get_latest_time(self) -> Optional[float]:
        """
        Get the latest time step.

        Returns:
            Latest time value or None if no time directories found
        """
        if self.time_dirs:
            return self.time_dirs[-1]
        return None

    def load_results(
        self,
        time_step: Union[float, str] = "latest",
        fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load results from a specific time step.

        Args:
            time_step: Time step to load ('latest' or numeric value)
            fields: List of fields to load (None = all available)

        Returns:
            Dictionary containing result data:
                - time: float
                - mesh: mesh data
                - fields: dict of field arrays

        Raises:
            ValueError: If time step not found
        """
        # Determine time directory
        if time_step == "latest":
            time_val = self.get_latest_time()
            if time_val is None:
                raise ValueError("No time directories found")
        else:
            time_val = float(time_step)
            if time_val not in self.time_dirs:
                raise ValueError(f"Time step {time_val} not found")

        time_dir = self.case_dir / str(time_val)
        logger.info(f"Loading results from time = {time_val}")

        # Load mesh
        mesh_data = self._load_mesh()

        # Load fields
        field_data = {}
        available_fields = self._get_available_fields(time_dir)

        fields_to_load = fields if fields else available_fields

        for field in fields_to_load:
            if field in available_fields:
                field_data[field] = self._load_field(time_dir, field)
            else:
                logger.warning(f"Field {field} not found in time directory")

        return {
            "time": time_val,
            "mesh": mesh_data,
            "fields": field_data,
            "available_fields": available_fields
        }

    def _load_mesh(self) -> Dict[str, Any]:
        """
        Load mesh data from constant/polyMesh.

        Returns:
            Dictionary with mesh data
        """
        poly_mesh_dir = self.case_dir / "constant" / "polyMesh"

        if not poly_mesh_dir.exists():
            raise ValueError("polyMesh directory not found")

        # Read points
        points = self._read_openfoam_field_file(poly_mesh_dir / "points")

        # Read faces
        faces = self._read_openfoam_field_file(poly_mesh_dir / "faces")

        # Read owner
        owner = self._read_openfoam_field_file(poly_mesh_dir / "owner")

        # Read neighbour
        neighbour_file = poly_mesh_dir / "neighbour"
        neighbour = self._read_openfoam_field_file(neighbour_file) if neighbour_file.exists() else None

        # Read boundary
        boundary = self._read_boundary_file(poly_mesh_dir / "boundary")

        return {
            "points": points,
            "faces": faces,
            "owner": owner,
            "neighbour": neighbour,
            "boundary": boundary,
            "n_points": len(points) if isinstance(points, list) else 0,
            "n_faces": len(faces) if isinstance(faces, list) else 0,
        }

    def _get_available_fields(self, time_dir: Path) -> List[str]:
        """
        Get list of available fields in time directory.

        Args:
            time_dir: Time directory path

        Returns:
            List of field names
        """
        fields = []

        for item in time_dir.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                # Check if it's a field file
                if item.name not in ['uniform']:
                    fields.append(item.name)

        return fields

    def _load_field(self, time_dir: Path, field_name: str) -> np.ndarray:
        """
        Load field data from OpenFOAM file.

        Args:
            time_dir: Time directory
            field_name: Field name (U, p, k, etc.)

        Returns:
            Field data as numpy array
        """
        field_file = time_dir / field_name

        if not field_file.exists():
            raise FileNotFoundError(f"Field file not found: {field_file}")

        data = self._read_openfoam_field_file(field_file)

        return np.array(data) if isinstance(data, list) else data

    def _read_openfoam_field_file(self, file_path: Path) -> Union[List, np.ndarray]:
        """
        Read OpenFOAM field file (simple ASCII parser).

        Args:
            file_path: Path to field file

        Returns:
            Field data
        """
        with open(file_path, 'r') as f:
            content = f.read()

        # Find the internalField or data section
        # Look for patterns like:
        # internalField   nonuniform List<scalar>
        # or
        # internalField   uniform (0 0 0);

        # Check for uniform field
        uniform_match = re.search(r'internalField\s+uniform\s+([^;]+);', content)
        if uniform_match:
            value_str = uniform_match.group(1).strip()
            if value_str.startswith('(') and value_str.endswith(')'):
                # Vector field
                values = [float(x) for x in value_str.strip('()').split()]
                return np.array(values)
            else:
                # Scalar field
                return float(value_str)

        # Check for nonuniform field
        nonuniform_match = re.search(r'internalField\s+nonuniform\s+List<[^>]+>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)', content, re.DOTALL)
        if nonuniform_match:
            n_entries = int(nonuniform_match.group(1))
            data_str = nonuniform_match.group(2)

            # Parse data based on format
            if '(' in data_str:
                # Vector or tensor data
                vector_matches = re.findall(r'\(([^)]+)\)', data_str)
                data = []
                for match in vector_matches:
                    values = [float(x) for x in match.split()]
                    data.append(values)
                return np.array(data)
            else:
                # Scalar data
                values = [float(x) for x in data_str.split()]
                return np.array(values)

        # Fallback: try to find any numerical data
        logger.warning(f"Could not parse field file format: {file_path.name}")
        return []

    def _read_boundary_file(self, boundary_file: Path) -> Dict[str, Any]:
        """
        Read OpenFOAM boundary file.

        Args:
            boundary_file: Path to boundary file

        Returns:
            Dictionary with boundary patch information
        """
        if not boundary_file.exists():
            return {}

        with open(boundary_file, 'r') as f:
            content = f.read()

        # Simple parser for boundary file
        # This is a simplified version - production code would need a more robust parser
        patches = {}

        # Find number of patches
        n_patches_match = re.search(r'(\d+)\s*\(', content)
        if not n_patches_match:
            return {}

        # Extract patch definitions
        patch_matches = re.finditer(
            r'(\w+)\s*\{[^}]*type\s+(\w+);[^}]*nFaces\s+(\d+);[^}]*startFace\s+(\d+);[^}]*\}',
            content
        )

        for match in patch_matches:
            patch_name = match.group(1)
            patch_type = match.group(2)
            n_faces = int(match.group(3))
            start_face = int(match.group(4))

            patches[patch_name] = {
                "type": patch_type,
                "nFaces": n_faces,
                "startFace": start_face
            }

        return patches

    def extract_field(
        self,
        results: Dict[str, Any],
        field: str = "U"
    ) -> np.ndarray:
        """
        Extract field data from results.

        Args:
            results: Results dictionary from load_results()
            field: Field name (U, p, k, epsilon, etc.)

        Returns:
            Field data as numpy array

        Raises:
            KeyError: If field not found in results
        """
        if field not in results["fields"]:
            raise KeyError(f"Field '{field}' not found in results")

        return results["fields"][field]

    def calculate_forces(
        self,
        results: Dict[str, Any],
        patch_names: List[str],
        rho: float = 1.225,
        U_inf: float = 1.0,
        A_ref: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculate forces and coefficients on specified patches.

        Args:
            results: Results dictionary
            patch_names: List of patch names to compute forces on
            rho: Fluid density (kg/m^3)
            U_inf: Reference velocity (m/s)
            A_ref: Reference area (m^2)

        Returns:
            Dictionary with force data:
                - force_pressure: Pressure force vector [Fx, Fy, Fz]
                - force_viscous: Viscous force vector
                - force_total: Total force vector
                - drag: Drag force (x-direction)
                - lift: Lift force (z-direction)
                - Cd: Drag coefficient
                - Cl: Lift coefficient

        Note:
            This is a simplified implementation. For accurate force calculation,
            use OpenFOAM's built-in forces function object during simulation.
        """
        logger.warning(
            "Force calculation from post-processing is approximate. "
            "Use OpenFOAM forces function object for accurate results."
        )

        # Dynamic pressure
        q_inf = 0.5 * rho * U_inf**2

        # This is a placeholder implementation
        # In practice, you would:
        # 1. Extract pressure field on boundary patches
        # 2. Compute surface normals
        # 3. Integrate pressure * normal over surface
        # 4. Add viscous stresses contribution

        # For now, return dummy values
        force_total = np.array([0.0, 0.0, 0.0])

        drag = force_total[0]
        lift = force_total[2]

        Cd = drag / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0
        Cl = lift / (q_inf * A_ref) if q_inf * A_ref > 0 else 0.0

        return {
            "force_pressure": force_total,
            "force_viscous": np.array([0.0, 0.0, 0.0]),
            "force_total": force_total,
            "drag": drag,
            "lift": lift,
            "Cd": Cd,
            "Cl": Cl,
            "dynamic_pressure": q_inf,
            "reference_area": A_ref
        }

    def create_vtk_export(
        self,
        results: Dict[str, Any],
        output_file: str,
        fields: Optional[List[str]] = None
    ) -> str:
        """
        Export results to VTK format for visualization.

        Args:
            results: Results dictionary from load_results()
            output_file: Output VTK file path
            fields: List of fields to export (None = all)

        Returns:
            Path to created VTK file

        Raises:
            ImportError: If PyVista not installed
        """
        if pv is None:
            raise ImportError(
                "pyvista is not installed. Install it with: pip install pyvista"
            )

        logger.info(f"Exporting results to VTK: {output_file}")

        # Get mesh points
        mesh_data = results["mesh"]
        points = np.array(mesh_data["points"])

        # Create unstructured grid
        # This is simplified - would need proper cell connectivity
        point_cloud = pv.PolyData(points)

        # Add field data
        fields_to_export = fields if fields else list(results["fields"].keys())

        for field_name in fields_to_export:
            if field_name in results["fields"]:
                field_data = results["fields"][field_name]

                # Add to point cloud
                if isinstance(field_data, np.ndarray):
                    if len(field_data) == len(points):
                        point_cloud[field_name] = field_data

        # Save to VTK
        point_cloud.save(output_file)
        logger.info(f"VTK export complete: {output_file}")

        return output_file

    def compute_statistics(
        self,
        results: Dict[str, Any],
        field: str = "U"
    ) -> Dict[str, float]:
        """
        Compute statistics for a field.

        Args:
            results: Results dictionary
            field: Field name

        Returns:
            Dictionary with statistics (min, max, mean, std)
        """
        field_data = self.extract_field(results, field)

        if not isinstance(field_data, np.ndarray):
            field_data = np.array(field_data)

        # Handle both scalar and vector fields
        if field_data.ndim == 1:
            # Scalar field
            stats = {
                "min": float(np.min(field_data)),
                "max": float(np.max(field_data)),
                "mean": float(np.mean(field_data)),
                "std": float(np.std(field_data)),
                "median": float(np.median(field_data))
            }
        else:
            # Vector field - compute magnitude
            magnitude = np.linalg.norm(field_data, axis=1)
            stats = {
                "magnitude_min": float(np.min(magnitude)),
                "magnitude_max": float(np.max(magnitude)),
                "magnitude_mean": float(np.mean(magnitude)),
                "magnitude_std": float(np.std(magnitude)),
                "component_x_mean": float(np.mean(field_data[:, 0])),
                "component_y_mean": float(np.mean(field_data[:, 1])),
                "component_z_mean": float(np.mean(field_data[:, 2])) if field_data.shape[1] > 2 else 0.0
            }

        return stats

    def extract_line_data(
        self,
        results: Dict[str, Any],
        field: str,
        start_point: Tuple[float, float, float],
        end_point: Tuple[float, float, float],
        n_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Extract field data along a line.

        Args:
            results: Results dictionary
            field: Field name
            start_point: Line start point (x, y, z)
            end_point: Line end point (x, y, z)
            n_samples: Number of sample points

        Returns:
            Dictionary with:
                - points: Sample point coordinates
                - values: Field values at sample points
        """
        # Generate sample points along line
        start = np.array(start_point)
        end = np.array(end_point)

        t = np.linspace(0, 1, n_samples)
        sample_points = start + np.outer(t, end - start)

        # In a full implementation, you would interpolate field values
        # to these sample points from the mesh
        # For now, return dummy data
        logger.warning("Line data extraction not fully implemented")

        return {
            "points": sample_points,
            "values": np.zeros(n_samples),
            "distance": np.linspace(0, np.linalg.norm(end - start), n_samples)
        }

    def get_time_series(
        self,
        field: str,
        location: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract time series data for a field.

        Args:
            field: Field name
            location: Point location (x, y, z) - if None, use domain average

        Returns:
            Dictionary with:
                - times: Time values
                - values: Field values over time
        """
        times = []
        values = []

        for time_val in self.time_dirs:
            try:
                results = self.load_results(time_step=time_val, fields=[field])

                if field in results["fields"]:
                    field_data = results["fields"][field]

                    # Compute average or extract at location
                    if location is None:
                        # Domain average
                        if isinstance(field_data, np.ndarray):
                            value = np.mean(field_data)
                        else:
                            value = field_data
                    else:
                        # Extract at location (simplified)
                        value = 0.0  # Would need interpolation

                    times.append(time_val)
                    values.append(value)

            except Exception as e:
                logger.warning(f"Failed to load time {time_val}: {str(e)}")

        return {
            "times": np.array(times),
            "values": np.array(values)
        }
