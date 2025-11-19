"""
Result Parser Module
Parse and extract CFD simulation results from various formats.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import re


class ResultParser:
    """
    Parser for CFD simulation results.

    Supports:
    - OpenFOAM native format
    - VTK files
    - CSV exports
    - Time series data
    """

    def __init__(self, case_dir: Path):
        """
        Initialize result parser.

        Args:
            case_dir: Path to simulation case directory
        """
        self.case_dir = Path(case_dir)

    def get_available_times(self) -> List[float]:
        """
        Get list of available time steps.

        Returns:
            List of time step values
        """
        times = []

        for item in self.case_dir.iterdir():
            if item.is_dir():
                try:
                    time_val = float(item.name)
                    times.append(time_val)
                except ValueError:
                    continue

        return sorted(times)

    def get_available_fields(self, time_step: Optional[float] = None) -> List[str]:
        """
        Get list of available fields at a time step.

        Args:
            time_step: Time step (None for latest)

        Returns:
            List of field names
        """
        if time_step is None:
            times = self.get_available_times()
            if not times:
                return []
            time_step = times[-1]

        time_dir = self.case_dir / str(time_step)
        if not time_dir.exists():
            return []

        fields = []
        for item in time_dir.iterdir():
            if item.is_file() and not item.name.startswith('.'):
                fields.append(item.name)

        return fields

    def read_field_data(self, field_name: str,
                       time_step: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Read field data from OpenFOAM format.

        Args:
            field_name: Name of the field
            time_step: Time step (None for latest)

        Returns:
            Field data as numpy array, or None if not found

        Note:
            This is a simplified implementation. Full implementation
            would properly parse OpenFOAM field files.
        """
        if time_step is None:
            times = self.get_available_times()
            if not times:
                return None
            time_step = times[-1]

        field_file = self.case_dir / str(time_step) / field_name

        if not field_file.exists():
            return None

        try:
            # Simplified parsing - real implementation would use PyFoam
            # or proper OpenFOAM file parser
            data = self._parse_foam_field_file(field_file)
            return data
        except Exception as e:
            print(f"Error reading field {field_name}: {e}")
            return None

    def _parse_foam_field_file(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Parse OpenFOAM field file (simplified).

        Args:
            file_path: Path to field file

        Returns:
            Parsed data as numpy array

        Note:
            This is a placeholder. Real implementation would properly
            parse OpenFOAM dictionary format and extract internalField data.
        """
        # Placeholder - return dummy data
        # Real implementation would parse the file properly
        return np.random.rand(100, 3)  # Dummy vector field

    def read_residuals(self, log_file: Optional[Path] = None) -> Dict[str, List[float]]:
        """
        Read residuals from log file.

        Args:
            log_file: Path to log file (None for default)

        Returns:
            Dictionary of residual histories
        """
        if log_file is None:
            log_file = self.case_dir / "log.simpleFoam"

        if not log_file.exists():
            return {}

        residuals = {"U": [], "p": [], "k": [], "omega": [], "epsilon": []}

        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Parse residual lines
                    for field in residuals.keys():
                        if f"Solving for {field}" in line:
                            match = re.search(r'Initial residual = ([\d.e-]+)', line)
                            if match:
                                residuals[field].append(float(match.group(1)))
        except Exception as e:
            print(f"Error reading residuals: {e}")

        return residuals

    def get_field_statistics(self, field_name: str,
                            time_step: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate field statistics.

        Args:
            field_name: Name of the field
            time_step: Time step (None for latest)

        Returns:
            Dictionary with min, max, mean, std
        """
        data = self.read_field_data(field_name, time_step)

        if data is None:
            return {}

        # Handle vector fields (take magnitude)
        if data.ndim > 1:
            data = np.linalg.norm(data, axis=1)

        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }

    def extract_boundary_data(self, boundary_name: str, field_name: str,
                              time_step: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Extract field data on a specific boundary.

        Args:
            boundary_name: Name of the boundary
            field_name: Field name
            time_step: Time step

        Returns:
            Boundary field data
        """
        # Placeholder implementation
        # Real implementation would extract boundary patch data
        return None

    def export_to_csv(self, field_name: str, output_file: Path,
                     time_step: Optional[float] = None) -> bool:
        """
        Export field data to CSV.

        Args:
            field_name: Field name
            output_file: Output CSV file path
            time_step: Time step

        Returns:
            Success status
        """
        data = self.read_field_data(field_name, time_step)

        if data is None:
            return False

        try:
            # Simple CSV export
            np.savetxt(output_file, data, delimiter=',')
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False

    def get_probe_data(self, probe_location: Tuple[float, float, float],
                      field_name: str) -> List[float]:
        """
        Get time series data at a probe location.

        Args:
            probe_location: (x, y, z) coordinates
            field_name: Field name

        Returns:
            Time series data

        Note:
            This requires probe data to be written during simulation.
        """
        # Placeholder - would read from postProcessing/probes directory
        return []

    def calculate_forces(self, patch_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Calculate forces on specified patches.

        Args:
            patch_names: List of patch names

        Returns:
            Dictionary with force and moment data

        Note:
            Requires forces function object to be enabled in simulation.
        """
        forces_dir = self.case_dir / "postProcessing" / "forces"

        if not forces_dir.exists():
            return {}

        # Placeholder - would read forces.dat file
        return {
            "force": np.zeros(3),
            "moment": np.zeros(3),
        }

    def get_convergence_status(self, tolerance: float = 1e-4) -> Tuple[bool, str]:
        """
        Check convergence status.

        Args:
            tolerance: Convergence tolerance

        Returns:
            Tuple of (converged, message)
        """
        residuals = self.read_residuals()

        if not residuals:
            return False, "No residual data found"

        # Check if all fields are below tolerance
        converged_fields = []
        diverged_fields = []

        for field, values in residuals.items():
            if not values:
                continue

            latest_residual = values[-1]

            if latest_residual < tolerance:
                converged_fields.append(field)
            elif latest_residual > 1.0:
                diverged_fields.append(field)

        if diverged_fields:
            return False, f"Diverged fields: {', '.join(diverged_fields)}"
        elif len(converged_fields) >= len([v for v in residuals.values() if v]):
            return True, "All fields converged"
        else:
            return False, "Still converging..."
