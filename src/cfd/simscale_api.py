"""
SimScale API Connector Module

Provides cloud-based CFD simulation capabilities through SimScale API.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimScaleConnector:
    """
    SimScale cloud CFD platform connector.

    This class provides methods to:
    - Authenticate with SimScale API
    - Upload geometry files
    - Create and configure simulations
    - Run cloud simulations
    - Monitor simulation status
    - Download results

    Note:
        Requires SimScale API key. Get one at: https://www.simscale.com/
    """

    BASE_URL = "https://api.simscale.com/v0"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize SimScale connector.

        Args:
            api_key: SimScale API key (or set SIMSCALE_API_KEY env variable)

        Raises:
            ImportError: If requests library not installed
            ValueError: If API key not provided
        """
        if requests is None:
            raise ImportError(
                "requests is not installed. Install it with: pip install requests"
            )

        self.api_key = api_key or os.getenv("SIMSCALE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SimScale API key required. "
                "Provide it as argument or set SIMSCALE_API_KEY environment variable. "
                "Get your API key at: https://www.simscale.com/docs/simscale/api/"
            )

        self.headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Verify connection
        self._verify_connection()

    def _verify_connection(self) -> bool:
        """
        Verify API connection and authentication.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/user")

            if response.status_code == 200:
                user_data = response.json()
                logger.info(f"Connected to SimScale as: {user_data.get('username', 'Unknown')}")
                return True
            elif response.status_code == 401:
                raise ConnectionError("Invalid API key")
            else:
                raise ConnectionError(f"Connection failed: {response.status_code}")

        except requests.RequestException as e:
            logger.error(f"Failed to connect to SimScale: {str(e)}")
            raise ConnectionError(f"Failed to connect to SimScale: {str(e)}")

    def create_project(self, name: str, description: str = "") -> str:
        """
        Create a new project.

        Args:
            name: Project name
            description: Project description

        Returns:
            Project ID

        Raises:
            RuntimeError: If project creation fails
        """
        payload = {
            "name": name,
            "description": description
        }

        try:
            response = self.session.post(
                f"{self.BASE_URL}/projects",
                json=payload
            )

            if response.status_code in [200, 201]:
                project_data = response.json()
                project_id = project_data["projectId"]
                logger.info(f"Project created: {name} (ID: {project_id})")
                return project_id
            else:
                raise RuntimeError(
                    f"Failed to create project: {response.status_code} - {response.text}"
                )

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to create project: {str(e)}")

    def upload_geometry(
        self,
        project_id: str,
        geometry_file: str,
        geometry_name: Optional[str] = None
    ) -> str:
        """
        Upload geometry file to project.

        Args:
            project_id: Project ID
            geometry_file: Path to geometry file (STEP, STL, etc.)
            geometry_name: Name for geometry (defaults to filename)

        Returns:
            Geometry ID

        Raises:
            FileNotFoundError: If geometry file doesn't exist
            RuntimeError: If upload fails
        """
        if not os.path.exists(geometry_file):
            raise FileNotFoundError(f"Geometry file not found: {geometry_file}")

        file_path = Path(geometry_file)
        name = geometry_name or file_path.stem

        logger.info(f"Uploading geometry: {name}")

        try:
            # Get upload URL
            response = self.session.post(
                f"{self.BASE_URL}/projects/{project_id}/geometries",
                json={"name": name}
            )

            if response.status_code not in [200, 201]:
                raise RuntimeError(f"Failed to initiate upload: {response.text}")

            geometry_data = response.json()
            geometry_id = geometry_data["geometryId"]
            upload_url = geometry_data.get("uploadUrl")

            if not upload_url:
                raise RuntimeError("No upload URL provided")

            # Upload file
            with open(geometry_file, 'rb') as f:
                upload_response = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": "application/octet-stream"}
                )

            if upload_response.status_code not in [200, 201]:
                raise RuntimeError(f"Failed to upload file: {upload_response.text}")

            # Confirm upload
            confirm_response = self.session.post(
                f"{self.BASE_URL}/projects/{project_id}/geometries/{geometry_id}/confirm"
            )

            if confirm_response.status_code in [200, 201]:
                logger.info(f"Geometry uploaded successfully (ID: {geometry_id})")
                return geometry_id
            else:
                raise RuntimeError(f"Failed to confirm upload: {confirm_response.text}")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to upload geometry: {str(e)}")

    def create_simulation(
        self,
        project_id: str,
        geometry_id: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Create CFD simulation.

        Args:
            project_id: Project ID
            geometry_id: Geometry ID
            config: Simulation configuration
                Example:
                {
                    "name": "Flow Analysis",
                    "type": "INCOMPRESSIBLE",
                    "model": {
                        "turbulenceModel": "K_EPSILON",
                        "fluid": "air"
                    },
                    "boundaryConditions": [...],
                    "numerics": {...}
                }

        Returns:
            Simulation ID

        Raises:
            RuntimeError: If simulation creation fails
        """
        logger.info(f"Creating simulation: {config.get('name', 'Unnamed')}")

        payload = {
            "name": config.get("name", "CFD Simulation"),
            "geometryId": geometry_id,
            "type": config.get("type", "INCOMPRESSIBLE"),
            "model": config.get("model", {}),
            "boundaryConditions": config.get("boundaryConditions", []),
            "numerics": config.get("numerics", {})
        }

        try:
            response = self.session.post(
                f"{self.BASE_URL}/projects/{project_id}/simulations",
                json=payload
            )

            if response.status_code in [200, 201]:
                sim_data = response.json()
                sim_id = sim_data["simulationId"]
                logger.info(f"Simulation created (ID: {sim_id})")
                return sim_id
            else:
                raise RuntimeError(
                    f"Failed to create simulation: {response.status_code} - {response.text}"
                )

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to create simulation: {str(e)}")

    def run_simulation(
        self,
        project_id: str,
        simulation_id: str,
        wait_for_completion: bool = False,
        poll_interval: int = 30
    ) -> Dict[str, Any]:
        """
        Run simulation.

        Args:
            project_id: Project ID
            simulation_id: Simulation ID
            wait_for_completion: Wait until simulation completes
            poll_interval: Status polling interval in seconds

        Returns:
            Dictionary with run information:
                - run_id: Run ID
                - status: Current status
                - progress: Progress percentage (if available)

        Raises:
            RuntimeError: If simulation start fails
        """
        logger.info(f"Starting simulation: {simulation_id}")

        try:
            # Start simulation
            response = self.session.post(
                f"{self.BASE_URL}/projects/{project_id}/simulations/{simulation_id}/runs"
            )

            if response.status_code not in [200, 201]:
                raise RuntimeError(f"Failed to start simulation: {response.text}")

            run_data = response.json()
            run_id = run_data["runId"]

            logger.info(f"Simulation run started (Run ID: {run_id})")

            if wait_for_completion:
                return self._wait_for_completion(
                    project_id,
                    simulation_id,
                    run_id,
                    poll_interval
                )

            return {
                "run_id": run_id,
                "status": "QUEUED",
                "progress": 0
            }

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to run simulation: {str(e)}")

    def _wait_for_completion(
        self,
        project_id: str,
        simulation_id: str,
        run_id: str,
        poll_interval: int
    ) -> Dict[str, Any]:
        """
        Wait for simulation to complete.

        Args:
            project_id: Project ID
            simulation_id: Simulation ID
            run_id: Run ID
            poll_interval: Polling interval in seconds

        Returns:
            Final status information
        """
        logger.info("Waiting for simulation to complete...")

        while True:
            status = self.get_simulation_status(project_id, simulation_id, run_id)

            current_status = status.get("status", "UNKNOWN")
            progress = status.get("progress", 0)

            logger.info(f"Status: {current_status}, Progress: {progress}%")

            if current_status in ["FINISHED", "COMPLETED"]:
                logger.info("Simulation completed successfully")
                return status
            elif current_status in ["FAILED", "CANCELED", "CANCELLED"]:
                logger.error(f"Simulation {current_status.lower()}")
                return status

            time.sleep(poll_interval)

    def get_simulation_status(
        self,
        project_id: str,
        simulation_id: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Get simulation run status.

        Args:
            project_id: Project ID
            simulation_id: Simulation ID
            run_id: Run ID

        Returns:
            Status information dictionary

        Raises:
            RuntimeError: If status check fails
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/projects/{project_id}/simulations/{simulation_id}/runs/{run_id}"
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"Failed to get status: {response.text}")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get simulation status: {str(e)}")

    def download_results(
        self,
        project_id: str,
        simulation_id: str,
        run_id: str,
        output_dir: str,
        result_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        Download simulation results.

        Args:
            project_id: Project ID
            simulation_id: Simulation ID
            run_id: Run ID
            output_dir: Output directory for results
            result_types: Types of results to download (None = all)

        Returns:
            List of downloaded file paths

        Raises:
            RuntimeError: If download fails
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading results to: {output_dir}")

        try:
            # Get available results
            response = self.session.get(
                f"{self.BASE_URL}/projects/{project_id}/simulations/{simulation_id}/runs/{run_id}/results"
            )

            if response.status_code != 200:
                raise RuntimeError(f"Failed to get results list: {response.text}")

            results_data = response.json()
            available_results = results_data.get("results", [])

            downloaded_files = []

            for result in available_results:
                result_type = result.get("type")
                result_id = result.get("resultId")

                # Filter by result types if specified
                if result_types and result_type not in result_types:
                    continue

                # Get download URL
                download_response = self.session.get(
                    f"{self.BASE_URL}/projects/{project_id}/simulations/{simulation_id}/runs/{run_id}/results/{result_id}/download"
                )

                if download_response.status_code == 200:
                    download_data = download_response.json()
                    download_url = download_data.get("downloadUrl")

                    if download_url:
                        # Download file
                        filename = f"{result_type}_{result_id}.zip"
                        output_file = output_path / filename

                        file_response = requests.get(download_url, stream=True)
                        with open(output_file, 'wb') as f:
                            for chunk in file_response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        downloaded_files.append(str(output_file))
                        logger.info(f"Downloaded: {filename}")

            logger.info(f"Downloaded {len(downloaded_files)} result files")
            return downloaded_files

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download results: {str(e)}")

    def list_projects(self) -> List[Dict[str, Any]]:
        """
        List all projects.

        Returns:
            List of project dictionaries
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/projects")

            if response.status_code == 200:
                return response.json().get("projects", [])
            else:
                logger.error(f"Failed to list projects: {response.text}")
                return []

        except requests.RequestException as e:
            logger.error(f"Failed to list projects: {str(e)}")
            return []

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.

        Args:
            project_id: Project ID

        Returns:
            True if deletion successful

        Raises:
            RuntimeError: If deletion fails
        """
        try:
            response = self.session.delete(
                f"{self.BASE_URL}/projects/{project_id}"
            )

            if response.status_code in [200, 204]:
                logger.info(f"Project deleted: {project_id}")
                return True
            else:
                raise RuntimeError(f"Failed to delete project: {response.text}")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to delete project: {str(e)}")

    def get_project_info(self, project_id: str) -> Dict[str, Any]:
        """
        Get project information.

        Args:
            project_id: Project ID

        Returns:
            Project information dictionary
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/projects/{project_id}"
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"Failed to get project info: {response.text}")

        except requests.RequestException as e:
            raise RuntimeError(f"Failed to get project info: {str(e)}")

    def create_standard_cfd_config(
        self,
        name: str = "CFD Analysis",
        inlet_velocity: float = 10.0,
        fluid: str = "air",
        turbulence_model: str = "K_EPSILON"
    ) -> Dict[str, Any]:
        """
        Create a standard CFD configuration template.

        Args:
            name: Simulation name
            inlet_velocity: Inlet velocity (m/s)
            fluid: Fluid type
            turbulence_model: Turbulence model

        Returns:
            Configuration dictionary
        """
        config = {
            "name": name,
            "type": "INCOMPRESSIBLE",
            "model": {
                "turbulenceModel": turbulence_model,
                "fluid": fluid,
                "velocity": inlet_velocity
            },
            "boundaryConditions": [
                {
                    "name": "inlet",
                    "type": "VELOCITY_INLET",
                    "velocity": [inlet_velocity, 0, 0]
                },
                {
                    "name": "outlet",
                    "type": "PRESSURE_OUTLET",
                    "pressure": 0
                },
                {
                    "name": "walls",
                    "type": "WALL",
                    "wallType": "NO_SLIP"
                }
            ],
            "numerics": {
                "relaxationFactors": {
                    "pressure": 0.3,
                    "velocity": 0.7,
                    "turbulence": 0.7
                },
                "convergenceCriteria": {
                    "residuals": {
                        "pressure": 1e-4,
                        "velocity": 1e-4,
                        "turbulence": 1e-4
                    }
                }
            }
        }

        return config
