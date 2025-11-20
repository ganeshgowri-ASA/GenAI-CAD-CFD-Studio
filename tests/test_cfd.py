"""
CFD Module Tests

Comprehensive tests for the CFD pipeline including:
- Mesh generation (Gmsh)
- OpenFOAM case setup
- Result parsing
- SimScale API integration
"""

import os
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import CFD modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cfd.gmsh_mesher import GmshMesher
from cfd.pyfoam_wrapper import PyFoamWrapper
from cfd.result_parser import ResultParser
from cfd.simscale_api import SimScaleConnector


class TestGmshMesher(unittest.TestCase):
    """Test cases for GmshMesher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_mesher_initialization(self):
        """Test GmshMesher initialization."""
        try:
            mesher = GmshMesher(verbose=False)
            self.assertIsNotNone(mesher)
            self.assertFalse(mesher.verbose)
            self.assertFalse(mesher.initialized)
        except ImportError as e:
            self.skipTest(f"Gmsh not available: {str(e)}")

    def test_generate_mesh_missing_file(self):
        """Test mesh generation with missing STEP file."""
        try:
            mesher = GmshMesher(verbose=False)
            with self.assertRaises(FileNotFoundError):
                mesher.generate_mesh(
                    step_file="/nonexistent/file.step",
                    mesh_size=0.1
                )
        except ImportError as e:
            self.skipTest(f"Gmsh not available: {str(e)}")

    @patch('cfd.gmsh_mesher.gmsh')
    def test_generate_mesh_with_refinement_zones(self, mock_gmsh):
        """Test mesh generation with refinement zones."""
        try:
            # Create a dummy STEP file
            step_file = os.path.join(self.temp_dir, "test.step")
            with open(step_file, "w") as f:
                f.write("ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;")

            mesher = GmshMesher(verbose=False)

            refinement_zones = [
                {
                    "type": "box",
                    "x_min": 0,
                    "x_max": 1,
                    "y_min": 0,
                    "y_max": 1,
                    "z_min": 0,
                    "z_max": 1,
                    "size": 0.05
                }
            ]

            # Mock gmsh methods
            mock_gmsh.initialize = Mock()
            mock_gmsh.finalize = Mock()
            mock_gmsh.clear = Mock()
            mock_gmsh.model.add = Mock()
            mock_gmsh.model.occ.importShapes = Mock()
            mock_gmsh.model.occ.synchronize = Mock()
            mock_gmsh.model.mesh.generate = Mock()
            mock_gmsh.write = Mock()
            mock_gmsh.option.setNumber = Mock()
            mock_gmsh.model.getEntities = Mock(return_value=[])

            output_file = mesher.generate_mesh(
                step_file=step_file,
                mesh_size=0.1,
                refinement_zones=refinement_zones
            )

            self.assertTrue(output_file.endswith(".msh"))
            mock_gmsh.model.mesh.generate.assert_called_once_with(3)

        except ImportError as e:
            self.skipTest(f"Gmsh not available: {str(e)}")

    def test_get_mesh_stats_missing_file(self):
        """Test mesh statistics with missing file."""
        try:
            mesher = GmshMesher(verbose=False)
            with self.assertRaises(FileNotFoundError):
                mesher.get_mesh_stats("/nonexistent/mesh.msh")
        except ImportError as e:
            self.skipTest(f"Gmsh not available: {str(e)}")


class TestPyFoamWrapper(unittest.TestCase):
    """Test cases for PyFoamWrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_wrapper_initialization(self):
        """Test PyFoamWrapper initialization."""
        wrapper = PyFoamWrapper()
        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.openfoam_env, {})

    def test_create_case(self):
        """Test OpenFOAM case creation."""
        wrapper = PyFoamWrapper()
        case_dir = os.path.join(self.temp_dir, "test_case")

        case_path = wrapper.create_case(
            case_dir=case_dir,
            solver_type="simpleFoam"
        )

        self.assertTrue(os.path.exists(case_path))
        self.assertTrue(os.path.exists(os.path.join(case_path, "0")))
        self.assertTrue(os.path.exists(os.path.join(case_path, "constant")))
        self.assertTrue(os.path.exists(os.path.join(case_path, "system")))
        self.assertTrue(os.path.exists(os.path.join(case_path, "system", "controlDict")))
        self.assertTrue(os.path.exists(os.path.join(case_path, "system", "fvSchemes")))
        self.assertTrue(os.path.exists(os.path.join(case_path, "system", "fvSolution")))

    def test_set_boundary_conditions(self):
        """Test setting boundary conditions."""
        wrapper = PyFoamWrapper()
        case_dir = os.path.join(self.temp_dir, "test_case")
        wrapper.create_case(case_dir=case_dir)

        bc_dict = {
            "inlet": {
                "U": {"type": "fixedValue", "value": [10, 0, 0]},
                "p": {"type": "zeroGradient"}
            },
            "outlet": {
                "U": {"type": "zeroGradient"},
                "p": {"type": "fixedValue", "value": 0}
            },
            "walls": {
                "U": {"type": "noSlip"},
                "p": {"type": "zeroGradient"}
            }
        }

        wrapper.set_boundary_conditions(case_dir, bc_dict)

        # Check if boundary condition files were created
        self.assertTrue(os.path.exists(os.path.join(case_dir, "0", "U")))
        self.assertTrue(os.path.exists(os.path.join(case_dir, "0", "p")))

        # Verify content
        with open(os.path.join(case_dir, "0", "U"), "r") as f:
            content = f.read()
            self.assertIn("inlet", content)
            self.assertIn("fixedValue", content)

    def test_set_fluid_properties(self):
        """Test setting fluid properties."""
        wrapper = PyFoamWrapper()
        case_dir = os.path.join(self.temp_dir, "test_case")
        wrapper.create_case(case_dir=case_dir)

        wrapper.set_fluid_properties(
            case_dir=case_dir,
            fluid="air",
            turbulence_model="k-epsilon"
        )

        # Check if property files were created
        self.assertTrue(os.path.exists(
            os.path.join(case_dir, "constant", "transportProperties")
        ))
        self.assertTrue(os.path.exists(
            os.path.join(case_dir, "constant", "turbulenceProperties")
        ))

        # Verify content
        with open(os.path.join(case_dir, "constant", "transportProperties"), "r") as f:
            content = f.read()
            self.assertIn("nu", content)
            self.assertIn("1.5e-05", content)  # Air viscosity

    def test_convert_mesh_missing_file(self):
        """Test mesh conversion with missing file."""
        wrapper = PyFoamWrapper()
        case_dir = os.path.join(self.temp_dir, "test_case")
        wrapper.create_case(case_dir=case_dir)

        with self.assertRaises(FileNotFoundError):
            wrapper.convert_mesh(
                mesh_file="/nonexistent/mesh.msh",
                case_dir=case_dir
            )

    def test_fluid_properties_dict(self):
        """Test fluid properties dictionary."""
        wrapper = PyFoamWrapper()

        # Check that standard fluids are defined
        self.assertIn("air", wrapper.FLUID_PROPERTIES)
        self.assertIn("water", wrapper.FLUID_PROPERTIES)
        self.assertIn("oil", wrapper.FLUID_PROPERTIES)

        # Check air properties
        air_props = wrapper.FLUID_PROPERTIES["air"]
        self.assertEqual(air_props["density"], 1.225)
        self.assertAlmostEqual(air_props["kinematic_viscosity"], 1.5e-5)

    def test_turbulence_models(self):
        """Test turbulence model mapping."""
        wrapper = PyFoamWrapper()

        self.assertIn("k-epsilon", wrapper.TURBULENCE_MODELS)
        self.assertIn("k-omega-sst", wrapper.TURBULENCE_MODELS)
        self.assertIn("laminar", wrapper.TURBULENCE_MODELS)


class TestResultParser(unittest.TestCase):
    """Test cases for ResultParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a mock case structure
        self.case_dir = os.path.join(self.temp_dir, "test_case")
        os.makedirs(self.case_dir)

        # Create constant/polyMesh
        poly_mesh_dir = os.path.join(self.case_dir, "constant", "polyMesh")
        os.makedirs(poly_mesh_dir)

        # Create time directories
        for time_val in [0, 100, 200]:
            time_dir = os.path.join(self.case_dir, str(time_val))
            os.makedirs(time_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_parser_initialization(self):
        """Test ResultParser initialization."""
        parser = ResultParser(self.case_dir)
        self.assertIsNotNone(parser)
        self.assertEqual(parser.case_dir, Path(self.case_dir))

    def test_parser_initialization_invalid_dir(self):
        """Test ResultParser initialization with invalid directory."""
        with self.assertRaises(ValueError):
            ResultParser("/nonexistent/directory")

    def test_find_time_directories(self):
        """Test finding time directories."""
        parser = ResultParser(self.case_dir)
        self.assertEqual(parser.time_dirs, [0.0, 100.0, 200.0])

    def test_get_latest_time(self):
        """Test getting latest time step."""
        parser = ResultParser(self.case_dir)
        latest_time = parser.get_latest_time()
        self.assertEqual(latest_time, 200.0)

    def test_compute_statistics_scalar(self):
        """Test statistics computation for scalar field."""
        parser = ResultParser(self.case_dir)

        # Create mock results with scalar field
        results = {
            "time": 100.0,
            "mesh": {},
            "fields": {
                "p": np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            }
        }

        stats = parser.compute_statistics(results, field="p")

        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["median"], 3.0)

    def test_compute_statistics_vector(self):
        """Test statistics computation for vector field."""
        parser = ResultParser(self.case_dir)

        # Create mock results with vector field
        results = {
            "time": 100.0,
            "mesh": {},
            "fields": {
                "U": np.array([
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0]
                ])
            }
        }

        stats = parser.compute_statistics(results, field="U")

        self.assertIn("magnitude_min", stats)
        self.assertIn("magnitude_max", stats)
        self.assertIn("magnitude_mean", stats)
        self.assertEqual(stats["magnitude_min"], 1.0)
        self.assertEqual(stats["magnitude_max"], 3.0)
        self.assertEqual(stats["magnitude_mean"], 2.0)

    def test_calculate_forces(self):
        """Test force calculation."""
        parser = ResultParser(self.case_dir)

        # Create mock results
        results = {
            "time": 100.0,
            "mesh": {},
            "fields": {}
        }

        forces = parser.calculate_forces(
            results=results,
            patch_names=["walls"],
            rho=1.225,
            U_inf=10.0,
            A_ref=1.0
        )

        self.assertIn("force_total", forces)
        self.assertIn("drag", forces)
        self.assertIn("lift", forces)
        self.assertIn("Cd", forces)
        self.assertIn("Cl", forces)


class TestSimScaleConnector(unittest.TestCase):
    """Test cases for SimScaleConnector class."""

    def test_connector_initialization_no_api_key(self):
        """Test connector initialization without API key."""
        # Temporarily clear environment variable
        original_key = os.environ.get("SIMSCALE_API_KEY")
        if original_key:
            del os.environ["SIMSCALE_API_KEY"]

        with self.assertRaises(ValueError):
            SimScaleConnector()

        # Restore original key
        if original_key:
            os.environ["SIMSCALE_API_KEY"] = original_key

    @patch('cfd.simscale_api.requests')
    def test_connector_initialization_with_api_key(self, mock_requests):
        """Test connector initialization with API key."""
        try:
            # Mock the session
            mock_session = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"username": "test_user"}
            mock_session.get.return_value = mock_response
            mock_requests.Session.return_value = mock_session

            connector = SimScaleConnector(api_key="test_key")
            self.assertIsNotNone(connector)
            self.assertEqual(connector.api_key, "test_key")

        except ImportError as e:
            self.skipTest(f"requests not available: {str(e)}")

    def test_create_standard_cfd_config(self):
        """Test creating standard CFD configuration."""
        try:
            # Create connector with dummy key (won't connect)
            os.environ["SIMSCALE_API_KEY"] = "dummy_key"
            with patch('cfd.simscale_api.requests'):
                with patch.object(SimScaleConnector, '_verify_connection'):
                    connector = SimScaleConnector(api_key="dummy_key")

                    config = connector.create_standard_cfd_config(
                        name="Test CFD",
                        inlet_velocity=10.0,
                        fluid="air",
                        turbulence_model="K_EPSILON"
                    )

                    self.assertIn("name", config)
                    self.assertIn("type", config)
                    self.assertIn("model", config)
                    self.assertIn("boundaryConditions", config)
                    self.assertIn("numerics", config)

                    self.assertEqual(config["name"], "Test CFD")
                    self.assertEqual(config["model"]["velocity"], 10.0)
                    self.assertEqual(config["model"]["fluid"], "air")

        except ImportError as e:
            self.skipTest(f"requests not available: {str(e)}")
        finally:
            if "SIMSCALE_API_KEY" in os.environ:
                del os.environ["SIMSCALE_API_KEY"]


class TestCFDIntegration(unittest.TestCase):
    """Integration tests for CFD pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_case_setup_pipeline(self):
        """Test full pipeline: case creation + BC setup + fluid properties."""
        wrapper = PyFoamWrapper()

        # Create case
        case_dir = os.path.join(self.temp_dir, "integration_case")
        case_path = wrapper.create_case(case_dir=case_dir, solver_type="simpleFoam")

        # Set boundary conditions
        bc_dict = {
            "inlet": {
                "U": {"type": "fixedValue", "value": [10, 0, 0]},
                "p": {"type": "zeroGradient"},
                "k": {"type": "fixedValue", "value": 0.1},
                "epsilon": {"type": "fixedValue", "value": 0.01}
            },
            "outlet": {
                "U": {"type": "zeroGradient"},
                "p": {"type": "fixedValue", "value": 0},
                "k": {"type": "zeroGradient"},
                "epsilon": {"type": "zeroGradient"}
            }
        }

        wrapper.set_boundary_conditions(case_dir, bc_dict)

        # Set fluid properties
        wrapper.set_fluid_properties(
            case_dir=case_dir,
            fluid="air",
            turbulence_model="k-epsilon"
        )

        # Verify all files exist
        required_files = [
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "constant/transportProperties",
            "constant/turbulenceProperties",
            "0/U",
            "0/p",
            "0/k",
            "0/epsilon"
        ]

        for file_path in required_files:
            full_path = os.path.join(case_dir, file_path)
            self.assertTrue(
                os.path.exists(full_path),
                f"Required file missing: {file_path}"
            )


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGmshMesher))
    suite.addTests(loader.loadTestsFromTestCase(TestPyFoamWrapper))
    suite.addTests(loader.loadTestsFromTestCase(TestResultParser))
    suite.addTests(loader.loadTestsFromTestCase(TestSimScaleConnector))
    suite.addTests(loader.loadTestsFromTestCase(TestCFDIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
