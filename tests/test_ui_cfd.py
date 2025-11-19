"""
Unit tests for CFD Analysis UI components.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.components.simulation_wizard import SimulationWizard
from src.ui.components.mesh_configurator import MeshConfigurator
from src.ui.components.boundary_condition_form import BCForm, BCType
from src.ui.components.results_viewer import ResultsViewer
from src.cfd.gmsh_mesher import GmshMesher
from src.cfd.pyfoam_wrapper import PyFoamWrapper, SolverType, TurbulenceModel, SimulationStatus
from src.cfd.result_parser import ResultParser
from src.visualization.pyvista_viewer import PyVistaViewer


class TestSimulationWizard:
    """Test cases for SimulationWizard component."""

    def test_wizard_initialization(self):
        """Test wizard initialization."""
        steps = ["Step 1", "Step 2", "Step 3"]
        wizard = SimulationWizard(steps, session_key="test_wizard")

        assert wizard.steps == steps
        assert wizard.total_steps == 3
        assert wizard.session_key == "test_wizard"

    def test_wizard_navigation(self):
        """Test wizard step navigation."""
        steps = ["Step 1", "Step 2", "Step 3"]
        wizard = SimulationWizard(steps, session_key="test_nav")

        # Initial step should be 0
        assert wizard.current_step == 0

        # Move forward
        wizard.current_step = 1
        assert wizard.current_step == 1

        # Move backward
        wizard.current_step = 0
        assert wizard.current_step == 0

        # Boundary check - can't go negative
        wizard.current_step = -1
        assert wizard.current_step == 0

        # Boundary check - can't exceed max
        wizard.current_step = 10
        assert wizard.current_step == 2  # Should clamp to max

    def test_wizard_data_storage(self):
        """Test wizard data storage and retrieval."""
        wizard = SimulationWizard(["Step 1"], session_key="test_data")

        # Set and get data
        wizard.set_data("test_key", "test_value")
        assert wizard.get_data("test_key") == "test_value"

        # Get with default
        assert wizard.get_data("nonexistent", "default") == "default"

        # Set multiple values
        wizard.set_data("key1", 123)
        wizard.set_data("key2", {"nested": "dict"})

        assert wizard.get_data("key1") == 123
        assert wizard.get_data("key2") == {"nested": "dict"}


class TestMeshConfigurator:
    """Test cases for MeshConfigurator component."""

    def test_mesh_configurator_initialization(self):
        """Test mesh configurator initialization."""
        configurator = MeshConfigurator(session_key="test_mesh")

        assert configurator.session_key == "test_mesh"
        assert configurator.settings["quality"] == "Medium"
        assert configurator.settings["global_size"] == 0.05

    def test_mesh_presets(self):
        """Test mesh quality presets."""
        assert "Coarse" in MeshConfigurator.MESH_PRESETS
        assert "Medium" in MeshConfigurator.MESH_PRESETS
        assert "Fine" in MeshConfigurator.MESH_PRESETS

        # Check preset values
        coarse = MeshConfigurator.MESH_PRESETS["Coarse"]
        assert coarse["size"] == 0.1
        assert coarse["min_size"] == 0.05

    def test_mesh_statistics_estimation(self):
        """Test mesh statistics estimation."""
        configurator = MeshConfigurator(session_key="test_stats")

        # Estimate for simple geometry
        stats = configurator.estimate_mesh_statistics(
            geometry_volume=1.0,
            geometry_surface_area=6.0
        )

        assert "nodes" in stats
        assert "volume_elements" in stats
        assert "surface_elements" in stats
        assert "total_elements" in stats

        assert stats["nodes"] > 0
        assert stats["total_elements"] > 0

    def test_refinement_zone_addition(self):
        """Test adding refinement zones."""
        configurator = MeshConfigurator(session_key="test_refinement")

        # Add a box refinement zone
        box_zone = {
            "name": "Test Box",
            "type": "Box",
            "size": 0.01,
            "center": [0.0, 0.0, 0.0],
            "dims": [1.0, 1.0, 1.0]
        }

        configurator.update_setting("refinement_zones", [box_zone])

        zones = configurator.settings["refinement_zones"]
        assert len(zones) == 1
        assert zones[0]["name"] == "Test Box"
        assert zones[0]["type"] == "Box"

    def test_zone_volume_estimation(self):
        """Test refinement zone volume estimation."""
        configurator = MeshConfigurator(session_key="test_volume")

        # Box zone
        box_zone = {
            "type": "Box",
            "dims": [2.0, 2.0, 2.0]
        }
        volume = configurator._estimate_zone_volume(box_zone)
        assert volume == 8.0  # 2*2*2

        # Sphere zone
        sphere_zone = {
            "type": "Sphere",
            "radius": 1.0
        }
        volume = configurator._estimate_zone_volume(sphere_zone)
        assert abs(volume - (4/3 * np.pi)) < 0.01


class TestBoundaryConditionForm:
    """Test cases for BoundaryConditionForm component."""

    def test_bc_form_initialization(self):
        """Test BC form initialization."""
        form = BCForm(session_key="test_bc")

        assert form.session_key == "test_bc"
        assert isinstance(form.conditions, dict)
        assert isinstance(form.patches, list)

    def test_bc_templates(self):
        """Test BC templates."""
        assert "External Flow (Wind Tunnel)" in BCForm.BC_TEMPLATES
        assert "Internal Flow (Pipe)" in BCForm.BC_TEMPLATES

        # Check template structure
        wind_tunnel = BCForm.BC_TEMPLATES["External Flow (Wind Tunnel)"]
        assert "inlet" in wind_tunnel
        assert "outlet" in wind_tunnel
        assert wind_tunnel["inlet"]["type"] == "INLET"

    def test_patch_management(self):
        """Test patch management."""
        form = BCForm(session_key="test_patches")

        # Set patches
        patches = ["inlet", "outlet", "wall"]
        form.set_patches(patches)

        assert form.patches == patches

    def test_condition_update(self):
        """Test boundary condition update."""
        form = BCForm(session_key="test_update")

        # Update condition
        condition = {
            "type": "INLET",
            "velocity": 10.0,
            "turbulence_intensity": 0.05
        }
        form.update_condition("inlet", condition)

        assert "inlet" in form.conditions
        assert form.conditions["inlet"]["velocity"] == 10.0

    def test_validation(self):
        """Test BC validation."""
        form = BCForm(session_key="test_validation")

        # Set patches
        form.set_patches(["inlet", "outlet", "wall"])

        # Empty conditions should fail validation
        is_valid, errors = form.validate()
        assert not is_valid
        assert len(errors) > 0

        # Add inlet and outlet conditions
        form.update_condition("inlet", {
            "type": "INLET",
            "velocity": 10.0
        })
        form.update_condition("outlet", {
            "type": "OUTLET",
            "pressure": 0.0
        })
        form.update_condition("wall", {
            "type": "WALL"
        })

        is_valid, errors = form.validate()
        assert is_valid
        assert len(errors) == 0


class TestResultsViewer:
    """Test cases for ResultsViewer component."""

    def test_results_viewer_initialization(self):
        """Test results viewer initialization."""
        viewer = ResultsViewer(session_key="test_viewer")

        assert viewer.session_key == "test_viewer"
        assert viewer.settings["field"] == "Velocity"
        assert viewer.settings["viz_type"] == "Contours"
        assert viewer.settings["colormap"] == "viridis"

    def test_visualization_types(self):
        """Test visualization types."""
        assert "Contours" in ResultsViewer.VIZ_TYPES
        assert "Vectors" in ResultsViewer.VIZ_TYPES
        assert "Streamlines" in ResultsViewer.VIZ_TYPES

    def test_colormap_options(self):
        """Test colormap options."""
        assert "viridis" in ResultsViewer.COLORMAPS
        assert "jet" in ResultsViewer.COLORMAPS
        assert "coolwarm" in ResultsViewer.COLORMAPS

    def test_settings_update(self):
        """Test settings update."""
        viewer = ResultsViewer(session_key="test_settings")

        viewer.update_setting("field", "Pressure")
        assert viewer.settings["field"] == "Pressure"

        viewer.update_setting("colormap", "jet")
        assert viewer.settings["colormap"] == "jet"

    def test_field_statistics_calculation(self):
        """Test field statistics calculation (placeholder)."""
        viewer = ResultsViewer(session_key="test_stats")

        # This would test actual statistics calculation with field data
        # For now, just verify the method exists
        assert hasattr(viewer, 'render_statistics')


class TestGmshMesher:
    """Test cases for GmshMesher."""

    def test_mesher_initialization(self):
        """Test mesher initialization."""
        mesher = GmshMesher()

        assert not mesher.initialized
        assert not mesher.geometry_loaded
        assert not mesher.mesh_generated

    def test_mesher_lifecycle(self):
        """Test mesher initialization and finalization."""
        mesher = GmshMesher()

        mesher.initialize(verbose=False)
        assert mesher.initialized

        mesher.finalize()
        assert not mesher.initialized


class TestPyFoamWrapper:
    """Test cases for PyFoamWrapper."""

    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        case_dir = Path("/tmp/test_case")
        wrapper = PyFoamWrapper(case_dir)

        assert wrapper.case_dir == case_dir
        assert wrapper.status == SimulationStatus.NOT_STARTED
        assert wrapper.process is None

    def test_solver_types(self):
        """Test solver type enumeration."""
        assert SolverType.SIMPLE_FOAM.value == "simpleFoam"
        assert SolverType.PIMPLE_FOAM.value == "pimpleFoam"
        assert SolverType.ICOFOAM.value == "icoFoam"

    def test_turbulence_models(self):
        """Test turbulence model enumeration."""
        assert TurbulenceModel.LAMINAR.value == "laminar"
        assert TurbulenceModel.K_EPSILON.value == "kEpsilon"
        assert TurbulenceModel.K_OMEGA_SST.value == "kOmegaSST"

    def test_case_structure_setup(self):
        """Test case directory structure creation."""
        import tempfile
        import shutil

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            wrapper = PyFoamWrapper(temp_dir / "test_case")
            success = wrapper.setup_case_structure()

            assert success
            assert (temp_dir / "test_case" / "0").exists()
            assert (temp_dir / "test_case" / "constant").exists()
            assert (temp_dir / "test_case" / "system").exists()
            assert (temp_dir / "test_case" / "constant" / "polyMesh").exists()

        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

    def test_residuals_storage(self):
        """Test residuals storage."""
        wrapper = PyFoamWrapper(Path("/tmp/test"))

        assert "U" in wrapper.residuals
        assert "p" in wrapper.residuals
        assert isinstance(wrapper.residuals["U"], list)


class TestResultParser:
    """Test cases for ResultParser."""

    def test_parser_initialization(self):
        """Test parser initialization."""
        case_dir = Path("/tmp/test_results")
        parser = ResultParser(case_dir)

        assert parser.case_dir == case_dir

    def test_field_statistics(self):
        """Test field statistics calculation."""
        parser = ResultParser(Path("/tmp/test"))

        # Test with None data
        stats = parser.get_field_statistics("U", time_step=0)
        assert isinstance(stats, dict)


class TestPyVistaViewer:
    """Test cases for PyVistaViewer."""

    def test_viewer_initialization(self):
        """Test viewer initialization."""
        viewer = PyVistaViewer()

        assert viewer.mesh is None
        assert viewer.plotter is None
        assert isinstance(viewer.field_data, dict)

    def test_viewer_methods_exist(self):
        """Test that all required methods exist."""
        viewer = PyVistaViewer()

        assert hasattr(viewer, 'load_mesh')
        assert hasattr(viewer, 'add_field_data')
        assert hasattr(viewer, 'create_contour_plot')
        assert hasattr(viewer, 'create_vector_plot')
        assert hasattr(viewer, 'create_streamlines')
        assert hasattr(viewer, 'create_slice')


# Integration tests
class TestIntegration:
    """Integration tests for complete workflow."""

    def test_wizard_mesh_integration(self):
        """Test wizard and mesh configurator integration."""
        wizard = SimulationWizard(["Mesh Config"], session_key="test_integration")
        configurator = MeshConfigurator(session_key="test_integration_mesh")

        # Store mesh config in wizard
        mesh_config = configurator.get_mesh_config()
        wizard.set_data("mesh_config", mesh_config)

        # Retrieve and verify
        retrieved_config = wizard.get_data("mesh_config")
        assert retrieved_config == mesh_config

    def test_wizard_bc_integration(self):
        """Test wizard and BC form integration."""
        wizard = SimulationWizard(["BC Setup"], session_key="test_bc_integration")
        bc_form = BCForm(session_key="test_bc_integration_form")

        # Setup BCs
        bc_form.set_patches(["inlet", "outlet"])
        bc_form.update_condition("inlet", {"type": "INLET", "velocity": 10.0})
        bc_form.update_condition("outlet", {"type": "OUTLET", "pressure": 0.0})

        # Store in wizard
        wizard.set_data("boundary_conditions", bc_form.get_conditions())

        # Verify
        bcs = wizard.get_data("boundary_conditions")
        assert "inlet" in bcs
        assert bcs["inlet"]["velocity"] == 10.0


def test_bc_type_enum():
    """Test BCType enumeration."""
    assert BCType.INLET.value == "Inlet"
    assert BCType.OUTLET.value == "Outlet"
    assert BCType.WALL.value == "Wall"
    assert BCType.SYMMETRY.value == "Symmetry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
