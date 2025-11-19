"""Comprehensive unit tests for the utils module.

Tests cover all utility modules with > 80% code coverage including
edge cases and error conditions.
"""

import pytest
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config import ConfigManager
from utils.validation import (
    ValidationResult,
    validate_dimensions,
    validate_file_type,
    validate_coordinates,
    validate_api_key,
    validate_email,
    validate_url,
    validate_range,
)
from utils.geometry_utils import (
    calculate_bounding_box,
    calculate_centroid,
    calculate_area,
    calculate_volume,
    convert_units,
    calculate_distance,
    calculate_normal,
    calculate_sphere_volume,
    calculate_cylinder_volume,
    calculate_cone_volume,
)


# ============================================================================
# ConfigManager Tests
# ============================================================================

class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_init_creates_directory(self):
        """Test that ConfigManager creates config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "configs")
            config = ConfigManager(config_dir)
            assert os.path.exists(config_dir)
            assert config.config_dir == Path(config_dir)

    def test_load_valid_yaml(self):
        """Test loading a valid YAML configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigManager(tmpdir)

            # Create a test config file
            config_data = {"database": {"host": "localhost", "port": 5432}}
            config_file = os.path.join(tmpdir, "test.yaml")
            with open(config_file, 'w') as f:
                yaml.safe_dump(config_data, f)

            config.load("test.yaml")
            assert config.get("database.host") == "localhost"
            assert config.get("database.port") == 5432

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent configuration file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigManager(tmpdir)
            with pytest.raises(FileNotFoundError):
                config.load("nonexistent.yaml")

    def test_get_with_dot_notation(self):
        """Test getting values using dot notation."""
        config = ConfigManager()
        config.config = {
            "database": {
                "host": "localhost",
                "connection": {"timeout": 30}
            }
        }
        assert config.get("database.host") == "localhost"
        assert config.get("database.connection.timeout") == 30
        assert config.get("database.nonexistent", "default") == "default"

    def test_set_with_dot_notation(self):
        """Test setting values using dot notation."""
        config = ConfigManager()
        config.set("database.host", "localhost")
        config.set("database.connection.timeout", 30)

        assert config.get("database.host") == "localhost"
        assert config.get("database.connection.timeout") == 30

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigManager(tmpdir)

            # Set environment variable
            os.environ["TEST_HOST"] = "testhost"
            os.environ["TEST_PORT"] = "9999"

            # Create config with env vars
            config_data = {
                "database": {
                    "host": "${TEST_HOST}",
                    "port": "${TEST_PORT:5432}"
                }
            }
            config_file = os.path.join(tmpdir, "test.yaml")
            with open(config_file, 'w') as f:
                yaml.safe_dump(config_data, f)

            config.load("test.yaml")
            assert config.get("database.host") == "testhost"
            assert config.get("database.port") == "9999"

            # Clean up
            del os.environ["TEST_HOST"]
            del os.environ["TEST_PORT"]

    def test_environment_variable_default(self):
        """Test environment variable default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigManager(tmpdir)

            # Create config with env var that doesn't exist
            config_data = {"setting": "${NONEXISTENT_VAR:default_value}"}
            config_file = os.path.join(tmpdir, "test.yaml")
            with open(config_file, 'w') as f:
                yaml.safe_dump(config_data, f)

            config.load("test.yaml")
            assert config.get("setting") == "default_value"

    def test_validate_schema_success(self):
        """Test successful schema validation."""
        config = ConfigManager()
        config.config = {
            "database": {"host": "localhost", "port": 5432},
            "debug": True
        }

        schema = {
            "database.host": str,
            "database.port": int,
            "debug": bool
        }

        assert config.validate_schema(schema) is True

    def test_validate_schema_missing_key(self):
        """Test schema validation with missing key."""
        config = ConfigManager()
        config.config = {"database": {"host": "localhost"}}

        schema = {"database.port": int}

        with pytest.raises(ValueError, match="Required configuration key missing"):
            config.validate_schema(schema)

    def test_validate_schema_wrong_type(self):
        """Test schema validation with wrong type."""
        config = ConfigManager()
        config.config = {"database": {"port": "5432"}}  # String instead of int

        schema = {"database.port": int}

        with pytest.raises(ValueError, match="Invalid type"):
            config.validate_schema(schema)

    def test_get_all(self):
        """Test getting all configuration data."""
        config = ConfigManager()
        config.config = {"key1": "value1", "key2": "value2"}
        all_config = config.get_all()

        assert all_config == {"key1": "value1", "key2": "value2"}
        # Ensure it's a copy
        all_config["key3"] = "value3"
        assert "key3" not in config.config

    def test_clear(self):
        """Test clearing configuration."""
        config = ConfigManager()
        config.config = {"key1": "value1", "key2": "value2"}
        config.clear()
        assert config.config == {}

    def test_save(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigManager(tmpdir)
            config.config = {"database": {"host": "localhost", "port": 5432}}

            config.save("output.yaml")

            # Verify file was created and contains correct data
            output_file = os.path.join(tmpdir, "output.yaml")
            assert os.path.exists(output_file)

            with open(output_file, 'r') as f:
                loaded = yaml.safe_load(f)
                assert loaded == {"database": {"host": "localhost", "port": 5432}}

    def test_repr(self):
        """Test string representation."""
        config = ConfigManager()
        config.config = {"key1": "value1"}
        repr_str = repr(config)
        assert "ConfigManager" in repr_str
        assert "key1" in repr_str


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Test suite for validation functions."""

    def test_validation_result_ok(self):
        """Test ValidationResult.ok factory method."""
        result = ValidationResult.ok(value="test")
        assert result.success is True
        assert result.error is None
        assert result.value == "test"

    def test_validation_result_fail(self):
        """Test ValidationResult.fail factory method."""
        result = ValidationResult.fail("Error message")
        assert result.success is False
        assert result.error == "Error message"
        assert result.value is None

    def test_validate_dimensions_success(self):
        """Test successful dimension validation."""
        result = validate_dimensions(10.5, 5.2, 3.0, 'm')
        assert result.success is True
        assert result.value['length'] == 10.5
        assert result.value['width'] == 5.2
        assert result.value['height'] == 3.0

    def test_validate_dimensions_invalid_unit(self):
        """Test dimension validation with invalid unit."""
        result = validate_dimensions(10, 5, 3, 'invalid')
        assert result.success is False
        assert "Unsupported unit" in result.error

    def test_validate_dimensions_negative(self):
        """Test dimension validation with negative values."""
        result = validate_dimensions(-10, 5, 3, 'm')
        assert result.success is False
        assert "must be positive" in result.error

    def test_validate_dimensions_zero(self):
        """Test dimension validation with zero values."""
        result = validate_dimensions(0, 5, 3, 'm')
        assert result.success is False
        assert "must be positive" in result.error

    def test_validate_dimensions_non_numeric(self):
        """Test dimension validation with non-numeric values."""
        result = validate_dimensions("abc", 5, 3, 'm')
        assert result.success is False
        assert "must be numeric" in result.error

    def test_validate_dimensions_too_large(self):
        """Test dimension validation with excessively large values."""
        result = validate_dimensions(10000000, 5, 3, 'm')
        assert result.success is False
        assert "exceed maximum" in result.error

    def test_validate_file_type_success_extension(self):
        """Test file type validation with direct extension."""
        result = validate_file_type("model.stl", [".stl", ".obj"])
        assert result.success is True
        assert result.value['extension'] == ".stl"

    def test_validate_file_type_success_category(self):
        """Test file type validation with file category."""
        result = validate_file_type("model.stl", ["cad", "mesh"])
        assert result.success is True

    def test_validate_file_type_invalid(self):
        """Test file type validation with invalid type."""
        result = validate_file_type("document.pdf", ["cad"])
        assert result.success is False
        assert "not allowed" in result.error

    def test_validate_file_type_no_extension(self):
        """Test file type validation with no extension."""
        result = validate_file_type("filename", [".stl"])
        assert result.success is False
        assert "no extension" in result.error

    def test_validate_file_type_empty_filename(self):
        """Test file type validation with empty filename."""
        result = validate_file_type("", [".stl"])
        assert result.success is False
        assert "cannot be empty" in result.error

    def test_validate_file_type_unknown_category(self):
        """Test file type validation with unknown category."""
        result = validate_file_type("file.txt", ["unknown_category"])
        assert result.success is False
        assert "Unknown file type category" in result.error

    def test_validate_coordinates_success(self):
        """Test successful coordinate validation."""
        result = validate_coordinates(40.7128, -74.0060)  # New York
        assert result.success is True
        assert result.value['latitude'] == 40.7128
        assert result.value['longitude'] == -74.0060

    def test_validate_coordinates_invalid_latitude(self):
        """Test coordinate validation with invalid latitude."""
        result = validate_coordinates(100, -74)
        assert result.success is False
        assert "Latitude must be between -90 and 90" in result.error

    def test_validate_coordinates_invalid_longitude(self):
        """Test coordinate validation with invalid longitude."""
        result = validate_coordinates(40, 200)
        assert result.success is False
        assert "Longitude must be between -180 and 180" in result.error

    def test_validate_coordinates_non_numeric(self):
        """Test coordinate validation with non-numeric values."""
        result = validate_coordinates("abc", "def")
        assert result.success is False
        assert "must be numeric" in result.error

    def test_validate_api_key_openai_success(self):
        """Test OpenAI API key validation."""
        # Valid format (48 chars after sk-)
        valid_key = "sk-" + "A" * 48
        result = validate_api_key(valid_key, "openai")
        assert result.success is True

    def test_validate_api_key_invalid_format(self):
        """Test API key validation with invalid format."""
        result = validate_api_key("invalid-key", "openai")
        assert result.success is False
        assert "Invalid API key format" in result.error

    def test_validate_api_key_empty(self):
        """Test API key validation with empty key."""
        result = validate_api_key("", "openai")
        assert result.success is False
        assert "cannot be empty" in result.error

    def test_validate_api_key_unknown_provider(self):
        """Test API key validation with unknown provider."""
        result = validate_api_key("sk-test123", "unknown_provider")
        assert result.success is False
        assert "Unknown API provider" in result.error

    def test_validate_api_key_too_short(self):
        """Test API key validation with too short key."""
        result = validate_api_key("short", "zoo")
        assert result.success is False
        # Zoo pattern requires 20+ chars, so this fails pattern match
        assert ("Invalid API key format" in result.error or "too short" in result.error)

    def test_validate_email_success(self):
        """Test successful email validation."""
        result = validate_email("user@example.com")
        assert result.success is True
        assert result.value['email'] == "user@example.com"

    def test_validate_email_invalid(self):
        """Test email validation with invalid format."""
        result = validate_email("invalid-email")
        assert result.success is False
        assert "Invalid email format" in result.error

    def test_validate_email_empty(self):
        """Test email validation with empty string."""
        result = validate_email("")
        assert result.success is False
        assert "cannot be empty" in result.error

    def test_validate_url_success(self):
        """Test successful URL validation."""
        result = validate_url("https://example.com")
        assert result.success is True
        assert result.value['url'] == "https://example.com"

    def test_validate_url_http(self):
        """Test URL validation with HTTP."""
        result = validate_url("http://example.com")
        assert result.success is True

    def test_validate_url_require_https(self):
        """Test URL validation requiring HTTPS."""
        result = validate_url("http://example.com", require_https=True)
        assert result.success is False
        assert "HTTPS protocol is required" in result.error

    def test_validate_url_invalid(self):
        """Test URL validation with invalid format."""
        result = validate_url("not-a-url")
        assert result.success is False
        assert "Invalid URL format" in result.error

    def test_validate_url_empty(self):
        """Test URL validation with empty string."""
        result = validate_url("")
        assert result.success is False
        assert "cannot be empty" in result.error

    def test_validate_range_success(self):
        """Test successful range validation."""
        result = validate_range(5.5, min_value=0, max_value=10)
        assert result.success is True
        assert result.value['value'] == 5.5

    def test_validate_range_below_min(self):
        """Test range validation below minimum."""
        result = validate_range(-1, min_value=0, max_value=10)
        assert result.success is False
        assert "less than minimum" in result.error

    def test_validate_range_above_max(self):
        """Test range validation above maximum."""
        result = validate_range(15, min_value=0, max_value=10)
        assert result.success is False
        assert "greater than maximum" in result.error

    def test_validate_range_non_inclusive(self):
        """Test range validation with non-inclusive bounds."""
        result = validate_range(0, min_value=0, max_value=10, inclusive=False)
        assert result.success is False
        assert "must be greater than" in result.error

    def test_validate_range_no_bounds(self):
        """Test range validation with no bounds."""
        result = validate_range(999999)
        assert result.success is True

    def test_validate_range_non_numeric(self):
        """Test range validation with non-numeric value."""
        result = validate_range("abc", min_value=0, max_value=10)
        assert result.success is False
        assert "must be numeric" in result.error


# ============================================================================
# Geometry Utils Tests
# ============================================================================

class TestGeometryUtils:
    """Test suite for geometry utility functions."""

    def test_calculate_bounding_box_simple(self):
        """Test bounding box calculation with simple points."""
        points = [(0, 0, 0), (1, 2, 3), (4, 1, 2)]
        bbox = calculate_bounding_box(points)

        assert bbox['min'] == (0, 0, 0)
        assert bbox['max'] == (4, 2, 3)
        assert bbox['center'] == (2, 1, 1.5)
        assert bbox['dimensions'] == (4, 2, 3)

    def test_calculate_bounding_box_negative(self):
        """Test bounding box with negative coordinates."""
        points = [(-1, -2, -3), (1, 2, 3)]
        bbox = calculate_bounding_box(points)

        assert bbox['min'] == (-1, -2, -3)
        assert bbox['max'] == (1, 2, 3)

    def test_calculate_bounding_box_empty(self):
        """Test bounding box with empty point list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_bounding_box([])

    def test_calculate_centroid_rectangle(self):
        """Test centroid calculation for a rectangle."""
        polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]
        centroid = calculate_centroid(polygon)

        assert abs(centroid[0] - 2.0) < 0.01
        assert abs(centroid[1] - 1.5) < 0.01

    def test_calculate_centroid_triangle(self):
        """Test centroid calculation for a triangle."""
        polygon = [(0, 0), (3, 0), (0, 4)]
        centroid = calculate_centroid(polygon)

        assert abs(centroid[0] - 1.0) < 0.01
        assert abs(centroid[1] - 1.33) < 0.01

    def test_calculate_centroid_too_few_points(self):
        """Test centroid with too few points."""
        with pytest.raises(ValueError, match="at least 3 points"):
            calculate_centroid([(0, 0), (1, 1)])

    def test_calculate_centroid_degenerate(self):
        """Test centroid with degenerate polygon (all points on a line)."""
        polygon = [(0, 0), (1, 0), (2, 0)]
        centroid = calculate_centroid(polygon)
        # Should return average of points
        assert centroid is not None

    def test_calculate_area_rectangle(self):
        """Test area calculation for a rectangle."""
        polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]
        area = calculate_area(polygon)

        assert abs(area - 12.0) < 0.01

    def test_calculate_area_triangle(self):
        """Test area calculation for a triangle."""
        polygon = [(0, 0), (4, 0), (0, 3)]
        area = calculate_area(polygon)

        assert abs(area - 6.0) < 0.01

    def test_calculate_area_too_few_points(self):
        """Test area with too few points."""
        with pytest.raises(ValueError, match="at least 3 points"):
            calculate_area([(0, 0), (1, 1)])

    def test_calculate_volume_cube(self):
        """Test volume calculation for a unit cube."""
        vertices = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
        ]
        faces = [
            (0, 1, 2), (0, 2, 3),  # Bottom
            (4, 6, 5), (4, 7, 6),  # Top
            (0, 4, 5), (0, 5, 1),  # Front
            (2, 6, 7), (2, 7, 3),  # Back
            (0, 3, 7), (0, 7, 4),  # Left
            (1, 5, 6), (1, 6, 2),  # Right
        ]
        mesh = {'vertices': vertices, 'faces': faces}
        volume = calculate_volume(mesh)

        assert abs(volume - 1.0) < 0.01

    def test_calculate_volume_invalid_mesh(self):
        """Test volume with invalid mesh structure."""
        with pytest.raises(ValueError, match="must contain"):
            calculate_volume({})

    def test_calculate_volume_empty_mesh(self):
        """Test volume with empty mesh."""
        with pytest.raises(ValueError, match="at least one"):
            calculate_volume({'vertices': [], 'faces': []})

    def test_calculate_volume_non_triangular_face(self):
        """Test volume with non-triangular face."""
        vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        faces = [(0, 1, 2, 3)]  # Quad instead of triangle
        mesh = {'vertices': vertices, 'faces': faces}

        with pytest.raises(ValueError, match="must be triangles"):
            calculate_volume(mesh)

    def test_convert_units_length(self):
        """Test unit conversion for length."""
        result = convert_units(1000, 'mm', 'm')
        assert abs(result - 1.0) < 0.01

        result = convert_units(1, 'ft', 'cm')
        assert abs(result - 30.48) < 0.01

    def test_convert_units_area(self):
        """Test unit conversion for area."""
        result = convert_units(1, 'm2', 'cm2')
        assert abs(result - 10000.0) < 0.01

    def test_convert_units_volume(self):
        """Test unit conversion for volume."""
        result = convert_units(1, 'm3', 'cm3')
        assert abs(result - 1000000.0) < 0.01

    def test_convert_units_invalid_source(self):
        """Test unit conversion with invalid source unit."""
        with pytest.raises(ValueError, match="Unsupported source unit"):
            convert_units(1, 'invalid', 'm')

    def test_convert_units_invalid_target(self):
        """Test unit conversion with invalid target unit."""
        with pytest.raises(ValueError, match="Unsupported target unit"):
            convert_units(1, 'm', 'invalid')

    def test_convert_units_incompatible(self):
        """Test unit conversion between incompatible dimensions."""
        with pytest.raises(ValueError, match="incompatible dimensions"):
            convert_units(1, 'm', 'm2')

    def test_calculate_distance(self):
        """Test distance calculation between two points."""
        p1 = (0, 0, 0)
        p2 = (3, 4, 0)
        distance = calculate_distance(p1, p2)

        assert abs(distance - 5.0) < 0.01

    def test_calculate_distance_3d(self):
        """Test distance calculation in 3D."""
        p1 = (1, 2, 3)
        p2 = (4, 6, 8)
        distance = calculate_distance(p1, p2)

        expected = ((4-1)**2 + (6-2)**2 + (8-3)**2) ** 0.5
        assert abs(distance - expected) < 0.01

    def test_calculate_normal(self):
        """Test normal vector calculation for a triangle."""
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (0, 1, 0)
        normal = calculate_normal(p1, p2, p3)

        # Normal should point in +Z direction
        assert abs(normal[0] - 0.0) < 0.01
        assert abs(normal[1] - 0.0) < 0.01
        assert abs(normal[2] - 1.0) < 0.01

    def test_calculate_normal_degenerate(self):
        """Test normal calculation for degenerate triangle."""
        # All points on a line
        p1 = (0, 0, 0)
        p2 = (1, 0, 0)
        p3 = (2, 0, 0)
        normal = calculate_normal(p1, p2, p3)

        assert normal == (0.0, 0.0, 0.0)

    def test_calculate_sphere_volume(self):
        """Test sphere volume calculation."""
        volume = calculate_sphere_volume(1.0)
        expected = (4.0 / 3.0) * 3.14159265359 * 1.0 ** 3
        assert abs(volume - expected) < 0.01

    def test_calculate_sphere_volume_negative_radius(self):
        """Test sphere volume with negative radius."""
        with pytest.raises(ValueError, match="must be non-negative"):
            calculate_sphere_volume(-1.0)

    def test_calculate_cylinder_volume(self):
        """Test cylinder volume calculation."""
        volume = calculate_cylinder_volume(2.0, 5.0)
        expected = 3.14159265359 * 2.0 ** 2 * 5.0
        assert abs(volume - expected) < 0.01

    def test_calculate_cylinder_volume_negative(self):
        """Test cylinder volume with negative values."""
        with pytest.raises(ValueError, match="must be non-negative"):
            calculate_cylinder_volume(-2.0, 5.0)

    def test_calculate_cone_volume(self):
        """Test cone volume calculation."""
        volume = calculate_cone_volume(3.0, 4.0)
        expected = (1.0 / 3.0) * 3.14159265359 * 3.0 ** 2 * 4.0
        assert abs(volume - expected) < 0.01

    def test_calculate_cone_volume_negative(self):
        """Test cone volume with negative values."""
        with pytest.raises(ValueError, match="must be non-negative"):
            calculate_cone_volume(3.0, -4.0)


# ============================================================================
# Logger Tests (Basic, since logger requires live streamlit)
# ============================================================================

class TestLogger:
    """Test suite for logger module."""

    def test_logger_imports(self):
        """Test that logger modules can be imported."""
        from utils.logger import (
            setup_logger,
            get_logger,
            set_trace_context,
            clear_trace_context,
            get_trace_context,
            LoggerContext,
        )
        assert setup_logger is not None
        assert get_logger is not None

    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        from utils.logger import setup_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger(
                "test_logger",
                log_level="DEBUG",
                log_dir=tmpdir,
                console_output=False,
                file_output=True
            )
            assert logger is not None
            assert logger.name == "test_logger"

    def test_trace_context(self):
        """Test trace context management."""
        from utils.logger import set_trace_context, get_trace_context, clear_trace_context

        set_trace_context(request_id="123", user_id="user456")
        context = get_trace_context()

        assert context["request_id"] == "123"
        assert context["user_id"] == "user456"

        clear_trace_context()
        context = get_trace_context()
        assert context == {}

    def test_logger_context_manager(self):
        """Test logger context manager."""
        from utils.logger import LoggerContext, get_trace_context

        with LoggerContext(request_id="abc"):
            context = get_trace_context()
            assert context["request_id"] == "abc"

        # Context should be cleared after exiting
        context = get_trace_context()
        assert "request_id" not in context


# ============================================================================
# Session Manager Tests (Mock streamlit)
# ============================================================================

class TestSessionManager:
    """Test suite for StreamlitSessionManager."""

    @pytest.fixture
    def mock_streamlit(self):
        """Create a mock streamlit module."""
        mock_st = MagicMock()
        mock_st.session_state = MagicMock()

        # Mock the session_state to behave like a dict
        state_dict = {}

        def getattr_side_effect(key, default=None):
            return state_dict.get(key, default)

        def setattr_side_effect(key, value):
            state_dict[key] = value

        def contains_side_effect(key):
            return key in state_dict

        def delattr_side_effect(key):
            if key in state_dict:
                del state_dict[key]

        def keys_side_effect():
            return state_dict.keys()

        mock_st.session_state.__getattr__ = getattr_side_effect
        mock_st.session_state.__setattr__ = setattr_side_effect
        mock_st.session_state.__contains__ = contains_side_effect
        mock_st.session_state.__delattr__ = delattr_side_effect
        mock_st.session_state.keys.return_value = state_dict.keys()

        return mock_st, state_dict

    def test_session_manager_imports(self):
        """Test that session manager can be imported."""
        # This will fail if streamlit is not installed, but we can at least check import
        try:
            from utils.session_manager import StreamlitSessionManager
            assert StreamlitSessionManager is not None
        except Exception:
            # Skip if streamlit is not available
            pytest.skip("Streamlit not available")


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the utils module."""

    def test_import_all(self):
        """Test importing all utilities from main module."""
        import utils
        assert hasattr(utils, 'ConfigManager')
        assert hasattr(utils, 'setup_logger')
        assert hasattr(utils, 'ValidationResult')
        assert hasattr(utils, 'calculate_bounding_box')
        assert hasattr(utils, '__version__')

    def test_version_info(self):
        """Test version information."""
        from utils import __version__, __author__, __license__
        assert isinstance(__version__, str)
        assert isinstance(__author__, str)
        assert isinstance(__license__, str)

    def test_get_version(self):
        """Test get_version function."""
        from utils import get_version
        version = get_version()
        assert isinstance(version, str)

    def test_get_module_info(self):
        """Test get_module_info function."""
        from utils import get_module_info
        info = get_module_info()
        assert 'version' in info
        assert 'author' in info
        assert 'license' in info


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=utils", "--cov-report=term-missing"])
