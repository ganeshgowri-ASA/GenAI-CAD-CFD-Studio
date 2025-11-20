"""GenAI CAD/CFD Studio - Utility Module.

This module provides comprehensive utilities for the GenAI CAD/CFD Studio
platform including configuration management, logging, session management,
validation, and geometry calculations.
"""

# Version information
__version__ = "0.1.0"
__author__ = "GenAI CAD/CFD Studio Team"
__license__ = "MIT"

# Configuration management
from .config import ConfigManager

# Logging utilities
from .logger import (
    setup_logger,
    get_logger,
    set_trace_context,
    clear_trace_context,
    get_trace_context,
    LoggerContext,
    ColoredFormatter,
    ContextLogger,
    default_logger,
    # Convenience functions
    debug,
    info,
    warning,
    error,
    critical,
)

# Session management
from .session_manager import (
    StreamlitSessionManager,
    session,
)

# Validation utilities
from .validation import (
    ValidationResult,
    validate_dimensions,
    validate_file_type,
    validate_coordinates,
    validate_api_key,
    validate_email,
    validate_url,
    validate_range,
    # Constants
    SUPPORTED_UNITS,
    FILE_TYPE_EXTENSIONS,
    API_KEY_PATTERNS,
)

# Geometry utilities
from .geometry_utils import (
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
    # Type aliases
    Point3D,
    Point2D,
    BoundingBox,
    # Constants
    UNIT_TO_METERS,
)

# Define what should be exported when using "from utils import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Config
    "ConfigManager",
    # Logger
    "setup_logger",
    "get_logger",
    "set_trace_context",
    "clear_trace_context",
    "get_trace_context",
    "LoggerContext",
    "ColoredFormatter",
    "ContextLogger",
    "default_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    # Session
    "StreamlitSessionManager",
    "session",
    # Validation
    "ValidationResult",
    "validate_dimensions",
    "validate_file_type",
    "validate_coordinates",
    "validate_api_key",
    "validate_email",
    "validate_url",
    "validate_range",
    "SUPPORTED_UNITS",
    "FILE_TYPE_EXTENSIONS",
    "API_KEY_PATTERNS",
    # Geometry
    "calculate_bounding_box",
    "calculate_centroid",
    "calculate_area",
    "calculate_volume",
    "convert_units",
    "calculate_distance",
    "calculate_normal",
    "calculate_sphere_volume",
    "calculate_cylinder_volume",
    "calculate_cone_volume",
    "Point3D",
    "Point2D",
    "BoundingBox",
    "UNIT_TO_METERS",
]


def get_version() -> str:
    """Get the version of the utils module.

    Returns:
        Version string.
    """
    return __version__


def get_module_info() -> dict:
    """Get information about the utils module.

    Returns:
        Dictionary containing version, author, and license information.
    """
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
    }
