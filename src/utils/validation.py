"""Validation utilities for GenAI CAD/CFD Studio.

This module provides various validators for input data validation including
dimensions, file types, coordinates, and API keys.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import re


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        success: Whether validation passed.
        error: Error message if validation failed, None otherwise.
        value: Validated/normalized value if successful.
    """
    success: bool
    error: Optional[str] = None
    value: Optional[any] = None

    @classmethod
    def ok(cls, value: any = None) -> 'ValidationResult':
        """Create a successful validation result.

        Args:
            value: Optional validated/normalized value.

        Returns:
            ValidationResult with success=True.
        """
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> 'ValidationResult':
        """Create a failed validation result.

        Args:
            error: Error message describing the validation failure.

        Returns:
            ValidationResult with success=False.
        """
        return cls(success=False, error=error)


# Supported units for dimension validation
SUPPORTED_UNITS = {
    'length': ['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi'],
    'area': ['mm2', 'cm2', 'm2', 'km2', 'in2', 'ft2', 'yd2', 'mi2'],
    'volume': ['mm3', 'cm3', 'm3', 'km3', 'in3', 'ft3', 'yd3', 'mi3'],
}

# Common file type mappings
FILE_TYPE_EXTENSIONS = {
    'cad': ['.step', '.stp', '.iges', '.igs', '.stl', '.obj', '.fbx', '.3ds'],
    'mesh': ['.stl', '.obj', '.ply', '.off', '.mesh'],
    'cfd': ['.foam', '.msh', '.cas', '.dat', '.cgns'],
    'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'],
    'document': ['.pdf', '.doc', '.docx', '.txt', '.md'],
    'data': ['.csv', '.json', '.xml', '.yaml', '.yml'],
}


def validate_dimensions(
    length: float,
    width: float,
    height: float,
    unit: str = 'm'
) -> ValidationResult:
    """Validate dimensional values for CAD/CFD modeling.

    Args:
        length: Length dimension.
        width: Width dimension.
        height: Height dimension.
        unit: Unit of measurement. Defaults to 'm' (meters).

    Returns:
        ValidationResult indicating success or failure with error message.

    Example:
        >>> result = validate_dimensions(10.5, 5.2, 3.0, 'cm')
        >>> if result.success:
        ...     print(f"Valid dimensions: {result.value}")
        >>> else:
        ...     print(f"Error: {result.error}")
    """
    # Check if unit is supported
    if unit not in SUPPORTED_UNITS['length']:
        return ValidationResult.fail(
            f"Unsupported unit '{unit}'. Supported units: {', '.join(SUPPORTED_UNITS['length'])}"
        )

    # Validate that dimensions are positive numbers
    try:
        length = float(length)
        width = float(width)
        height = float(height)
    except (ValueError, TypeError):
        return ValidationResult.fail("Dimensions must be numeric values")

    if length <= 0:
        return ValidationResult.fail(f"Length must be positive, got {length}")
    if width <= 0:
        return ValidationResult.fail(f"Width must be positive, got {width}")
    if height <= 0:
        return ValidationResult.fail(f"Height must be positive, got {height}")

    # Check for reasonable maximum values (1000 km in any dimension)
    max_value_meters = {
        'mm': 1_000_000_000,
        'cm': 100_000_000,
        'm': 1_000_000,
        'km': 1_000,
        'in': 39_370_079,
        'ft': 3_280_840,
        'yd': 1_093_613,
        'mi': 621.371,
    }

    max_val = max_value_meters.get(unit, float('inf'))
    if length > max_val or width > max_val or height > max_val:
        return ValidationResult.fail(
            f"Dimensions exceed maximum reasonable value ({max_val} {unit})"
        )

    return ValidationResult.ok({
        'length': length,
        'width': width,
        'height': height,
        'unit': unit
    })


def validate_file_type(
    filename: str,
    allowed_types: List[str]
) -> ValidationResult:
    """Validate file type against allowed types.

    Args:
        filename: Name or path of the file to validate.
        allowed_types: List of allowed file type categories or extensions.
            Can be categories like 'cad', 'mesh', 'image' or specific
            extensions like '.stl', '.png'.

    Returns:
        ValidationResult indicating success or failure.

    Example:
        >>> result = validate_file_type("model.stl", ["cad", "mesh"])
        >>> if result.success:
        ...     print("Valid file type")

        >>> result = validate_file_type("data.csv", [".csv", ".json"])
        >>> if result.success:
        ...     print("Valid data file")
    """
    if not filename:
        return ValidationResult.fail("Filename cannot be empty")

    # Get file extension
    file_path = Path(filename)
    extension = file_path.suffix.lower()

    if not extension:
        return ValidationResult.fail(f"File '{filename}' has no extension")

    # Build list of allowed extensions
    allowed_extensions = set()

    for allowed_type in allowed_types:
        if allowed_type.startswith('.'):
            # Direct extension
            allowed_extensions.add(allowed_type.lower())
        elif allowed_type in FILE_TYPE_EXTENSIONS:
            # File type category
            allowed_extensions.update(FILE_TYPE_EXTENSIONS[allowed_type])
        else:
            return ValidationResult.fail(
                f"Unknown file type category: '{allowed_type}'"
            )

    # Check if file extension is allowed
    if extension not in allowed_extensions:
        return ValidationResult.fail(
            f"File type '{extension}' not allowed. "
            f"Allowed types: {', '.join(sorted(allowed_extensions))}"
        )

    return ValidationResult.ok({
        'filename': filename,
        'extension': extension,
        'allowed': True
    })


def validate_coordinates(
    lat: float,
    lon: float
) -> ValidationResult:
    """Validate geographic coordinates (latitude and longitude).

    Args:
        lat: Latitude value (-90 to 90).
        lon: Longitude value (-180 to 180).

    Returns:
        ValidationResult indicating success or failure.

    Example:
        >>> result = validate_coordinates(40.7128, -74.0060)  # New York
        >>> if result.success:
        ...     print(f"Valid coordinates: {result.value}")
    """
    try:
        lat = float(lat)
        lon = float(lon)
    except (ValueError, TypeError):
        return ValidationResult.fail("Coordinates must be numeric values")

    # Validate latitude range
    if lat < -90 or lat > 90:
        return ValidationResult.fail(
            f"Latitude must be between -90 and 90, got {lat}"
        )

    # Validate longitude range
    if lon < -180 or lon > 180:
        return ValidationResult.fail(
            f"Longitude must be between -180 and 180, got {lon}"
        )

    return ValidationResult.ok({
        'latitude': lat,
        'longitude': lon
    })


# API key validation patterns
API_KEY_PATTERNS = {
    'openai': r'^sk-[A-Za-z0-9]{48}$',
    'anthropic': r'^sk-ant-[A-Za-z0-9\-]{95}$',
    'google': r'^AIza[0-9A-Za-z\-_]{35}$',
    'aws': r'^AKIA[0-9A-Z]{16}$',
    'azure': r'^[A-Za-z0-9]{32,}$',
    'zoo': r'^[A-Za-z0-9\-_]{20,}$',  # Zoo.dev API key pattern
}


def validate_api_key(
    key: str,
    provider: str
) -> ValidationResult:
    """Validate API key format for various providers.

    Args:
        key: API key to validate.
        provider: API provider name (e.g., 'openai', 'anthropic', 'google').

    Returns:
        ValidationResult indicating success or failure.

    Example:
        >>> result = validate_api_key("sk-abc123...", "openai")
        >>> if result.success:
        ...     print("Valid API key")
        >>> else:
        ...     print(f"Invalid: {result.error}")
    """
    if not key:
        return ValidationResult.fail("API key cannot be empty")

    if not isinstance(key, str):
        return ValidationResult.fail("API key must be a string")

    # Normalize provider name
    provider = provider.lower()

    # Check if provider is supported
    if provider not in API_KEY_PATTERNS:
        return ValidationResult.fail(
            f"Unknown API provider: '{provider}'. "
            f"Supported providers: {', '.join(API_KEY_PATTERNS.keys())}"
        )

    # Validate key format using regex
    pattern = API_KEY_PATTERNS[provider]
    if not re.match(pattern, key):
        return ValidationResult.fail(
            f"Invalid API key format for provider '{provider}'. "
            f"Key should match pattern: {pattern}"
        )

    # Additional security checks
    if len(key) < 20:
        return ValidationResult.fail("API key is too short (minimum 20 characters)")

    if len(key) > 200:
        return ValidationResult.fail("API key is too long (maximum 200 characters)")

    return ValidationResult.ok({
        'provider': provider,
        'valid': True,
        'length': len(key)
    })


def validate_email(email: str) -> ValidationResult:
    """Validate email address format.

    Args:
        email: Email address to validate.

    Returns:
        ValidationResult indicating success or failure.

    Example:
        >>> result = validate_email("user@example.com")
        >>> if result.success:
        ...     print("Valid email")
    """
    if not email:
        return ValidationResult.fail("Email cannot be empty")

    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(pattern, email):
        return ValidationResult.fail(f"Invalid email format: '{email}'")

    return ValidationResult.ok({'email': email})


def validate_url(url: str, require_https: bool = False) -> ValidationResult:
    """Validate URL format.

    Args:
        url: URL to validate.
        require_https: Whether to require HTTPS protocol. Defaults to False.

    Returns:
        ValidationResult indicating success or failure.

    Example:
        >>> result = validate_url("https://example.com")
        >>> if result.success:
        ...     print("Valid URL")
    """
    if not url:
        return ValidationResult.fail("URL cannot be empty")

    # URL regex pattern
    pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$'

    if not re.match(pattern, url):
        return ValidationResult.fail(f"Invalid URL format: '{url}'")

    if require_https and not url.startswith('https://'):
        return ValidationResult.fail("HTTPS protocol is required")

    return ValidationResult.ok({'url': url})


def validate_range(
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inclusive: bool = True
) -> ValidationResult:
    """Validate that a value is within a specified range.

    Args:
        value: Value to validate.
        min_value: Minimum allowed value (None for no minimum).
        max_value: Maximum allowed value (None for no maximum).
        inclusive: Whether to include boundaries. Defaults to True.

    Returns:
        ValidationResult indicating success or failure.

    Example:
        >>> result = validate_range(5.5, min_value=0, max_value=10)
        >>> if result.success:
        ...     print("Value in range")
    """
    try:
        value = float(value)
    except (ValueError, TypeError):
        return ValidationResult.fail("Value must be numeric")

    if min_value is not None:
        if inclusive and value < min_value:
            return ValidationResult.fail(
                f"Value {value} is less than minimum {min_value}"
            )
        elif not inclusive and value <= min_value:
            return ValidationResult.fail(
                f"Value {value} must be greater than {min_value}"
            )

    if max_value is not None:
        if inclusive and value > max_value:
            return ValidationResult.fail(
                f"Value {value} is greater than maximum {max_value}"
            )
        elif not inclusive and value >= max_value:
            return ValidationResult.fail(
                f"Value {value} must be less than {max_value}"
            )

    return ValidationResult.ok({'value': value})
