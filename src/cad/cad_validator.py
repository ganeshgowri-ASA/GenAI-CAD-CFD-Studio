"""
CAD Validator - Geometry validation and quality checking.

This module provides functions to validate CAD geometry, check for common
issues, and suggest fixes for problems like invalid volumes, self-intersections,
and non-manifold geometry.
"""

from __future__ import annotations
from typing import List, Optional, Any, Dict, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

try:
    from build123d import Part, Shape
    BUILD123D_AVAILABLE = True
except ImportError:
    BUILD123D_AVAILABLE = False
    Part = Any
    Shape = Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the geometry."""
    severity: str  # 'error', 'warning', 'info'
    issue_type: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of geometry validation.

    Contains validation status, list of issues found, and suggested fixes.
    """
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == 'error' for issue in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == 'warning' for issue in self.issues)

    def get_errors(self) -> List[ValidationIssue]:
        """Get all error-level issues."""
        return [issue for issue in self.issues if issue.severity == 'error']

    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning-level issues."""
        return [issue for issue in self.issues if issue.severity == 'warning']

    def summary(self) -> str:
        """Get a summary of validation results."""
        status = "VALID" if self.is_valid else "INVALID"
        error_count = len(self.get_errors())
        warning_count = len(self.get_warnings())

        summary = f"Validation Status: {status}\n"
        summary += f"Errors: {error_count}\n"
        summary += f"Warnings: {warning_count}\n"

        if self.metrics:
            summary += "\nMetrics:\n"
            for key, value in self.metrics.items():
                summary += f"  {key}: {value}\n"

        if self.issues:
            summary += "\nIssues:\n"
            for i, issue in enumerate(self.issues, 1):
                summary += f"{i}. [{issue.severity.upper()}] {issue.issue_type}: {issue.message}\n"
                if issue.suggestion:
                    summary += f"   Suggestion: {issue.suggestion}\n"

        return summary


def validate_geometry(part: Part) -> ValidationResult:
    """
    Validate CAD geometry for common issues.

    Checks:
    - Volume > 0
    - No self-intersections
    - Manifold geometry
    - Valid topology

    Args:
        part: The Part object to validate

    Returns:
        ValidationResult: Comprehensive validation result

    Raises:
        ImportError: If build123d is not available
    """
    if not BUILD123D_AVAILABLE:
        raise ImportError(
            "build123d is required for validation. "
            "Install with: pip install build123d>=0.10.0"
        )

    result = ValidationResult(is_valid=True)
    issues = []

    logger.info("Starting geometry validation...")

    # Check 1: Valid part object
    if part is None:
        issues.append(ValidationIssue(
            severity='error',
            issue_type='null_part',
            message='Part object is None',
            suggestion='Ensure the part was created successfully'
        ))
        result.is_valid = False
        result.issues = issues
        return result

    try:
        # Check 2: Volume > 0
        try:
            volume = part.volume
            result.metrics['volume'] = volume

            if volume <= 0:
                issues.append(ValidationIssue(
                    severity='error',
                    issue_type='invalid_volume',
                    message=f'Part has non-positive volume: {volume}',
                    suggestion='Check that the part is properly formed and not inverted'
                ))
                result.is_valid = False
            elif volume < 1e-6:
                issues.append(ValidationIssue(
                    severity='warning',
                    issue_type='small_volume',
                    message=f'Part has very small volume: {volume}',
                    suggestion='Consider if this is the intended scale'
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                severity='error',
                issue_type='volume_calculation_failed',
                message=f'Failed to calculate volume: {str(e)}',
                suggestion='Part geometry may be invalid or corrupted'
            ))
            result.is_valid = False

        # Check 3: Bounding box
        try:
            bbox = part.bounding_box()
            if bbox:
                result.metrics['bounding_box'] = {
                    'min': (bbox.min.X, bbox.min.Y, bbox.min.Z),
                    'max': (bbox.max.X, bbox.max.Y, bbox.max.Z),
                    'size': (
                        bbox.max.X - bbox.min.X,
                        bbox.max.Y - bbox.min.Y,
                        bbox.max.Z - bbox.min.Z
                    )
                }
        except Exception as e:
            logger.warning(f"Could not calculate bounding box: {e}")

        # Check 4: Valid topology (manifold check)
        try:
            # In build123d, we can check if the shape is valid
            if hasattr(part, 'wrapped') and hasattr(part.wrapped, 'isValid'):
                is_valid_topology = part.wrapped.isValid()
                result.metrics['valid_topology'] = is_valid_topology

                if not is_valid_topology:
                    issues.append(ValidationIssue(
                        severity='error',
                        issue_type='invalid_topology',
                        message='Part has invalid topology',
                        suggestion='Review boolean operations and ensure proper geometry construction'
                    ))
                    result.is_valid = False
        except Exception as e:
            logger.warning(f"Could not check topology: {e}")

        # Check 5: Face count (sanity check)
        try:
            if hasattr(part, 'faces'):
                face_count = len(part.faces())
                result.metrics['face_count'] = face_count

                if face_count == 0:
                    issues.append(ValidationIssue(
                        severity='error',
                        issue_type='no_faces',
                        message='Part has no faces',
                        suggestion='Ensure the part geometry was created successfully'
                    ))
                    result.is_valid = False
                elif face_count > 10000:
                    issues.append(ValidationIssue(
                        severity='warning',
                        issue_type='high_face_count',
                        message=f'Part has very high face count: {face_count}',
                        suggestion='Consider simplifying geometry for better performance'
                    ))
        except Exception as e:
            logger.warning(f"Could not count faces: {e}")

        # Check 6: Edge count
        try:
            if hasattr(part, 'edges'):
                edge_count = len(part.edges())
                result.metrics['edge_count'] = edge_count
        except Exception as e:
            logger.warning(f"Could not count edges: {e}")

        # Check 7: Vertex count
        try:
            if hasattr(part, 'vertices'):
                vertex_count = len(part.vertices())
                result.metrics['vertex_count'] = vertex_count
        except Exception as e:
            logger.warning(f"Could not count vertices: {e}")

        # Check 8: Check for very small edges (can cause issues)
        try:
            if hasattr(part, 'edges'):
                edges = part.edges()
                small_edges = []
                for i, edge in enumerate(edges):
                    try:
                        length = edge.length
                        if length < 1e-6:
                            small_edges.append((i, length))
                    except:
                        pass

                if small_edges:
                    issues.append(ValidationIssue(
                        severity='warning',
                        issue_type='small_edges',
                        message=f'Found {len(small_edges)} very small edges',
                        suggestion='Small edges can cause meshing issues; consider cleanup'
                    ))
                    result.metrics['small_edge_count'] = len(small_edges)
        except Exception as e:
            logger.warning(f"Could not check edge lengths: {e}")

    except Exception as e:
        issues.append(ValidationIssue(
            severity='error',
            issue_type='validation_error',
            message=f'Unexpected error during validation: {str(e)}',
            suggestion='Check that the part object is valid'
        ))
        result.is_valid = False

    result.issues = issues

    # If we have any errors, mark as invalid
    if result.has_errors():
        result.is_valid = False

    logger.info(f"Validation complete: {'VALID' if result.is_valid else 'INVALID'}")
    return result


def suggest_fixes(validation_result: ValidationResult) -> List[str]:
    """
    Generate specific fix suggestions based on validation results.

    Args:
        validation_result: The validation result to analyze

    Returns:
        List of suggested fixes
    """
    fixes = []

    for issue in validation_result.issues:
        if issue.suggestion and issue.suggestion not in fixes:
            fixes.append(issue.suggestion)

        # Add specific fix suggestions based on issue type
        if issue.issue_type == 'invalid_volume':
            if 'Check face orientations' not in fixes:
                fixes.append('Check face orientations - normals may be inverted')

        elif issue.issue_type == 'invalid_topology':
            if 'Try rebuilding with simpler operations' not in fixes:
                fixes.append('Try rebuilding with simpler operations')
            if 'Check for self-intersecting geometry' not in fixes:
                fixes.append('Check for self-intersecting geometry')

        elif issue.issue_type == 'small_edges':
            if 'Use a cleanup/healing function' not in fixes:
                fixes.append('Use a cleanup/healing function to merge small edges')

        elif issue.issue_type == 'no_faces':
            if 'Verify extrusion/sweep operations completed' not in fixes:
                fixes.append('Verify extrusion/sweep operations completed successfully')

    return fixes


def quick_validate(part: Part) -> bool:
    """
    Quick validation check - returns True if part is valid.

    Args:
        part: The Part object to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        result = validate_geometry(part)
        return result.is_valid
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def validate_with_report(part: Part) -> str:
    """
    Validate geometry and return formatted report.

    Args:
        part: The Part object to validate

    Returns:
        str: Formatted validation report
    """
    result = validate_geometry(part)
    report = result.summary()

    # Add fix suggestions
    fixes = suggest_fixes(result)
    if fixes:
        report += "\nSuggested Fixes:\n"
        for i, fix in enumerate(fixes, 1):
            report += f"{i}. {fix}\n"

    return report
