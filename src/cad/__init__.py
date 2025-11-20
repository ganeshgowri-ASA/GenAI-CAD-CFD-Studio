"""
CAD Generation Module - Universal AI-Powered CAD Platform.

This module provides a unified interface for multiple CAD generation engines:
- Build123D: Direct parametric CAD modeling
- Zoo.dev: KCL-based text-to-CAD generation
- Adam.new: Conversational AI CAD generation

Example usage:
    from cad import UnifiedCADInterface

    # Initialize with API keys
    interface = UnifiedCADInterface(
        zoo_api_key='your_zoo_key',
        adam_api_key='your_adam_key'
    )

    # Generate with auto-selection
    result = interface.generate("Create a box 10x10x10")

    # Export to STEP
    result.export_step("output.step")

    # Or use specific engines directly
    from cad import Build123DEngine

    engine = Build123DEngine()
    part = engine.generate_from_params({
        'type': 'box',
        'length': 10,
        'width': 10,
        'height': 10
    })
"""

__version__ = "0.1.0"
__author__ = "GenAI-CAD-CFD-Studio"

# Core engines
from .build123d_engine import Build123DEngine
from .zoo_connector import ZooDevConnector
from .adam_connector import AdamNewConnector

# Multi-modal model generator
from .model_generator import CADModelGenerator

# Unified interface
from .agent_interface import (
    CADAgent,
    CADResult,
    UnifiedCADInterface
)

# Validation
from .cad_validator import (
    ValidationIssue,
    ValidationResult,
    validate_geometry,
    suggest_fixes,
    quick_validate,
    validate_with_report
)

__all__ = [
    # Version
    '__version__',

    # Engines
    'Build123DEngine',
    'ZooDevConnector',
    'AdamNewConnector',

    # Multi-modal generator
    'CADModelGenerator',

    # Unified interface
    'CADAgent',
    'CADResult',
    'UnifiedCADInterface',

    # Validation
    'ValidationIssue',
    'ValidationResult',
    'validate_geometry',
    'suggest_fixes',
    'quick_validate',
    'validate_with_report',
]
