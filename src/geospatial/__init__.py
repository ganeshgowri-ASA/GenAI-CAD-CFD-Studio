"""
Geospatial module for Solar PV plant layout optimization.

This module provides tools for:
- Layout optimization and grid generation
- Shadow analysis and sun position calculations
- Geospatial data processing and transformation
"""

from .layout_optimizer import LayoutOptimizer
from .shadow_analysis import ShadowAnalyzer
from .map_processor import MapProcessor

__all__ = [
    'LayoutOptimizer',
    'ShadowAnalyzer',
    'MapProcessor',
]

__version__ = '0.1.0'
