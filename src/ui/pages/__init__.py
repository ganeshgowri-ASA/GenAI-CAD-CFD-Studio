"""
UI Pages Module

Contains individual page modules for the GenAI CAD CFD Studio application.
"""

from . import rendering
from . import mesh_generation
from . import cfd_analysis

__all__ = ['rendering', 'mesh_generation', 'cfd_analysis']
