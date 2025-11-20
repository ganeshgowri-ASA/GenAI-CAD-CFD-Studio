"""
Setup script for GenAI-CAD-CFD-Studio.

Install in development mode:
    pip install -e .

Install with dependencies:
    pip install -e .[dev]
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#') and not line.startswith('trimesh[easy]')
        ]

setup(
    name="genai-cad-cfd-studio",
    version="0.1.0",
    author="GenAI CAD-CFD Studio Team",
    description="Universal AI-Powered CAD & CFD Platform with Multi-Format File I/O",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "pyglet>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Computer Aided Design (CAD)",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="cad cfd generative-ai build123d parametric-design simulation file-import",
    entry_points={
        "console_scripts": [
            "cad-import=io.universal_importer:main",
        ],
    },
)
