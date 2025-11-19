"""
Setup script for GenAI CAD-CFD Studio.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name='genai-cad-cfd-studio',
    version='1.0.0',
    author='GenAI CAD-CFD Studio Team',
    description='Comprehensive CAD file processing and CFD analysis platform',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Manufacturing',
        'Topic :: Scientific/Engineering :: Computer Aided Design (CAD)',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'viz': [
            'matplotlib>=3.4.0',
            'pyglet>=1.5.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'cad-import=src.io.universal_importer:main',
        ],
    },
)
