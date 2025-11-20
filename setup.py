"""
Setup configuration for GenAI-CAD-CFD-Studio.
"""

from setuptools import setup, find_packages
import os

# Read README
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='genai-cad-cfd-studio',
    version='0.1.0',
    description='Universal AI-Powered CAD & CFD Platform',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='GenAI CAD CFD Team',
    author_email='contact@example.com',
    url='https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
    ],
    keywords='cad cfd solar pv geospatial optimization',
    project_urls={
        'Source': 'https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio',
        'Bug Reports': 'https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio/issues',
    },
)
