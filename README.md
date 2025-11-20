# GenAI-CAD-CFD-Studio

[![CI](https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio/actions/workflows/ci.yml/badge.svg)](https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🚀 **Universal AI-Powered CAD & CFD Platform** | Democratizing 3D Design & Simulation

Transform natural language into parametric 3D models and simulations. Build complex CAD designs and run CFD analyses without deep technical expertise.

---

## 🌟 Features

- **Natural Language Interface**: Describe your design in plain English
- **Parametric 3D Modeling**: Powered by Build123d for precise CAD operations
- **Multi-Format File Import**: Support for 20+ CAD and mesh formats (DXF, STEP, STL, VTK, etc.)
- **CFD Simulation**: Integrated OpenFOAM for computational fluid dynamics
- **AI-Driven Design**: Leveraging Zoo.dev and Adam.new APIs
- **Real-time Collaboration**: Streamlit-based interactive interface
- **Digital Twins**: Create simulation-ready models of physical systems
- **Specialized Templates**: Solar PV systems, test chambers, HVAC, and more

---

## 🏗️ Architecture Overview

```
GenAI-CAD-CFD-Studio/
├── src/                    # Source code
│   ├── ai/                # AI model integrations
│   ├── cad/               # CAD modeling engine
│   ├── cfd/               # CFD simulation engine
│   ├── io/                # Multi-format file import/export
│   ├── ui/                # Streamlit interface
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docs/                  # Documentation
├── examples/              # Example projects
└── .github/               # CI/CD workflows
```

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- (Optional) Docker for containerized deployment
- (Optional) Conda for STEP file support

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio.git
   cd GenAI-CAD-CFD-Studio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **For STEP file support (optional)**
   ```bash
   conda install -c conda-forge pythonocc-core
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

6. **Install pre-commit hooks** (optional, for contributors)
   ```bash
   pre-commit install
   ```

---

## 🚀 Usage

### Launch the Streamlit App

```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

### Example: Create a Solar Panel Array

```python
# In the Streamlit interface, use natural language:
"Create a 10x5 solar panel array with 2-degree tilt facing south"

# The system will:
# 1. Parse your request using AI
# 2. Generate parametric 3D model
# 3. Prepare CFD-ready geometry
# 4. Display interactive visualization
```

---

## 📁 Multi-Format File Import Engine

A comprehensive CAD and mesh file import system supporting 20+ formats with unified geometry output.

### Supported Formats

**CAD Formats:**
- **DXF** (Drawing Exchange Format) - 2D/3D drawings, R12-R2018
- **STEP/STP** (Standard for Exchange of Product Data) - 3D CAD models
- **IGES/IGS** (Initial Graphics Exchange Specification)
- **BREP** (Boundary Representation)

**Mesh Formats:**
- **STL** (Stereolithography) - ASCII and binary
- **OBJ** (Wavefront OBJ)
- **PLY** (Stanford Polygon File Format)
- **OFF** (Object File Format)

**FEA/CFD Mesh Formats:**
- **VTK** family (.vtk, .vtu, .vts, .vtr, .vtp, .pvtu)
- **Gmsh** (.msh)
- **ANSYS** (.ans)
- **Abaqus** (.inp)
- **CGNS** (.cgns)
- **Exodus** (.e, .exo)
- **FLAC3D** (.f3grid)
- **H5M** (.h5m)
- **Nastran** (.bdf, .nas)
- **Tecplot** (.dat)
- **XDMF** (.xdmf, .xmf)
- And more...

### Import Examples

#### Import Any Supported File

```python
from src.io import UniversalImporter

# Create importer
importer = UniversalImporter()

# Import file (format auto-detected)
geometry = importer.import_file('model.step')

# Access unified geometry data
print(f"Vertices: {len(geometry['vertices'])}")
print(f"Volume: {geometry['volume']}")
print(f"Bounds: {geometry['bounds']}")
print(f"Format: {geometry['format']}")
```

#### Use Specific Handlers

```python
from src.io import DXFParser, STEPHandler, STLHandler, MeshConverter

# DXF files
dxf_parser = DXFParser()
dxf_data = dxf_parser.parse('drawing.dxf')
print(f"Lines: {len(dxf_data['lines'])}")
print(f"Circles: {len(dxf_data['circles'])}")

# STEP files
step_handler = STEPHandler()
shape = step_handler.import_step('model.step')
properties = step_handler.get_properties(shape)
print(f"Volume: {properties['volume']}")

# STL files
stl_handler = STLHandler()
mesh = stl_handler.load_mesh('part.stl')
validation = stl_handler.validate_mesh(mesh)
print(f"Watertight: {validation['is_watertight']}")

# Mesh conversion
converter = MeshConverter()
converter.convert('input.vtk', 'output.stl')
```

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_io.py -v
```

### Code Quality

```bash
# Format code with black
black .

# Run linter
flake8 .

# Run all pre-commit hooks
pre-commit run --all-files
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Maintain test coverage above 80%
- Write descriptive commit messages
- Add docstrings to all functions and classes

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Build123d](https://github.com/gumyr/build123d) - Parametric CAD modeling
- [OpenFOAM](https://www.openfoam.com/) - CFD simulation
- [Streamlit](https://streamlit.io/) - Interactive web framework
- [Zoo.dev](https://zoo.dev/) - AI-powered CAD API
- [Adam.new](https://adam.new/) - Generative design platform

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

**Built with ❤️ for democratizing engineering design and simulation**
