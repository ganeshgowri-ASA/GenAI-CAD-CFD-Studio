# GenAI-CAD-CFD-Studio
🚀 Universal AI-Powered CAD &amp; CFD Platform | Democratizing 3D Design &amp; Simulation | Natural Language → Parametric Models | Build123d + Zoo.dev + Adam.new + OpenFOAM | Solar PV, Test Chambers, Digital Twins &amp; More

## Multi-Format File Import Engine

A comprehensive CAD and mesh file import system supporting 20+ formats with unified geometry output.

### Features

#### Supported Formats

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

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio.git
cd GenAI-CAD-CFD-Studio

# Install dependencies
pip install -r requirements.txt

# For STEP file support (requires conda)
conda install -c conda-forge pythonocc-core

# Install package in development mode
pip install -e .
```

### Quick Start

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

#### Progress Callbacks for Large Files

```python
def progress_callback(message, progress):
    print(f"{message}: {progress*100:.1f}%")

importer.set_progress_callback(progress_callback)
geometry = importer.import_file('large_model.step')
```

### Module Documentation

#### UniversalImporter

Unified interface for importing any supported file format.

**Key Methods:**
- `import_file(filepath)` - Auto-detect and import file
- `detect_format(filepath)` - Detect file format from extension
- `is_format_supported(filepath)` - Check if format is supported
- `get_format_info(filepath)` - Get format metadata

**Returns:**
```python
{
    'vertices': np.ndarray,      # N x 3 vertex coordinates
    'faces': np.ndarray,         # M x 3 face indices (for meshes)
    'edges': list,               # Edge definitions (for 2D/wireframe)
    'bounds': tuple,             # (xmin, xmax, ymin, ymax, zmin, zmax)
    'volume': float,             # Volume (if applicable)
    'surface_area': float,       # Surface area (if applicable)
    'metadata': dict,            # Format-specific metadata
    'format': str                # Detected format type
}
```

#### DXFParser

Parse AutoCAD DXF files (R12-R2018).

**Key Methods:**
- `parse(filepath)` - Parse DXF file
- `get_entity_count()` - Get entity type counts

**Extracts:**
- Lines, arcs, circles, polylines
- Dimensions, layers, blocks
- Drawing units and version

#### STEPHandler

Handle STEP CAD files using pythonocc-core.

**Key Methods:**
- `import_step(filepath)` - Import STEP file
- `export_step(shape, filepath)` - Export to STEP
- `export_to_format(shape, filepath, format)` - Export to other formats
- `get_properties(shape)` - Get geometric properties
- `get_volume(shape)` - Calculate volume
- `get_surface_area(shape)` - Calculate surface area
- `get_bounding_box(shape)` - Get bounding box
- `is_valid(shape)` - Validate geometry

#### STLHandler

Load and manipulate STL mesh files using trimesh.

**Key Methods:**
- `load_mesh(filepath)` - Load STL file
- `save_mesh(filepath, mesh)` - Save mesh
- `is_watertight(mesh)` - Check if watertight
- `validate_mesh(mesh)` - Comprehensive validation
- `repair_mesh(mesh)` - Repair mesh defects
- `calculate_properties(mesh)` - Get all properties
- `simplify_mesh(mesh, target_faces)` - Reduce mesh complexity

#### MeshConverter

Universal mesh format converter using meshio.

**Key Methods:**
- `read(filepath)` - Read any supported mesh format
- `write(filepath, mesh)` - Write to any format
- `convert(input_file, output_file)` - Convert between formats
- `get_mesh_info(mesh)` - Get mesh structure info
- `scale_mesh(factor, mesh)` - Scale mesh
- `translate_mesh(vector, mesh)` - Translate mesh
- `get_supported_formats()` - List all supported formats

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_io.py -v

# Run specific test class
pytest tests/test_io.py::TestSTLHandler -v
```

### Architecture

```
src/io/
├── __init__.py              # Module exports and convenience functions
├── dxf_parser.py            # DXF file parsing (ezdxf)
├── step_handler.py          # STEP file handling (pythonocc-core)
├── stl_handler.py           # STL mesh handling (trimesh)
├── mesh_converter.py        # Universal mesh conversion (meshio)
└── universal_importer.py    # Unified import interface

tests/
└── test_io.py               # Comprehensive unit tests (>80% coverage)
```

### Examples

#### Batch Convert Files

```python
from src.io import MeshConverter
from pathlib import Path

converter = MeshConverter()

# Convert all VTK files to STL
for vtk_file in Path('meshes').glob('*.vtk'):
    stl_file = vtk_file.with_suffix('.stl')
    converter.convert(str(vtk_file), str(stl_file))
    print(f"Converted {vtk_file.name} -> {stl_file.name}")
```

#### Analyze CAD Model

```python
from src.io import STEPHandler

handler = STEPHandler()
shape = handler.import_step('engine_part.step')

props = handler.get_properties(shape)
print(f"Volume: {props['volume']:.2f} mm³")
print(f"Surface Area: {props['surface_area']:.2f} mm²")
print(f"Center of Mass: {props['center_of_mass']}")
print(f"Topology: {props['topology']}")
```

#### Validate and Repair STL Mesh

```python
from src.io import STLHandler

handler = STLHandler()
mesh = handler.load_mesh('broken_mesh.stl')

# Validate
validation = handler.validate_mesh(mesh)
if not validation['is_valid']:
    print("Issues found:", validation['issues'])

    # Repair
    repaired = handler.repair_mesh(
        mesh,
        remove_degenerate=True,
        remove_duplicate=True,
        fill_holes=True
    )

    # Save repaired mesh
    handler.save_mesh('repaired_mesh.stl', repaired)
```

### Dependencies

- **numpy** - Numerical operations
- **ezdxf** (≥1.3.0) - DXF parsing
- **pythonocc-core** (≥7.8.0) - STEP/IGES handling (conda install)
- **trimesh** (≥4.0.0) - STL mesh operations
- **meshio** (≥5.3.0) - Universal mesh I/O

### License

MIT License - see LICENSE file for details.
