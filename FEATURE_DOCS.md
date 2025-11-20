# File Import UI with 3D Preview - Feature Documentation

## Overview

This feature implements a comprehensive File Import and Conversion UI for the GenAI-CAD-CFD-Studio platform with full 3D preview capabilities.

## Features Implemented

### 1. Multi-Format File Support

**Supported Formats:**
- **DXF** - AutoCAD Drawing Exchange Format
- **DWG** - AutoCAD Drawing Database
- **STEP/STP** - ISO 10303 STEP format
- **IGES/IGS** - Initial Graphics Exchange Specification
- **STL** - Stereolithography (both ASCII and Binary)
- **OBJ** - Wavefront Object
- **PLY** - Polygon File Format
- **BREP** - Boundary Representation

### 2. File Import Functionality

#### `src/io/universal_importer.py`
- **Universal file parser** with automatic format detection
- **GeometryData class** for structured geometry storage:
  - Vertices (Nx3 numpy array)
  - Faces (Mx3 or Mx4 array for triangles/quads)
  - Normals (for STL files)
  - Layer information (for DXF files)
  - Metadata (file info, notes)

**Key Functions:**
- `parse(file_path, file_type)` - Main parsing function
- Format-specific parsers:
  - `_parse_stl()` - Supports binary and ASCII STL
  - `_parse_obj()` - Wavefront OBJ parser
  - `_parse_ply()` - PLY format parser
  - `_parse_dxf()` - Basic DXF parser (placeholder with example geometry)
  - `_parse_dwg()` - DWG parser (placeholder)
  - `_parse_step()` - STEP parser (placeholder)
  - `_parse_iges()` - IGES parser (placeholder)
  - `_parse_brep()` - BREP parser (placeholder)

**GeometryData Properties:**
- `num_vertices` - Vertex count
- `num_faces` - Face count
- `bounding_box` - Min/max coordinates
- `bounding_box_dimensions` - Width, height, depth
- `volume` - Estimated volume
- `surface_area` - Calculated surface area

### 3. 3D Visualization

#### `src/visualization/preview_basic.py`
Interactive 3D visualization using Plotly with:

**Display Modes:**
- **Solid** - Shaded mesh with lighting
- **Wireframe** - Edge-only display
- **Both** - Combined solid + wireframe

**Interactive Controls:**
- Rotate (left-click + drag)
- Pan (right-click + drag)
- Zoom (scroll wheel)
- Reset view (double-click)

**Camera Presets:**
- Isometric view
- Top view
- Front view
- Side view

**Key Functions:**
- `plot_mesh_3d()` - Create interactive 3D plot
- `plot_geometry_data()` - Plot from GeometryData object
- `create_camera_controls()` - Generate camera presets
- `get_mesh_statistics()` - Calculate mesh statistics

### 4. Custom UI Components

#### `src/ui/components/file_uploader.py`
Styled file uploader with:

**Features:**
- Custom CSS styling with gradient backgrounds
- Drag-and-drop interface
- Progress indicators
- File size formatting
- Geometry metrics display
- Download buttons for export

**Components:**
- `custom_file_uploader()` - Enhanced file uploader
- `display_file_info()` - Show file details
- `display_geometry_metrics()` - Show geometry statistics
- `show_upload_progress()` - Progress bar and status
- `create_download_button()` - Styled download buttons

### 5. Main File Import UI

#### `src/ui/file_import.py`
Complete file import interface with:

**Sections:**
1. **File Upload**
   - Drag-and-drop uploader
   - Format validation
   - Progress tracking

2. **File Information Display**
   - File name, size, format
   - Upload status

3. **Geometry Information**
   - Vertices and faces count
   - Bounding box dimensions
   - Volume and surface area
   - Layer information (for DXF)

4. **3D Preview**
   - Interactive visualization
   - View mode selection
   - Camera controls
   - Axis toggle

5. **Export & Conversion**
   - Export to STL (binary)
   - Export to OBJ
   - Export to PLY
   - Export vertices as CSV
   - Export metrics as JSON
   - Generate analysis report (TXT)

**Key Functions:**
- `render_file_import_tab()` - Main rendering function
- `process_uploaded_file()` - File processing workflow
- `display_geometry_info()` - Show geometry details
- `display_3d_preview()` - Render 3D visualization
- `display_export_section()` - Export options
- Export functions: `export_as_stl()`, `export_as_obj()`, etc.

### 6. Testing

#### `tests/test_ui_file_import.py`
Comprehensive test suite:

**Test Classes:**
- `TestUniversalImporter` - File parsing tests
- `TestGeometryData` - Geometry calculation tests
- `TestVisualization` - 3D rendering tests
- `TestFileUploaderComponents` - UI component tests
- `TestTriangulation` - Face triangulation tests
- `TestIntegration` - End-to-end workflow tests

**Test Coverage:**
- STL parsing (binary and ASCII)
- OBJ parsing
- Error handling
- Bounding box calculations
- Volume and surface area
- Visualization rendering
- File format validation
- Complete import-to-visualization workflow

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

**Core Requirements:**
- `numpy>=1.24.0` - Numerical computations
- `streamlit>=1.28.0` - Web UI framework
- `plotly>=5.18.0` - Interactive 3D visualization
- `matplotlib>=3.8.0` - Additional plotting (optional)
- `scipy>=1.11.0` - Scientific computing

**Optional Libraries:**
For advanced CAD format support:
- `ezdxf>=1.1.0` - Enhanced DXF parsing
- `pythonOCC-core>=7.7.0` - STEP, IGES, BREP support

## Usage

### Running the Application

```bash
streamlit run app.py
```

### Using the File Import Tab

1. **Upload a File**
   - Drag and drop a CAD/CFD file
   - Or click to browse
   - Supported formats: DXF, DWG, STEP, IGES, STL, OBJ, PLY, BREP
   - Max size: 200MB

2. **View File Information**
   - File name, size, and format
   - Automatic format detection

3. **Explore Geometry**
   - Vertices and faces count
   - Bounding box dimensions
   - Volume and surface area calculations
   - Layer information (if available)

4. **Interactive 3D Preview**
   - Choose display mode (solid/wireframe/both)
   - Select camera view preset
   - Interact with mouse:
     - Rotate: Left-click + drag
     - Pan: Right-click + drag
     - Zoom: Scroll wheel

5. **Export to Other Formats**
   - Convert to STL, OBJ, PLY
   - Export vertices as CSV
   - Generate analysis reports
   - Download metrics as JSON

### Programmatic Usage

```python
from src.io import universal_importer
from src.visualization import preview_basic

# Parse a file
geometry = universal_importer.parse("model.stl")

# Get geometry info
print(f"Vertices: {geometry.num_vertices}")
print(f"Faces: {geometry.num_faces}")
print(f"Volume: {geometry.volume}")
print(f"Surface Area: {geometry.surface_area}")

# Create 3D visualization
fig = preview_basic.plot_geometry_data(geometry, mode='solid')

# In Streamlit
import streamlit as st
st.plotly_chart(fig)
```

## Architecture

```
GenAI-CAD-CFD-Studio/
├── src/
│   ├── io/
│   │   ├── __init__.py
│   │   └── universal_importer.py      # File parsing
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── preview_basic.py           # 3D visualization
│   └── ui/
│       ├── __init__.py
│       ├── file_import.py             # Main UI
│       └── components/
│           ├── __init__.py
│           └── file_uploader.py       # Custom components
├── tests/
│   ├── __init__.py
│   └── test_ui_file_import.py         # Test suite
├── app.py                              # Main application
├── requirements.txt                    # Dependencies
└── FEATURE_DOCS.md                     # This file
```

## File Format Support Details

### Fully Supported (with parsing)
- **STL** - Complete binary and ASCII support
- **OBJ** - Complete support with face parsing
- **PLY** - ASCII format support

### Placeholder Support (basic geometry)
- **DXF** - Generates sample geometry, layer detection
- **DWG** - Generates sample geometry
- **STEP** - Generates sample geometry
- **IGES** - Generates sample geometry
- **BREP** - Generates sample geometry

**Note:** For full DXF/DWG/STEP/IGES/BREP support, install optional libraries:
- `ezdxf` for DXF
- `pythonOCC-core` for STEP, IGES, BREP

## Performance Considerations

- **File Size Limit:** 200MB (configurable)
- **Vertex Limit:** No hard limit, but 1M+ vertices may slow down visualization
- **Face Limit:** No hard limit, but 500K+ faces may impact performance

**Optimization Tips:**
- Use wireframe mode for large models
- Reduce face count before upload if possible
- Consider decimation for preview purposes

## Error Handling

The system includes comprehensive error handling:

1. **File Not Found**
   - Clear error message
   - Verification of file path

2. **Unsupported Format**
   - Lists supported formats
   - Suggests alternatives

3. **Parsing Errors**
   - Detailed error messages
   - Troubleshooting tips
   - Format-specific guidance

4. **Visualization Errors**
   - Graceful fallbacks
   - Error reporting
   - Empty geometry handling

## Future Enhancements

### Planned Features
1. **Advanced CAD Parsing**
   - Full DXF support with ezdxf
   - STEP/IGES parsing with pythonOCC
   - Assembly support

2. **Mesh Processing**
   - Mesh repair tools
   - Decimation/simplification
   - Normal recalculation
   - Quality metrics

3. **Enhanced Visualization**
   - Multiple lighting modes
   - Material properties
   - Animation support
   - Section views

4. **Export Options**
   - More format support
   - Batch conversion
   - Custom export settings
   - Compression options

5. **Analysis Tools**
   - Mesh quality analysis
   - Interference detection
   - Volume calculations
   - Mass properties

## Testing

### Running Tests

```bash
# Run all tests
python tests/test_ui_file_import.py

# With pytest (if installed)
pytest tests/test_ui_file_import.py -v
```

### Test Coverage
- File parsing: STL, OBJ, format detection
- Geometry calculations: bbox, volume, surface area
- Visualization: all display modes, camera controls
- Integration: complete workflow
- Error handling: invalid files, unsupported formats

## Contributing

When extending this feature:

1. **Adding New File Formats**
   - Create parser function in `universal_importer.py`
   - Add format to `SUPPORTED_FORMATS`
   - Add tests for the new format

2. **Enhancing Visualization**
   - Extend `preview_basic.py`
   - Add new display modes or controls
   - Test with various geometry types

3. **UI Improvements**
   - Update `file_import.py`
   - Maintain responsive design
   - Test with different file sizes

## License

This feature is part of the GenAI-CAD-CFD-Studio project.

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check existing documentation
- Review test cases for examples

---

**Version:** 1.0.0
**Last Updated:** 2025-11-19
**Author:** GenAI-CAD-CFD-Studio Team
