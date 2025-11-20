# CAD Model Generation Feature

## Overview

The CAD Model Generation feature provides comprehensive, multi-modal CAD model creation from various input types:

- **Natural Language Text**: Describe models in plain English
- **Images & Sketches**: Upload hand-drawn sketches or reference photos
- **Technical Drawings**: Import DXF, DWG, or PDF technical drawings
- **Hybrid Inputs**: Combine multiple sources for enhanced accuracy

## Architecture

### Core Components

#### 1. `CADModelGenerator` (src/cad/model_generator.py)

Main orchestrator that coordinates all CAD generation activities.

**Key Features:**
- Multi-modal input processing (text, image, drawing, hybrid)
- Intelligent engine selection (Build123d, Zoo.dev)
- Parameter extraction and validation
- Multi-format export (STEP, STL, KCL, DXF)

**Integrations:**
- Claude API for advanced NLP and vision analysis
- Zoo.dev API for KCL-based parametric CAD
- Build123d for Python-native CAD modeling
- OpenCV for image/sketch analysis

#### 2. AI Processing Pipeline

**SketchInterpreter** (`src/ai/sketch_interpreter.py`)
- Edge detection using Canny algorithm
- Contour extraction and shape recognition
- Geometry conversion for CAD generation

**DimensionExtractor** (`src/ai/dimension_extractor.py`)
- Parse dimensions from text (multiple formats)
- Unit conversion (mm, cm, m, in, ft)
- Dimension validation and suggestions

**ClaudeSkills** (`src/ai/claude_skills.py`)
- Object type detection
- Parameter extraction from natural language
- Design suggestions and material recommendations

#### 3. CAD Engines

**Build123d Engine** (`src/cad/build123d_engine.py`)
- Parametric CAD modeling in Python
- Primitive shapes (box, cylinder, sphere, cone)
- Boolean operations (union, subtract, intersect)
- Export to STEP and STL

**Zoo.dev Connector** (`src/cad/zoo_connector.py`)
- KCL (KittyCAD Language) code generation
- Text-to-CAD via Zoo.dev API
- Rate limiting and retry logic
- Model download and caching

## Usage

### Basic Text-to-CAD

```python
from src.cad.model_generator import CADModelGenerator

# Initialize generator
generator = CADModelGenerator(
    claude_api_key="your_anthropic_key",  # Optional
    zoo_api_key="your_zoo_key",           # Optional
    default_engine='build123d',
    default_unit='mm'
)

# Generate from text
result = generator.generate_from_text(
    description="Create a box 100mm x 50mm x 30mm",
    export_formats=['step', 'stl']
)

if result.success:
    print(f"Model generated: {result.export_paths}")
else:
    print(f"Failed: {result.message}")
```

### Image/Sketch-to-CAD

```python
# Generate from sketch
result = generator.generate_from_image(
    image_path="sketch.png",
    image_type="sketch",
    description="Make it 100mm tall",
    export_formats=['step']
)
```

### Technical Drawing Import

```python
# Import DXF drawing
result = generator.generate_from_drawing(
    drawing_path="technical_drawing.dxf",
    drawing_format="dxf",
    export_formats=['step']
)
```

### Hybrid Multi-Modal

```python
# Combine multiple inputs
result = generator.generate_from_hybrid(
    inputs={
        'text': 'Create a mounting bracket',
        'image': 'sketch.png',
        'specs': {
            'material': 'Aluminum',
            'thickness': 5
        }
    },
    export_formats=['step', 'stl']
)
```

## UI Integration

The Design Studio UI (`src/ui/design_studio.py`) provides an interactive interface with:

### Features

1. **Multiple Input Tabs**
   - Text Input: Natural language descriptions
   - Image/Sketch: Upload and analyze images
   - Technical Drawing: DXF/DWG/PDF import
   - Hybrid Input: Combine multiple sources
   - History: Track previous generations

2. **Configuration Sidebar**
   - API key management
   - Engine selection
   - Unit preferences
   - Export format selection

3. **Real-time Results**
   - Parameter extraction display
   - Export file downloads
   - 3D preview (coming soon)
   - Generation history

### Running the UI

```bash
streamlit run app.py
```

Navigate to the "AI Design Studio" tab.

## API Reference

### CADModelGenerator Class

#### Constructor

```python
CADModelGenerator(
    claude_api_key: Optional[str] = None,
    zoo_api_key: Optional[str] = None,
    default_engine: str = "build123d",
    default_unit: str = "mm",
    output_dir: Optional[str] = None
)
```

#### Methods

**generate_from_text(description, engine, export_formats, **kwargs)**
- Generate CAD model from text description
- Returns: `CADGenerationResult`

**generate_from_image(image_path, image_type, description, engine, export_formats, **kwargs)**
- Generate CAD model from image/sketch
- Returns: `CADGenerationResult`

**generate_from_drawing(drawing_path, drawing_format, description, engine, export_formats, **kwargs)**
- Generate CAD model from technical drawing
- Returns: `CADGenerationResult`

**generate_from_hybrid(inputs, engine, export_formats, **kwargs)**
- Generate CAD model from multiple input sources
- Returns: `CADGenerationResult`

### CADGenerationResult Class

**Attributes:**
- `success` (bool): Generation success status
- `message` (str): Status message
- `parameters` (dict): Extracted CAD parameters
- `part` (object): Build123d Part object (if using Build123d)
- `kcl_code` (str): Generated KCL code (if using Zoo.dev)
- `model_url` (str): Model URL (if using Zoo.dev)
- `export_paths` (dict): Exported file paths by format
- `metadata` (dict): Additional metadata
- `timestamp` (str): Generation timestamp

**Methods:**
- `to_dict()`: Convert result to dictionary
- `__repr__()`: String representation

## Input Formats

### Text Descriptions

Supported formats:
- Dimensional: "10cm x 5cm x 3cm"
- Labeled: "length: 100mm, width: 50mm, height: 30mm"
- Natural language: "Create a cylindrical pipe 50mm diameter and 200mm long"

### Image Types

- **Sketch**: Hand-drawn sketches with shapes and dimensions
- **Photo**: Reference photos of objects
- **Technical**: Technical drawing images

### Drawing Formats

- **DXF**: AutoCAD Drawing Exchange Format
- **DWG**: AutoCAD Drawing (via DXF conversion)
- **PDF**: PDF technical drawings (experimental)

## Examples

### Example 1: Simple Box

```python
result = generator.generate_from_text(
    "Create a box 100mm x 50mm x 30mm",
    export_formats=['step', 'stl']
)
```

### Example 2: Cylinder with Material

```python
result = generator.generate_from_text(
    "Make a cylindrical pipe with 50mm diameter and 200mm length",
    export_formats=['step'],
    material='Aluminum 6061'
)
```

### Example 3: Complex Geometry with Zoo.dev

```python
result = generator.generate_from_text(
    "Create a mounting bracket with 4 holes and rounded corners",
    engine='zoo',
    export_formats=['kcl', 'step']
)
```

### Example 4: Sketch Analysis

```python
# Upload a hand-drawn sketch
result = generator.generate_from_image(
    image_path="hand_sketch.png",
    image_type="sketch",
    description="Add 5mm thickness",
    export_formats=['step']
)
```

### Example 5: DXF Import

```python
result = generator.generate_from_drawing(
    drawing_path="part_drawing.dxf",
    description="Extrude to 10mm height",
    export_formats=['step']
)
```

## Testing

### Run Unit Tests

```bash
# Using pytest
pytest tests/test_model_generator.py -v

# Using unittest
python tests/test_model_generator.py
```

### Run Demo

```bash
python examples/cad_generation_demo.py
```

## Configuration

### Environment Variables

```bash
# Anthropic Claude API (optional but recommended)
export ANTHROPIC_API_KEY="your_api_key"

# Zoo.dev API (optional)
export ZOO_API_KEY="your_zoo_api_key"
```

### API Keys

1. **Anthropic Claude**: Required for advanced NLP and vision analysis
   - Get key: https://console.anthropic.com/

2. **Zoo.dev**: Optional for KCL-based generation
   - Get key: https://zoo.dev/

## Dependencies

### Required
- `build123d>=0.10.0` - Python CAD library
- `opencv-python>=4.9.0` - Image processing
- `pillow>=10.0.0` - Image handling
- `ezdxf>=1.3.0` - DXF parsing
- `PyPDF2>=3.0.0` - PDF parsing
- `requests>=2.31.0` - HTTP requests

### Optional
- `anthropic>=0.18.0` - Claude API integration
- `numpy>=1.24.0` - Numerical computations
- `plotly>=5.18.0` - 3D visualization
- `pyvista>=0.43.0` - Advanced 3D visualization

## Limitations

1. **Build123d Availability**: Some environments may not support Build123d
2. **API Rate Limits**: Claude and Zoo.dev APIs have rate limits
3. **PDF Parsing**: PDF technical drawing parsing is experimental
4. **Image Quality**: Sketch analysis quality depends on image clarity
5. **Complex Geometry**: Very complex geometries may require manual refinement

## Troubleshooting

### Issue: "Engine not available"
**Solution**: Ensure Build123d is installed or configure Zoo.dev API key

### Issue: "Parameter extraction failed"
**Solution**: Provide more explicit dimensions in the description

### Issue: "Image analysis failed"
**Solution**: Ensure image is clear and high-contrast

### Issue: "DXF parsing error"
**Solution**: Verify DXF file is valid and not corrupted

## Future Enhancements

- [ ] Advanced 3D preview in UI
- [ ] CAD assembly support
- [ ] Direct integration with CAD software
- [ ] Machine learning-based shape recognition
- [ ] Collaborative editing features
- [ ] Cloud-based model storage
- [ ] Real-time collaboration

## Contributing

To contribute to this feature:

1. Follow the existing code structure
2. Add unit tests for new functionality
3. Update documentation
4. Ensure backward compatibility

## License

Part of GenAI CAD-CFD Studio project.

## Support

For issues or questions:
1. Check this documentation
2. Review example scripts
3. Run verification: `python verify_installation.py`
4. Check logs for detailed error messages
