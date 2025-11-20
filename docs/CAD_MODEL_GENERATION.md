# CAD Model Generation - Multi-Modal AI-Powered CAD

## Overview

The CAD Model Generation feature provides comprehensive AI-powered CAD model generation from multiple input modalities:

- 📝 **Text Descriptions** - Natural language to CAD
- 🖼️ **Images/Sketches** - Computer vision-based geometry extraction
- 📐 **Technical Drawings** - DXF/DWG parsing
- 🔀 **Hybrid Multi-Modal** - Combine multiple input sources

## Features

### ✨ Core Capabilities

- **Natural Language Processing**: Extracts dimensions, shapes, and features from text using Claude AI
- **Computer Vision**: Detects edges, contours, and shapes from images using OpenCV
- **Drawing Parsing**: Converts DXF files to 3D CAD models
- **Parameter Merging**: Intelligently combines information from multiple sources
- **Multi-Engine Support**:
  - Build123d for direct Python CAD modeling
  - Zoo.dev KCL for parametric text-to-CAD
- **Export Options**: STEP, STL formats
- **Validation**: Automatic parameter validation and conflict resolution

---

## Installation

### Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

### Required Dependencies

```bash
# Core dependencies (already in requirements.txt)
pip install streamlit anthropic build123d opencv-python ezdxf plotly pyvista
```

### API Keys

```bash
# Set environment variables
export ANTHROPIC_API_KEY="your_anthropic_key_here"
export ZOO_API_KEY="your_zoo_key_here"  # Optional, for Zoo.dev KCL
```

---

## Quick Start

### 1. Text-to-CAD Generation

```python
from src.cad.model_generator import CADModelGenerator

# Initialize generator
generator = CADModelGenerator(mock_mode=False)

# Generate from text
result = generator.generate_from_text(
    description="Create a box 100mm x 50mm x 30mm",
    output_format='step'
)

print(f"Generated: {result['files']}")
print(f"Parameters: {result['parameters']}")
```

### 2. Image-to-CAD Generation

```python
# Generate from image/sketch
result = generator.generate_from_image(
    image_path='sketch.png',
    image_type='sketch',
    additional_context='This is a mounting bracket',
    output_format='both'
)

print(f"Detected shapes: {result['detected_geometry']['num_shapes']}")
print(f"Generated files: {result['files']}")
```

### 3. Drawing-to-CAD Generation

```python
# Generate from DXF drawing
result = generator.generate_from_drawing(
    drawing_path='part.dxf',
    drawing_format='dxf',
    output_format='step'
)

print(f"Parsed geometry: {result['parsed_geometry']}")
```

### 4. Hybrid Multi-Modal Generation

```python
# Combine multiple sources
result = generator.generate_from_hybrid(
    text_description="Mounting bracket for motor",
    image_path='reference_photo.png',
    drawing_path='dimensions.dxf',
    specifications={
        'material': 'Aluminum 6061',
        'thickness': '5mm',
        'tolerance': '±0.1mm'
    },
    output_format='both'
)

print(f"Input sources: {result['parameter_sources']}")
print(f"Merged parameters: {result['parameters']}")
```

---

## Streamlit UI

### Launch the UI

```bash
streamlit run streamlit_app.py
```

Navigate to the **AI Design Studio** tab.

### UI Features

- **Text Input Tab**: Describe your design in natural language
- **Image Input Tab**: Upload sketches or photos
- **Drawing Input Tab**: Upload DXF files
- **Hybrid Tab**: Combine multiple input sources
- **Settings Tab**: Configure API keys and preferences

### Example Workflow

1. Go to "Text Description" tab
2. Enter: "Create a cylindrical rod with 25mm diameter and 150mm length"
3. Select output format (STEP/STL/Both)
4. Click "Generate CAD Model"
5. Download generated files

---

## API Reference

### CADModelGenerator

Main class for multi-modal CAD generation.

```python
from src.cad.model_generator import CADModelGenerator

generator = CADModelGenerator(
    anthropic_api_key=None,  # Uses ANTHROPIC_API_KEY env var if None
    zoo_api_key=None,        # Uses ZOO_API_KEY env var if None
    use_zoo_dev=False,       # Use Zoo.dev KCL engine
    mock_mode=False          # Use mock responses for testing
)
```

### Methods

#### `generate_from_text(description, output_format, output_path)`

Generate CAD model from natural language description.

**Parameters:**
- `description` (str): Natural language description
- `output_format` (str): 'step', 'stl', or 'both'
- `output_path` (str, optional): Custom output path

**Returns:**
```python
{
    'parameters': dict,      # Extracted CAD parameters
    'model': Part,          # Build123d Part object (if applicable)
    'files': list,          # List of generated file paths
    'metadata': dict,       # Generation metadata
    'input_type': 'text',
    'input_description': str
}
```

#### `generate_from_image(image_path, image_type, additional_context, output_format, output_path)`

Generate CAD model from image or sketch.

**Parameters:**
- `image_path` (str): Path to image file
- `image_type` (str): 'sketch', 'photo', or 'drawing'
- `additional_context` (str, optional): Additional text description
- `output_format` (str): 'step', 'stl', or 'both'
- `output_path` (str, optional): Custom output path

**Returns:**
```python
{
    'parameters': dict,
    'detected_geometry': dict,  # CV-detected geometry
    'files': list,
    'input_type': 'image',
    'image_path': str,
    'image_type': str
}
```

#### `generate_from_drawing(drawing_path, drawing_format, output_format, output_path)`

Generate CAD model from technical drawing.

**Parameters:**
- `drawing_path` (str): Path to drawing file
- `drawing_format` (str): 'dxf' (dwg, pdf coming soon)
- `output_format` (str): 'step', 'stl', or 'both'
- `output_path` (str, optional): Custom output path

**Returns:**
```python
{
    'parameters': dict,
    'parsed_geometry': dict,  # Parsed drawing geometry
    'files': list,
    'input_type': 'drawing'
}
```

#### `generate_from_hybrid(text_description, image_path, drawing_path, specifications, output_format, output_path)`

Generate CAD model from hybrid multi-modal inputs.

**Parameters:**
- `text_description` (str, optional): Text description
- `image_path` (str, optional): Path to image
- `drawing_path` (str, optional): Path to drawing
- `specifications` (dict, optional): Additional specifications
- `output_format` (str): 'step', 'stl', or 'both'
- `output_path` (str, optional): Custom output path

**Returns:**
```python
{
    'parameters': dict,
    'parameter_sources': list,  # List of input sources used
    'files': list,
    'input_type': 'hybrid'
}
```

---

## Examples

### Example 1: Simple Box

```python
generator = CADModelGenerator()

result = generator.generate_from_text(
    description="Create a box 100mm x 50mm x 30mm",
    output_format='step'
)

# Output: outputs/cad/model_TIMESTAMP.step
```

### Example 2: Cylinder with Hole

```python
result = generator.generate_from_text(
    description="Cylindrical rod 50mm diameter, 200mm length with 10mm hole through center",
    output_format='both'
)

# Outputs:
# - outputs/cad/model_TIMESTAMP.step
# - outputs/cad/model_TIMESTAMP.stl
```

### Example 3: Complex Bracket

```python
description = """
Create a mounting bracket with:
- Base plate: 80mm x 60mm x 5mm thick
- Two mounting holes: 6mm diameter, 10mm from edges
- Vertical support: 50mm high, 5mm thick
"""

result = generator.generate_from_text(
    description=description,
    output_format='step'
)
```

### Example 4: Sketch Analysis

```python
result = generator.generate_from_image(
    image_path='hand_sketch.png',
    image_type='sketch',
    additional_context='Mounting plate for Arduino',
    output_format='step'
)

# System detects rectangles, circles, dimensions
print(result['detected_geometry'])
```

### Example 5: DXF Conversion

```python
result = generator.generate_from_drawing(
    drawing_path='technical_drawing.dxf',
    drawing_format='dxf',
    output_format='step'
)

# Converts 2D DXF to 3D STEP
```

### Example 6: Multi-Modal Combination

```python
result = generator.generate_from_hybrid(
    text_description="Create a motor mounting bracket",
    image_path='reference_sketch.png',
    specifications={
        'material': 'Aluminum 6061-T6',
        'thickness': '5mm',
        'mounting_holes': '8mm M8',
        'finish': 'Anodized'
    },
    output_format='both'
)

# Combines all inputs for comprehensive model
```

---

## Mock Mode (Testing)

Use mock mode to test without API keys:

```python
generator = CADModelGenerator(mock_mode=True)

result = generator.generate_from_text(
    description="Test box",
    output_format='step'
)

# Uses basic parameter extraction without Claude API
# Still generates real CAD files using Build123d
```

---

## Architecture

### System Flow

```
Input (Text/Image/Drawing)
         ↓
    Parameter Extraction
    - Claude NLP (text)
    - OpenCV CV (images)
    - ezdxf Parser (drawings)
         ↓
    Parameter Validation
         ↓
    Parameter Merging (hybrid)
         ↓
    CAD Generation
    - Build123d Engine
    - Zoo.dev KCL Engine
         ↓
    Export (STEP/STL)
         ↓
    Output Files
```

### Module Structure

```
src/cad/
├── model_generator.py      # Main orchestrator
├── build123d_engine.py     # Build123d integration
├── zoo_connector.py        # Zoo.dev KCL integration
└── __init__.py

src/ai/
├── sketch_interpreter.py   # Computer vision
├── dimension_extractor.py  # Text dimension parsing
└── claude_skills.py        # Claude integration

src/io/
├── dxf_parser.py          # DXF file parsing
├── step_handler.py        # STEP export
└── stl_handler.py         # STL export

src/ui/
└── design_studio.py       # Streamlit UI
```

---

## Advanced Usage

### Custom Output Paths

```python
result = generator.generate_from_text(
    description="Custom path example",
    output_format='step',
    output_path='/custom/path/my_model.step'
)
```

### Using Zoo.dev KCL

```python
generator = CADModelGenerator(use_zoo_dev=True)

result = generator.generate_from_text(
    description="Parametric gear with 20 teeth, 50mm diameter",
    output_format='step'
)

print(f"Generated KCL: {result['kcl_code']}")
```

### Parameter Validation

```python
# Parameters are automatically validated
# Invalid parameters will raise ValueError

try:
    result = generator.generate_from_text(
        description="Something vague",
        output_format='step'
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

---

## Troubleshooting

### Issue: "ANTHROPIC_API_KEY not set"

**Solution:**
```bash
export ANTHROPIC_API_KEY="your_key_here"
```

Or use mock mode:
```python
generator = CADModelGenerator(mock_mode=True)
```

### Issue: "Build123d not available"

**Solution:**
```bash
pip install build123d>=0.10.0
```

### Issue: "OpenCV not installed"

**Solution:**
```bash
pip install opencv-python>=4.9.0
```

### Issue: Generated files not found

**Solution:**
Check the outputs directory:
```bash
ls -la outputs/cad/
```

Create directory if needed:
```bash
mkdir -p outputs/cad
```

---

## Testing

### Run Unit Tests

```bash
# Run all tests
pytest tests/test_model_generator.py -v

# Run specific test class
pytest tests/test_model_generator.py::TestTextToCAD -v

# Run with coverage
pytest tests/test_model_generator.py --cov=src.cad.model_generator
```

### Run Demo

```bash
python examples/cad_model_generation_demo.py
```

---

## Roadmap

### Phase 1: Core Infrastructure ✅ (Complete)
- ✅ Multi-modal input processing
- ✅ Build123d integration
- ✅ Zoo.dev KCL integration
- ✅ Text/image/drawing support
- ✅ Streamlit UI
- ✅ Unit tests

### Phase 2: Enhanced Features (Next)
- 🔄 3D preview in UI (plotly/pyvista)
- 🔄 PDF drawing support
- 🔄 DWG file support
- 🔄 Advanced feature detection (fillets, chamfers)
- 🔄 Assembly generation

### Phase 3: Advanced Capabilities (Future)
- 🔜 Parametric constraint solving
- 🔜 Topology optimization
- 🔜 AI-powered design recommendations
- 🔜 Real-time collaboration
- 🔜 Version control integration

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Submit pull request

---

## License

See LICENSE file for details.

---

## Support

For issues, questions, or feature requests:

- GitHub Issues: https://github.com/yourusername/GenAI-CAD-CFD-Studio/issues
- Documentation: See this file and inline code documentation
- Examples: See `examples/cad_model_generation_demo.py`

---

## Credits

Built with:
- [Anthropic Claude](https://anthropic.com) - AI/NLP
- [Build123d](https://github.com/gumyr/build123d) - Python CAD
- [Zoo.dev](https://zoo.dev) - KCL text-to-CAD
- [OpenCV](https://opencv.org) - Computer vision
- [Streamlit](https://streamlit.io) - UI framework

---

**Last Updated:** 2025-11-20

**Version:** 1.0.0
