# GenAI-CAD-CFD-Studio

🚀 Universal AI-Powered CAD & CFD Platform | Democratizing 3D Design & Simulation | Natural Language → Parametric Models | Build123d + Zoo.dev + Adam.new + OpenFOAM | Solar PV, Test Chambers, Digital Twins & More

## Overview

GenAI-CAD-CFD-Studio is a comprehensive platform that combines multiple CAD generation engines with a unified interface, enabling designers and engineers to create 3D models using natural language, parametric code, or direct API calls.

## Features

### CAD Generation Engines

- **Build123D Engine**: Direct parametric CAD modeling with Python
  - Primitives: box, cylinder, sphere, cone
  - Operations: extrude, revolve, loft, sweep
  - Boolean operations: union, subtract, intersect
  - Export: STEP, STL with quality control

- **Zoo.dev Connector**: KCL-based text-to-CAD generation
  - Natural language to KCL code conversion
  - Cloud-based model execution
  - Rate limiting and error handling
  - Mock mode for testing

- **Adam.new Connector**: Conversational AI CAD generation
  - Natural language model generation
  - Iterative refinement with feedback
  - Multi-format export (STEP, STL, OBJ, GLB)
  - Conversation history tracking

- **Unified Interface**: Automatic engine selection
  - Auto-detects best engine for your prompt
  - Consistent API across all engines
  - Integrated validation and export

### Geometry Validation

- Volume validation (positive, non-zero)
- Topology checking (manifold geometry)
- Self-intersection detection
- Quality metrics and suggestions
- Automated fix recommendations

## Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio.git
cd GenAI-CAD-CFD-Studio

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e .[dev]
```

### Dependencies

- `build123d>=0.10.0` - Direct CAD modeling
- `requests>=2.31.0` - API communication
- `python-dotenv>=1.0.0` - Environment management
- `pytest>=7.4.0` - Testing (dev)
- `pytest-cov>=4.1.0` - Coverage (dev)

## Quick Start

### 1. Setup API Keys (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
# ZOO_API_KEY=your_zoo_dev_api_key_here
# ADAM_API_KEY=your_adam_new_api_key_here
```

**Note**: API keys are only required for Zoo.dev and Adam.new connectors. Build123D works locally without any API keys. All connectors support mock mode for testing.

### 2. Basic Usage

```python
from cad import UnifiedCADInterface

# Initialize (mock_mode=True for testing without API keys)
interface = UnifiedCADInterface(mock_mode=True)

# Generate with auto-selection
result = interface.generate("Create a box 10x10x10")

# Export to STEP
result.export_step("output.step")

# Export to STL
result.export_stl("output.stl", resolution='high')
```

### 3. Using Specific Engines

#### Build123D Engine (Local, No API Required)

```python
from cad import Build123DEngine, validate_geometry

engine = Build123DEngine()

# Create primitives
box = engine.generate_from_params({
    'type': 'box',
    'length': 100,
    'width': 50,
    'height': 30
})

cylinder = engine.generate_from_params({
    'type': 'cylinder',
    'radius': 20,
    'height': 100
})

# Boolean operations
result = engine.subtract(box, cylinder)

# Validate geometry
validation = validate_geometry(result)
print(validation.summary())

# Export
engine.export_step(result, 'part.step')
engine.export_stl(result, 'part.stl', resolution='high')
```

#### Zoo.dev Connector (KCL)

```python
from cad import ZooDevConnector
import os

connector = ZooDevConnector(
    api_key=os.getenv('ZOO_API_KEY'),
    mock_mode=True  # Set False to use real API
)

# Generate KCL code
kcl_code = connector.generate_kcl(
    "Create a mounting bracket with M6 screw holes"
)

# Execute KCL
model_url = connector.execute_kcl(kcl_code)

# Download model
connector.download_model(model_url, 'bracket.glb')
```

#### Adam.new Connector (Conversational)

```python
from cad import AdamNewConnector
import os

connector = AdamNewConnector(
    api_key=os.getenv('ADAM_API_KEY'),
    mock_mode=True  # Set False to use real API
)

# Generate from natural language
result = connector.generate_from_nl(
    "Design a solar panel mounting bracket"
)

# Refine the design
refined = connector.refine_model(
    result['model_id'],
    "Make it stronger with reinforcement ribs"
)

# Download in multiple formats
files = connector.download_formats(
    refined['model_id'],
    formats=['step', 'stl', 'obj'],
    output_dir='output'
)
```

### 4. Composite Parts

```python
from cad import Build123DEngine

engine = Build123DEngine()

operations = [
    {
        'type': 'primitive',
        'params': {'type': 'box', 'length': 100, 'width': 100, 'height': 20}
    },
    {
        'type': 'subtract',
        'params': {'type': 'cylinder', 'radius': 10, 'height': 25}
    },
    {
        'type': 'union',
        'params': {'type': 'box', 'length': 100, 'width': 10, 'height': 30}
    }
]

part = engine.create_composite(operations)
engine.export_step(part, 'composite.step')
```

## Testing

### Run All Tests

```bash
# Run tests with coverage
pytest

# Run specific test file
pytest tests/test_cad.py

# Run with verbose output
pytest -v

# Generate HTML coverage report
pytest --cov=src/cad --cov-report=html
```

### Coverage

The test suite achieves >80% code coverage across all modules:
- Build123D engine tests
- Zoo.dev connector tests (mock mode)
- Adam.new connector tests (mock mode)
- Unified interface tests
- Validation tests
- Integration tests

## Architecture

```
src/cad/
├── __init__.py              # Main exports
├── build123d_engine.py      # Direct CAD modeling
├── zoo_connector.py         # KCL-based generation
├── adam_connector.py        # Conversational AI
├── agent_interface.py       # Unified interface
└── cad_validator.py         # Geometry validation

tests/
└── test_cad.py             # Comprehensive tests

examples/
└── basic_usage.py          # Usage examples
```

## API Reference

### UnifiedCADInterface

```python
interface = UnifiedCADInterface(
    zoo_api_key=None,    # Optional
    adam_api_key=None,   # Optional
    mock_mode=False      # True for testing
)

# Auto-select engine
result = interface.generate(prompt, engine='auto')

# Specific engine
result = interface.generate(prompt, engine='build123d')
result = interface.generate(prompt, engine='zoo')
result = interface.generate(prompt, engine='adam')

# Refine (Adam only)
refined = interface.refine(model_id, feedback)
```

### CADResult

```python
result = interface.generate("Create a box")

# Access model
model = result.model

# Access metadata
metadata = result.get_metadata()
engine = result.engine
prompt = result.prompt

# Export
result.export_step('output.step')
result.export_stl('output.stl', resolution='high')
```

### Validation

```python
from cad import validate_geometry, quick_validate

# Full validation
validation = validate_geometry(part)
print(validation.summary())

# Quick check
is_valid = quick_validate(part)

# Get suggested fixes
if not validation.is_valid:
    fixes = suggest_fixes(validation)
    for fix in fixes:
        print(f"- {fix}")
```

## Examples

See `examples/basic_usage.py` for comprehensive examples:

```bash
python examples/basic_usage.py
```

## Contributing

Contributions are welcome! Please ensure:
- Tests pass: `pytest`
- Coverage >80%: `pytest --cov=src/cad`
- Code is formatted: `black src tests`
- Linting passes: `flake8 src tests`

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] CFD integration with OpenFOAM
- [ ] Advanced mesh generation
- [ ] Simulation automation
- [ ] Digital twin creation
- [ ] Solar PV design automation
- [ ] Test chamber modeling
- [ ] Multi-physics simulation

## Support

- Documentation: See this README
- Issues: https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio/issues
- Examples: `examples/basic_usage.py`

## Acknowledgments

- Build123D community
- Zoo.dev team
- Adam.new team
- OpenCASCADE technology
