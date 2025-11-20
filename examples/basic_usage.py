"""
Basic usage examples for CAD generation engines.

This script demonstrates how to use the different CAD generation engines
in the GenAI-CAD-CFD-Studio platform.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cad import (
    Build123DEngine,
    ZooDevConnector,
    AdamNewConnector,
    UnifiedCADInterface,
    validate_geometry
)
import os
from dotenv import load_dotenv


def example_build123d():
    """Example: Direct parametric CAD with Build123D."""
    print("\n=== Build123D Engine Example ===\n")

    try:
        engine = Build123DEngine()

        # Create a simple box
        box = engine.generate_from_params({
            'type': 'box',
            'length': 100,
            'width': 50,
            'height': 30
        })
        print(f"Created box with volume: {box.volume:.2f}")

        # Create a cylinder
        cylinder = engine.generate_from_params({
            'type': 'cylinder',
            'radius': 20,
            'height': 100
        })
        print(f"Created cylinder with volume: {cylinder.volume:.2f}")

        # Boolean subtraction: box with hole
        result = engine.subtract(box, cylinder)
        print(f"After subtraction, volume: {result.volume:.2f}")

        # Export to STEP
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)

        engine.export_step(result, output_dir / 'box_with_hole.step')
        print(f"Exported to {output_dir / 'box_with_hole.step'}")

        # Validate geometry
        validation = validate_geometry(result)
        print(f"Validation: {validation.is_valid}")

    except ImportError:
        print("Build123D not installed. Install with: pip install build123d")


def example_zoo_connector():
    """Example: KCL-based generation with Zoo.dev."""
    print("\n=== Zoo.dev Connector Example ===\n")

    load_dotenv()
    api_key = os.getenv('ZOO_API_KEY')

    # Use mock mode if no API key
    mock_mode = not bool(api_key)
    if mock_mode:
        print("No API key found, using mock mode")

    connector = ZooDevConnector(api_key=api_key, mock_mode=mock_mode)

    # Generate KCL code
    prompt = "Create a mounting bracket with holes for M6 screws"
    kcl_code = connector.generate_kcl(prompt)
    print(f"Generated KCL code:\n{kcl_code[:200]}...\n")

    # Execute KCL
    model_url = connector.execute_kcl(kcl_code)
    print(f"Model URL: {model_url}")

    # Download model
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    connector.download_model(model_url, output_dir / 'zoo_model.glb')
    print(f"Downloaded to {output_dir / 'zoo_model.glb'}")


def example_adam_connector():
    """Example: Conversational CAD with Adam.new."""
    print("\n=== Adam.new Connector Example ===\n")

    load_dotenv()
    api_key = os.getenv('ADAM_API_KEY')

    # Use mock mode if no API key
    mock_mode = not bool(api_key)
    if mock_mode:
        print("No API key found, using mock mode")

    connector = AdamNewConnector(api_key=api_key, mock_mode=mock_mode)

    # Generate from natural language
    prompt = "Design a solar panel mounting bracket for rooftop installation"
    result = connector.generate_from_nl(prompt)
    print(f"Generated model ID: {result['model_id']}")
    print(f"Status: {result['status']}")
    print(f"Available formats: {result['formats_available']}")

    # Refine the model
    feedback = "Make the bracket stronger with reinforcement ribs"
    refined = connector.refine_model(result['model_id'], feedback)
    print(f"\nRefined model ID: {refined['model_id']}")

    # Download in multiple formats
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    formats = ['step', 'stl', 'obj']
    downloaded = connector.download_formats(
        refined['model_id'],
        formats,
        output_dir
    )
    print(f"\nDownloaded formats: {list(downloaded.keys())}")


def example_unified_interface():
    """Example: Unified interface with auto-selection."""
    print("\n=== Unified CAD Interface Example ===\n")

    load_dotenv()
    interface = UnifiedCADInterface(
        zoo_api_key=os.getenv('ZOO_API_KEY'),
        adam_api_key=os.getenv('ADAM_API_KEY'),
        mock_mode=os.getenv('MOCK_MODE', 'true').lower() == 'true'
    )

    # Auto-select engine based on prompt
    prompts = [
        "box length 50 width 50 height 50",  # -> build123d
        "Create a parametric bracket design",  # -> adam
        "Generate KCL for a mounting plate"  # -> zoo
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        engine = interface.auto_select_engine(prompt)
        print(f"Selected engine: {engine}")

        # Generate with auto-selection
        result = interface.generate(prompt, engine='auto')
        print(f"Generated using: {result.engine}")
        print(f"Metadata: {list(result.metadata.keys())}")


def example_composite_part():
    """Example: Creating complex composite parts."""
    print("\n=== Composite Part Example ===\n")

    try:
        engine = Build123DEngine()

        # Define operations
        operations = [
            # Start with a base box
            {
                'type': 'primitive',
                'params': {
                    'type': 'box',
                    'length': 100,
                    'width': 100,
                    'height': 20
                }
            },
            # Subtract a cylinder (mounting hole)
            {
                'type': 'subtract',
                'params': {
                    'type': 'cylinder',
                    'radius': 10,
                    'height': 25
                }
            },
            # Union with a reinforcement rib
            {
                'type': 'union',
                'params': {
                    'type': 'box',
                    'length': 100,
                    'width': 10,
                    'height': 30
                }
            }
        ]

        part = engine.create_composite(operations)
        print(f"Created composite part with volume: {part.volume:.2f}")

        # Validate
        validation = validate_geometry(part)
        print(f"Validation: {validation.is_valid}")
        print(f"Metrics: {validation.metrics}")

        # Export
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)

        engine.export_step(part, output_dir / 'composite.step')
        engine.export_stl(part, output_dir / 'composite.stl', resolution='high')
        print(f"Exported to {output_dir}")

    except ImportError:
        print("Build123D not installed. Install with: pip install build123d")


def main():
    """Run all examples."""
    print("=" * 60)
    print("GenAI-CAD-CFD-Studio - CAD Generation Examples")
    print("=" * 60)

    # Run examples
    example_build123d()
    example_unified_interface()
    example_composite_part()

    # API examples (will use mock mode if no keys)
    example_zoo_connector()
    example_adam_connector()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
