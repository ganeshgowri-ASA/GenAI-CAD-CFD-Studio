"""
CAD Model Generation Demo

Demonstrates comprehensive usage of the CADModelGenerator class
for multi-modal CAD model generation.

This script shows:
1. Text-to-CAD generation
2. Image-to-CAD generation
3. Drawing-to-CAD generation
4. Hybrid multi-modal generation
5. Export options

Requirements:
- Set ANTHROPIC_API_KEY environment variable for full functionality
- Optional: Set ZOO_API_KEY for Zoo.dev KCL generation
- Run in mock mode without API keys for testing
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from cad.model_generator import CADModelGenerator
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_text_to_cad(mock_mode=True):
    """Demonstrate text-to-CAD generation."""

    print("\n" + "="*80)
    print("DEMO 1: Text-to-CAD Generation")
    print("="*80)

    # Initialize generator
    generator = CADModelGenerator(mock_mode=mock_mode)

    # Example 1: Simple box
    print("\nExample 1: Simple box")
    description = "Create a rectangular box 100mm x 50mm x 30mm"

    result = generator.generate_from_text(
        description=description,
        output_format='step'
    )

    print(f"✓ Generated: {result['files']}")
    print(f"  Parameters: {result['parameters']}")
    print(f"  Engine: {result.get('engine')}")

    # Example 2: Cylinder with hole
    print("\nExample 2: Cylinder with hole")
    description = "Cylindrical rod with 25mm diameter and 100mm length with a 5mm diameter hole through the center"

    result = generator.generate_from_text(
        description=description,
        output_format='both'
    )

    print(f"✓ Generated: {result['files']}")
    print(f"  Parameters: {result['parameters']}")

    # Example 3: Complex shape
    print("\nExample 3: Complex shape")
    description = """
    Create a mounting bracket with the following specifications:
    - Base plate: 80mm x 60mm x 5mm thick
    - Two mounting holes: 6mm diameter, positioned 10mm from edges
    - Vertical support: 50mm high, 5mm thick
    """

    result = generator.generate_from_text(
        description=description,
        output_format='step'
    )

    print(f"✓ Generated: {result['files']}")
    print(f"  Parameters: {result['parameters']}")


def demo_image_to_cad(mock_mode=True):
    """Demonstrate image-to-CAD generation."""

    print("\n" + "="*80)
    print("DEMO 2: Image-to-CAD Generation")
    print("="*80)

    # Note: This requires an actual image file
    # For demo purposes, we'll show how it would be used

    print("\nExample: Sketch-to-CAD")
    print("To use image-to-CAD:")
    print("""
    generator = CADModelGenerator(mock_mode=False)

    result = generator.generate_from_image(
        image_path='path/to/sketch.png',
        image_type='sketch',
        additional_context='This is a mounting bracket',
        output_format='step'
    )

    # The system will:
    # 1. Use OpenCV to detect edges and contours
    # 2. Identify shapes (rectangles, circles, etc.)
    # 3. Use Claude vision to interpret the sketch
    # 4. Generate CAD model from extracted geometry
    """)


def demo_drawing_to_cad(mock_mode=True):
    """Demonstrate technical drawing-to-CAD generation."""

    print("\n" + "="*80)
    print("DEMO 3: Drawing-to-CAD Generation")
    print("="*80)

    print("\nExample: DXF-to-CAD")
    print("To convert a DXF drawing to 3D CAD:")
    print("""
    generator = CADModelGenerator()

    result = generator.generate_from_drawing(
        drawing_path='path/to/drawing.dxf',
        drawing_format='dxf',
        output_format='step'
    )

    # The system will:
    # 1. Parse DXF file to extract geometry
    # 2. Convert 2D entities to 3D parameters
    # 3. Generate 3D CAD model
    # 4. Export to STEP format
    """)


def demo_hybrid_generation(mock_mode=True):
    """Demonstrate hybrid multi-modal CAD generation."""

    print("\n" + "="*80)
    print("DEMO 4: Hybrid Multi-Modal Generation")
    print("="*80)

    generator = CADModelGenerator(mock_mode=mock_mode)

    print("\nExample: Combining text + specifications")

    # Text description
    description = "Create a mounting bracket for a motor"

    # Additional specifications
    specs = {
        'material': 'Aluminum 6061',
        'wall_thickness': '5mm',
        'tolerance': '±0.1mm',
        'finish': 'Anodized'
    }

    result = generator.generate_from_hybrid(
        text_description=description,
        specifications=specs,
        output_format='step'
    )

    print(f"✓ Generated: {result['files']}")
    print(f"  Input sources: {result.get('parameter_sources')}")
    print(f"  Merged parameters: {result['parameters']}")

    print("\nIn a real scenario, you could combine:")
    print("  - Text: 'Mounting bracket for motor'")
    print("  - Image: photo of existing bracket")
    print("  - Drawing: DXF with precise dimensions")
    print("  - Specs: material, tolerance, finish")


def demo_zoo_dev_generation(mock_mode=True):
    """Demonstrate Zoo.dev KCL-based generation."""

    print("\n" + "="*80)
    print("DEMO 5: Zoo.dev KCL Generation")
    print("="*80)

    if mock_mode:
        print("\nRunning in mock mode - Zoo.dev will return simulated responses")

    # Initialize with Zoo.dev enabled
    generator = CADModelGenerator(
        use_zoo_dev=True,
        mock_mode=mock_mode
    )

    print("\nExample: Generate with KCL")
    description = "Create a parametric gear with 20 teeth, 50mm diameter"

    result = generator.generate_from_text(
        description=description,
        output_format='step'
    )

    print(f"✓ Generated: {result['files']}")
    print(f"  Engine: {result.get('engine')}")

    if 'kcl_code' in result:
        print(f"\nGenerated KCL code:")
        print("-" * 40)
        print(result['kcl_code'][:500] + "...")  # Show first 500 chars


def demo_export_options():
    """Demonstrate various export options."""

    print("\n" + "="*80)
    print("DEMO 6: Export Options")
    print("="*80)

    print("\nSupported export formats:")
    print("  - STEP (.step, .stp) - Industry standard for CAD exchange")
    print("  - STL (.stl) - For 3D printing and mesh visualization")
    print("  - Both - Export to both formats simultaneously")

    print("\nExport quality settings (STL):")
    print("  - Low: Fast export, larger triangles")
    print("  - Medium: Balanced quality/speed")
    print("  - High: Best quality, smaller triangles")

    print("\nExample usage:")
    print("""
    # Export to STEP only
    result = generator.generate_from_text(
        description="box 10x10x10 cm",
        output_format='step'
    )

    # Export to STL only
    result = generator.generate_from_text(
        description="cylinder 5cm diameter, 10cm height",
        output_format='stl'
    )

    # Export to both formats
    result = generator.generate_from_text(
        description="sphere 50mm radius",
        output_format='both'
    )
    """)


def demo_error_handling():
    """Demonstrate error handling and validation."""

    print("\n" + "="*80)
    print("DEMO 7: Error Handling & Validation")
    print("="*80)

    generator = CADModelGenerator(mock_mode=True)

    # Example 1: Invalid description
    print("\nExample 1: Handling invalid input")
    try:
        result = generator.generate_from_text(
            description="",  # Empty description
            output_format='step'
        )
    except ValueError as e:
        print(f"✓ Caught error: {e}")

    # Example 2: Validation
    print("\nExample 2: Parameter validation")
    description = "Create something vague"  # Vague description

    try:
        result = generator.generate_from_text(
            description=description,
            output_format='step'
        )
        print(f"✓ Generated with defaults: {result['parameters']}")
    except Exception as e:
        print(f"✓ Validation prevented invalid generation: {e}")


def main():
    """Run all demos."""

    print("\n" + "="*80)
    print("CAD MODEL GENERATION - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nThis demo showcases multi-modal CAD generation capabilities.")
    print("Running in MOCK MODE (no API keys required)")
    print("\nTo run with real AI:")
    print("  export ANTHROPIC_API_KEY=your_key_here")
    print("  export ZOO_API_KEY=your_key_here (optional)")
    print("="*80)

    # Run demos
    demo_text_to_cad(mock_mode=True)
    demo_image_to_cad(mock_mode=True)
    demo_drawing_to_cad(mock_mode=True)
    demo_hybrid_generation(mock_mode=True)
    demo_zoo_dev_generation(mock_mode=True)
    demo_export_options()
    demo_error_handling()

    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Set API keys for real AI-powered generation")
    print("  2. Try with your own images, sketches, and descriptions")
    print("  3. Explore the Streamlit UI: streamlit run streamlit_app.py")
    print("  4. Check generated files in: outputs/cad/")
    print("="*80)


if __name__ == "__main__":
    main()
