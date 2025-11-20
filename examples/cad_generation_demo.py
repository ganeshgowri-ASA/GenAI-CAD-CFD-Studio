"""
CAD Model Generation Demo

Demonstrates the comprehensive CAD generation capabilities:
1. Text-to-CAD generation
2. Image/sketch-to-CAD generation
3. Technical drawing import
4. Hybrid multi-modal generation
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cad.model_generator import CADModelGenerator
from src.ai.dimension_extractor import DimensionExtractor
from src.ai.claude_skills import ClaudeSkills


def print_separator(title=""):
    """Print a visual separator."""
    width = 80
    if title:
        print("\n" + "=" * width)
        print(f"  {title}")
        print("=" * width + "\n")
    else:
        print("\n" + "-" * width + "\n")


def demo_text_generation():
    """Demo: Generate CAD models from text descriptions."""
    print_separator("DEMO 1: Text-to-CAD Generation")

    # Initialize generator
    # Note: Set environment variables ANTHROPIC_API_KEY and ZOO_API_KEY for full functionality
    generator = CADModelGenerator(
        claude_api_key=os.getenv('ANTHROPIC_API_KEY'),
        zoo_api_key=os.getenv('ZOO_API_KEY'),
        default_engine='build123d',
        default_unit='mm',
        output_dir='./cad_output/demo'
    )

    print("CAD Model Generator initialized!\n")

    # Example 1: Simple box
    print("Example 1: Simple Box")
    print("Description: 'Create a box 100mm x 50mm x 30mm'")

    result = generator.generate_from_text(
        description="Create a box 100mm x 50mm x 30mm",
        export_formats=['step', 'stl']
    )

    print(f"Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    print(f"Message: {result.message}")
    if result.parameters:
        print(f"Parameters: {result.parameters}")
    if result.export_paths:
        print(f"Exported files: {result.export_paths}")

    print_separator()

    # Example 2: Cylinder with material
    print("Example 2: Cylinder with Specifications")
    print("Description: 'Make a cylindrical pipe with 50mm diameter and 200mm length'")

    result = generator.generate_from_text(
        description="Make a cylindrical pipe with 50mm diameter and 200mm length",
        export_formats=['step'],
        material='Aluminum 6061'
    )

    print(f"Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    print(f"Message: {result.message}")
    if result.parameters:
        print(f"Parameters: {result.parameters}")

    print_separator()

    # Example 3: Using Zoo.dev for complex geometry
    if generator.zoo_connector:
        print("Example 3: Complex Geometry with Zoo.dev")
        print("Description: 'Create a mounting bracket with holes and rounded corners'")

        result = generator.generate_from_text(
            description="Create a mounting bracket with holes and rounded corners",
            engine='zoo',
            export_formats=['kcl']
        )

        print(f"Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
        print(f"Message: {result.message}")
        if result.kcl_code:
            print(f"KCL Code generated: {len(result.kcl_code)} characters")
    else:
        print("Example 3: Skipped (Zoo.dev API key not configured)")


def demo_parameter_extraction():
    """Demo: Parameter extraction from text."""
    print_separator("DEMO 2: Parameter Extraction")

    extractor = DimensionExtractor()
    skills = ClaudeSkills()

    # Test various description formats
    descriptions = [
        "10cm x 5cm x 3cm",
        "length: 100mm, width: 50mm, height: 30mm",
        "Create a cylinder with radius 25mm and height 100mm",
        "Make a sphere 50mm in diameter",
        "Box 6 inches by 4 inches by 2 inches"
    ]

    for desc in descriptions:
        print(f"Description: '{desc}'")

        # Extract using DimensionExtractor
        dims = extractor.parse_dimensions(desc)
        print(f"  DimensionExtractor: {dims}")

        # Extract using ClaudeSkills
        params = skills.extract_dimensions(desc)
        print(f"  ClaudeSkills: {params}")

        # Validate
        is_valid = extractor.validate_dimensions(dims)
        print(f"  Valid: {is_valid}")

        if not is_valid:
            suggestions = extractor.suggest_corrections(dims)
            print(f"  Suggestions: {suggestions}")

        print()


def demo_hybrid_generation():
    """Demo: Hybrid multi-modal generation."""
    print_separator("DEMO 3: Hybrid Multi-Modal Generation")

    generator = CADModelGenerator(
        claude_api_key=os.getenv('ANTHROPIC_API_KEY'),
        output_dir='./cad_output/demo'
    )

    # Example: Combine text + specifications
    print("Example: Text + Specifications")

    inputs = {
        'text': 'Create a mounting plate',
        'specs': {
            'length': 150,
            'width': 100,
            'thickness': 10,
            'material': 'Steel',
            'holes': 4
        }
    }

    result = generator.generate_from_hybrid(
        inputs=inputs,
        export_formats=['step']
    )

    print(f"Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    print(f"Message: {result.message}")
    if result.parameters:
        print(f"Combined Parameters: {result.parameters}")


def demo_image_analysis():
    """Demo: Image analysis capabilities."""
    print_separator("DEMO 4: Image Analysis (Mock)")

    # Note: This would require actual image files
    print("Image analysis capabilities:")
    print("  ✓ Edge detection using Canny algorithm")
    print("  ✓ Contour extraction and shape recognition")
    print("  ✓ Automatic dimension inference from sketches")
    print("  ✓ Claude Vision API for enhanced interpretation")
    print("\nTo use: upload an image file via the UI or call generator.generate_from_image()")


def demo_workflow_pipeline():
    """Demo: Complete end-to-end workflow."""
    print_separator("DEMO 5: Complete Workflow Pipeline")

    generator = CADModelGenerator(output_dir='./cad_output/demo')

    print("Step 1: Natural language input")
    description = "Create a cylindrical container 80mm diameter, 120mm tall"
    print(f"  Input: '{description}'")

    print("\nStep 2: Parameter extraction")
    params = generator._extract_parameters_from_text(description)
    print(f"  Extracted: {params}")

    print("\nStep 3: Parameter validation")
    validated = generator._validate_parameters(params)
    print(f"  Validated: {validated}")

    print("\nStep 4: Engine selection")
    engine = generator._select_engine(validated)
    print(f"  Selected engine: {engine}")

    print("\nStep 5: Model generation")
    result = generator.generate_from_text(
        description=description,
        export_formats=['step', 'stl']
    )

    print(f"  Status: {'✓ SUCCESS' if result.success else '✗ FAILED'}")
    print(f"  Message: {result.message}")

    print("\nStep 6: Export results")
    if result.export_paths:
        for fmt, path in result.export_paths.items():
            print(f"  {fmt.upper()}: {path}")
    else:
        print("  No files exported (engine may not be available)")


def demo_api_integration():
    """Demo: API integration status."""
    print_separator("DEMO 6: API Integration Status")

    # Check environment variables
    claude_key = os.getenv('ANTHROPIC_API_KEY')
    zoo_key = os.getenv('ZOO_API_KEY')

    print("API Configuration:")
    print(f"  Anthropic (Claude): {'✓ Configured' if claude_key else '✗ Not configured'}")
    print(f"  Zoo.dev (KCL): {'✓ Configured' if zoo_key else '✗ Not configured'}")

    print("\nCapabilities:")
    print("  ✓ Build123d parametric CAD (always available)")
    print(f"  {'✓' if claude_key else '✗'} Advanced NLP with Claude API")
    print(f"  {'✓' if claude_key else '✗'} Vision analysis with Claude Vision")
    print(f"  {'✓' if zoo_key else '✗'} KCL-based generation with Zoo.dev")
    print("  ✓ OpenCV-based sketch interpretation")
    print("  ✓ DXF/DWG technical drawing import")
    print("  ✓ Multi-format export (STEP, STL, etc.)")

    if not claude_key:
        print("\n💡 Tip: Set ANTHROPIC_API_KEY environment variable for advanced features")
    if not zoo_key:
        print("💡 Tip: Set ZOO_API_KEY environment variable for KCL generation")


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              GenAI CAD-CFD Studio - Model Generation Demo                   ║
║                                                                              ║
║  Comprehensive CAD generation from multiple input types:                    ║
║    • Natural language text descriptions                                     ║
║    • Reference photos and hand-drawn sketches                               ║
║    • Technical drawings (DXF, DWG, PDF)                                     ║
║    • Hybrid multi-modal inputs                                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Run demos
    try:
        demo_api_integration()
        demo_parameter_extraction()
        demo_text_generation()
        demo_hybrid_generation()
        demo_image_analysis()
        demo_workflow_pipeline()

        print_separator("Demo Complete!")
        print("✓ All demonstrations completed successfully!")
        print("\nNext steps:")
        print("  1. Configure API keys for full functionality")
        print("  2. Run the Streamlit UI: streamlit run app.py")
        print("  3. Try uploading sketches and technical drawings")
        print("  4. Explore the Design Studio tab for interactive CAD generation")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
