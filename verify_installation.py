"""
Verification script for CAD Model Generator installation
"""

import sys
from pathlib import Path

def check_import(module_name, package=None):
    """Check if a module can be imported."""
    try:
        if package:
            exec(f"from {package} import {module_name}")
        else:
            exec(f"import {module_name}")
        return True, "✓"
    except ImportError as e:
        return False, f"✗ ({str(e)})"
    except Exception as e:
        return False, f"✗ ({type(e).__name__})"


def main():
    """Run verification checks."""
    print("=" * 80)
    print("  CAD Model Generator - Installation Verification")
    print("=" * 80)
    print()

    # Check core modules
    print("Core CAD Modules:")
    modules = [
        ('model_generator', 'src.cad'),
        ('CADModelGenerator', 'src.cad.model_generator'),
        ('CADGenerationResult', 'src.cad.model_generator'),
        ('Build123DEngine', 'src.cad.build123d_engine'),
        ('ZooDevConnector', 'src.cad.zoo_connector'),
    ]

    all_passed = True
    for module, package in modules:
        success, status = check_import(module, package)
        print(f"  {status} {package}.{module}")
        all_passed = all_passed and success

    print()
    print("AI Modules:")
    ai_modules = [
        ('SketchInterpreter', 'src.ai.sketch_interpreter'),
        ('DimensionExtractor', 'src.ai.dimension_extractor'),
        ('ClaudeSkills', 'src.ai.claude_skills'),
    ]

    for module, package in ai_modules:
        success, status = check_import(module, package)
        print(f"  {status} {package}.{module}")
        all_passed = all_passed and success

    print()
    print("IO Modules:")
    io_modules = [
        ('DXFParser', 'src.io.dxf_parser'),
        ('STLHandler', 'src.io.stl_handler'),
        ('STEPHandler', 'src.io.step_handler'),
    ]

    for module, package in io_modules:
        success, status = check_import(module, package)
        print(f"  {status} {package}.{module}")
        all_passed = all_passed and success

    print()
    print("UI Modules:")
    ui_modules = [
        ('design_studio', 'src.ui'),
    ]

    for module, package in ui_modules:
        success, status = check_import(module, package)
        print(f"  {status} {package}.{module}")
        all_passed = all_passed and success

    print()
    print("=" * 80)
    print("Optional Dependencies:")
    print("=" * 80)

    optional = [
        'anthropic',
        'build123d',
        'cv2',
        'PIL',
        'ezdxf',
        'PyPDF2',
    ]

    for module in optional:
        success, status = check_import(module)
        availability = "Available" if success else "Not installed"
        print(f"  {status} {module:20s} - {availability}")

    print()
    print("=" * 80)
    if all_passed:
        print("✓ All core modules imported successfully!")
        print("\nYou can now:")
        print("  1. Run the demo: python examples/cad_generation_demo.py")
        print("  2. Start the UI: streamlit run app.py")
        print("  3. Use the API programmatically")
    else:
        print("✗ Some imports failed. Please check the error messages above.")

    print("=" * 80)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
