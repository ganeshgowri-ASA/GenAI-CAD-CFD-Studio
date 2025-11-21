"""
Test Script for Image Upload Handler

Validates the image upload functionality including:
- File format validation
- Size validation
- Image integrity
- Type-specific validation

Author: GenAI CAD CFD Studio
Version: 1.0.0
"""

import sys
from pathlib import Path
import logging
import io
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cad.image_upload_handler import ImageUploadHandler, ImageType, ImageValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_basic_validation():
    """Test basic image validation."""
    print("\n" + "="*60)
    print("TEST 1: Basic Image Validation")
    print("="*60)

    handler = ImageUploadHandler()

    # Create a test image in memory
    img = Image.new('RGB', (800, 600), color='white')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Test validation
    result = handler.validate_and_process(buffer, image_type=ImageType.SKETCH)

    print(f"\nValidation Result: {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Metadata: {result['metadata']}")

    assert result['is_valid'], "Basic validation should pass"
    print("\n✅ Test 1 PASSED")


def test_size_validation():
    """Test file size validation."""
    print("\n" + "="*60)
    print("TEST 2: File Size Validation")
    print("="*60)

    handler = ImageUploadHandler(max_file_size=1024)  # 1KB limit

    # Create a large image
    img = Image.new('RGB', (2000, 2000), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    # Test validation
    result = handler.validate_and_process(buffer)

    print(f"\nValidation Result: {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")

    # Should have warnings about size
    assert len(result['warnings']) > 0, "Should have size warnings"
    print("\n✅ Test 2 PASSED")


def test_dimension_validation():
    """Test dimension validation."""
    print("\n" + "="*60)
    print("TEST 3: Dimension Validation")
    print("="*60)

    handler = ImageUploadHandler()

    # Test minimum dimensions - too small
    img = Image.new('RGB', (32, 32), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    result = handler.validate_and_process(buffer)

    print(f"\nValidation Result (small image): {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Errors: {result['errors']}")

    assert not result['is_valid'], "Too small image should fail"

    # Test valid dimensions
    img = Image.new('RGB', (800, 600), color='green')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    result = handler.validate_and_process(buffer)

    print(f"\nValidation Result (valid image): {'PASSED' if result['is_valid'] else 'FAILED'}")
    assert result['is_valid'], "Valid size should pass"

    print("\n✅ Test 3 PASSED")


def test_quality_analysis():
    """Test image quality analysis."""
    print("\n" + "="*60)
    print("TEST 4: Quality Analysis")
    print("="*60)

    handler = ImageUploadHandler()

    # Create image with varying quality
    img = Image.new('RGB', (800, 600), color='white')
    # Add some variation
    pixels = img.load()
    for i in range(100):
        for j in range(100):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256)

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    result = handler.validate_and_process(buffer)

    print(f"\nValidation Result: {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Quality Metrics: {result['metadata'].get('quality', {})}")

    assert result['is_valid'], "Quality analysis should pass"
    assert 'quality' in result['metadata'], "Should have quality metrics"

    print("\n✅ Test 4 PASSED")


def test_type_specific_validation():
    """Test type-specific validation."""
    print("\n" + "="*60)
    print("TEST 5: Type-Specific Validation")
    print("="*60)

    handler = ImageUploadHandler()

    # Test sketch validation
    img = Image.new('L', (800, 600), color=128)  # Grayscale
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    result = handler.validate_and_process(buffer, image_type=ImageType.SKETCH)

    print(f"\nSketch Validation: {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Warnings: {result['warnings']}")

    # Test photo validation
    img = Image.new('RGB', (800, 600), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    result = handler.validate_and_process(buffer, image_type=ImageType.PHOTO)

    print(f"\nPhoto Validation: {'PASSED' if result['is_valid'] else 'FAILED'}")
    print(f"Warnings: {result['warnings']}")

    print("\n✅ Test 5 PASSED")


def test_metrics_tracking():
    """Test metrics tracking."""
    print("\n" + "="*60)
    print("TEST 6: Metrics Tracking")
    print("="*60)

    handler = ImageUploadHandler()

    # Perform multiple validations
    for i in range(5):
        img = Image.new('RGB', (800, 600), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        handler.validate_and_process(buffer)

    # Get metrics
    metrics = handler.get_metrics()

    print(f"\nMetrics:")
    print(f"Total Validations: {metrics['total_validations']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Failed: {metrics['failed']}")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")

    assert metrics['total_validations'] == 5, "Should have 5 validations"
    assert metrics['success_rate'] == 1.0, "All should succeed"

    print("\n✅ Test 6 PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("IMAGE UPLOAD HANDLER - TEST SUITE")
    print("="*60)

    tests = [
        test_basic_validation,
        test_size_validation,
        test_dimension_validation,
        test_quality_analysis,
        test_type_specific_validation,
        test_metrics_tracking
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ Test FAILED: {e}")
            logger.exception("Test failed")

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
