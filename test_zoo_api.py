#!/usr/bin/env python3
"""
Test script to verify Zoo.dev API integration and diagnose deployment issues.

This script checks:
1. API key configuration
2. Zoo.dev API endpoint connectivity
3. Async task polling
4. File download functionality
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cad.zoo_connector import ZooDevConnector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_api_key_configuration():
    """Test if API keys are properly configured."""
    logger.info("=" * 60)
    logger.info("TEST 1: API Key Configuration")
    logger.info("=" * 60)

    zoo_api_key = os.getenv('ZOO_API_KEY') or os.getenv('ZOO_API_TOKEN')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

    results = {
        'zoo_key_found': bool(zoo_api_key),
        'anthropic_key_found': bool(anthropic_api_key),
    }

    logger.info(f"✓ ZOO_API_KEY/ZOO_API_TOKEN found: {results['zoo_key_found']}")
    logger.info(f"✓ ANTHROPIC_API_KEY found: {results['anthropic_key_found']}")

    if zoo_api_key:
        logger.info(f"  ZOO_API_KEY length: {len(zoo_api_key)} chars")
        logger.info(f"  ZOO_API_KEY prefix: {zoo_api_key[:10]}...")

    if anthropic_api_key:
        logger.info(f"  ANTHROPIC_API_KEY length: {len(anthropic_api_key)} chars")
        logger.info(f"  ANTHROPIC_API_KEY prefix: {anthropic_api_key[:10]}...")

    return results


def test_zoo_connector_initialization():
    """Test if ZooDevConnector can be initialized."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: ZooDevConnector Initialization")
    logger.info("=" * 60)

    zoo_api_key = os.getenv('ZOO_API_KEY') or os.getenv('ZOO_API_TOKEN')

    try:
        # Try real mode
        if zoo_api_key:
            connector = ZooDevConnector(api_key=zoo_api_key, mock_mode=False)
            logger.info("✓ ZooDevConnector initialized successfully (real mode)")
            logger.info(f"  API Base URL: {connector.API_BASE_URL}")
            return {'success': True, 'mode': 'real', 'connector': connector}
        else:
            # Try mock mode
            connector = ZooDevConnector(api_key=None, mock_mode=True)
            logger.info("⚠ ZooDevConnector initialized in MOCK mode (no API key)")
            logger.info(f"  API Base URL: {connector.API_BASE_URL}")
            return {'success': True, 'mode': 'mock', 'connector': connector}

    except Exception as e:
        logger.error(f"✗ Failed to initialize ZooDevConnector: {e}")
        return {'success': False, 'error': str(e)}


def test_api_endpoint():
    """Test the Zoo.dev API endpoint with a simple request."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Zoo.dev API Endpoint Test")
    logger.info("=" * 60)

    zoo_api_key = os.getenv('ZOO_API_KEY') or os.getenv('ZOO_API_TOKEN')

    if not zoo_api_key:
        logger.warning("⚠ Skipping API test (no API key configured)")
        logger.info("  Running in MOCK mode instead...")

        # Test mock mode
        try:
            connector = ZooDevConnector(api_key=None, mock_mode=True)
            result = connector.generate_cad_model(
                prompt="Create a simple box 10mm x 10mm x 10mm",
                output_format="step"
            )
            logger.info("✓ Mock API test successful")
            logger.info(f"  Mock Task ID: {result.get('id')}")
            logger.info(f"  Mock Status: {result.get('status')}")
            logger.info(f"  Mock Outputs: {list(result.get('outputs', {}).keys())}")
            return {'success': True, 'mode': 'mock', 'result': result}
        except Exception as e:
            logger.error(f"✗ Mock API test failed: {e}")
            return {'success': False, 'mode': 'mock', 'error': str(e)}

    # Test real API
    try:
        connector = ZooDevConnector(api_key=zoo_api_key, mock_mode=False)
        logger.info("Testing real Zoo.dev API endpoint...")
        logger.info("  Endpoint: POST https://api.zoo.dev/ai/text-to-cad/step")
        logger.info("  Prompt: Create a simple box 10mm x 10mm x 10mm")
        logger.info("  This may take 10-30 seconds...")

        result = connector.generate_cad_model(
            prompt="Create a simple box 10mm x 10mm x 10mm",
            output_format="step"
        )

        logger.info("✓ Real API test successful!")
        logger.info(f"  Task ID: {result.get('id')}")
        logger.info(f"  Status: {result.get('status')}")
        logger.info(f"  Output files: {list(result.get('outputs', {}).keys())}")

        if 'code' in result:
            logger.info(f"  Generated KCL code: {len(result['code'])} chars")

        return {'success': True, 'mode': 'real', 'result': result}

    except Exception as e:
        logger.error(f"✗ Real API test failed: {e}")
        logger.error(f"  Error type: {type(e).__name__}")

        if "404" in str(e):
            logger.error("  → This is a 404 error - the endpoint may be incorrect")
        elif "401" in str(e) or "403" in str(e):
            logger.error("  → This is an authentication error - check your API key")
        elif "timeout" in str(e).lower():
            logger.error("  → Request timed out - API may be slow or unavailable")

        return {'success': False, 'mode': 'real', 'error': str(e)}


def test_backward_compatibility():
    """Test backward compatibility with old API."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Backward Compatibility")
    logger.info("=" * 60)

    try:
        connector = ZooDevConnector(api_key=None, mock_mode=True)
        result = connector.generate_model(
            prompt="Create a cylinder with radius 5mm and height 20mm",
            output_format="step"
        )

        # Check for backward-compatible fields
        has_kcl_code = 'kcl_code' in result
        has_model_url = 'model_url' in result
        has_outputs = 'outputs' in result

        logger.info(f"✓ generate_model() method works")
        logger.info(f"  Has 'kcl_code' field (backward compat): {has_kcl_code}")
        logger.info(f"  Has 'model_url' field (backward compat): {has_model_url}")
        logger.info(f"  Has 'outputs' field (new API): {has_outputs}")

        if has_kcl_code and has_model_url:
            logger.info("✓ Backward compatibility: PASS")
        else:
            logger.warning("⚠ Backward compatibility: Some fields missing")

        return {'success': True, 'backward_compat': has_kcl_code and has_model_url}

    except Exception as e:
        logger.error(f"✗ Backward compatibility test failed: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Zoo.dev API Integration Test Suite")
    logger.info("=" * 60)
    logger.info("\n")

    results = {}

    # Run tests
    results['api_keys'] = test_api_key_configuration()
    results['initialization'] = test_zoo_connector_initialization()
    results['api_endpoint'] = test_api_endpoint()
    results['backward_compat'] = test_backward_compatibility()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = all(
        r.get('success', False)
        for r in results.values()
        if isinstance(r, dict)
    )

    if results['api_keys']['zoo_key_found']:
        logger.info("✓ API Key Configuration: PASS")
    else:
        logger.warning("⚠ API Key Configuration: NO KEY (using mock mode)")

    if results['initialization']['success']:
        logger.info(f"✓ Connector Initialization: PASS ({results['initialization']['mode']} mode)")
    else:
        logger.error("✗ Connector Initialization: FAIL")

    if results['api_endpoint']['success']:
        logger.info(f"✓ API Endpoint Test: PASS ({results['api_endpoint']['mode']} mode)")
    else:
        logger.error(f"✗ API Endpoint Test: FAIL")

    if results['backward_compat']['success']:
        logger.info("✓ Backward Compatibility: PASS")
    else:
        logger.error("✗ Backward Compatibility: FAIL")

    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)

    if not results['api_keys']['zoo_key_found']:
        logger.info("""
⚠ MISSING API KEY:
   1. Set environment variable: export ZOO_API_KEY='your_key_here'
   2. Or for Streamlit Cloud: Add to Secrets:
      - Go to: https://share.streamlit.io/
      - Navigate to your app settings
      - Add to secrets.toml:
        ZOO_API_KEY = "your_key_here"
        ANTHROPIC_API_KEY = "your_key_here"
   3. Get Zoo.dev API key from: https://zoo.dev/account
""")

    if not results['api_endpoint']['success'] and results['api_keys']['zoo_key_found']:
        logger.info("""
⚠ API ENDPOINT FAILED:
   1. Check if your API key is valid at: https://zoo.dev/account
   2. Verify the API endpoint is accessible
   3. Check for any API service outages
   4. Try running this test again
""")

    if all_passed and results['api_keys']['zoo_key_found']:
        logger.info("""
✓ ALL TESTS PASSED!
  Your Zoo.dev API integration is working correctly.

  If Streamlit app still shows errors:
  1. Restart the Streamlit app
  2. Check Streamlit Cloud deployment logs
  3. Verify the latest code is deployed (main branch)
""")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
