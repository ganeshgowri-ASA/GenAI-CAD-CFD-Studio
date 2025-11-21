#!/usr/bin/env python3
"""
Standalone test script to verify Zoo.dev API integration.
This version has minimal dependencies and can run without full environment setup.
"""

import os
import sys
import time
import logging
import requests
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

    logger.info(f"ZOO_API_KEY/ZOO_API_TOKEN found: {results['zoo_key_found']}")
    logger.info(f"ANTHROPIC_API_KEY found: {results['anthropic_key_found']}")

    if zoo_api_key:
        logger.info(f"  ZOO_API_KEY length: {len(zoo_api_key)} chars")
        logger.info(f"  ZOO_API_KEY prefix: {zoo_api_key[:10]}...")
    else:
        logger.warning("  No Zoo.dev API key found!")

    if anthropic_api_key:
        logger.info(f"  ANTHROPIC_API_KEY length: {len(anthropic_api_key)} chars")
        logger.info(f"  ANTHROPIC_API_KEY prefix: {anthropic_api_key[:10]}...")
    else:
        logger.warning("  No Anthropic API key found!")

    return results


def test_zoo_api_endpoint_directly():
    """Test the Zoo.dev API endpoint with a direct HTTP request."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Direct Zoo.dev API Endpoint Test")
    logger.info("=" * 60)

    zoo_api_key = os.getenv('ZOO_API_KEY') or os.getenv('ZOO_API_TOKEN')

    if not zoo_api_key:
        logger.warning("⚠ Skipping API test (no API key configured)")
        logger.info("\nTo test with a real API key:")
        logger.info("  export ZOO_API_KEY='your_key_here'")
        logger.info("  python test_zoo_api_standalone.py")
        return {'success': False, 'reason': 'no_api_key'}

    # Test the actual API endpoint
    try:
        logger.info("Testing Zoo.dev API endpoint...")
        logger.info("  Endpoint: POST https://api.zoo.dev/ai/text-to-cad/step")
        logger.info("  Prompt: Create a simple box 10mm x 10mm x 10mm")

        headers = {
            'Authorization': f'Bearer {zoo_api_key}',
            'Content-Type': 'application/json'
        }

        payload = {
            'prompt': 'Create a simple box 10mm x 10mm x 10mm'
        }

        logger.info("\nStep 1: Submitting text-to-CAD request...")
        response = requests.post(
            'https://api.zoo.dev/ai/text-to-cad/step',
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"  Response status: {response.status_code}")

        if response.status_code == 404:
            logger.error("✗ 404 Not Found - API endpoint may be incorrect!")
            logger.error(f"  Response: {response.text[:200]}")
            return {'success': False, 'error': '404_not_found', 'response': response.text}

        if response.status_code == 401 or response.status_code == 403:
            logger.error("✗ Authentication failed - check your API key!")
            logger.error(f"  Response: {response.text[:200]}")
            return {'success': False, 'error': 'auth_failed', 'response': response.text}

        response.raise_for_status()
        data = response.json()

        task_id = data.get('id')
        if not task_id:
            logger.error("✗ No task ID received from API")
            logger.error(f"  Response: {data}")
            return {'success': False, 'error': 'no_task_id'}

        logger.info(f"✓ Task submitted successfully!")
        logger.info(f"  Task ID: {task_id}")
        logger.info(f"  Initial status: {data.get('status', 'unknown')}")

        # Poll for completion
        logger.info("\nStep 2: Polling for task completion...")
        max_wait = 60  # 1 minute for this test
        poll_interval = 5
        start_time = time.time()

        while time.time() - start_time < max_wait:
            logger.info(f"  Checking status... ({int(time.time() - start_time)}s elapsed)")

            status_response = requests.get(
                f'https://api.zoo.dev/async/operations/{task_id}',
                headers=headers,
                timeout=30
            )
            status_response.raise_for_status()
            status_data = status_response.json()

            status = status_data.get('status', '').lower()
            logger.info(f"  Status: {status}")

            if status == 'completed':
                logger.info("✓ Task completed successfully!")
                outputs = status_data.get('outputs', {})
                logger.info(f"  Output files: {list(outputs.keys())}")

                if 'code' in status_data:
                    logger.info(f"  Generated KCL code: {len(status_data['code'])} chars")

                return {
                    'success': True,
                    'task_id': task_id,
                    'status': status,
                    'outputs': list(outputs.keys()),
                    'elapsed_time': int(time.time() - start_time)
                }

            elif status == 'failed':
                error_msg = status_data.get('error', 'Unknown error')
                logger.error(f"✗ Task failed: {error_msg}")
                return {'success': False, 'error': 'task_failed', 'message': error_msg}

            # Still in progress
            time.sleep(poll_interval)

        logger.warning("⚠ Task did not complete within timeout")
        return {'success': False, 'error': 'timeout'}

    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ Connection error: {e}")
        logger.error("  Check your internet connection and Zoo.dev service status")
        return {'success': False, 'error': 'connection_error', 'message': str(e)}

    except requests.exceptions.Timeout as e:
        logger.error(f"✗ Request timeout: {e}")
        return {'success': False, 'error': 'timeout', 'message': str(e)}

    except requests.exceptions.HTTPError as e:
        logger.error(f"✗ HTTP error: {e}")
        logger.error(f"  Response: {e.response.text[:500]}")
        return {'success': False, 'error': 'http_error', 'message': str(e)}

    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        logger.error(f"  Error type: {type(e).__name__}")
        return {'success': False, 'error': 'unexpected', 'message': str(e)}


def test_code_version():
    """Check if the code has the updated API endpoint."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Code Version Check")
    logger.info("=" * 60)

    try:
        with open('src/cad/zoo_connector.py', 'r') as f:
            content = f.read()

        # Check for new API URL
        has_correct_url = 'https://api.zoo.dev' in content and 'https://api.zoo.dev/v1' not in content
        has_new_endpoint = '/ai/text-to-cad/' in content
        has_old_endpoint = '/text-to-cad/generate-kcl' in content
        has_polling = '_poll_task_status' in content

        logger.info(f"Correct base URL (without /v1): {has_correct_url}")
        logger.info(f"New endpoint (/ai/text-to-cad/): {has_new_endpoint}")
        logger.info(f"Old endpoint (should be removed): {has_old_endpoint}")
        logger.info(f"Has async polling: {has_polling}")

        if has_correct_url and has_new_endpoint and not has_old_endpoint and has_polling:
            logger.info("✓ Code version: UP TO DATE")
            return {'success': True, 'up_to_date': True}
        else:
            logger.warning("⚠ Code version: MAY BE OUTDATED")
            return {'success': True, 'up_to_date': False}

    except Exception as e:
        logger.error(f"✗ Could not check code version: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Zoo.dev API Integration Test Suite (Standalone)")
    logger.info("=" * 60)
    logger.info("\n")

    results = {}

    # Run tests
    results['api_keys'] = test_api_key_configuration()
    results['code_version'] = test_code_version()
    results['api_endpoint'] = test_zoo_api_endpoint_directly()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    if results['api_keys']['zoo_key_found']:
        logger.info("✓ API Key Configuration: CONFIGURED")
    else:
        logger.warning("⚠ API Key Configuration: NOT CONFIGURED")

    if results['code_version'].get('up_to_date'):
        logger.info("✓ Code Version: UP TO DATE")
    else:
        logger.warning("⚠ Code Version: CHECK NEEDED")

    if results['api_endpoint'].get('success'):
        logger.info("✓ API Endpoint Test: PASS")
        elapsed = results['api_endpoint'].get('elapsed_time', 0)
        logger.info(f"  Completed in {elapsed} seconds")
    elif results['api_endpoint'].get('reason') == 'no_api_key':
        logger.warning("⚠ API Endpoint Test: SKIPPED (no API key)")
    else:
        logger.error("✗ API Endpoint Test: FAIL")
        error = results['api_endpoint'].get('error', 'unknown')
        logger.error(f"  Error: {error}")

    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS FOR STREAMLIT DEPLOYMENT")
    logger.info("=" * 60)

    if not results['api_keys']['zoo_key_found']:
        logger.info("""
⚠ ACTION REQUIRED: Set up API keys in Streamlit Cloud

1. Go to: https://share.streamlit.io/
2. Select your app: GenAI-CAD-CFD-Studio
3. Click "Settings" → "Secrets"
4. Add the following to secrets.toml:

   ZOO_API_KEY = "your_zoo_api_key_here"
   ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

5. Click "Save"
6. Restart the app

Get Zoo.dev API key from: https://zoo.dev/account
Get Anthropic API key from: https://console.anthropic.com/
""")

    if not results['code_version'].get('up_to_date'):
        logger.info("""
⚠ ACTION REQUIRED: Update code to latest version

1. Ensure the latest code is merged to main branch
2. In Streamlit Cloud, trigger a redeployment:
   - Go to app settings
   - Click "Reboot app" or "Redeploy"
3. Check deployment logs for any errors
""")

    if results['api_endpoint'].get('success'):
        logger.info("""
✓ API is working correctly!

If Streamlit app still shows errors:
1. Check Streamlit Cloud logs for specific error messages
2. Verify the app is using the correct branch (main)
3. Try rebooting the app in Streamlit Cloud settings
4. Clear browser cache and reload the app
""")

    return 0


if __name__ == '__main__':
    sys.exit(main())
