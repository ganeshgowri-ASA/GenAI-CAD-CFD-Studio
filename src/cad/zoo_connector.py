"""
Zoo.dev API Connector - Text-to-CAD generation.

This module provides an interface to Zoo.dev's text-to-CAD API,
generating 3D CAD models from natural language descriptions.
Supports multiple output formats (STEP, STL, OBJ, GLTF, GLB, etc.)
with async task polling and automatic file downloading.
"""

import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import requests
from collections import deque

logger = logging.getLogger(__name__)


class PaymentRequiredError(Exception):
    """Exception raised when Zoo.dev returns 402 Payment Required."""
    pass


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()

    def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)

        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        # Wait if at limit
        if len(self.requests) >= self.max_requests:
            oldest_request = self.requests[0]
            wait_time = (oldest_request + timedelta(seconds=self.time_window) - now).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                # Recursively call to re-check
                self.acquire()
                return

        self.requests.append(now)


class ZooDevConnector:
    """
    Connector for Zoo.dev text-to-CAD API.

    Handles CAD model generation from text prompts with async task polling,
    multiple output format support, and automatic file downloading.
    Includes rate limiting and comprehensive error handling.
    """

    API_BASE_URL = "https://api.zoo.dev"

    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize Zoo.dev connector.

        Args:
            api_key: Zoo.dev API key (required unless mock_mode=True)
            mock_mode: If True, use mock responses for testing
        """
        self.mock_mode = mock_mode
        self.api_key = api_key
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)

        if not mock_mode and not api_key:
            raise ValueError("api_key is required when mock_mode=False")

        self.session = requests.Session()
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })

        logger.info(f"Zoo.dev connector initialized (mock_mode={mock_mode})")

    def _poll_task_status(
        self,
        task_id: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Poll async task status until completion.

        Args:
            task_id: The async task ID to poll
            timeout: Maximum time to wait in seconds (default: 300)
            poll_interval: Time between polls in seconds (default: 5)

        Returns:
            dict: Task result data including outputs

        Raises:
            TimeoutError: If task doesn't complete within timeout
            ValueError: If task fails
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

            try:
                response = self.session.get(
                    f"{self.API_BASE_URL}/async/operations/{task_id}",
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                status = data.get('status', '').lower()
                logger.info(f"Task {task_id} status: {status}")

                if status == 'completed':
                    logger.info(f"Task {task_id} completed successfully")
                    return data
                elif status == 'failed':
                    error_msg = data.get('error', 'Unknown error')
                    raise ValueError(f"Task {task_id} failed: {error_msg}")

                # Status is 'queued' or 'in_progress', continue polling
                time.sleep(poll_interval)

            except requests.RequestException as e:
                logger.warning(f"Error polling task {task_id}: {e}")
                time.sleep(poll_interval)

    def generate_cad_model(
        self,
        prompt: str,
        output_format: str = "step",
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate CAD model from natural language prompt using Zoo.dev API.

        Args:
            prompt: Natural language description of desired CAD model
            output_format: Output format (step, stl, obj, gltf, glb, etc.)
            max_retries: Maximum number of retry attempts on failure

        Returns:
            dict: Task result with 'outputs', 'id', 'status', etc.

        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If prompt is empty or task fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.mock_mode:
            return self._mock_generate_cad_model(prompt, output_format)

        self.rate_limiter.acquire()

        for attempt in range(max_retries):
            try:
                # Submit text-to-CAD request
                response = self.session.post(
                    f"{self.API_BASE_URL}/ai/text-to-cad/{output_format}",
                    json={'prompt': prompt},
                    timeout=30
                )

                # Check for 402 Payment Required specifically
                if response.status_code == 402:
                    logger.warning("Zoo.dev returned 402 Payment Required")
                    raise PaymentRequiredError("Zoo.dev API requires payment or quota exceeded")

                response.raise_for_status()

                data = response.json()
                task_id = data.get('id')

                if not task_id:
                    raise ValueError("No task ID received from API")

                logger.info(f"Text-to-CAD task created: {task_id}")

                # Poll for completion
                result = self._poll_task_status(task_id)
                return result

            except PaymentRequiredError:
                # Don't retry on payment errors, raise immediately
                logger.error("Zoo.dev payment required - no retry")
                raise
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts exhausted")
                    raise

    def download_output_file(
        self,
        outputs: Dict[str, Any],
        file_key: str,
        output_path: str,
        max_retries: int = 3
    ) -> None:
        """
        Download a specific output file from task results.

        Args:
            outputs: The 'outputs' dict from completed task
            file_key: Key of the file to download (e.g., 'source.step')
            output_path: Local path to save the file
            max_retries: Maximum number of retry attempts on failure

        Raises:
            requests.RequestException: If download fails after retries
            KeyError: If file_key not found in outputs
        """
        if self.mock_mode:
            self._mock_download_model(file_key, output_path)
            return

        if file_key not in outputs:
            available_keys = list(outputs.keys())
            raise KeyError(
                f"File '{file_key}' not found in outputs. "
                f"Available files: {available_keys}"
            )

        file_data = outputs[file_key]

        # Handle both direct bytes and URL references
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # If file_data is bytes, write directly
        if isinstance(file_data, bytes):
            with open(output_path, 'wb') as f:
                f.write(file_data)
            logger.info(f"File saved successfully to {output_path}")
            return

        # If it's a string (URL), download from URL
        if isinstance(file_data, str):
            for attempt in range(max_retries):
                try:
                    response = self.session.get(file_data, stream=True, timeout=120)
                    response.raise_for_status()

                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    logger.info(f"File downloaded successfully to {output_path}")
                    return

                except requests.RequestException as e:
                    logger.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error("All download attempts exhausted")
                        raise

        raise ValueError(f"Unexpected file data type: {type(file_data)}")

    def generate_model(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        output_format: str = "step"
    ) -> Dict[str, Any]:
        """
        Complete workflow: generate CAD model and optionally download.

        Args:
            prompt: Natural language description of desired CAD model
            output_path: If provided, download model to this path
            output_format: Output format (step, stl, obj, gltf, glb, etc.)

        Returns:
            dict: Contains task result with 'outputs', 'id', 'status', etc.
                  Includes backward-compatible 'kcl_code' and 'model_url' fields.
                  If output_path provided, also includes 'local_path'
        """
        # Generate CAD model using new API
        logger.info(f"Generating {output_format.upper()} model from prompt: {prompt[:100]}...")
        result = self.generate_cad_model(prompt, output_format)

        # Add backward compatibility fields
        if 'code' in result:
            result['kcl_code'] = result['code']

        # Try to extract a primary model URL from outputs
        outputs = result.get('outputs', {})
        if outputs:
            # Prefer the requested format, or use first available
            file_key = f"source.{output_format}"
            if file_key not in outputs:
                # Find any matching format
                for key in outputs.keys():
                    if key.endswith(f".{output_format}"):
                        file_key = key
                        break
                else:
                    # Use first available output
                    file_key = next(iter(outputs.keys()))

            result['model_url'] = outputs.get(file_key, '')

        # Download if requested
        if output_path and outputs:
            file_key = f"source.{output_format}"

            # If exact key not found, try to find any matching format
            if file_key not in outputs:
                for key in outputs.keys():
                    if key.endswith(f".{output_format}"):
                        file_key = key
                        break

            logger.info(f"Downloading {file_key} to {output_path}...")
            self.download_output_file(outputs, file_key, output_path)
            result['local_path'] = str(output_path)

        logger.info("Model generation complete")
        return result

    # Mock methods for testing
    def _mock_generate_cad_model(
        self,
        prompt: str,
        output_format: str
    ) -> Dict[str, Any]:
        """Generate mock CAD model result for testing."""
        logger.info(f"[MOCK] Generating {output_format} model for: {prompt[:50]}...")
        time.sleep(1)  # Simulate processing time

        mock_kcl = f"""// Mock KCL generated from prompt: {prompt[:50]}
const box = startSketchOn('XY')
  |> startProfileAt([0, 0], %)
  |> line([0, 10], %)
  |> line([10, 0], %)
  |> line([0, -10], %)
  |> close(%)
  |> extrude(5, %)
"""

        mock_id = f"mock-task-{hash(prompt) % 1000000}"
        mock_outputs = {
            f"source.{output_format}": f"https://mock-zoo.example.com/files/{mock_id}.{output_format}",
            "source.gltf": f"https://mock-zoo.example.com/files/{mock_id}.gltf"
        }

        return {
            'id': mock_id,
            'status': 'Completed',
            'prompt': prompt,
            'output_format': output_format,
            'outputs': mock_outputs,
            'code': mock_kcl
        }

    def _mock_download_model(self, file_key: str, output_path: str) -> None:
        """Create mock model file for testing."""
        logger.info(f"[MOCK] Downloading {file_key} to {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"Mock model data for {file_key}\n")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
