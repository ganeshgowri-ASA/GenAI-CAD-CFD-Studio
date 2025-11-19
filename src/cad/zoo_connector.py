"""
Zoo.dev API Connector - KCL-based CAD generation.

This module provides an interface to Zoo.dev's text-to-CAD API,
generating KCL (KittyCAD Language) code and executing it to create 3D models.
"""

import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import requests
from collections import deque

logger = logging.getLogger(__name__)


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

    Handles KCL code generation, execution, and model downloading
    with rate limiting and error handling.
    """

    API_BASE_URL = "https://api.zoo.dev/v1"

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

    def generate_kcl(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate KCL code from natural language prompt.

        Args:
            prompt: Natural language description of desired CAD model
            max_retries: Maximum number of retry attempts on failure

        Returns:
            str: Generated KCL code

        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If prompt is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.mock_mode:
            return self._mock_generate_kcl(prompt)

        self.rate_limiter.acquire()

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.API_BASE_URL}/text-to-cad/generate-kcl",
                    json={'prompt': prompt},
                    timeout=30
                )
                response.raise_for_status()

                data = response.json()
                kcl_code = data.get('kcl_code', '')

                if not kcl_code:
                    raise ValueError("Empty KCL code received from API")

                logger.info(f"KCL code generated successfully (length: {len(kcl_code)})")
                return kcl_code

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts exhausted")
                    raise

    def execute_kcl(self, code: str, max_retries: int = 3) -> str:
        """
        Execute KCL code and get model URL.

        Args:
            code: KCL code to execute
            max_retries: Maximum number of retry attempts on failure

        Returns:
            str: URL to the generated 3D model

        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If code is empty
        """
        if not code or not code.strip():
            raise ValueError("KCL code cannot be empty")

        if self.mock_mode:
            return self._mock_execute_kcl(code)

        self.rate_limiter.acquire()

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.API_BASE_URL}/text-to-cad/execute-kcl",
                    json={'code': code},
                    timeout=60
                )
                response.raise_for_status()

                data = response.json()
                model_url = data.get('model_url', '')

                if not model_url:
                    raise ValueError("Empty model URL received from API")

                logger.info(f"KCL code executed successfully: {model_url}")
                return model_url

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts exhausted")
                    raise

    def download_model(
        self,
        url: str,
        output_path: str,
        max_retries: int = 3
    ) -> None:
        """
        Download model from URL to local file.

        Args:
            url: URL of the model to download
            output_path: Local path to save the model
            max_retries: Maximum number of retry attempts on failure

        Raises:
            requests.RequestException: If download fails after retries
        """
        if self.mock_mode:
            self._mock_download_model(url, output_path)
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, stream=True, timeout=120)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"Model downloaded successfully to {output_path}")
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

    def generate_model(
        self,
        prompt: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete workflow: generate KCL, execute, and optionally download.

        Args:
            prompt: Natural language description of desired CAD model
            output_path: If provided, download model to this path

        Returns:
            dict: Contains 'kcl_code', 'model_url', and optionally 'local_path'
        """
        result = {}

        # Generate KCL
        logger.info(f"Generating KCL from prompt: {prompt[:100]}...")
        kcl_code = self.generate_kcl(prompt)
        result['kcl_code'] = kcl_code

        # Execute KCL
        logger.info("Executing KCL code...")
        model_url = self.execute_kcl(kcl_code)
        result['model_url'] = model_url

        # Download if requested
        if output_path:
            logger.info(f"Downloading model to {output_path}...")
            self.download_model(model_url, output_path)
            result['local_path'] = str(output_path)

        logger.info("Model generation complete")
        return result

    # Mock methods for testing
    def _mock_generate_kcl(self, prompt: str) -> str:
        """Generate mock KCL code for testing."""
        logger.info(f"[MOCK] Generating KCL for: {prompt[:50]}...")
        return f"""// Mock KCL generated from prompt: {prompt[:50]}
const box = startSketchOn('XY')
  |> startProfileAt([0, 0], %)
  |> line([0, 10], %)
  |> line([10, 0], %)
  |> line([0, -10], %)
  |> close(%)
  |> extrude(5, %)
"""

    def _mock_execute_kcl(self, code: str) -> str:
        """Generate mock model URL for testing."""
        logger.info("[MOCK] Executing KCL code...")
        mock_hash = hash(code) % 1000000
        return f"https://mock-zoo-dev.example.com/models/mock-{mock_hash}.glb"

    def _mock_download_model(self, url: str, output_path: str) -> None:
        """Create mock model file for testing."""
        logger.info(f"[MOCK] Downloading from {url} to {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"Mock model data from {url}\n")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
