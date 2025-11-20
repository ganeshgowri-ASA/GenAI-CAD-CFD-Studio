"""
Adam.new API Connector - Conversational AI CAD generation.

This module provides an interface to Adam.new's conversational CAD API,
supporting natural language model generation, refinement, and multi-format export.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
import requests

logger = logging.getLogger(__name__)


@dataclass
class ConversationHistory:
    """Track conversation history for iterative refinement."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    model_id: Optional[str] = None

    def add_user_message(self, content: str) -> None:
        """Add user message to history."""
        self.messages.append({
            'role': 'user',
            'content': content,
            'timestamp': time.time()
        })

    def add_assistant_message(self, content: str, model_id: Optional[str] = None) -> None:
        """Add assistant message to history."""
        self.messages.append({
            'role': 'assistant',
            'content': content,
            'timestamp': time.time()
        })
        if model_id:
            self.model_id = model_id

    def get_context(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get conversation context, optionally limited to last N messages."""
        if last_n:
            return self.messages[-last_n:]
        return self.messages.copy()

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.model_id = None


class AdamNewConnector:
    """
    Connector for Adam.new conversational CAD API.

    Supports natural language model generation, iterative refinement,
    and multi-format export with conversation tracking.
    """

    API_BASE_URL = "https://api.adam.new/v1"

    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize Adam.new connector.

        Args:
            api_key: Adam.new API key (required unless mock_mode=True)
            mock_mode: If True, use mock responses for testing
        """
        self.mock_mode = mock_mode
        self.api_key = api_key
        self.conversation = ConversationHistory()

        if not mock_mode and not api_key:
            raise ValueError("api_key is required when mock_mode=False")

        self.session = requests.Session()
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })

        logger.info(f"Adam.new connector initialized (mock_mode={mock_mode})")

    def generate_from_nl(
        self,
        prompt: str,
        max_retries: int = 3,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Generate CAD model from natural language prompt.

        Args:
            prompt: Natural language description of desired CAD model
            max_retries: Maximum number of retry attempts on failure
            include_context: Whether to include conversation history

        Returns:
            dict: Contains 'model_id', 'model_data', 'preview_url', etc.

        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If prompt is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.mock_mode:
            return self._mock_generate_from_nl(prompt)

        # Add to conversation history
        self.conversation.add_user_message(prompt)

        # Prepare request payload
        payload = {'prompt': prompt}
        if include_context and self.conversation.messages:
            payload['context'] = self.conversation.get_context(last_n=10)

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.API_BASE_URL}/cad/generate",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                data = response.json()
                model_id = data.get('model_id')

                if not model_id:
                    raise ValueError("No model_id received from API")

                # Add to conversation history
                self.conversation.add_assistant_message(
                    f"Generated model: {model_id}",
                    model_id=model_id
                )

                logger.info(f"Model generated successfully: {model_id}")
                return data

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts exhausted")
                    raise

    def refine_model(
        self,
        model_id: str,
        feedback: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Refine existing model with feedback.

        Args:
            model_id: ID of the model to refine
            feedback: Natural language feedback for refinement
            max_retries: Maximum number of retry attempts on failure

        Returns:
            dict: Contains updated 'model_id', 'model_data', etc.

        Raises:
            requests.RequestException: If API request fails after retries
            ValueError: If model_id or feedback is empty
        """
        if not model_id or not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if not feedback or not feedback.strip():
            raise ValueError("feedback cannot be empty")

        if self.mock_mode:
            return self._mock_refine_model(model_id, feedback)

        # Add to conversation history
        self.conversation.add_user_message(feedback)

        payload = {
            'model_id': model_id,
            'feedback': feedback,
            'context': self.conversation.get_context(last_n=10)
        }

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.API_BASE_URL}/cad/refine",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                data = response.json()
                new_model_id = data.get('model_id')

                # Update conversation history
                self.conversation.add_assistant_message(
                    f"Refined model: {new_model_id}",
                    model_id=new_model_id
                )

                logger.info(f"Model refined successfully: {new_model_id}")
                return data

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("All retry attempts exhausted")
                    raise

    def download_formats(
        self,
        model_id: str,
        formats: List[str],
        output_dir: str = ".",
        max_retries: int = 3
    ) -> Dict[str, str]:
        """
        Download model in multiple formats.

        Args:
            model_id: ID of the model to download
            formats: List of desired formats (e.g., ['step', 'stl', 'obj'])
            output_dir: Directory to save downloaded files
            max_retries: Maximum number of retry attempts on failure

        Returns:
            dict: Mapping of format to local file path

        Raises:
            requests.RequestException: If download fails after retries
            ValueError: If model_id or formats is empty
        """
        if not model_id or not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if not formats:
            raise ValueError("formats list cannot be empty")

        if self.mock_mode:
            return self._mock_download_formats(model_id, formats, output_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = {}

        for fmt in formats:
            fmt = fmt.lower()
            logger.info(f"Downloading {fmt.upper()} format for model {model_id}...")

            for attempt in range(max_retries):
                try:
                    response = self.session.get(
                        f"{self.API_BASE_URL}/cad/download/{model_id}",
                        params={'format': fmt},
                        stream=True,
                        timeout=120
                    )
                    response.raise_for_status()

                    # Determine filename
                    filename = f"{model_id}.{fmt}"
                    filepath = output_dir / filename

                    # Download file
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    downloaded_files[fmt] = str(filepath)
                    logger.info(f"Downloaded {fmt.upper()} to {filepath}")
                    break

                except requests.RequestException as e:
                    logger.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed to download {fmt.upper()} format")
                        raise

        return downloaded_files

    def get_model_info(self, model_id: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_id: ID of the model
            max_retries: Maximum number of retry attempts on failure

        Returns:
            dict: Model information including status, formats available, etc.
        """
        if self.mock_mode:
            return self._mock_get_model_info(model_id)

        for attempt in range(max_retries):
            try:
                response = self.session.get(
                    f"{self.API_BASE_URL}/cad/models/{model_id}",
                    timeout=30
                )
                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation.clear()
        logger.info("Conversation history cleared")

    # Mock methods for testing
    def _mock_generate_from_nl(self, prompt: str) -> Dict[str, Any]:
        """Generate mock model data for testing."""
        logger.info(f"[MOCK] Generating model from: {prompt[:50]}...")
        model_id = f"mock-model-{hash(prompt) % 1000000}"

        self.conversation.add_assistant_message(
            f"Generated model: {model_id}",
            model_id=model_id
        )

        return {
            'model_id': model_id,
            'status': 'completed',
            'preview_url': f'https://mock-adam.example.com/preview/{model_id}.png',
            'formats_available': ['step', 'stl', 'obj', 'glb'],
            'metadata': {
                'prompt': prompt,
                'created_at': time.time()
            }
        }

    def _mock_refine_model(self, model_id: str, feedback: str) -> Dict[str, Any]:
        """Generate mock refined model data for testing."""
        logger.info(f"[MOCK] Refining model {model_id} with: {feedback[:50]}...")
        new_model_id = f"{model_id}-refined-{hash(feedback) % 1000}"

        self.conversation.add_assistant_message(
            f"Refined model: {new_model_id}",
            model_id=new_model_id
        )

        return {
            'model_id': new_model_id,
            'status': 'completed',
            'preview_url': f'https://mock-adam.example.com/preview/{new_model_id}.png',
            'formats_available': ['step', 'stl', 'obj', 'glb'],
            'metadata': {
                'parent_model': model_id,
                'feedback': feedback,
                'refined_at': time.time()
            }
        }

    def _mock_download_formats(
        self,
        model_id: str,
        formats: List[str],
        output_dir: str
    ) -> Dict[str, str]:
        """Create mock model files for testing."""
        logger.info(f"[MOCK] Downloading formats {formats} for model {model_id}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = {}
        for fmt in formats:
            filename = f"{model_id}.{fmt.lower()}"
            filepath = output_dir / filename
            filepath.write_text(f"Mock {fmt.upper()} data for model {model_id}\n")
            downloaded_files[fmt.lower()] = str(filepath)
            logger.info(f"[MOCK] Created {filepath}")

        return downloaded_files

    def _mock_get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Generate mock model info for testing."""
        logger.info(f"[MOCK] Getting info for model {model_id}")
        return {
            'model_id': model_id,
            'status': 'completed',
            'formats_available': ['step', 'stl', 'obj', 'glb'],
            'metadata': {
                'created_at': time.time()
            }
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
