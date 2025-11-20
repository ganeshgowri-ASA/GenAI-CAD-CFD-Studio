"""
CAD Agent Interface - Unified interface for all CAD generation engines.

This module provides an abstract base class for CAD agents and a unified
interface that can auto-select the appropriate engine based on the prompt.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CADResult:
    """
    Result of CAD generation operation.

    Contains the generated model, metadata, and convenience methods
    for exporting to different formats.
    """
    engine: str
    model: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None

    def export_step(self, filepath: str) -> None:
        """
        Export to STEP format.

        Args:
            filepath: Output file path
        """
        from .build123d_engine import Build123DEngine, BUILD123D_AVAILABLE

        if self.engine == 'build123d':
            if not BUILD123D_AVAILABLE:
                raise RuntimeError("build123d not available for export")
            engine = Build123DEngine()
            engine.export_step(self.model, filepath)
        else:
            # For other engines, the model might already be a file path
            # or we need to handle it differently
            if isinstance(self.model, (str, Path)):
                # Model is already a file path, copy/convert if needed
                source_path = Path(self.model)
                dest_path = Path(filepath)
                if source_path.suffix.lower() in ['.step', '.stp']:
                    dest_path.write_bytes(source_path.read_bytes())
                else:
                    logger.warning(
                        f"Model from {self.engine} is not in STEP format, "
                        "conversion not implemented"
                    )
            else:
                raise NotImplementedError(
                    f"STEP export not implemented for engine: {self.engine}"
                )

    def export_stl(self, filepath: str, resolution: str = 'high') -> None:
        """
        Export to STL format.

        Args:
            filepath: Output file path
            resolution: Quality of mesh ('low', 'medium', 'high')
        """
        from .build123d_engine import Build123DEngine, BUILD123D_AVAILABLE

        if self.engine == 'build123d':
            if not BUILD123D_AVAILABLE:
                raise RuntimeError("build123d not available for export")
            engine = Build123DEngine()
            engine.export_stl(self.model, filepath, resolution)
        else:
            # Similar handling as export_step
            if isinstance(self.model, (str, Path)):
                source_path = Path(self.model)
                dest_path = Path(filepath)
                if source_path.suffix.lower() == '.stl':
                    dest_path.write_bytes(source_path.read_bytes())
                else:
                    logger.warning(
                        f"Model from {self.engine} is not in STL format, "
                        "conversion not implemented"
                    )
            else:
                raise NotImplementedError(
                    f"STL export not implemented for engine: {self.engine}"
                )

    def get_metadata(self, key: Optional[str] = None) -> Any:
        """
        Get metadata value(s).

        Args:
            key: Specific metadata key, or None to get all metadata

        Returns:
            Metadata value or entire metadata dict
        """
        if key:
            return self.metadata.get(key)
        return self.metadata.copy()


class CADAgent(ABC):
    """Abstract base class for CAD generation agents."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> CADResult:
        """
        Generate CAD model from prompt.

        Args:
            prompt: Natural language description or parameters
            **kwargs: Additional engine-specific parameters

        Returns:
            CADResult: The generation result
        """
        pass

    @abstractmethod
    def get_engine_name(self) -> str:
        """Get the name of this engine."""
        pass


class UnifiedCADInterface:
    """
    Unified interface for all CAD generation engines.

    Automatically selects the appropriate engine based on prompt
    characteristics and provides a consistent API.
    """

    def __init__(
        self,
        zoo_api_key: Optional[str] = None,
        adam_api_key: Optional[str] = None,
        mock_mode: bool = False
    ):
        """
        Initialize unified CAD interface.

        Args:
            zoo_api_key: API key for Zoo.dev (optional)
            adam_api_key: API key for Adam.new (optional)
            mock_mode: If True, use mock mode for API connectors
        """
        self.mock_mode = mock_mode
        self.zoo_api_key = zoo_api_key
        self.adam_api_key = adam_api_key

        # Initialize engines lazily
        self._build123d_engine = None
        self._zoo_connector = None
        self._adam_connector = None

        logger.info(f"Unified CAD interface initialized (mock_mode={mock_mode})")

    @property
    def build123d_engine(self):
        """Lazy load Build123D engine."""
        if self._build123d_engine is None:
            from .build123d_engine import Build123DEngine
            self._build123d_engine = Build123DEngine()
        return self._build123d_engine

    @property
    def zoo_connector(self):
        """Lazy load Zoo connector."""
        if self._zoo_connector is None:
            from .zoo_connector import ZooDevConnector
            self._zoo_connector = ZooDevConnector(
                api_key=self.zoo_api_key,
                mock_mode=self.mock_mode
            )
        return self._zoo_connector

    @property
    def adam_connector(self):
        """Lazy load Adam connector."""
        if self._adam_connector is None:
            from .adam_connector import AdamNewConnector
            self._adam_connector = AdamNewConnector(
                api_key=self.adam_api_key,
                mock_mode=self.mock_mode
            )
        return self._adam_connector

    def auto_select_engine(self, prompt: str) -> str:
        """
        Automatically select the best engine based on prompt characteristics.

        Args:
            prompt: Natural language prompt or parameter dict

        Returns:
            str: Selected engine name ('build123d', 'zoo', or 'adam')

        Selection logic:
        - If prompt contains specific dimensions/parameters -> build123d
        - If prompt mentions KCL or parametric code -> zoo
        - If prompt is conversational or iterative -> adam
        - Default: adam (most flexible)
        """
        prompt_lower = prompt.lower()

        # Check for structured parameters (suggests build123d)
        param_keywords = [
            'length', 'width', 'height', 'radius', 'diameter',
            'box', 'cylinder', 'sphere', 'cone',
            'extrude', 'revolve', 'union', 'subtract'
        ]

        # Count parameter keywords
        param_score = sum(1 for kw in param_keywords if kw in prompt_lower)

        # Check for KCL/code keywords (suggests zoo)
        kcl_keywords = ['kcl', 'sketch', 'parametric', 'code', 'script']
        kcl_score = sum(1 for kw in kcl_keywords if kw in prompt_lower)

        # Check for conversational keywords (suggests adam)
        conv_keywords = [
            'create', 'design', 'make', 'generate', 'build',
            'refine', 'modify', 'adjust', 'change',
            'like', 'similar', 'want', 'need'
        ]
        conv_score = sum(1 for kw in conv_keywords if kw in prompt_lower)

        # Check if prompt looks like structured data (JSON/dict)
        if re.match(r'^\s*\{.*\}\s*$', prompt, re.DOTALL):
            logger.info("Auto-selected build123d (structured parameters)")
            return 'build123d'

        # Decision logic
        if param_score >= 3:
            logger.info(f"Auto-selected build123d (param_score={param_score})")
            return 'build123d'
        elif kcl_score >= 2:
            logger.info(f"Auto-selected zoo (kcl_score={kcl_score})")
            return 'zoo'
        elif conv_score >= 2 or len(prompt.split()) > 10:
            logger.info(f"Auto-selected adam (conv_score={conv_score})")
            return 'adam'
        else:
            # Default to adam for flexibility
            logger.info("Auto-selected adam (default)")
            return 'adam'

    def generate(
        self,
        prompt: str,
        engine: str = 'auto',
        output_path: Optional[str] = None,
        **kwargs
    ) -> CADResult:
        """
        Generate CAD model using specified or auto-selected engine.

        Args:
            prompt: Natural language description or parameters
            engine: Engine to use ('auto', 'build123d', 'zoo', 'adam')
            output_path: Optional path to save generated model
            **kwargs: Additional engine-specific parameters

        Returns:
            CADResult: The generation result

        Raises:
            ValueError: If engine is unknown
        """
        # Auto-select engine if needed
        if engine == 'auto':
            engine = self.auto_select_engine(prompt)

        logger.info(f"Generating CAD model using engine: {engine}")

        # Generate based on selected engine
        if engine == 'build123d':
            return self._generate_build123d(prompt, output_path, **kwargs)
        elif engine == 'zoo':
            return self._generate_zoo(prompt, output_path, **kwargs)
        elif engine == 'adam':
            return self._generate_adam(prompt, output_path, **kwargs)
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def _generate_build123d(
        self,
        prompt: str,
        output_path: Optional[str],
        **kwargs
    ) -> CADResult:
        """Generate using Build123D engine."""
        # Parse prompt as parameters
        # This is a simplified parser - could be enhanced
        import json

        try:
            params = json.loads(prompt)
        except json.JSONDecodeError:
            # Simple keyword extraction as fallback
            params = self._extract_parameters(prompt)

        part = self.build123d_engine.generate_from_params(params)

        # Export if path provided
        if output_path:
            output_path = Path(output_path)
            if output_path.suffix.lower() in ['.step', '.stp']:
                self.build123d_engine.export_step(part, output_path)
            elif output_path.suffix.lower() == '.stl':
                self.build123d_engine.export_stl(part, output_path)

        return CADResult(
            engine='build123d',
            model=part,
            metadata={
                'parameters': params,
                'output_path': str(output_path) if output_path else None
            },
            prompt=prompt
        )

    def _generate_zoo(
        self,
        prompt: str,
        output_path: Optional[str],
        **kwargs
    ) -> CADResult:
        """Generate using Zoo.dev connector."""
        result = self.zoo_connector.generate_model(prompt, output_path)

        return CADResult(
            engine='zoo',
            model=result.get('model_url'),
            metadata={
                'kcl_code': result.get('kcl_code'),
                'model_url': result.get('model_url'),
                'local_path': result.get('local_path')
            },
            prompt=prompt
        )

    def _generate_adam(
        self,
        prompt: str,
        output_path: Optional[str],
        **kwargs
    ) -> CADResult:
        """Generate using Adam.new connector."""
        formats = kwargs.get('formats', ['step'])

        # Generate model
        result = self.adam_connector.generate_from_nl(prompt)
        model_id = result['model_id']

        # Download if path provided
        local_files = {}
        if output_path:
            output_dir = Path(output_path).parent
            local_files = self.adam_connector.download_formats(
                model_id,
                formats,
                str(output_dir)
            )

        return CADResult(
            engine='adam',
            model=model_id,
            metadata={
                'model_id': model_id,
                'status': result.get('status'),
                'preview_url': result.get('preview_url'),
                'formats_available': result.get('formats_available'),
                'local_files': local_files
            },
            prompt=prompt
        )

    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """
        Extract parameters from natural language prompt.

        This is a simple implementation - could be enhanced with NLP.
        """
        params = {'type': 'box'}  # Default

        # Try to extract shape type
        for shape in ['box', 'cylinder', 'sphere', 'cone']:
            if shape in prompt.lower():
                params['type'] = shape
                break

        # Try to extract dimensions using regex
        import re

        # Look for patterns like "10mm", "5 cm", "length: 10", etc.
        number_pattern = r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|inch|in)?'

        # Extract length, width, height
        for dim in ['length', 'width', 'height', 'radius', 'diameter']:
            pattern = rf'{dim}\s*[:=]?\s*{number_pattern}'
            match = re.search(pattern, prompt.lower())
            if match:
                params[dim] = float(match.group(1))

        # If no specific dimensions found, try to extract any numbers
        if len(params) == 1:  # Only has 'type'
            numbers = re.findall(number_pattern, prompt)
            if numbers:
                # Assign to common dimensions based on shape type
                if params['type'] == 'box' and len(numbers) >= 3:
                    params['length'] = float(numbers[0])
                    params['width'] = float(numbers[1])
                    params['height'] = float(numbers[2])
                elif params['type'] == 'cylinder' and len(numbers) >= 2:
                    params['radius'] = float(numbers[0])
                    params['height'] = float(numbers[1])
                elif params['type'] == 'sphere' and len(numbers) >= 1:
                    params['radius'] = float(numbers[0])

        logger.info(f"Extracted parameters: {params}")
        return params

    def refine(
        self,
        model_id: str,
        feedback: str,
        engine: str = 'adam'
    ) -> CADResult:
        """
        Refine an existing model (only supported by Adam engine).

        Args:
            model_id: ID of the model to refine
            feedback: Natural language feedback
            engine: Engine to use (currently only 'adam' supported)

        Returns:
            CADResult: The refined model result

        Raises:
            ValueError: If engine doesn't support refinement
        """
        if engine != 'adam':
            raise ValueError(f"Engine '{engine}' does not support refinement")

        result = self.adam_connector.refine_model(model_id, feedback)

        return CADResult(
            engine='adam',
            model=result['model_id'],
            metadata={
                'model_id': result['model_id'],
                'status': result.get('status'),
                'preview_url': result.get('preview_url'),
                'parent_model': model_id
            },
            prompt=feedback
        )
