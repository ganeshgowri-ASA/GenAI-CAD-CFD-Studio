"""
Comprehensive CAD Model Generator

This module provides a unified interface for generating CAD models from multiple input types:
- Natural language text descriptions
- Reference photos and images  - Technical drawings (PDF, DXF, DWG)
- Hand-drawn sketches
- Part/assembly specifications

Integrates with:
- Claude API for NLP processing
- Zoo.dev for KCL parametric CAD
- Build123d for Python-native CAD modeling
- OpenCV for image analysis
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import base64
import io

# Import existing modules
from .build123d_engine import Build123DEngine, BUILD123D_AVAILABLE
from .zoo_connector import ZooDevConnector, PaymentRequiredError
from ..ai.sketch_interpreter import SketchInterpreter
from ..ai.dimension_extractor import DimensionExtractor
from ..ai.claude_skills import ClaudeSkills
from ..io.dxf_parser import DXFParser, HAS_EZDXF

# Optional imports
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    logging.warning("anthropic package not installed. Claude API integration disabled.")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("OpenCV not installed. Image analysis disabled.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not installed. Image processing disabled.")

try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    logging.warning("PyPDF2 not installed. PDF parsing disabled.")


logger = logging.getLogger(__name__)


class CADGenerationResult:
    """Container for CAD generation results."""

    def __init__(
        self,
        success: bool,
        message: str,
        parameters: Optional[Dict[str, Any]] = None,
        part: Optional[Any] = None,
        kcl_code: Optional[str] = None,
        model_url: Optional[str] = None,
        export_paths: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.message = message
        self.parameters = parameters or {}
        self.part = part
        self.kcl_code = kcl_code
        self.model_url = model_url
        self.export_paths = export_paths or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'message': self.message,
            'parameters': self.parameters,
            'kcl_code': self.kcl_code,
            'model_url': self.model_url,
            'export_paths': self.export_paths,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"CADGenerationResult({status}: {self.message})"


class CADModelGenerator:
    """
    Comprehensive CAD model generation engine.

    Supports multiple input types and generation backends with intelligent
    routing, parameter extraction, validation, and export.
    """

    def __init__(
        self,
        claude_api_key: Optional[str] = None,
        zoo_api_key: Optional[str] = None,
        default_engine: str = "build123d",
        default_unit: str = "mm",
        output_dir: Optional[str] = None
    ):
        """
        Initialize CAD Model Generator.

        Args:
            claude_api_key: Anthropic API key for Claude integration
            zoo_api_key: Zoo.dev API key for KCL generation
            default_engine: Default CAD engine ('build123d' or 'zoo')
            default_unit: Default unit for dimensions ('mm', 'cm', 'm', etc.)
            output_dir: Directory for output files (default: './cad_output')
        """
        self.claude_api_key = claude_api_key
        self.zoo_api_key = zoo_api_key
        self.default_engine = default_engine
        self.default_unit = default_unit
        self.output_dir = Path(output_dir or './cad_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.claude_client = None
        if claude_api_key and HAS_ANTHROPIC:
            self.claude_client = anthropic.Anthropic(api_key=claude_api_key)

        self.build123d_engine = None
        if BUILD123D_AVAILABLE:
            try:
                self.build123d_engine = Build123DEngine()
            except Exception as e:
                logger.warning(f"Build123d engine initialization failed: {e}")

        self.zoo_connector = None
        if zoo_api_key:
            self.zoo_connector = ZooDevConnector(api_key=zoo_api_key)

        self.sketch_interpreter = SketchInterpreter() if HAS_CV2 else None
        self.dimension_extractor = DimensionExtractor()
        self.claude_skills = ClaudeSkills(api_key=claude_api_key)

        logger.info(f"CADModelGenerator initialized (engine={default_engine}, unit={default_unit})")

    def generate_from_text(
        self,
        description: str,
        engine: Optional[str] = None,
        export_formats: Optional[List[str]] = None,
        **kwargs
    ) -> CADGenerationResult:
        """
        Generate CAD model from natural language text description.

        Args:
            description: Natural language description of the model
            engine: CAD engine to use ('build123d', 'zoo', or None for auto)
            export_formats: List of formats to export ('step', 'stl', 'dxf')
            **kwargs: Additional parameters

        Returns:
            CADGenerationResult with generated model and metadata

        Example:
            >>> generator = CADModelGenerator()
            >>> result = generator.generate_from_text(
            ...     "Create a box 100mm x 50mm x 30mm",
            ...     export_formats=['step', 'stl']
            ... )
        """
        logger.info(f"Generating CAD model from text: {description[:100]}...")

        try:
            # Step 1: Extract parameters using Claude API or fallback
            parameters = self._extract_parameters_from_text(description, **kwargs)

            # Step 2: Validate and normalize parameters
            parameters = self._validate_parameters(parameters)

            # Step 3: Select engine
            engine = engine or self._select_engine(parameters)

            # Step 4: Generate model with fallback on 402 error
            if engine == 'zoo' and self.zoo_connector:
                try:
                    result = self._generate_with_zoo(description, parameters, **kwargs)
                except PaymentRequiredError:
                    logger.warning("Zoo.dev returned 402 Payment Required, falling back to Build123d")
                    # Fallback to Build123d
                    if self.build123d_engine:
                        result = self._generate_with_build123d(parameters, **kwargs)
                        # Add metadata to indicate fallback was used
                        result.metadata['fallback'] = 'build123d_after_zoo_402'
                        result.message = "Model generated with Build123d (fallback due to Zoo.dev payment required)"
                    else:
                        return CADGenerationResult(
                            success=False,
                            message="Zoo.dev requires payment and Build123d is not available as fallback",
                            parameters=parameters
                        )
            elif engine == 'build123d' and self.build123d_engine:
                result = self._generate_with_build123d(parameters, **kwargs)
            else:
                return CADGenerationResult(
                    success=False,
                    message=f"Engine '{engine}' not available or not configured",
                    parameters=parameters
                )

            # Step 5: Export to requested formats
            if result.success and export_formats:
                result = self._export_model(result, export_formats)

            logger.info(f"CAD generation completed: {result.message}")
            return result

        except Exception as e:
            logger.error(f"CAD generation failed: {e}", exc_info=True)
            return CADGenerationResult(
                success=False,
                message=f"Generation failed: {str(e)}"
            )

    def generate_from_image(
        self,
        image_path: Union[str, Path],
        image_type: str = "sketch",
        description: Optional[str] = None,
        engine: Optional[str] = None,
        export_formats: Optional[List[str]] = None,
        **kwargs
    ) -> CADGenerationResult:
        """
        Generate CAD model from image (sketch, photo, or technical drawing).

        Args:
            image_path: Path to image file
            image_type: Type of image ('sketch', 'photo', 'technical')
            description: Optional text description to supplement image
            engine: CAD engine to use
            export_formats: List of formats to export
            **kwargs: Additional parameters

        Returns:
            CADGenerationResult with generated model and metadata

        Example:
            >>> result = generator.generate_from_image(
            ...     "sketch.png",
            ...     image_type="sketch",
            ...     description="Make it 100mm tall"
            ... )
        """
        logger.info(f"Generating CAD model from image: {image_path}")

        if not HAS_CV2 or not self.sketch_interpreter:
            return CADGenerationResult(
                success=False,
                message="OpenCV not available for image analysis"
            )

        try:
            # Step 1: Analyze image
            image_params = self._analyze_image(image_path, image_type, **kwargs)

            # Step 2: Use Claude Vision API if available
            if self.claude_client and HAS_PIL:
                claude_params = self._analyze_image_with_claude(image_path, description)
                # Merge parameters, preferring Claude's analysis
                image_params.update(claude_params)

            # Step 3: Merge with text description if provided
            if description:
                text_params = self._extract_parameters_from_text(description, **kwargs)
                # Merge, preferring explicit text parameters
                for key, value in text_params.items():
                    if value is not None:
                        image_params[key] = value

            # Step 4: Validate and generate
            image_params = self._validate_parameters(image_params)
            engine = engine or self._select_engine(image_params)

            # Try Zoo first if selected, with fallback to Build123d on 402 error
            if engine == 'zoo' and self.zoo_connector:
                try:
                    # Create a combined prompt for Zoo
                    combined_prompt = self._create_prompt_from_params(image_params, description)
                    result = self._generate_with_zoo(combined_prompt, image_params, **kwargs)
                except PaymentRequiredError:
                    logger.warning("Zoo.dev returned 402 Payment Required, falling back to Build123d + Anthropic Vision")
                    # Fallback to Build123d
                    if self.build123d_engine:
                        result = self._generate_with_build123d(image_params, **kwargs)
                        # Add metadata to indicate fallback was used
                        result.metadata['fallback'] = 'build123d_after_zoo_402'
                        result.message = "Model generated with Build123d (fallback due to Zoo.dev payment required)"
                    else:
                        return CADGenerationResult(
                            success=False,
                            message="Zoo.dev requires payment and Build123d is not available as fallback",
                            parameters=image_params
                        )
            elif engine == 'build123d' and self.build123d_engine:
                result = self._generate_with_build123d(image_params, **kwargs)
            else:
                return CADGenerationResult(
                    success=False,
                    message=f"Engine '{engine}' not available",
                    parameters=image_params
                )

            # Step 5: Export
            if result.success and export_formats:
                result = self._export_model(result, export_formats)

            return result

        except Exception as e:
            logger.error(f"Image-based generation failed: {e}", exc_info=True)
            return CADGenerationResult(
                success=False,
                message=f"Image generation failed: {str(e)}"
            )

    def generate_from_drawing(
        self,
        drawing_path: Union[str, Path],
        drawing_format: Optional[str] = None,
        description: Optional[str] = None,
        engine: Optional[str] = None,
        export_formats: Optional[List[str]] = None,
        **kwargs
    ) -> CADGenerationResult:
        """
        Generate CAD model from technical drawing (DXF, DWG, PDF).

        Args:
            drawing_path: Path to drawing file
            drawing_format: Format hint ('dxf', 'dwg', 'pdf', or None for auto-detect)
            description: Optional supplementary description
            engine: CAD engine to use
            export_formats: List of formats to export
            **kwargs: Additional parameters

        Returns:
            CADGenerationResult with generated model and metadata

        Example:
            >>> result = generator.generate_from_drawing(
            ...     "technical_drawing.dxf",
            ...     export_formats=['step']
            ... )
        """
        logger.info(f"Generating CAD model from drawing: {drawing_path}")

        drawing_path = Path(drawing_path)
        if not drawing_path.exists():
            return CADGenerationResult(
                success=False,
                message=f"Drawing file not found: {drawing_path}"
            )

        # Auto-detect format
        if not drawing_format:
            drawing_format = drawing_path.suffix.lower().lstrip('.')

        try:
            # Parse drawing based on format
            if drawing_format in ['dxf', 'dwg']:
                if not HAS_EZDXF:
                    return CADGenerationResult(
                        success=False,
                        message="ezdxf not installed for DXF/DWG parsing"
                    )
                drawing_params = self._parse_dxf(drawing_path)

            elif drawing_format == 'pdf':
                if not HAS_PYPDF:
                    return CADGenerationResult(
                        success=False,
                        message="PyPDF2 not installed for PDF parsing"
                    )
                drawing_params = self._parse_pdf(drawing_path, description)

            else:
                return CADGenerationResult(
                    success=False,
                    message=f"Unsupported drawing format: {drawing_format}"
                )

            # Merge with text description
            if description:
                text_params = self._extract_parameters_from_text(description, **kwargs)
                drawing_params.update(text_params)

            # Validate and generate
            drawing_params = self._validate_parameters(drawing_params)
            engine = engine or self._select_engine(drawing_params)

            if engine == 'build123d' and self.build123d_engine:
                result = self._generate_with_build123d(drawing_params, **kwargs)
            else:
                return CADGenerationResult(
                    success=False,
                    message=f"Engine '{engine}' not available for drawing import",
                    parameters=drawing_params
                )

            # Export
            if result.success and export_formats:
                result = self._export_model(result, export_formats)

            return result

        except Exception as e:
            logger.error(f"Drawing-based generation failed: {e}", exc_info=True)
            return CADGenerationResult(
                success=False,
                message=f"Drawing generation failed: {str(e)}"
            )

    def generate_from_hybrid(
        self,
        inputs: Dict[str, Any],
        engine: Optional[str] = None,
        export_formats: Optional[List[str]] = None,
        **kwargs
    ) -> CADGenerationResult:
        """
        Generate CAD model from multiple input sources (hybrid approach).

        Args:
            inputs: Dictionary with keys like 'text', 'image', 'drawing', 'specs'
            engine: CAD engine to use
            export_formats: List of formats to export
            **kwargs: Additional parameters

        Returns:
            CADGenerationResult with generated model and metadata

        Example:
            >>> result = generator.generate_from_hybrid({
            ...     'text': 'Create a mounting bracket',
            ...     'image': 'sketch.png',
            ...     'specs': {'material': 'aluminum', 'thickness': 5}
            ... })
        """
        logger.info("Generating CAD model from hybrid inputs")

        try:
            # Collect parameters from all sources
            all_params = {}

            # Process text input
            if 'text' in inputs and inputs['text']:
                text_params = self._extract_parameters_from_text(inputs['text'], **kwargs)
                all_params.update(text_params)

            # Process image input
            if 'image' in inputs and inputs['image']:
                image_params = self._analyze_image(inputs['image'], 'sketch', **kwargs)
                # Don't overwrite explicit text parameters
                for key, value in image_params.items():
                    if key not in all_params:
                        all_params[key] = value

            # Process drawing input
            if 'drawing' in inputs and inputs['drawing']:
                drawing_path = Path(inputs['drawing'])
                if drawing_path.suffix.lower() in ['.dxf', '.dwg']:
                    drawing_params = self._parse_dxf(drawing_path)
                    for key, value in drawing_params.items():
                        if key not in all_params:
                            all_params[key] = value

            # Process explicit specifications
            if 'specs' in inputs and inputs['specs']:
                all_params.update(inputs['specs'])

            # Validate and generate
            all_params = self._validate_parameters(all_params)
            engine = engine or self._select_engine(all_params)

            # Generate model with fallback on 402 error
            if engine == 'zoo' and self.zoo_connector:
                try:
                    prompt = self._create_prompt_from_params(all_params, inputs.get('text'))
                    result = self._generate_with_zoo(prompt, all_params, **kwargs)
                except PaymentRequiredError:
                    logger.warning("Zoo.dev returned 402 Payment Required, falling back to Build123d")
                    # Fallback to Build123d
                    if self.build123d_engine:
                        result = self._generate_with_build123d(all_params, **kwargs)
                        # Add metadata to indicate fallback was used
                        result.metadata['fallback'] = 'build123d_after_zoo_402'
                        result.message = "Model generated with Build123d (fallback due to Zoo.dev payment required)"
                    else:
                        return CADGenerationResult(
                            success=False,
                            message="Zoo.dev requires payment and Build123d is not available as fallback",
                            parameters=all_params
                        )
            elif engine == 'build123d' and self.build123d_engine:
                result = self._generate_with_build123d(all_params, **kwargs)
            else:
                return CADGenerationResult(
                    success=False,
                    message=f"Engine '{engine}' not available",
                    parameters=all_params
                )

            # Export
            if result.success and export_formats:
                result = self._export_model(result, export_formats)

            return result

        except Exception as e:
            logger.error(f"Hybrid generation failed: {e}", exc_info=True)
            return CADGenerationResult(
                success=False,
                message=f"Hybrid generation failed: {str(e)}"
            )

    # ==================== Private Helper Methods ====================

    def _extract_parameters_from_text(
        self,
        description: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract CAD parameters from text using Claude API or fallback."""
        # First try Claude API for advanced NLP
        if self.claude_client:
            try:
                params = self._extract_with_claude_api(description, **kwargs)
                if params:
                    return params
            except Exception as e:
                logger.warning(f"Claude API extraction failed, using fallback: {e}")

        # Fallback to local extraction
        params = self.claude_skills.extract_dimensions(description)

        # Also use dimension extractor
        dim_params = self.dimension_extractor.parse_dimensions(description)
        params.update(dim_params)

        return params

    def _extract_with_claude_api(
        self,
        description: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Use Claude API for advanced parameter extraction."""
        prompt = f"""Extract CAD modeling parameters from this description:

"{description}"

Extract and return ONLY a JSON object with these fields (use null for unknown values):
{{
  "object_type": "box|cylinder|sphere|cone|custom",
  "length": number (in mm),
  "width": number (in mm),
  "height": number (in mm),
  "radius": number (in mm),
  "diameter": number (in mm),
  "thickness": number (in mm),
  "unit": "mm|cm|m|in|ft",
  "features": ["hole", "fillet", "chamfer", etc.],
  "material": "string",
  "notes": "any additional requirements"
}}

Return ONLY valid JSON, no other text."""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Extract JSON from response
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            params = json.loads(response_text)
            logger.info(f"Claude API extracted parameters: {params}")
            return params

        except Exception as e:
            logger.error(f"Claude API parameter extraction failed: {e}")
            raise

    def _analyze_image(
        self,
        image_path: Union[str, Path],
        image_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Analyze image using OpenCV-based sketch interpreter."""
        if not self.sketch_interpreter:
            raise ValueError("Sketch interpreter not available")

        # Load and process image
        self.sketch_interpreter.load_image(str(image_path))
        self.sketch_interpreter.detect_edges()
        self.sketch_interpreter.extract_contours()

        # Get CAD specifications
        specs = self.sketch_interpreter.get_cad_specifications()

        # Convert to standardized parameters
        params = {
            'type': 'custom',
            'num_shapes': specs['num_shapes'],
            'shapes': specs['shapes']
        }

        # Try to infer basic dimensions from largest shape
        if specs['shapes']:
            largest_shape = specs['shapes'][0]
            params['object_type'] = largest_shape['type']

            if largest_shape['type'] == 'rectangle':
                props = largest_shape['properties']
                params['length'] = props['width']
                params['width'] = props['height']
                # Default height if not specified
                params['height'] = min(props['width'], props['height']) * 0.5

            elif largest_shape['type'] == 'circle':
                props = largest_shape['properties']
                params['radius'] = props['radius']
                params['height'] = props['radius'] * 2  # Default

        return params

    def _analyze_image_with_claude(
        self,
        image_path: Union[str, Path],
        description: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze image using Claude Vision API."""
        if not HAS_PIL:
            return {}

        try:
            # Load and encode image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize if too large (max 5MB)
                max_size = (1568, 1568)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Create prompt
            prompt = f"""Analyze this technical drawing/sketch and extract CAD parameters.

{description or 'Extract all visible dimensions and features.'}

Return ONLY a JSON object with these fields:
{{
  "object_type": "box|cylinder|sphere|cone|custom",
  "length": number (in mm),
  "width": number (in mm),
  "height": number (in mm),
  "radius": number (in mm),
  "features": ["list", "of", "features"],
  "notes": "observations about the sketch"
}}

Return ONLY valid JSON."""

            # Call Claude Vision API
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            response_text = response.content[0].text.strip()

            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            params = json.loads(response_text)
            logger.info(f"Claude Vision extracted parameters: {params}")
            return params

        except Exception as e:
            logger.warning(f"Claude Vision analysis failed: {e}")
            return {}

    def _parse_dxf(self, dxf_path: Path) -> Dict[str, Any]:
        """Parse DXF file and extract CAD parameters."""
        parser = DXFParser()
        dxf_data = parser.parse(str(dxf_path))

        # Convert DXF data to CAD parameters
        params = {
            'type': 'custom',
            'dxf_data': dxf_data,
            'bounds': dxf_data['bounds']
        }

        # Try to infer dimensions from bounds
        xmin, xmax, ymin, ymax, zmin, zmax = dxf_data['bounds']
        params['length'] = xmax - xmin
        params['width'] = ymax - ymin
        params['height'] = max(zmax - zmin, 10)  # Default height if 2D

        return params

    def _parse_pdf(
        self,
        pdf_path: Path,
        description: Optional[str]
    ) -> Dict[str, Any]:
        """Parse PDF technical drawing."""
        # For PDF, we'd typically convert to image and use vision analysis
        # This is a simplified implementation
        logger.warning("PDF parsing is basic - consider using Claude Vision on PDF-to-image")

        # If Claude client available, could use it directly on PDF
        if self.claude_client and description:
            try:
                return self._extract_parameters_from_text(description)
            except Exception as e:
                logger.error(f"PDF analysis failed: {e}")

        return {'type': 'box', 'length': 100, 'width': 100, 'height': 100}

    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters."""
        # Ensure object type
        if 'object_type' not in params and 'type' not in params:
            params['type'] = 'box'
        elif 'object_type' in params and 'type' not in params:
            params['type'] = params['object_type']

        # Validate dimensions
        if self.dimension_extractor:
            is_valid = self.dimension_extractor.validate_dimensions(params)
            if not is_valid:
                logger.warning("Dimension validation failed, using defaults")
                suggestions = self.dimension_extractor.suggest_corrections(params)
                logger.info(f"Suggestions: {suggestions}")

        return params

    def _select_engine(self, params: Dict[str, Any]) -> str:
        """Select appropriate CAD engine based on parameters."""
        # Use Zoo for complex/custom descriptions
        if params.get('type') == 'custom' and self.zoo_connector:
            return 'zoo'

        # Use Build123d for standard primitives
        if params.get('type') in ['box', 'cylinder', 'sphere', 'cone'] and self.build123d_engine:
            return 'build123d'

        # Fall back to default
        return self.default_engine

    def _generate_with_build123d(
        self,
        params: Dict[str, Any],
        **kwargs
    ) -> CADGenerationResult:
        """Generate CAD model using Build123d."""
        try:
            part = self.build123d_engine.generate_from_params(params)

            return CADGenerationResult(
                success=True,
                message="Model generated successfully with Build123d",
                parameters=params,
                part=part,
                metadata={'engine': 'build123d'}
            )

        except Exception as e:
            logger.error(f"Build123d generation failed: {e}")
            return CADGenerationResult(
                success=False,
                message=f"Build123d generation failed: {str(e)}",
                parameters=params
            )

    def _generate_with_zoo(
        self,
        prompt: str,
        params: Dict[str, Any],
        **kwargs
    ) -> CADGenerationResult:
        """Generate CAD model using Zoo.dev."""
        try:
            result = self.zoo_connector.generate_model(prompt)

            return CADGenerationResult(
                success=True,
                message="Model generated successfully with Zoo.dev",
                parameters=params,
                kcl_code=result['kcl_code'],
                model_url=result['model_url'],
                metadata={'engine': 'zoo'}
            )

        except Exception as e:
            logger.error(f"Zoo.dev generation failed: {e}")
            return CADGenerationResult(
                success=False,
                message=f"Zoo.dev generation failed: {str(e)}",
                parameters=params
            )

    def _create_prompt_from_params(
        self,
        params: Dict[str, Any],
        description: Optional[str]
    ) -> str:
        """Create natural language prompt from parameters."""
        if description:
            return description

        obj_type = params.get('type', 'object')
        parts = [f"Create a {obj_type}"]

        if 'length' in params and 'width' in params and 'height' in params:
            parts.append(f"{params['length']}mm x {params['width']}mm x {params['height']}mm")
        elif 'radius' in params and 'height' in params:
            parts.append(f"with radius {params['radius']}mm and height {params['height']}mm")
        elif 'radius' in params:
            parts.append(f"with radius {params['radius']}mm")

        return " ".join(parts)

    def _export_model(
        self,
        result: CADGenerationResult,
        formats: List[str]
    ) -> CADGenerationResult:
        """Export model to requested formats."""
        export_paths = {}

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"model_{timestamp}"

            for fmt in formats:
                fmt = fmt.lower()

                if fmt == 'step' and result.part:
                    output_path = self.output_dir / f"{base_name}.step"
                    self.build123d_engine.export_step(result.part, output_path)
                    export_paths['step'] = str(output_path)

                elif fmt == 'stl' and result.part:
                    output_path = self.output_dir / f"{base_name}.stl"
                    self.build123d_engine.export_stl(result.part, output_path)
                    export_paths['stl'] = str(output_path)

                elif fmt == 'kcl' and result.kcl_code:
                    output_path = self.output_dir / f"{base_name}.kcl"
                    output_path.write_text(result.kcl_code)
                    export_paths['kcl'] = str(output_path)

                elif fmt in ['glb', 'gltf'] and result.model_url:
                    output_path = self.output_dir / f"{base_name}.{fmt}"
                    self.zoo_connector.download_model(result.model_url, str(output_path))
                    export_paths[fmt] = str(output_path)

            result.export_paths = export_paths
            logger.info(f"Exported to formats: {list(export_paths.keys())}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            result.metadata['export_error'] = str(e)

        return result
