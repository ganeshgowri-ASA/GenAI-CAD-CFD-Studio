"""
CAD Model Generator - Multi-modal CAD generation orchestrator.

This module provides a comprehensive interface for generating CAD models from:
- Natural language text descriptions
- Reference photos/images
- Technical drawings (PDF, DWG, DXF)
- Hand-drawn sketches
- Part/assembly specifications
- Hybrid combinations of the above

It orchestrates various AI and CAD generation engines to produce high-quality
parametric 3D models.
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import base64

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic not installed. Install with: pip install anthropic")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not installed")

try:
    # Try relative imports (when installed as package)
    from .build123d_engine import Build123DEngine, BUILD123D_AVAILABLE
    from .zoo_connector import ZooDevConnector
    from ..ai.sketch_interpreter import SketchInterpreter
    from ..ai.dimension_extractor import DimensionExtractor
    from ..io.dxf_parser import DXFParser
except ImportError:
    # Fall back to absolute imports (when run from examples/)
    from cad.build123d_engine import Build123DEngine, BUILD123D_AVAILABLE
    from cad.zoo_connector import ZooDevConnector
    from ai.sketch_interpreter import SketchInterpreter
    from ai.dimension_extractor import DimensionExtractor
    from io.dxf_parser import DXFParser

logger = logging.getLogger(__name__)


class CADModelGenerator:
    """
    Multi-modal CAD model generator.

    Orchestrates AI-powered analysis of various inputs and generates
    3D CAD models using Build123d and Zoo.dev KCL.

    Capabilities:
    - Text-to-CAD using Claude AI
    - Image/sketch-to-CAD using computer vision
    - Drawing-to-CAD (DXF/DWG parsing)
    - Hybrid multi-modal input processing
    - Parametric model generation
    - Export to STEP, STL, DXF formats
    """

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        zoo_api_key: Optional[str] = None,
        use_zoo_dev: bool = False,
        mock_mode: bool = False
    ):
        """
        Initialize the CAD Model Generator.

        Args:
            anthropic_api_key: Anthropic API key for Claude (uses env var if None)
            zoo_api_key: Zoo.dev API key for KCL generation (uses env var if None)
            use_zoo_dev: If True, use Zoo.dev for generation; otherwise use Build123d
            mock_mode: If True, use mock responses for testing

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If API keys are missing when not in mock mode
        """
        self.mock_mode = mock_mode
        self.use_zoo_dev = use_zoo_dev

        # Initialize Anthropic client for NLP processing
        if not mock_mode:
            if anthropic_api_key is None:
                anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

            if not anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY required (set env var or pass as argument)")

            self.claude_client = Anthropic(api_key=anthropic_api_key)
        else:
            self.claude_client = None

        # Initialize Build123d engine
        if BUILD123D_AVAILABLE:
            self.build123d = Build123DEngine()
        else:
            logger.warning("Build123d not available")
            self.build123d = None

        # Initialize Zoo.dev connector if requested
        if use_zoo_dev:
            if zoo_api_key is None:
                zoo_api_key = os.getenv('ZOO_API_KEY')
            self.zoo_connector = ZooDevConnector(api_key=zoo_api_key, mock_mode=mock_mode)
        else:
            self.zoo_connector = None

        # Initialize helper modules
        self.sketch_interpreter = SketchInterpreter()
        self.dimension_extractor = DimensionExtractor()

        logger.info(f"CADModelGenerator initialized (zoo_dev={use_zoo_dev}, mock={mock_mode})")

    def generate_from_text(
        self,
        description: str,
        output_format: str = 'step',
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CAD model from natural language text description.

        Args:
            description: Natural language description of the desired model
            output_format: Output format ('step', 'stl', 'both')
            output_path: Optional output file path (auto-generated if None)

        Returns:
            dict: Generation result containing:
                - parameters: Extracted CAD parameters
                - model: Generated model object (if Build123d used)
                - files: List of generated file paths
                - metadata: Generation metadata

        Example:
            >>> generator = CADModelGenerator()
            >>> result = generator.generate_from_text(
            ...     "Create a box 10cm x 5cm x 3cm with a 2cm diameter hole through the center"
            ... )
        """
        logger.info(f"Generating CAD from text: {description[:100]}...")

        # Step 1: Extract dimensions and parameters using Claude
        parameters = self._extract_parameters_from_text(description)
        logger.info(f"Extracted parameters: {parameters}")

        # Step 2: Validate parameters
        if not self._validate_parameters(parameters):
            raise ValueError(f"Invalid parameters extracted: {parameters}")

        # Step 3: Generate CAD model
        if self.use_zoo_dev and self.zoo_connector:
            # Use Zoo.dev KCL generation
            result = self._generate_with_zoo(description, parameters, output_path)
        else:
            # Use Build123d
            result = self._generate_with_build123d(parameters, output_format, output_path)

        result['input_type'] = 'text'
        result['input_description'] = description
        result['parameters'] = parameters

        return result

    def generate_from_image(
        self,
        image_path: str,
        image_type: str = 'sketch',
        additional_context: Optional[str] = None,
        output_format: str = 'step',
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CAD model from image (sketch, photo, or technical drawing).

        Args:
            image_path: Path to image file
            image_type: Type of image ('sketch', 'photo', 'drawing')
            additional_context: Optional text description to augment image analysis
            output_format: Output format ('step', 'stl', 'both')
            output_path: Optional output file path

        Returns:
            dict: Generation result with extracted geometry and generated files

        Example:
            >>> result = generator.generate_from_image(
            ...     "sketch.png",
            ...     image_type='sketch',
            ...     additional_context="This is a mounting bracket"
            ... )
        """
        logger.info(f"Generating CAD from image: {image_path} (type={image_type})")

        # Step 1: Analyze image using computer vision
        geometry = self._analyze_image(image_path, image_type)
        logger.info(f"Detected {len(geometry.get('shapes', []))} shapes")

        # Step 2: If additional context provided, combine with CV analysis
        if additional_context:
            parameters = self._combine_image_and_text(geometry, additional_context, image_path)
        else:
            parameters = self._geometry_to_parameters(geometry)

        # Step 3: Validate parameters
        if not self._validate_parameters(parameters):
            raise ValueError(f"Could not extract valid parameters from image")

        # Step 4: Generate CAD model
        if self.use_zoo_dev and self.zoo_connector:
            # Create enhanced description from geometry
            description = self._create_description_from_geometry(geometry, additional_context)
            result = self._generate_with_zoo(description, parameters, output_path)
        else:
            result = self._generate_with_build123d(parameters, output_format, output_path)

        result['input_type'] = 'image'
        result['image_path'] = image_path
        result['image_type'] = image_type
        result['detected_geometry'] = geometry
        result['parameters'] = parameters

        return result

    def generate_from_drawing(
        self,
        drawing_path: str,
        drawing_format: str = 'dxf',
        output_format: str = 'step',
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CAD model from technical drawing (DXF, DWG).

        Args:
            drawing_path: Path to drawing file
            drawing_format: Format of drawing ('dxf', 'dwg', 'pdf')
            output_format: Output format ('step', 'stl', 'both')
            output_path: Optional output file path

        Returns:
            dict: Generation result with parsed geometry and generated files

        Example:
            >>> result = generator.generate_from_drawing("part.dxf")
        """
        logger.info(f"Generating CAD from drawing: {drawing_path} (format={drawing_format})")

        # Step 1: Parse drawing file
        if drawing_format.lower() == 'dxf':
            geometry = self._parse_dxf(drawing_path)
        elif drawing_format.lower() == 'dwg':
            # DWG parsing would require additional library (e.g., ezdxf with ODA)
            raise NotImplementedError("DWG parsing not yet implemented. Convert to DXF first.")
        elif drawing_format.lower() == 'pdf':
            # PDF parsing would require OCR and image extraction
            raise NotImplementedError("PDF parsing not yet implemented")
        else:
            raise ValueError(f"Unsupported drawing format: {drawing_format}")

        # Step 2: Convert geometry to parameters
        parameters = self._geometry_to_parameters(geometry)

        # Step 3: Validate parameters
        if not self._validate_parameters(parameters):
            raise ValueError(f"Could not extract valid parameters from drawing")

        # Step 4: Generate CAD model
        result = self._generate_with_build123d(parameters, output_format, output_path)

        result['input_type'] = 'drawing'
        result['drawing_path'] = drawing_path
        result['drawing_format'] = drawing_format
        result['parsed_geometry'] = geometry
        result['parameters'] = parameters

        return result

    def generate_from_hybrid(
        self,
        text_description: Optional[str] = None,
        image_path: Optional[str] = None,
        drawing_path: Optional[str] = None,
        specifications: Optional[Dict[str, Any]] = None,
        output_format: str = 'step',
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate CAD model from hybrid multi-modal inputs.

        Combines information from multiple sources to create a comprehensive
        CAD model specification.

        Args:
            text_description: Natural language description
            image_path: Path to reference image/sketch
            drawing_path: Path to technical drawing
            specifications: Additional structured specifications
            output_format: Output format ('step', 'stl', 'both')
            output_path: Optional output file path

        Returns:
            dict: Generation result with combined parameters and generated files

        Example:
            >>> result = generator.generate_from_hybrid(
            ...     text_description="Mounting bracket for motor",
            ...     image_path="sketch.png",
            ...     specifications={'material': 'aluminum', 'thickness': '5mm'}
            ... )
        """
        logger.info("Generating CAD from hybrid multi-modal inputs")

        # Collect all inputs
        inputs = {
            'text': text_description,
            'image': image_path,
            'drawing': drawing_path,
            'specs': specifications
        }

        # Process each input type
        all_parameters = []

        if text_description:
            text_params = self._extract_parameters_from_text(text_description)
            all_parameters.append(('text', text_params))
            logger.info(f"Text parameters: {text_params}")

        if image_path:
            geometry = self._analyze_image(image_path, 'sketch')
            image_params = self._geometry_to_parameters(geometry)
            all_parameters.append(('image', image_params))
            logger.info(f"Image parameters: {image_params}")

        if drawing_path:
            geometry = self._parse_dxf(drawing_path)
            drawing_params = self._geometry_to_parameters(geometry)
            all_parameters.append(('drawing', drawing_params))
            logger.info(f"Drawing parameters: {drawing_params}")

        if specifications:
            all_parameters.append(('specs', specifications))
            logger.info(f"Specifications: {specifications}")

        # Merge and resolve conflicts
        merged_parameters = self._merge_parameters(all_parameters)
        logger.info(f"Merged parameters: {merged_parameters}")

        # Validate merged parameters
        if not self._validate_parameters(merged_parameters):
            raise ValueError(f"Could not create valid parameters from hybrid inputs")

        # Generate CAD model
        if self.use_zoo_dev and self.zoo_connector and text_description:
            # Use Zoo.dev with enhanced description
            result = self._generate_with_zoo(text_description, merged_parameters, output_path)
        else:
            result = self._generate_with_build123d(merged_parameters, output_format, output_path)

        result['input_type'] = 'hybrid'
        result['inputs'] = inputs
        result['parameters'] = merged_parameters
        result['parameter_sources'] = [source for source, _ in all_parameters]

        return result

    # ========================================================================
    # Internal Helper Methods
    # ========================================================================

    def _extract_parameters_from_text(self, text: str) -> Dict[str, Any]:
        """Extract CAD parameters from text using Claude AI."""

        # First, try simple dimension extraction
        dims = self.dimension_extractor.parse_dimensions(text)

        if self.mock_mode or not self.claude_client:
            # In mock mode, return basic extraction
            return self._enhance_basic_dimensions(dims, text)

        # Use Claude for comprehensive parameter extraction
        prompt = f"""Analyze this CAD model description and extract all relevant parameters:

Description: {text}

Extract and return a JSON object with:
1. type: The primary shape type (box, cylinder, sphere, cone, composite)
2. dimensions: All dimensions with units
3. features: List of features (holes, fillets, chamfers, etc.)
4. operations: Boolean operations needed (union, subtract, intersect)
5. constraints: Any constraints or relationships

Be precise with dimensions and return valid JSON only."""

        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract JSON from response
            content = response.content[0].text

            # Try to find JSON in response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
            else:
                # Fallback to basic extraction
                params = self._enhance_basic_dimensions(dims, text)

            logger.info(f"Claude extracted parameters: {params}")
            return params

        except Exception as e:
            logger.warning(f"Claude API error: {e}, falling back to basic extraction")
            return self._enhance_basic_dimensions(dims, text)

    def _enhance_basic_dimensions(self, dims: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Enhance basic dimensions with shape type detection."""
        text_lower = text.lower()

        # Detect shape type from text
        if any(word in text_lower for word in ['box', 'cube', 'rectangular', 'square']):
            shape_type = 'box'
        elif any(word in text_lower for word in ['cylinder', 'tube', 'pipe', 'rod']):
            shape_type = 'cylinder'
        elif any(word in text_lower for word in ['sphere', 'ball', 'round']):
            shape_type = 'sphere'
        elif any(word in text_lower for word in ['cone', 'conical', 'tapered']):
            shape_type = 'cone'
        else:
            shape_type = 'box'  # Default

        params = {
            'type': shape_type,
            'dimensions': dims,
            'features': [],
            'operations': []
        }

        # Detect holes
        if any(word in text_lower for word in ['hole', 'drill', 'through']):
            # Try to extract hole dimensions
            import re
            hole_match = re.search(r'(\d+(?:\.\d+)?)\s*(cm|mm|m|in)?\s*(?:diameter\s+)?hole', text_lower)
            if hole_match:
                hole_size = float(hole_match.group(1))
                hole_unit = hole_match.group(2) or dims.get('original_unit', 'mm')
                hole_diameter = self.dimension_extractor._convert_to_meters(hole_size, hole_unit)

                params['features'].append({
                    'type': 'hole',
                    'diameter': hole_diameter,
                    'position': 'center'
                })

        return params

    def _analyze_image(self, image_path: str, image_type: str) -> Dict[str, Any]:
        """Analyze image using computer vision."""

        # Load and process image
        self.sketch_interpreter.load_image(image_path)

        # Detect edges
        edges = self.sketch_interpreter.detect_edges()

        # Extract contours
        contours = self.sketch_interpreter.extract_contours(min_area=50)

        # Convert to geometry
        geometries = self.sketch_interpreter.contour_to_geometry()

        # Get specifications
        specs = self.sketch_interpreter.get_cad_specifications()

        return specs

    def _combine_image_and_text(
        self,
        geometry: Dict[str, Any],
        text: str,
        image_path: str
    ) -> Dict[str, Any]:
        """Combine image analysis with text description using Claude vision."""

        if self.mock_mode or not self.claude_client:
            # Fallback to basic combination
            text_params = self._extract_parameters_from_text(text)
            image_params = self._geometry_to_parameters(geometry)
            return self._merge_parameters([('text', text_params), ('image', image_params)])

        # Use Claude vision API to analyze image with context
        try:
            # Read image and encode to base64
            with open(image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode('utf-8')

            # Determine image media type
            ext = Path(image_path).suffix.lower()
            media_type_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(ext, 'image/png')

            prompt = f"""Analyze this image and the description to extract CAD parameters:

Description: {text}

Computer vision detected {geometry.get('num_shapes', 0)} shapes: {[s.get('type') for s in geometry.get('shapes', [])]}

Extract and return a JSON object with:
1. type: Primary shape type
2. dimensions: All dimensions (use pixels to estimate if no scale)
3. features: List of features visible in the image
4. operations: Boolean operations needed

Return valid JSON only."""

            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
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

            # Extract JSON from response
            content = response.content[0].text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                params = json.loads(json_match.group())
                logger.info(f"Claude vision extracted parameters: {params}")
                return params
            else:
                raise ValueError("No JSON in response")

        except Exception as e:
            logger.warning(f"Claude vision error: {e}, falling back to basic combination")
            text_params = self._extract_parameters_from_text(text)
            image_params = self._geometry_to_parameters(geometry)
            return self._merge_parameters([('text', text_params), ('image', image_params)])

    def _geometry_to_parameters(self, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert detected geometry to CAD parameters."""

        shapes = geometry.get('shapes', [])
        if not shapes:
            return {'type': 'box', 'dimensions': {}, 'features': [], 'operations': []}

        # Use the first (largest) shape as primary
        primary_shape = shapes[0]
        shape_type = primary_shape.get('type', 'box')

        # Convert geometry to dimensions
        dimensions = {}

        if shape_type == 'rectangle':
            props = primary_shape.get('properties', {})
            # Convert pixels to mm (assume 1 pixel = 1mm for now)
            dimensions['length'] = props.get('width', 100) * 0.001  # to meters
            dimensions['width'] = props.get('height', 100) * 0.001
            dimensions['height'] = 0.01  # Default 10mm thickness
            shape_type = 'box'

        elif shape_type == 'circle':
            props = primary_shape.get('properties', {})
            radius = props.get('radius', 50) * 0.001  # to meters
            dimensions['radius'] = radius
            dimensions['height'] = 0.01  # Default 10mm thickness
            shape_type = 'cylinder'

        params = {
            'type': shape_type,
            'dimensions': dimensions,
            'features': [],
            'operations': []
        }

        # Additional shapes could be features (holes, etc.)
        for i, shape in enumerate(shapes[1:], start=1):
            if shape.get('type') == 'circle':
                props = shape.get('properties', {})
                params['features'].append({
                    'type': 'hole',
                    'diameter': props.get('radius', 10) * 2 * 0.001,
                    'position': shape.get('center', (0, 0))
                })

        return params

    def _parse_dxf(self, dxf_path: str) -> Dict[str, Any]:
        """Parse DXF file to extract geometry."""
        try:
            parser = DXFParser()
            result = parser.parse(dxf_path)
            return result
        except Exception as e:
            logger.error(f"DXF parsing error: {e}")
            raise

    def _create_description_from_geometry(
        self,
        geometry: Dict[str, Any],
        additional_context: Optional[str] = None
    ) -> str:
        """Create text description from geometry for Zoo.dev."""

        shapes = geometry.get('shapes', [])
        description_parts = []

        if additional_context:
            description_parts.append(additional_context)

        for shape in shapes[:3]:  # Top 3 shapes
            shape_type = shape.get('type')
            props = shape.get('properties', {})

            if shape_type == 'rectangle':
                desc = f"Rectangle {props.get('width', 0)}x{props.get('height', 0)} pixels"
                description_parts.append(desc)
            elif shape_type == 'circle':
                desc = f"Circle with radius {props.get('radius', 0)} pixels"
                description_parts.append(desc)

        return ". ".join(description_parts) if description_parts else "Generic part"

    def _merge_parameters(self, param_list: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Merge parameters from multiple sources with conflict resolution."""

        merged = {
            'type': None,
            'dimensions': {},
            'features': [],
            'operations': []
        }

        # Priority order: specs > drawing > text > image
        priority = {'specs': 4, 'drawing': 3, 'text': 2, 'image': 1}

        # Sort by priority
        param_list_sorted = sorted(param_list, key=lambda x: priority.get(x[0], 0))

        for source, params in param_list_sorted:
            if params.get('type') and not merged['type']:
                merged['type'] = params['type']

            # Merge dimensions (higher priority overwrites)
            if 'dimensions' in params:
                merged['dimensions'].update(params['dimensions'])

            # Accumulate features
            if 'features' in params:
                merged['features'].extend(params['features'])

            # Accumulate operations
            if 'operations' in params:
                merged['operations'].extend(params['operations'])

        # Set default type if not found
        if not merged['type']:
            merged['type'] = 'box'

        return merged

    def _validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate CAD parameters."""

        if not params.get('type'):
            logger.warning("No shape type specified")
            return False

        dims = params.get('dimensions', {})

        # Validate based on shape type
        shape_type = params['type']

        if shape_type == 'box':
            required = ['length', 'width', 'height']
            if not all(d in dims for d in required):
                logger.warning(f"Box requires {required}, got {list(dims.keys())}")
                # Try to use defaults
                if not dims:
                    return False

        elif shape_type == 'cylinder':
            if 'radius' not in dims and 'diameter' not in dims:
                logger.warning("Cylinder requires radius or diameter")
                return False
            if 'height' not in dims:
                logger.warning("Cylinder requires height")
                return False

        elif shape_type == 'sphere':
            if 'radius' not in dims and 'diameter' not in dims:
                logger.warning("Sphere requires radius or diameter")
                return False

        # Use dimension extractor for validation
        return self.dimension_extractor.validate_dimensions(dims)

    def _generate_with_build123d(
        self,
        params: Dict[str, Any],
        output_format: str,
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """Generate CAD model using Build123d."""

        if not self.build123d:
            raise RuntimeError("Build123d not available")

        # Generate base shape
        shape_type = params['type']
        dims = params['dimensions']

        # Convert dimensions to Build123d format
        build_params = {'type': shape_type}
        build_params.update(dims)

        # Generate primary part
        part = self.build123d.generate_from_params(build_params)

        # Apply features (holes, etc.)
        for feature in params.get('features', []):
            if feature.get('type') == 'hole':
                # Create hole as cylinder subtraction
                hole_params = {
                    'type': 'cylinder',
                    'radius': feature.get('diameter', 0.01) / 2,
                    'height': dims.get('height', 0.1) * 1.2  # Slightly taller than part
                }
                hole_part = self.build123d.generate_from_params(hole_params)
                part = self.build123d.subtract(part, hole_part)

        # Export
        files = []
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs/cad")
            output_dir.mkdir(parents=True, exist_ok=True)
            base_path = output_dir / f"model_{timestamp}"
        else:
            base_path = Path(output_path).with_suffix('')

        if output_format in ['step', 'both']:
            step_path = base_path.with_suffix('.step')
            self.build123d.export_step(part, str(step_path))
            files.append(str(step_path))
            logger.info(f"Exported STEP: {step_path}")

        if output_format in ['stl', 'both']:
            stl_path = base_path.with_suffix('.stl')
            self.build123d.export_stl(part, str(stl_path))
            files.append(str(stl_path))
            logger.info(f"Exported STL: {stl_path}")

        return {
            'model': part,
            'files': files,
            'engine': 'build123d',
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'shape_type': shape_type,
                'num_features': len(params.get('features', []))
            }
        }

    def _generate_with_zoo(
        self,
        description: str,
        params: Dict[str, Any],
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """Generate CAD model using Zoo.dev KCL."""

        if not self.zoo_connector:
            raise RuntimeError("Zoo.dev connector not initialized")

        # Generate enhanced prompt
        prompt = f"{description}\n\nParameters: {json.dumps(params, indent=2)}"

        # Set output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("outputs/cad")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"model_{timestamp}.glb"

        # Generate model
        result = self.zoo_connector.generate_model(prompt, str(output_path))

        return {
            'kcl_code': result['kcl_code'],
            'model_url': result['model_url'],
            'files': [result.get('local_path')] if 'local_path' in result else [],
            'engine': 'zoo_kcl',
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'prompt': description
            }
        }

    def export_model(
        self,
        model: Any,
        output_path: str,
        format: str = 'step'
    ) -> str:
        """
        Export a model to specified format.

        Args:
            model: Model object (Build123d Part)
            output_path: Output file path
            format: Export format ('step', 'stl')

        Returns:
            str: Path to exported file
        """
        if not self.build123d:
            raise RuntimeError("Build123d not available for export")

        output_path = Path(output_path)

        if format.lower() == 'step':
            self.build123d.export_step(model, str(output_path))
        elif format.lower() == 'stl':
            self.build123d.export_stl(model, str(output_path))
        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {format.upper()} to {output_path}")
        return str(output_path)
