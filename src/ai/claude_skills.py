"""
Claude AI integration for CAD generation.

This module provides Claude-powered natural language understanding for
extracting intent, dimensions, and generating CAD descriptions.
"""

import json
from typing import Dict, List, Optional
import anthropic

from .prompt_templates import (
    DIMENSION_EXTRACTION_TEMPLATE,
    CLARIFICATION_TEMPLATE,
    format_prompt,
    get_template_for_agent
)
from .dimension_extractor import DimensionExtractor


class ClaudeSkills:
    """
    Claude AI integration for natural language CAD generation.

    Provides methods for:
    - Extracting intent and dimensions from natural language
    - Generating CAD descriptions
    - Clarifying ambiguous requests
    """

    def __init__(self, api_key: str):
        """
        Initialize Claude integration.

        Args:
            api_key: Anthropic API key for Claude access

        Raises:
            ValueError: If api_key is empty or invalid
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.dimension_extractor = DimensionExtractor()
        self.model = "claude-3-5-sonnet-20241022"  # Latest Claude model

    def extract_intent_and_dimensions(self, prompt: str) -> Dict:
        """
        Extract intent and dimensional information from a natural language prompt.

        Uses Claude with structured output to identify:
        - Object type (box, cylinder, sphere, custom shape, etc.)
        - Dimensions with units
        - Material specifications
        - Design constraints

        Args:
            prompt: Natural language CAD request

        Returns:
            Dictionary with extracted information:
            {
                'type': str (object type),
                'dimensions': dict (dimensional parameters),
                'unit': str (measurement unit),
                'materials': str (material specification),
                'constraints': list (design constraints),
                'confidence': float (0-1, extraction confidence),
                'ambiguities': list (unclear aspects)
            }

        Example:
            >>> skills = ClaudeSkills(api_key="...")
            >>> result = skills.extract_intent_and_dimensions("Create a 10cm x 5cm x 3cm box")
            >>> result['type']  # Returns 'box'
            >>> result['dimensions']  # Returns {'length': 0.1, 'width': 0.05, 'height': 0.03}
        """
        # First try to extract dimensions using regex
        basic_dims = self.dimension_extractor.parse_dimensions(prompt)

        # Prepare the extraction prompt
        extraction_prompt = format_prompt(
            DIMENSION_EXTRACTION_TEMPLATE,
            {'prompt': prompt}
        )

        # Define the JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "object_type": {
                    "type": "string",
                    "description": "Type of CAD object (box, cylinder, sphere, cone, custom, etc.)"
                },
                "dimensions": {
                    "type": "object",
                    "description": "Dimensional parameters",
                    "properties": {
                        "length": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                        "diameter": {"type": "number"},
                        "radius": {"type": "number"},
                        "thickness": {"type": "number"}
                    }
                },
                "unit": {
                    "type": "string",
                    "description": "Measurement unit (mm, cm, m, in, ft)"
                },
                "materials": {
                    "type": "string",
                    "description": "Material specification"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Design constraints or requirements"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score 0-1"
                },
                "ambiguities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Unclear or missing information"
                }
            },
            "required": ["object_type", "unit", "confidence"]
        }

        try:
            # Call Claude with structured output
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent extraction
                system=(
                    "You are a CAD expert that extracts structured information from "
                    "natural language descriptions. Always return valid JSON matching "
                    "the specified schema. Be conservative with confidence scores."
                )
            )

            # Parse the response
            response_text = response.content[0].text

            # Try to extract JSON from the response
            json_match = None
            if '```json' in response_text:
                # Extract JSON from markdown code block
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_match = response_text[start:end].strip()
            elif '{' in response_text:
                # Try to find JSON object
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_match = response_text[start:end]

            if json_match:
                extracted_data = json.loads(json_match)
            else:
                # Fallback if JSON parsing fails
                extracted_data = {
                    'object_type': 'custom',
                    'unit': basic_dims.get('original_unit', 'mm'),
                    'confidence': 0.3,
                    'ambiguities': ['Failed to parse Claude response']
                }

            # Merge with regex-extracted dimensions
            if 'dimensions' not in extracted_data:
                extracted_data['dimensions'] = {}

            # Convert dimensions to meters if we have them from regex
            for key in ['length', 'width', 'height', 'diameter', 'radius']:
                if key in basic_dims and key not in extracted_data['dimensions']:
                    extracted_data['dimensions'][key] = basic_dims[key]

            # Ensure required fields
            if 'materials' not in extracted_data:
                extracted_data['materials'] = 'default'
            if 'constraints' not in extracted_data:
                extracted_data['constraints'] = []
            if 'ambiguities' not in extracted_data:
                extracted_data['ambiguities'] = []

            # Rename object_type to type for consistency
            if 'object_type' in extracted_data:
                extracted_data['type'] = extracted_data.pop('object_type')

            return extracted_data

        except Exception as e:
            # Fallback to basic extraction on error
            return {
                'type': 'custom',
                'dimensions': basic_dims,
                'unit': basic_dims.get('original_unit', 'mm'),
                'materials': 'default',
                'constraints': [],
                'confidence': 0.5,
                'ambiguities': [f'Error during Claude extraction: {str(e)}']
            }

    def generate_cad_description(self, prompt: str, agent: str = 'build123d') -> str:
        """
        Generate a detailed CAD description or code for the specified agent.

        Args:
            prompt: Natural language CAD request
            agent: Target CAD agent ('zoo_kcl', 'adam_nl', 'build123d')

        Returns:
            Generated CAD description or code

        Example:
            >>> skills = ClaudeSkills(api_key="...")
            >>> code = skills.generate_cad_description(
            ...     "Create a cylinder with 5cm diameter and 10cm height",
            ...     agent='build123d'
            ... )
        """
        # First extract intent and dimensions
        extracted = self.extract_intent_and_dimensions(prompt)

        # Get the appropriate template
        template = get_template_for_agent(agent)

        # Format dimensions as string
        dims_str = ', '.join(f"{k}: {v}" for k, v in extracted['dimensions'].items())

        # Format the prompt
        formatted_prompt = format_prompt(template, {
            'description': prompt,
            'object_type': extracted['type'],
            'dimensions': dims_str,
            'unit': extracted['unit'],
            'materials': extracted['materials'],
            'constraints': ', '.join(extracted['constraints']) if extracted['constraints'] else 'none'
        })

        try:
            # Generate CAD code/description
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                temperature=0.4,  # Slightly higher for more creative code generation
                system=(
                    f"You are an expert in {agent} CAD generation. "
                    "Generate clean, well-commented, production-ready code. "
                    "Follow best practices and ensure dimensional accuracy."
                )
            )

            return response.content[0].text

        except Exception as e:
            return f"# Error generating CAD description: {str(e)}\n# Original prompt: {prompt}"

    def clarify_ambiguity(self, prompt: str, missing_params: List[str]) -> str:
        """
        Generate clarifying questions for ambiguous or incomplete requests.

        Args:
            prompt: Original natural language request
            missing_params: List of missing or unclear parameters

        Returns:
            String containing clarifying questions

        Example:
            >>> skills = ClaudeSkills(api_key="...")
            >>> questions = skills.clarify_ambiguity(
            ...     "Create a box",
            ...     missing_params=['length', 'width', 'height', 'unit']
            ... )
            >>> print(questions)
            # What are the dimensions of the box?
            # What unit of measurement should I use?
            # ...
        """
        # Format the clarification prompt
        clarification_prompt = format_prompt(
            CLARIFICATION_TEMPLATE,
            {
                'prompt': prompt,
                'missing_params': missing_params
            }
        )

        try:
            # Generate clarifying questions
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": clarification_prompt
                    }
                ],
                temperature=0.5,
                system=(
                    "You are a helpful CAD assistant. Generate clear, specific "
                    "questions to gather missing information. Be friendly but technical."
                )
            )

            return response.content[0].text

        except Exception as e:
            # Fallback to basic questions
            questions = []
            for param in missing_params:
                questions.append(f"- What is the {param}?")

            return "I need some additional information:\n" + "\n".join(questions)

    def validate_and_extract(self, prompt: str) -> Dict:
        """
        Extract dimensions and validate them in one step.

        Args:
            prompt: Natural language CAD request

        Returns:
            Dictionary with extracted and validated dimensions, plus validation results

        Example:
            >>> result = skills.validate_and_extract("10cm box")
            >>> result['valid']  # True/False
            >>> result['suggestions']  # List of suggestions
        """
        extracted = self.extract_intent_and_dimensions(prompt)
        is_valid = self.dimension_extractor.validate_dimensions(extracted['dimensions'])
        suggestions = self.dimension_extractor.suggest_corrections(extracted['dimensions'])

        return {
            **extracted,
            'valid': is_valid,
            'suggestions': suggestions
        }
