"""
Prompt templates for different CAD generation agents.

This module provides template library for various CAD generation backends
including Zoo KCL, ADAM Natural Language, and Build123D Python.
"""

# Template for Zoo KCL (Sketch-based CAD Language)
ZOO_KCL_TEMPLATE = """Generate a KCL script for the following CAD model:

Description: {description}
Object Type: {object_type}
Dimensions: {dimensions}
Units: {unit}
Materials: {materials}
Constraints: {constraints}

Generate valid KCL code that creates this 3D model with proper sketches and extrusions.
Include all necessary transformations and ensure dimensional accuracy.
"""

# Template for ADAM Natural Language CAD
ADAM_NL_TEMPLATE = """Create a CAD model using natural language instructions:

Model: {object_type}
Requirements:
- Dimensions: {dimensions}
- Units: {unit}
- Materials: {materials}
- Design Constraints: {constraints}

Additional Context: {description}

Generate clear, step-by-step instructions for creating this model that can be
interpreted by the ADAM natural language CAD system.
"""

# Template for Build123D Python CAD
BUILD123D_PYTHON_TEMPLATE = """Write a Build123D Python script to create:

Object: {object_type}
Dimensions: {dimensions}
Unit: {unit}
Materials: {materials}
Constraints: {constraints}

Description: {description}

Generate Python code using Build123D library that creates this 3D model.
Include proper imports, dimensional parameters, and export to STEP format.
Use best practices for parametric modeling.
"""

# Additional specialized templates
DIMENSION_EXTRACTION_TEMPLATE = """Extract dimensional information from the following CAD request:

"{prompt}"

Identify and extract:
1. Object type (e.g., box, cylinder, sphere, custom shape)
2. All dimensions with their units
3. Material specifications
4. Design constraints or special requirements
5. Any ambiguities that need clarification

Return structured data in JSON format.
"""

CLARIFICATION_TEMPLATE = """The following CAD request is missing some required parameters:

Original Request: "{prompt}"
Missing Parameters: {missing_params}

Generate 3-5 clear, specific questions to gather the missing information.
Questions should be technical but user-friendly.
"""


def format_prompt(template: str, params: dict) -> str:
    """
    Format a prompt template with given parameters.

    Args:
        template: Template string with {placeholder} markers
        params: Dictionary of parameter values

    Returns:
        Formatted prompt string with placeholders replaced

    Example:
        >>> params = {'object_type': 'box', 'dimensions': '10x5x3', 'unit': 'cm'}
        >>> formatted = format_prompt(ZOO_KCL_TEMPLATE, params)
    """
    # Provide defaults for missing parameters
    defaults = {
        'description': 'No additional description provided',
        'object_type': 'custom shape',
        'dimensions': 'not specified',
        'unit': 'mm',
        'materials': 'default material',
        'constraints': 'none',
        'prompt': '',
        'missing_params': []
    }

    # Merge provided params with defaults
    full_params = {**defaults, **params}

    # Format missing_params if it's a list
    if isinstance(full_params['missing_params'], list):
        full_params['missing_params'] = ', '.join(full_params['missing_params'])

    try:
        return template.format(**full_params)
    except KeyError as e:
        raise ValueError(f"Missing required parameter for template: {e}")


def optimize_for_agent(prompt: str, agent: str) -> str:
    """
    Optimize a prompt for a specific CAD generation agent.

    Args:
        prompt: Original user prompt
        agent: Target agent ('zoo_kcl', 'adam_nl', 'build123d')

    Returns:
        Optimized prompt string for the specified agent

    Raises:
        ValueError: If agent type is not recognized

    Example:
        >>> optimized = optimize_for_agent("Create a 10cm box", "build123d")
    """
    agent = agent.lower().strip()

    agent_instructions = {
        'zoo_kcl': (
            "Focus on sketch-based operations and extrusions. "
            "KCL uses a functional programming approach with sketch planes and transformations."
        ),
        'adam_nl': (
            "Use clear, natural language descriptions. "
            "Break down complex operations into simple, sequential steps."
        ),
        'build123d': (
            "Generate parametric Python code using Build123D API. "
            "Focus on object-oriented design and proper method chaining."
        )
    }

    if agent not in agent_instructions:
        raise ValueError(
            f"Unknown agent type: {agent}. "
            f"Supported agents: {', '.join(agent_instructions.keys())}"
        )

    instruction = agent_instructions[agent]

    optimized = f"{instruction}\n\nUser Request: {prompt}\n\nGenerate appropriate code/instructions."

    return optimized


def get_template_for_agent(agent: str) -> str:
    """
    Get the appropriate template for a specific agent.

    Args:
        agent: Agent type ('zoo_kcl', 'adam_nl', 'build123d')

    Returns:
        Template string for the specified agent

    Raises:
        ValueError: If agent type is not recognized
    """
    templates = {
        'zoo_kcl': ZOO_KCL_TEMPLATE,
        'adam_nl': ADAM_NL_TEMPLATE,
        'build123d': BUILD123D_PYTHON_TEMPLATE
    }

    agent = agent.lower().strip()
    if agent not in templates:
        raise ValueError(
            f"Unknown agent type: {agent}. "
            f"Supported agents: {', '.join(templates.keys())}"
        )

    return templates[agent]
