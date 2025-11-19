"""
Agent Registry for GenAI CAD CFD Studio
Manages registration and loading of AI agents
"""

from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
import importlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an AI agent."""

    name: str
    display_name: str
    agent_type: str  # 'build123d', 'zoo', 'adam', 'claude', 'simscale', 'custom'
    api_endpoint: Optional[str] = None
    auth_method: str = 'api_key'  # 'api_key', 'bearer', 'oauth', 'none'
    model_name: Optional[str] = None
    default_params: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 60  # requests per minute
    timeout: int = 30  # seconds
    custom_prompt_template: Optional[str] = None
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentConfig':
        """Create from dictionary."""
        return cls(**data)


class AgentRegistry:
    """
    Registry for managing AI agents.

    Provides functionality to register, retrieve, and manage agent configurations
    and instances dynamically.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the AgentRegistry.

        Args:
            config_path: Optional path to agent configuration file
        """
        if config_path is None:
            self.config_path = Path.home() / ".streamlit" / "agents.json"
        else:
            self.config_path = Path(config_path)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Storage for agent configs and instances
        self._configs: Dict[str, AgentConfig] = {}
        self._instances: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}

        # Load configurations
        self._load_configs()

        # Register built-in agents
        self._register_builtin_agents()

    def _load_configs(self):
        """Load agent configurations from file."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                for name, config_dict in data.items():
                    self._configs[name] = AgentConfig.from_dict(config_dict)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load agent configs: {e}")

    def _save_configs(self):
        """Save agent configurations to file."""
        data = {name: config.to_dict() for name, config in self._configs.items()}

        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _register_builtin_agents(self):
        """Register built-in agent configurations."""
        builtin_agents = [
            AgentConfig(
                name='build123d',
                display_name='Build123d Agent',
                agent_type='build123d',
                api_endpoint='https://api.build123d.io/v1',
                model_name='build123d-v1',
                rate_limit=60,
                timeout=30,
                custom_prompt_template='Generate 3D CAD model: {prompt}'
            ),
            AgentConfig(
                name='zoo_dev',
                display_name='Zoo.dev Agent',
                agent_type='zoo',
                api_endpoint='https://api.zoo.dev/v1',
                model_name='zoo-cad-v1',
                rate_limit=100,
                timeout=60,
                custom_prompt_template='Zoo CAD generation: {prompt}'
            ),
            AgentConfig(
                name='adam_new',
                display_name='Adam.new Agent',
                agent_type='adam',
                api_endpoint='https://api.adam.new/v1',
                model_name='adam-engineering-v1',
                rate_limit=50,
                timeout=45,
                custom_prompt_template='Adam engineering task: {prompt}'
            ),
            AgentConfig(
                name='anthropic_claude',
                display_name='Anthropic Claude',
                agent_type='claude',
                api_endpoint='https://api.anthropic.com/v1',
                model_name='claude-3-5-sonnet-20241022',
                rate_limit=50,
                timeout=120,
                default_params={
                    'max_tokens': 4096,
                    'temperature': 0.7
                }
            ),
            AgentConfig(
                name='simscale',
                display_name='SimScale CFD',
                agent_type='simscale',
                api_endpoint='https://api.simscale.com/v1',
                rate_limit=20,
                timeout=300,
                custom_prompt_template='CFD simulation: {prompt}'
            )
        ]

        for agent_config in builtin_agents:
            if agent_config.name not in self._configs:
                self._configs[agent_config.name] = agent_config

        self._save_configs()

    def register_agent(self, config: AgentConfig, factory: Optional[Callable] = None):
        """
        Register a new agent.

        Args:
            config: Agent configuration
            factory: Optional factory function to create agent instances
        """
        config.updated_at = datetime.now().isoformat()
        self._configs[config.name] = config

        if factory:
            self._factories[config.name] = factory

        self._save_configs()
        logger.info(f"Registered agent: {config.name}")

    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.

        Args:
            name: Agent name

        Returns:
            True if agent was unregistered, False if not found
        """
        if name in self._configs:
            # Remove from all storage
            del self._configs[name]
            self._instances.pop(name, None)
            self._factories.pop(name, None)

            self._save_configs()
            logger.info(f"Unregistered agent: {name}")
            return True

        return False

    def get_config(self, name: str) -> Optional[AgentConfig]:
        """
        Get agent configuration.

        Args:
            name: Agent name

        Returns:
            AgentConfig or None if not found
        """
        return self._configs.get(name)

    def update_config(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update agent configuration.

        Args:
            name: Agent name
            updates: Dictionary of fields to update

        Returns:
            True if updated, False if agent not found
        """
        if name not in self._configs:
            return False

        config = self._configs[name]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)

        config.updated_at = datetime.now().isoformat()
        self._save_configs()

        # Invalidate cached instance
        if name in self._instances:
            del self._instances[name]

        return True

    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent instance.

        Args:
            name: Agent name

        Returns:
            Agent instance or None if not found
        """
        # Return cached instance if available
        if name in self._instances:
            return self._instances[name]

        # Get configuration
        config = self.get_config(name)
        if not config or not config.enabled:
            return None

        # Try to create instance using factory
        if name in self._factories:
            try:
                instance = self._factories[name](config)
                self._instances[name] = instance
                return instance
            except Exception as e:
                logger.error(f"Failed to create agent instance for {name}: {e}")
                return None

        # Return config if no factory available
        return config

    def list_agents(self, enabled_only: bool = False) -> List[str]:
        """
        List all registered agents.

        Args:
            enabled_only: If True, only return enabled agents

        Returns:
            List of agent names
        """
        if enabled_only:
            return [name for name, config in self._configs.items() if config.enabled]
        return list(self._configs.keys())

    def list_agent_configs(self, enabled_only: bool = False) -> List[AgentConfig]:
        """
        List all agent configurations.

        Args:
            enabled_only: If True, only return enabled agents

        Returns:
            List of AgentConfig objects
        """
        if enabled_only:
            return [config for config in self._configs.values() if config.enabled]
        return list(self._configs.values())

    def enable_agent(self, name: str) -> bool:
        """
        Enable an agent.

        Args:
            name: Agent name

        Returns:
            True if enabled, False if not found
        """
        return self.update_config(name, {'enabled': True})

    def disable_agent(self, name: str) -> bool:
        """
        Disable an agent.

        Args:
            name: Agent name

        Returns:
            True if disabled, False if not found
        """
        # Also remove cached instance
        if name in self._instances:
            del self._instances[name]

        return self.update_config(name, {'enabled': False})

    def get_agents_by_type(self, agent_type: str) -> List[AgentConfig]:
        """
        Get all agents of a specific type.

        Args:
            agent_type: Type of agent

        Returns:
            List of matching AgentConfig objects
        """
        return [
            config for config in self._configs.values()
            if config.agent_type == agent_type
        ]


# Global instance
_registry_instance = None


def get_agent_registry(config_path: Optional[str] = None) -> AgentRegistry:
    """
    Get the global AgentRegistry instance.

    Args:
        config_path: Optional custom config path

    Returns:
        AgentRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = AgentRegistry(config_path)

    return _registry_instance
