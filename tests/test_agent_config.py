"""
Comprehensive tests for Agent Configuration system
Tests API key management, agent registry, and integration
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_key_manager import SecureKeyManager, get_key_manager
from src.agents.agent_registry import AgentRegistry, AgentConfig, get_agent_registry


class TestSecureKeyManager:
    """Tests for SecureKeyManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def key_manager(self, temp_dir):
        """Create a SecureKeyManager instance for testing."""
        storage_path = Path(temp_dir) / "test_secrets.json"
        return SecureKeyManager(storage_path=str(storage_path))

    def test_initialization(self, key_manager):
        """Test SecureKeyManager initialization."""
        assert key_manager is not None
        assert key_manager._fernet is not None
        assert key_manager._cipher_key is not None

    def test_encrypt_key(self, key_manager):
        """Test key encryption."""
        test_key = "test-api-key-12345"
        encrypted = key_manager.encrypt_key(test_key)

        assert encrypted is not None
        assert encrypted != test_key
        assert isinstance(encrypted, str)

    def test_decrypt_key(self, key_manager):
        """Test key decryption."""
        test_key = "test-api-key-12345"
        encrypted = key_manager.encrypt_key(test_key)
        decrypted = key_manager.decrypt_key(encrypted)

        assert decrypted == test_key

    def test_encrypt_empty_key_raises_error(self, key_manager):
        """Test that encrypting empty key raises error."""
        with pytest.raises(ValueError, match="Key cannot be empty"):
            key_manager.encrypt_key("")

    def test_decrypt_empty_string_raises_error(self, key_manager):
        """Test that decrypting empty string raises error."""
        with pytest.raises(ValueError, match="Encrypted string cannot be empty"):
            key_manager.decrypt_key("")

    def test_decrypt_invalid_string_raises_error(self, key_manager):
        """Test that decrypting invalid string raises error."""
        with pytest.raises(ValueError, match="Failed to decrypt key"):
            key_manager.decrypt_key("invalid-encrypted-string")

    def test_store_key(self, key_manager):
        """Test storing an API key."""
        service = "test_service"
        api_key = "test-api-key-12345"

        key_manager.store_key(service, api_key)

        # Verify key is in store
        assert service in key_manager._keys_store
        assert 'key' in key_manager._keys_store[service]
        assert 'updated_at' in key_manager._keys_store[service]

    def test_store_key_with_metadata(self, key_manager):
        """Test storing key with metadata."""
        service = "test_service"
        api_key = "test-api-key-12345"
        metadata = {"description": "Test API", "owner": "test_user"}

        key_manager.store_key(service, api_key, metadata)

        assert key_manager._keys_store[service]['metadata'] == metadata

    def test_get_key(self, key_manager):
        """Test retrieving an API key."""
        service = "test_service"
        api_key = "test-api-key-12345"

        key_manager.store_key(service, api_key)
        retrieved_key = key_manager.get_key(service)

        assert retrieved_key == api_key

    def test_get_nonexistent_key(self, key_manager):
        """Test retrieving a non-existent key returns None."""
        result = key_manager.get_key("nonexistent_service")
        assert result is None

    def test_delete_key(self, key_manager):
        """Test deleting an API key."""
        service = "test_service"
        api_key = "test-api-key-12345"

        key_manager.store_key(service, api_key)
        assert key_manager.key_exists(service)

        result = key_manager.delete_key(service)
        assert result is True
        assert not key_manager.key_exists(service)

    def test_delete_nonexistent_key(self, key_manager):
        """Test deleting non-existent key returns False."""
        result = key_manager.delete_key("nonexistent_service")
        assert result is False

    def test_list_services(self, key_manager):
        """Test listing all services."""
        services = ["service1", "service2", "service3"]
        for service in services:
            key_manager.store_key(service, f"key-{service}")

        listed = key_manager.list_services()
        assert set(listed) == set(services)

    def test_get_metadata(self, key_manager):
        """Test getting metadata."""
        service = "test_service"
        metadata = {"description": "Test", "version": "1.0"}

        key_manager.store_key(service, "test-key", metadata)
        retrieved_metadata = key_manager.get_metadata(service)

        assert retrieved_metadata == metadata

    def test_update_metadata(self, key_manager):
        """Test updating metadata."""
        service = "test_service"
        initial_metadata = {"description": "Initial"}
        updated_metadata = {"description": "Updated", "new_field": "value"}

        key_manager.store_key(service, "test-key", initial_metadata)
        key_manager.update_metadata(service, updated_metadata)

        assert key_manager.get_metadata(service) == updated_metadata

    def test_key_exists(self, key_manager):
        """Test checking if key exists."""
        service = "test_service"

        assert not key_manager.key_exists(service)

        key_manager.store_key(service, "test-key")
        assert key_manager.key_exists(service)

    def test_persistence(self, temp_dir):
        """Test that keys persist across instances."""
        storage_path = Path(temp_dir) / "test_secrets.json"

        # Create first instance and store key
        manager1 = SecureKeyManager(storage_path=str(storage_path))
        manager1.store_key("test_service", "test-key-12345")

        # Create second instance and retrieve key
        manager2 = SecureKeyManager(storage_path=str(storage_path))
        retrieved_key = manager2.get_key("test_service")

        assert retrieved_key == "test-key-12345"


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def registry(self, temp_dir):
        """Create an AgentRegistry instance for testing."""
        config_path = Path(temp_dir) / "test_agents.json"
        return AgentRegistry(config_path=str(config_path))

    def test_initialization(self, registry):
        """Test AgentRegistry initialization."""
        assert registry is not None
        # Should have built-in agents
        agents = registry.list_agents()
        assert len(agents) > 0
        assert 'build123d' in agents
        assert 'zoo_dev' in agents

    def test_register_agent(self, registry):
        """Test registering a new agent."""
        config = AgentConfig(
            name='test_agent',
            display_name='Test Agent',
            agent_type='custom',
            api_endpoint='https://api.test.com/v1',
            rate_limit=50
        )

        registry.register_agent(config)

        assert 'test_agent' in registry.list_agents()
        retrieved_config = registry.get_config('test_agent')
        assert retrieved_config.name == 'test_agent'
        assert retrieved_config.display_name == 'Test Agent'

    def test_register_agent_with_factory(self, registry):
        """Test registering agent with factory function."""
        config = AgentConfig(
            name='test_agent',
            display_name='Test Agent',
            agent_type='custom'
        )

        factory = Mock(return_value="test_instance")
        registry.register_agent(config, factory)

        instance = registry.get_agent('test_agent')
        assert instance == "test_instance"
        factory.assert_called_once()

    def test_unregister_agent(self, registry):
        """Test unregistering an agent."""
        config = AgentConfig(
            name='test_agent',
            display_name='Test Agent',
            agent_type='custom'
        )

        registry.register_agent(config)
        assert 'test_agent' in registry.list_agents()

        result = registry.unregister_agent('test_agent')
        assert result is True
        assert 'test_agent' not in registry.list_agents()

    def test_unregister_nonexistent_agent(self, registry):
        """Test unregistering non-existent agent returns False."""
        result = registry.unregister_agent('nonexistent_agent')
        assert result is False

    def test_get_config(self, registry):
        """Test getting agent configuration."""
        config = registry.get_config('build123d')
        assert config is not None
        assert config.name == 'build123d'
        assert config.agent_type == 'build123d'

    def test_get_nonexistent_config(self, registry):
        """Test getting non-existent config returns None."""
        config = registry.get_config('nonexistent_agent')
        assert config is None

    def test_update_config(self, registry):
        """Test updating agent configuration."""
        updates = {
            'rate_limit': 100,
            'timeout': 60,
            'enabled': False
        }

        result = registry.update_config('build123d', updates)
        assert result is True

        config = registry.get_config('build123d')
        assert config.rate_limit == 100
        assert config.timeout == 60
        assert config.enabled is False

    def test_update_nonexistent_config(self, registry):
        """Test updating non-existent config returns False."""
        result = registry.update_config('nonexistent', {'rate_limit': 100})
        assert result is False

    def test_list_agents(self, registry):
        """Test listing all agents."""
        agents = registry.list_agents()
        assert len(agents) > 0
        assert isinstance(agents, list)

    def test_list_enabled_agents_only(self, registry):
        """Test listing only enabled agents."""
        # Disable one agent
        registry.update_config('build123d', {'enabled': False})

        all_agents = registry.list_agents(enabled_only=False)
        enabled_agents = registry.list_agents(enabled_only=True)

        assert len(enabled_agents) < len(all_agents)
        assert 'build123d' not in enabled_agents

    def test_list_agent_configs(self, registry):
        """Test listing agent configurations."""
        configs = registry.list_agent_configs()
        assert len(configs) > 0
        assert all(isinstance(c, AgentConfig) for c in configs)

    def test_enable_agent(self, registry):
        """Test enabling an agent."""
        # First disable it
        registry.update_config('build123d', {'enabled': False})
        assert not registry.get_config('build123d').enabled

        # Then enable it
        result = registry.enable_agent('build123d')
        assert result is True
        assert registry.get_config('build123d').enabled

    def test_disable_agent(self, registry):
        """Test disabling an agent."""
        result = registry.disable_agent('build123d')
        assert result is True
        assert not registry.get_config('build123d').enabled

    def test_get_agents_by_type(self, registry):
        """Test getting agents by type."""
        build_agents = registry.get_agents_by_type('build123d')
        assert len(build_agents) >= 1
        assert all(c.agent_type == 'build123d' for c in build_agents)

        claude_agents = registry.get_agents_by_type('claude')
        assert any(c.name == 'anthropic_claude' for c in claude_agents)

    def test_persistence(self, temp_dir):
        """Test that agent configs persist across instances."""
        config_path = Path(temp_dir) / "test_agents.json"

        # Create first instance and register agent
        registry1 = AgentRegistry(config_path=str(config_path))
        test_config = AgentConfig(
            name='persistent_agent',
            display_name='Persistent Agent',
            agent_type='custom'
        )
        registry1.register_agent(test_config)

        # Create second instance and check agent exists
        registry2 = AgentRegistry(config_path=str(config_path))
        config = registry2.get_config('persistent_agent')

        assert config is not None
        assert config.name == 'persistent_agent'
        assert config.display_name == 'Persistent Agent'


class TestAPIConnectionMocking:
    """Tests for API connection testing with mocked requests."""

    @patch('src.ui.agent_config.requests.get')
    def test_api_connection_success(self, mock_get):
        """Test successful API connection."""
        from src.ui.agent_config import test_api_connection

        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        success, message = test_api_connection(
            'zoo_dev',
            'test-key',
            'https://api.zoo.dev/v1'
        )

        assert success is True
        assert "successful" in message.lower()

    @patch('src.ui.agent_config.requests.get')
    def test_api_connection_failure(self, mock_get):
        """Test failed API connection."""
        from src.ui.agent_config import test_api_connection

        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        success, message = test_api_connection(
            'zoo_dev',
            'invalid-key',
            'https://api.zoo.dev/v1'
        )

        assert success is False
        assert "401" in message

    @patch('src.ui.agent_config.requests.get')
    def test_api_connection_timeout(self, mock_get):
        """Test API connection timeout."""
        from src.ui.agent_config import test_api_connection
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        success, message = test_api_connection(
            'zoo_dev',
            'test-key',
            'https://api.zoo.dev/v1'
        )

        assert success is False
        assert "timeout" in message.lower()

    @patch('src.ui.agent_config.requests.post')
    def test_anthropic_api_connection(self, mock_post):
        """Test Anthropic API connection with POST request."""
        from src.ui.agent_config import test_api_connection

        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        success, message = test_api_connection(
            'anthropic_claude',
            'test-key',
            'https://api.anthropic.com/v1'
        )

        assert success is True
        mock_post.assert_called_once()


class TestAgentConfigIntegration:
    """Integration tests for the entire agent config system."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: store key, register agent, configure."""
        # Setup
        key_storage = Path(temp_dir) / "keys.json"
        agent_storage = Path(temp_dir) / "agents.json"

        key_manager = SecureKeyManager(storage_path=str(key_storage))
        registry = AgentRegistry(config_path=str(agent_storage))

        # Store API key
        api_key = "test-api-key-12345"
        key_manager.store_key('custom_service', api_key)

        # Register custom agent
        config = AgentConfig(
            name='custom_service',
            display_name='Custom Service',
            agent_type='custom',
            api_endpoint='https://api.custom.com/v1',
            rate_limit=100,
            timeout=45
        )
        registry.register_agent(config)

        # Verify key retrieval
        retrieved_key = key_manager.get_key('custom_service')
        assert retrieved_key == api_key

        # Verify agent config
        agent_config = registry.get_config('custom_service')
        assert agent_config.name == 'custom_service'
        assert agent_config.rate_limit == 100

        # Update configuration
        registry.update_config('custom_service', {'rate_limit': 200})
        updated_config = registry.get_config('custom_service')
        assert updated_config.rate_limit == 200

        # Clean up
        key_manager.delete_key('custom_service')
        registry.unregister_agent('custom_service')

        assert not key_manager.key_exists('custom_service')
        assert 'custom_service' not in registry.list_agents()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
