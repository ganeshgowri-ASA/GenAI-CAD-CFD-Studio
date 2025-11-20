# Agent Configuration System

## Overview

The Agent Configuration system provides a secure, user-friendly interface for managing API keys and configuring AI agents for the GenAI CAD CFD Studio.

## Features

### 🔐 Secure API Key Management
- **Encryption**: All API keys are encrypted using Fernet symmetric encryption (cryptography library)
- **Local Storage**: Keys are stored locally in `~/.streamlit/secrets.json` (encrypted)
- **Never Logged**: Keys are never logged or transmitted in plain text
- **Easy Management**: Simple UI for adding, viewing, testing, and deleting API keys

### ⚙️ Agent Configuration
- **Multiple Agents**: Support for Zoo.dev, Adam.new, Anthropic Claude, SimScale, and custom agents
- **Dynamic Loading**: Agents are registered and loaded dynamically
- **Customization**: Configure rate limits, timeouts, models, and prompt templates per agent
- **Enable/Disable**: Easily enable or disable agents without deleting configurations

### 📊 Usage Monitoring
- **API Call Tracking**: Monitor API calls per service
- **Credit Usage**: Track credit consumption
- **Rate Limit Status**: Visual indicators for rate limit utilization
- **Error Logging**: Comprehensive error logs with timestamps

### 🚦 Agent Status Dashboard
- **Health Checks**: Real-time status indicators (Green/Red)
- **Connection Testing**: Test API connections with one click
- **Last Call Tracking**: See when each service was last used
- **Error Visibility**: View recent errors per service

### ➕ Custom Agent Integration
- **Add Custom Agents**: Register your own AI agents
- **Flexible Configuration**: Set API endpoints, auth methods, and prompt templates
- **Factory Pattern**: Support for custom agent instantiation logic

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Required Dependencies

- `streamlit>=1.28.0` - Web UI framework
- `cryptography>=41.0.0` - Encryption for API keys
- `python-dotenv>=1.0.0` - Environment variable management
- `requests>=2.31.0` - HTTP requests for API testing

## Usage

### Running the Agent Configuration UI

#### Standalone Mode

```bash
streamlit run src/ui/agent_config.py
```

#### As Part of Main Application

```python
from src.ui.agent_config import render_agent_config_page

# In your Streamlit app
render_agent_config_page()
```

### Managing API Keys Programmatically

```python
from src.utils.api_key_manager import get_key_manager

# Get the key manager instance
key_manager = get_key_manager()

# Store an API key
key_manager.store_key('zoo_dev', 'your-api-key-here')

# Retrieve an API key
api_key = key_manager.get_key('zoo_dev')

# Delete an API key
key_manager.delete_key('zoo_dev')

# List all services with stored keys
services = key_manager.list_services()
```

### Managing Agents Programmatically

```python
from src.agents.agent_registry import get_agent_registry, AgentConfig

# Get the registry instance
registry = get_agent_registry()

# Create a custom agent configuration
config = AgentConfig(
    name='my_agent',
    display_name='My Custom Agent',
    agent_type='custom',
    api_endpoint='https://api.example.com/v1',
    auth_method='api_key',
    rate_limit=100,
    timeout=30
)

# Register the agent
registry.register_agent(config)

# Get agent configuration
agent_config = registry.get_config('my_agent')

# Update agent configuration
registry.update_config('my_agent', {
    'rate_limit': 200,
    'timeout': 60
})

# List all agents
all_agents = registry.list_agents()
enabled_agents = registry.list_agents(enabled_only=True)

# Get agent instance
agent = registry.get_agent('my_agent')
```

## Architecture

### Components

#### 1. SecureKeyManager (`src/utils/api_key_manager.py`)
- Handles encryption/decryption of API keys
- Manages key storage and retrieval
- Uses Fernet symmetric encryption
- Stores keys in `~/.streamlit/secrets.json`

#### 2. AgentRegistry (`src/agents/agent_registry.py`)
- Manages agent configurations
- Supports dynamic agent registration
- Handles agent lifecycle (enable/disable)
- Stores configurations in `~/.streamlit/agents.json`

#### 3. Agent Configuration UI (`src/ui/agent_config.py`)
- Streamlit-based user interface
- API key management interface
- Agent configuration forms
- Usage statistics and monitoring
- Agent status dashboard
- Custom agent integration

### Data Flow

```
User Input → UI Component → Key Manager/Registry → Encrypted Storage
                                                           ↓
                                                    Local JSON Files
```

### Security Model

1. **Encryption Key Generation**
   - Unique encryption key generated per installation
   - Stored in `~/.streamlit/.key` with 0o600 permissions

2. **Key Encryption**
   - API keys encrypted using Fernet (symmetric encryption)
   - Based on cryptography library (industry standard)

3. **Storage**
   - Encrypted keys stored in JSON format
   - Files have restrictive permissions (owner read/write only)

4. **Decryption**
   - Keys decrypted only when needed
   - Never logged or transmitted in plain text

## Testing

### Run All Tests

```bash
pytest tests/test_agent_config.py -v
```

### Run Specific Test Classes

```bash
# Test key manager
pytest tests/test_agent_config.py::TestSecureKeyManager -v

# Test agent registry
pytest tests/test_agent_config.py::TestAgentRegistry -v

# Test API connection mocking
pytest tests/test_agent_config.py::TestAPIConnectionMocking -v
```

### Test Coverage

```bash
pytest tests/test_agent_config.py --cov=src --cov-report=html
```

## Built-in Agents

The system comes pre-configured with the following agents:

1. **Build123d Agent**
   - Type: `build123d`
   - Purpose: 3D CAD model generation
   - Endpoint: `https://api.build123d.io/v1`

2. **Zoo.dev Agent**
   - Type: `zoo`
   - Purpose: CAD generation service
   - Endpoint: `https://api.zoo.dev/v1`

3. **Adam.new Agent**
   - Type: `adam`
   - Purpose: Engineering platform
   - Endpoint: `https://api.adam.new/v1`

4. **Anthropic Claude**
   - Type: `claude`
   - Purpose: AI assistant
   - Endpoint: `https://api.anthropic.com/v1`
   - Models: claude-3-5-sonnet, claude-3-opus, claude-3-haiku

5. **SimScale CFD**
   - Type: `simscale`
   - Purpose: CFD simulations
   - Endpoint: `https://api.simscale.com/v1`

## Custom Agent Integration

### Adding a Custom Agent via UI

1. Navigate to "Custom Agent Integration" section
2. Click "Add New Agent" expander
3. Fill in the form:
   - Agent Name (unique identifier)
   - Display Name (friendly name)
   - Agent Type
   - API Endpoint
   - Authentication Method
   - Model Name (optional)
   - Rate Limit and Timeout
   - Custom Prompt Template
4. Click "Add Custom Agent"

### Adding a Custom Agent via Code

```python
from src.agents.agent_registry import get_agent_registry, AgentConfig

registry = get_agent_registry()

config = AgentConfig(
    name='openai_gpt4',
    display_name='OpenAI GPT-4',
    agent_type='custom',
    api_endpoint='https://api.openai.com/v1',
    auth_method='bearer',
    model_name='gpt-4',
    rate_limit=60,
    timeout=30,
    custom_prompt_template='Generate: {prompt}'
)

registry.register_agent(config)
```

### Custom Agent Factory

For agents requiring custom initialization:

```python
def my_agent_factory(config):
    # Custom initialization logic
    return MyAgentClass(config)

registry.register_agent(config, factory=my_agent_factory)
```

## Security Best Practices

1. **Never Commit Secrets**
   - `.streamlit/secrets.json` is in `.gitignore`
   - `.streamlit/.key` is in `.gitignore`
   - Never commit API keys to version control

2. **File Permissions**
   - Secret files are automatically set to 0o600 (owner read/write only)
   - Encryption key has restrictive permissions

3. **Environment Variables**
   - Can use `.env` file for additional configuration
   - Never commit `.env` to version control

4. **API Key Rotation**
   - Regularly rotate API keys
   - Update keys through the UI or programmatically

5. **Access Control**
   - Restrict access to the application
   - Use authentication if deploying publicly

## Troubleshooting

### Issue: API Connection Test Fails

**Solution:**
1. Verify API key is correct
2. Check network connectivity
3. Verify API endpoint is correct
4. Check API service status
5. Review error logs in the UI

### Issue: Keys Not Persisting

**Solution:**
1. Check file permissions on `~/.streamlit/`
2. Verify write access to home directory
3. Check disk space
4. Review error logs

### Issue: Encryption/Decryption Errors

**Solution:**
1. Verify `.streamlit/.key` exists
2. Don't delete or modify the encryption key file
3. If key file is lost, you'll need to re-enter all API keys

### Issue: Import Errors

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Development

### Project Structure

```
GenAI-CAD-CFD-Studio/
├── src/
│   ├── ui/
│   │   ├── __init__.py
│   │   └── agent_config.py         # Main UI component
│   ├── utils/
│   │   ├── __init__.py
│   │   └── api_key_manager.py      # Key encryption/storage
│   └── agents/
│       ├── __init__.py
│       └── agent_registry.py        # Agent management
├── tests/
│   ├── __init__.py
│   └── test_agent_config.py         # Comprehensive tests
├── docs/
│   └── AGENT_CONFIG.md              # This file
├── .streamlit/                       # Created at runtime (gitignored)
│   ├── secrets.json                 # Encrypted API keys
│   ├── agents.json                  # Agent configurations
│   └── .key                         # Encryption key
├── requirements.txt                  # Python dependencies
└── .gitignore                       # Git ignore rules
```

### Contributing

1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Never commit secrets or keys
5. Use type hints
6. Add docstrings to all functions

## API Reference

### SecureKeyManager

```python
class SecureKeyManager:
    def __init__(self, storage_path: Optional[str] = None)
    def encrypt_key(self, key: str) -> str
    def decrypt_key(self, encrypted_str: str) -> str
    def store_key(self, service: str, key: str, metadata: Optional[Dict] = None)
    def get_key(self, service: str) -> Optional[str]
    def delete_key(self, service: str) -> bool
    def list_services(self) -> list
    def get_metadata(self, service: str) -> Optional[Dict]
    def update_metadata(self, service: str, metadata: Dict)
    def key_exists(self, service: str) -> bool
```

### AgentRegistry

```python
class AgentRegistry:
    def __init__(self, config_path: Optional[str] = None)
    def register_agent(self, config: AgentConfig, factory: Optional[Callable] = None)
    def unregister_agent(self, name: str) -> bool
    def get_config(self, name: str) -> Optional[AgentConfig]
    def update_config(self, name: str, updates: Dict[str, Any]) -> bool
    def get_agent(self, name: str) -> Optional[Any]
    def list_agents(self, enabled_only: bool = False) -> List[str]
    def list_agent_configs(self, enabled_only: bool = False) -> List[AgentConfig]
    def enable_agent(self, name: str) -> bool
    def disable_agent(self, name: str) -> bool
    def get_agents_by_type(self, agent_type: str) -> List[AgentConfig]
```

### AgentConfig

```python
@dataclass
class AgentConfig:
    name: str
    display_name: str
    agent_type: str
    api_endpoint: Optional[str] = None
    auth_method: str = 'api_key'
    model_name: Optional[str] = None
    default_params: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 60
    timeout: int = 30
    custom_prompt_template: Optional[str] = None
    enabled: bool = True
```

## License

See LICENSE file in the project root.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review test cases for usage examples
