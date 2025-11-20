# GenAI-CAD-CFD-Studio

🚀 Universal AI-Powered CAD & CFD Platform | Democratizing 3D Design & Simulation | Natural Language → Parametric Models | Build123d + Zoo.dev + Adam.new + OpenFOAM | Solar PV, Test Chambers, Digital Twins & More

## 🎯 Features

### ⚙️ Agent Configuration System (NEW!)

Secure, user-friendly interface for managing AI agents and API keys:

- 🔐 **Encrypted API Key Storage** - Fernet symmetric encryption for all API keys
- 🤖 **Multi-Agent Support** - Zoo.dev, Adam.new, Anthropic Claude, SimScale, and custom agents
- 📊 **Usage Monitoring** - Track API calls, credits, and rate limits
- 🚦 **Health Dashboard** - Real-time agent status and connection testing
- ➕ **Custom Integration** - Add your own AI agents with custom endpoints
- ⚙️ **Flexible Configuration** - Rate limits, timeouts, models, and prompt templates

[📖 Full Agent Configuration Documentation](docs/AGENT_CONFIG.md)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio.git
cd GenAI-CAD-CFD-Studio

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent Configuration UI

```bash
streamlit run src/ui/agent_config.py
```

### Configure Your First Agent

1. Open the Agent Configuration UI
2. Navigate to "API Key Management"
3. Enter your API key for Zoo.dev, Adam.new, or Claude
4. Click "Save" and "Test Connection"
5. Start using AI-powered CAD generation!

## 📁 Project Structure

```
GenAI-CAD-CFD-Studio/
├── src/
│   ├── ui/
│   │   └── agent_config.py         # Agent Configuration UI
│   ├── utils/
│   │   └── api_key_manager.py      # Secure key management
│   └── agents/
│       └── agent_registry.py        # Agent registration system
├── tests/
│   └── test_agent_config.py         # Comprehensive tests
├── docs/
│   └── AGENT_CONFIG.md              # Detailed documentation
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 🔐 Security

- All API keys are encrypted using industry-standard Fernet encryption
- Keys stored locally in `~/.streamlit/secrets.json` (encrypted)
- Never logged or transmitted in plain text
- Restrictive file permissions (owner read/write only)
- See [Security Best Practices](docs/AGENT_CONFIG.md#security-best-practices)

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_agent_config.py -v

# Run with coverage
pytest tests/test_agent_config.py --cov=src --cov-report=html
```

## 📚 Documentation

- [Agent Configuration Guide](docs/AGENT_CONFIG.md) - Complete setup and usage guide
- API Reference - See documentation for SecureKeyManager and AgentRegistry
- Troubleshooting - Common issues and solutions

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Update documentation
5. Submit a pull request

## 📄 License

See LICENSE file for details.
