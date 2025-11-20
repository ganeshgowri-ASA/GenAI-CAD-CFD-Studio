# GenAI-CAD-CFD-Studio
🚀 Universal AI-Powered CAD &amp; CFD Platform | Democratizing 3D Design &amp; Simulation | Natural Language → Parametric Models | Build123d + Zoo.dev + Adam.new + OpenFOAM | Solar PV, Test Chambers, Digital Twins &amp; More

## Features

### 📚 Project History & Version Control UI
Comprehensive project tracking and management system with:
- **GitHub Integration**: PR/branch status dashboard, commit history, and branch comparison
- **Audit Trail**: Complete action logging with search, filter, and export capabilities
- **Backup Management**: One-click backups, restore functionality, and project exports
- **Version Control**: Track CAD model history, simulation archives, and design iterations

[📖 Full Documentation](docs/PROJECT_HISTORY_UI.md)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Project History UI
streamlit run src/ui/project_history.py
```

## Quick Start

### Configure GitHub Integration (Optional)

Create `.streamlit/secrets.toml`:

```toml
[github]
token = "your_github_personal_access_token"
repo = "owner/repository_name"
```

See [PROJECT_HISTORY_UI.md](docs/PROJECT_HISTORY_UI.md) for detailed setup instructions.

## Testing

```bash
# Run all tests with coverage
pytest tests/test_project_history.py -v --cov=src --cov-report=term-missing
```

## Project Structure

```
GenAI-CAD-CFD-Studio/
├── src/
│   ├── ui/                  # Streamlit UI components
│   │   └── project_history.py
│   └── utils/              # Utility modules
│       ├── version_control.py
│       ├── audit_logger.py
│       └── project_archiver.py
├── tests/                  # Test suite
├── projects/               # Project data
│   ├── backups/
│   ├── models/
│   ├── results/
│   └── audit_logs/
└── docs/                   # Documentation
```

## License

See [LICENSE](LICENSE) file for details.
