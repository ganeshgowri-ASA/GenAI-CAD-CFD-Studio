# GenAI-CAD-CFD-Studio

[![CI](https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio/actions/workflows/ci.yml/badge.svg)](https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

🚀 **Universal AI-Powered CAD & CFD Platform** | Democratizing 3D Design & Simulation

Transform natural language into parametric 3D models and simulations. Build complex CAD designs and run CFD analyses without deep technical expertise.

---

## 🌟 Features

- **Natural Language Interface**: Describe your design in plain English
- **Parametric 3D Modeling**: Powered by Build123d for precise CAD operations
- **CFD Simulation**: Integrated OpenFOAM for computational fluid dynamics
- **AI-Driven Design**: Leveraging Zoo.dev and Adam.new APIs
- **Real-time Collaboration**: Streamlit-based interactive interface
- **Digital Twins**: Create simulation-ready models of physical systems
- **Specialized Templates**: Solar PV systems, test chambers, HVAC, and more

---

## 🏗️ Architecture Overview

```
GenAI-CAD-CFD-Studio/
├── src/                    # Source code
│   ├── ai/                # AI model integrations
│   ├── cad/               # CAD modeling engine
│   ├── cfd/               # CFD simulation engine
│   ├── ui/                # Streamlit interface
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docs/                  # Documentation
├── examples/              # Example projects
└── .github/               # CI/CD workflows
```

---

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- Git
- (Optional) Docker for containerized deployment

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio.git
   cd GenAI-CAD-CFD-Studio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Install pre-commit hooks** (optional, for contributors)
   ```bash
   pre-commit install
   ```

---

## 🚀 Usage

### Launch the Streamlit App

```bash
streamlit run src/app.py
```

The application will open in your browser at `http://localhost:8501`

### Example: Create a Solar Panel Array

```python
# In the Streamlit interface, use natural language:
"Create a 10x5 solar panel array with 2-degree tilt facing south"

# The system will:
# 1. Parse your request using AI
# 2. Generate parametric 3D model
# 3. Prepare CFD-ready geometry
# 4. Display interactive visualization
```

---

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_cad_engine.py
```

### Code Quality

```bash
# Format code with black
black .

# Run linter
flake8 .

# Run all pre-commit hooks
pre-commit run --all-files
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Maintain test coverage above 80%
- Write descriptive commit messages
- Add docstrings to all functions and classes

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Build123d](https://github.com/gumyr/build123d) - Parametric CAD modeling
- [OpenFOAM](https://www.openfoam.com/) - CFD simulation
- [Streamlit](https://streamlit.io/) - Interactive web framework
- [Zoo.dev](https://zoo.dev/) - AI-powered CAD API
- [Adam.new](https://adam.new/) - Generative design platform

---

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or contact the maintainers.

**Built with ❤️ for democratizing engineering design and simulation**
