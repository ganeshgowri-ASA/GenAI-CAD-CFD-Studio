# GenAI-CAD-CFD-Studio

🚀 Universal AI-Powered CAD & CFD Platform | Democratizing 3D Design & Simulation | Natural Language → Parametric Models | Build123d + Zoo.dev + Adam.new + OpenFOAM | Solar PV, Test Chambers, Digital Twins & More

## 🎨 AI Design Studio - NOW AVAILABLE!

The AI Design Studio is a revolutionary web-based interface for creating 3D CAD models using natural language. Simply describe what you want to create, and the AI will extract dimensions, generate parameters, and create an interactive 3D preview.

### ✨ Key Features

- **🤖 Conversational AI Interface**: Chat-based design input with intelligent dimension extraction
- **📏 Smart Parameter Extraction**: Automatically detects object types and dimensions from natural language
- **🚀 Multi-Engine Support**: Choose between Build123d, Zoo.dev, or Adam.new CAD engines
- **🎨 Interactive 3D Preview**: Real-time visualization with Plotly (solid, wireframe, shaded views)
- **📐 Editable Forms**: Review and adjust extracted parameters before generation
- **💾 Export Ready**: Prepare for STEP, STL, and OBJ exports

### 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the Design Studio
streamlit run app.py
```

Then open `http://localhost:8501` in your browser and start designing!

### 📖 Documentation

See [DESIGN_STUDIO_README.md](DESIGN_STUDIO_README.md) for complete documentation including:
- Detailed feature descriptions
- Usage guide and examples
- Architecture and component details
- Testing instructions
- API integration roadmap

### 🏗️ Project Structure

```
GenAI-CAD-CFD-Studio/
├── src/
│   ├── ui/
│   │   ├── design_studio.py          # Main UI orchestrator
│   │   └── components/               # Modular UI components
│   │       ├── chat_interface.py     # Chat interface
│   │       ├── agent_selector.py     # CAD engine selector
│   │       ├── dimension_form.py     # Parameter form
│   │       └── preview_3d.py         # 3D visualization
│   └── ai/
│       └── claude_skills.py          # AI dimension extraction
├── tests/
│   └── test_ui_design_studio.py      # Comprehensive test suite
├── app.py                            # Application entry point
└── requirements.txt                  # Dependencies
```

### 🧪 Testing

```bash
# Run all tests
pytest tests/test_ui_design_studio.py -v

# Run with coverage
pytest tests/test_ui_design_studio.py --cov=src
```

### 🎯 Example Usage

```
User: "Create a box 100mm x 50mm x 30mm"
AI: Extracts → object_type: box, length: 100, width: 50, height: 30, unit: mm
User: Reviews parameters → Clicks "Generate"
Result: Interactive 3D preview with export options
```

### 🛠️ Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **3D Visualization**: Plotly
- **AI/NLP**: Pattern matching (Claude API integration ready)
- **CAD Engines**: Build123d, Zoo.dev, Adam.new (integration ready)
- **Testing**: pytest

### 🗺️ Roadmap

#### Phase 1: UI Foundation ✅ (Current)
- ✅ Chat interface with message history
- ✅ Agent selection (Build123d, Zoo.dev, Adam.new)
- ✅ Dynamic dimension form with validation
- ✅ Interactive 3D preview (Plotly)
- ✅ Comprehensive test suite

#### Phase 2: CAD Engine Integration (Next)
- [ ] Build123d Python API integration
- [ ] Zoo.dev API integration
- [ ] Adam.new API integration
- [ ] Real STEP/STL/OBJ export

#### Phase 3: Advanced AI (Future)
- [ ] Anthropic Claude API integration
- [ ] Advanced dimension extraction
- [ ] Design suggestions and optimization
- [ ] Context-aware conversations

#### Phase 4: Enterprise Features (Future)
- [ ] Cloud storage and collaboration
- [ ] Version control for designs
- [ ] Material library and cost estimation
- [ ] Manufacturing constraints validation

### 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Submit a pull request

### 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### 🙏 Acknowledgments

Built with amazing open-source tools:
- Streamlit - Web framework
- Plotly - 3D visualization
- NumPy - Numerical computing
- Pytest - Testing framework

---

**Current Version**: 1.0.0 (Design Studio UI Complete)
**Status**: Production Ready 🚀
**Last Updated**: 2025-11-19
