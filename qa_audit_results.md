# QA AUDIT REPORT - GenAI CAD CFD Studio
Generated: Thu Nov 20 04:47:49 UTC 2025

## Branch 1/15: Setup Python Project
**Branch:** `claude/setup-python-project-019BFvJySXHQcU3bTP774heQ`

### Directory Structure
```
.
./.git
```

### Key Files
- Python files: **0**
- Test files: **0** ⚠️ (no tests/ directory)
- Dependencies: ⚠️ (no requirements.txt or pyproject.toml)

### Main Code Structure
```
```

### QA Status
**Overall: ⚠️ WARNING** (Minimal Python code - possibly foundation setup)

---

## Branch 2/15: Build Utils Core
**Branch:** `claude/build-utils-core-01Swij98AtE6hJS2h6tUExWX`

### Directory Structure
```
.
./.git
./tests
./configs
./src
./src/utils
```

### Key Files
- Python files: **9**
- Test files: **2** ✅
- Dependencies (requirements.txt): **6** ✅
- Python files with docstrings: **9/9**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 3/15: Streamlit Six Tab UI
**Branch:** `claude/streamlit-six-tab-ui-01BhFEpimZcMbsNkUodrnobC`

### Directory Structure
```
.
./.git
./src
./src/ui
./src/ui/components
```

### Key Files
- Python files: **13**
- Test files: **0** ⚠️ (no tests/ directory)
- Dependencies (requirements.txt): **4** ✅
- Python files with docstrings: **13/13**

### Main Code Structure
```
```

### QA Status
**Overall: ⚠️ WARNING** (Missing tests directory)

---

## Branch 4/15: Multi-Format File Import
**Branch:** `claude/multi-format-file-import-012sAEsumNXr3PvMs6gXKgBG`

### Directory Structure
```
.
./.git
./tests
./src
./src/io
```

### Key Files
- Python files: **10**
- Test files: **2** ✅
- Dependencies (requirements.txt): **18** ✅
- Python files with docstrings: **10/10**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 5/15: CAD Generation Engines
**Branch:** `claude/cad-generation-engines-01NPSqbUYgG87F9SQchn9g4z`

### Directory Structure
```
.
./.git
./tests
./src
./src/cad
./examples
```

### Key Files
- Python files: **9**
- Test files: **1** ✅
- Dependencies (requirements.txt): **5** ✅
- Python files with docstrings: **9/9**
- ⚠️ **WARNING:** Potential hardcoded API keys detected (4 occurrences)

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 6/15: File Import UI Preview
**Branch:** `claude/file-import-ui-preview-01H17yM5qz3cknno4y2Zn3xg`

### Directory Structure
```
.
./.git
./tests
./src
./src/visualization
./src/ui
./src/ui/components
./src/io
```

### Key Files
- Python files: **12**
- Test files: **2** ✅
- Dependencies (requirements.txt): **5** ✅
- Python files with docstrings: **6/12**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 7/15: AI Core Claude Integration
**Branch:** `claude/ai-core-claude-integration-018txitCJsd3gjqkMmzG592y`

### Directory Structure
```
.
./.git
./tests
./src
./src/ai
```

### Key Files
- Python files: **6**
- Test files: **1** ✅
- Dependencies (requirements.txt): **7** ✅
- Python files with docstrings: **6/6**
- ⚠️ **WARNING:** Potential hardcoded API keys detected (11 occurrences)

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 8/15: Design Studio UI
**Branch:** `claude/design-studio-ui-01Gh7WaMPA8UGFsnUcmxhCnQ`

### Directory Structure
```
.
./.git
./tests
./src
./src/ui
./src/ui/components
./src/ai
./.streamlit
```

### Key Files
- Python files: **13**
- Test files: **2** ✅
- Dependencies (requirements.txt): **10** ✅
- Python files with docstrings: **8/13**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 9/15: Geospatial Solar PV Layout
**Branch:** `claude/geospatial-solar-pv-layout-01Rz11DkZd2PHTrPkhDvJYKM`

### Directory Structure
```
.
./.git
./tests
./src
./src/geospatial
```

### Key Files
- Python files: **8**
- Test files: **2** ✅
- Dependencies (requirements.txt): **9** ✅
- Python files with docstrings: **8/8**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 10/15: Solar PV Layout Designer
**Branch:** `claude/solar-pv-layout-designer-01MZ4Fio6GmAeMJhZfdLLbg8`

### Directory Structure
```
.
./.git
./tests
./src
./src/geospatial
./src/ui
./src/ui/components
./.streamlit
```

### Key Files
- Python files: **11**
- Test files: **2** ✅
- Dependencies (requirements.txt): **15** ✅
- Python files with docstrings: **6/11**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 11/15: CFD OpenFOAM Integration
**Branch:** `claude/cfd-openfoam-integration-01PRUcYSf42F82BpfZVyZhtw`

### Directory Structure
```
.
./.git
./tests
./src
./src/cfd
```

### Key Files
- Python files: **8**
- Test files: **2** ✅
- Dependencies (requirements.txt): **19** ✅
- Python files with docstrings: **8/8**
- ⚠️ **WARNING:** Potential hardcoded API keys detected (2 occurrences)

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 12/15: CFD Analysis Wizard UI
**Branch:** `claude/cfd-analysis-wizard-ui-017uS87TprcbnrrUMtViWPLX`

### Directory Structure
```
.
./.git
./tests
./docs
./src
./src/visualization
./src/ui
./src/ui/components
./src/cfd
```

### Key Files
- Python files: **18**
- Test files: **2** ✅
- Dependencies (requirements.txt): **15** ✅
- Python files with docstrings: **12/18**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 13/15: PyVista Visualization Module
**Branch:** `claude/pyvista-visualization-module-01ALvxSxy7vBpRGPMeszhif2`

### Directory Structure
```
.
./.git
./tests
./src
./src/visualization
./examples
```

### Key Files
- Python files: **8**
- Test files: **1** ✅
- Dependencies (requirements.txt): **19** ✅
- Python files with docstrings: **8/8**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 14/15: Agent Config UI
**Branch:** `claude/agent-config-ui-01FqhoTAuBun5gNUD9MeE8UQ`

### Directory Structure
```
.
./.git
./tests
./docs
./src
./src/utils
./src/ui
./src/agents
```

### Key Files
- Python files: **9**
- Test files: **2** ✅
- Dependencies (requirements.txt): **14** ✅
- Python files with docstrings: **4/9**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

## Branch 15/15: Project History UI
**Branch:** `claude/project-history-ui-018BANKgvj5nq78gMdwETxFx`

### Directory Structure
```
.
./.git
./tests
./docs
./src
./src/utils
./src/ui
./.streamlit
```

### Key Files
- Python files: **9**
- Test files: **2** ✅
- Dependencies (requirements.txt): **7** ✅
- Python files with docstrings: **5/9**

### Main Code Structure
```
```

### QA Status
**Overall: ✅ PASS**

---

