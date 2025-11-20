#!/bin/bash
# QA Audit Script for all 15 feature branches
# This script checks out each branch and generates a QA report

OUTPUT_FILE="qa_audit_results.md"
echo "# QA AUDIT REPORT - GenAI CAD CFD Studio" > $OUTPUT_FILE
echo "Generated: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Define all branches in dependency order
declare -a BRANCHES=(
    "claude/setup-python-project-019BFvJySXHQcU3bTP774heQ"
    "claude/build-utils-core-01Swij98AtE6hJS2h6tUExWX"
    "claude/streamlit-six-tab-ui-01BhFEpimZcMbsNkUodrnobC"
    "claude/multi-format-file-import-012sAEsumNXr3PvMs6gXKgBG"
    "claude/cad-generation-engines-01NPSqbUYgG87F9SQchn9g4z"
    "claude/file-import-ui-preview-01H17yM5qz3cknno4y2Zn3xg"
    "claude/ai-core-claude-integration-018txitCJsd3gjqkMmzG592y"
    "claude/design-studio-ui-01Gh7WaMPA8UGFsnUcmxhCnQ"
    "claude/geospatial-solar-pv-layout-01Rz11DkZd2PHTrPkhDvJYKM"
    "claude/solar-pv-layout-designer-01MZ4Fio6GmAeMJhZfdLLbg8"
    "claude/cfd-openfoam-integration-01PRUcYSf42F82BpfZVyZhtw"
    "claude/cfd-analysis-wizard-ui-017uS87TprcbnrrUMtViWPLX"
    "claude/pyvista-visualization-module-01ALvxSxy7vBpRGPMeszhif2"
    "claude/agent-config-ui-01FqhoTAuBun5gNUD9MeE8UQ"
    "claude/project-history-ui-018BANKgvj5nq78gMdwETxFx"
)

declare -a BRANCH_NAMES=(
    "Setup Python Project"
    "Build Utils Core"
    "Streamlit Six Tab UI"
    "Multi-Format File Import"
    "CAD Generation Engines"
    "File Import UI Preview"
    "AI Core Claude Integration"
    "Design Studio UI"
    "Geospatial Solar PV Layout"
    "Solar PV Layout Designer"
    "CFD OpenFOAM Integration"
    "CFD Analysis Wizard UI"
    "PyVista Visualization Module"
    "Agent Config UI"
    "Project History UI"
)

# Function to audit a single branch
audit_branch() {
    local branch=$1
    local name=$2
    local index=$3

    echo "## Branch $index/15: $name" >> $OUTPUT_FILE
    echo "**Branch:** \`$branch\`" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE

    # Checkout branch
    git checkout $branch 2>&1 | grep -v "^Switched" | grep -v "^Already" > /dev/null

    # Check directory structure
    echo "### Directory Structure" >> $OUTPUT_FILE
    echo "\`\`\`" >> $OUTPUT_FILE
    find . -type d -not -path "./.git/*" | head -20 >> $OUTPUT_FILE
    echo "\`\`\`" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE

    # Check for key files
    echo "### Key Files" >> $OUTPUT_FILE

    # Python files
    py_count=$(find . -name "*.py" -not -path "./.git/*" | wc -l)
    echo "- Python files: **$py_count**" >> $OUTPUT_FILE

    # Tests
    if [ -d "tests" ]; then
        test_count=$(find tests -name "*.py" 2>/dev/null | wc -l)
        echo "- Test files: **$test_count** ✅" >> $OUTPUT_FILE
    else
        echo "- Test files: **0** ⚠️ (no tests/ directory)" >> $OUTPUT_FILE
    fi

    # Dependencies
    if [ -f "requirements.txt" ]; then
        dep_count=$(grep -v "^#" requirements.txt | grep -v "^$" | wc -l)
        echo "- Dependencies (requirements.txt): **$dep_count** ✅" >> $OUTPUT_FILE
    elif [ -f "pyproject.toml" ]; then
        echo "- Dependencies (pyproject.toml): ✅" >> $OUTPUT_FILE
    else
        echo "- Dependencies: ⚠️ (no requirements.txt or pyproject.toml)" >> $OUTPUT_FILE
    fi

    # Check for docstrings in Python files
    docstring_count=$(find . -name "*.py" -not -path "./.git/*" -exec grep -l "\"\"\"" {} \; 2>/dev/null | wc -l)
    if [ $py_count -gt 0 ]; then
        echo "- Python files with docstrings: **$docstring_count/$py_count**" >> $OUTPUT_FILE
    fi

    # Check for hardcoded values (basic check)
    hardcoded_check=$(grep -r "api_key\s*=\s*['\"]" --include="*.py" . 2>/dev/null | grep -v "os.getenv" | grep -v "config" | wc -l)
    if [ $hardcoded_check -gt 0 ]; then
        echo "- ⚠️ **WARNING:** Potential hardcoded API keys detected ($hardcoded_check occurrences)" >> $OUTPUT_FILE
    fi

    # List main directories
    echo "" >> $OUTPUT_FILE
    echo "### Main Code Structure" >> $OUTPUT_FILE
    echo "\`\`\`" >> $OUTPUT_FILE
    tree -L 2 -I '.git' 2>/dev/null || find . -maxdepth 2 -type f -not -path "./.git/*" | sort
    echo "\`\`\`" >> $OUTPUT_FILE

    # QA Status
    echo "" >> $OUTPUT_FILE
    echo "### QA Status" >> $OUTPUT_FILE

    # Determine status based on checks
    if [ -d "tests" ] && [ -f "requirements.txt" ] && [ $py_count -gt 0 ]; then
        echo "**Overall: ✅ PASS**" >> $OUTPUT_FILE
    elif [ $py_count -eq 0 ]; then
        echo "**Overall: ⚠️ WARNING** (Minimal Python code - possibly foundation setup)" >> $OUTPUT_FILE
    elif [ ! -d "tests" ]; then
        echo "**Overall: ⚠️ WARNING** (Missing tests directory)" >> $OUTPUT_FILE
    else
        echo "**Overall: ⚠️ WARNING** (Some checks failed)" >> $OUTPUT_FILE
    fi

    echo "" >> $OUTPUT_FILE
    echo "---" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
}

# Audit all branches
for i in "${!BRANCHES[@]}"; do
    audit_branch "${BRANCHES[$i]}" "${BRANCH_NAMES[$i]}" $((i+1))
done

# Return to original branch
git checkout claude/qa-audit-merge-features-01KkeoWNi7B7xBFRc5kws6sn

echo "Audit complete! Results saved to $OUTPUT_FILE"
