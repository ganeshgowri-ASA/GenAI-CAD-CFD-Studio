#!/bin/bash
# Systematic Merge Script - Merge all 15 feature branches to main
# This script merges branches in dependency order with conflict resolution

MERGE_LOG="merge_log.txt"
echo "=== MERGE LOG - GenAI CAD CFD Studio ===" > $MERGE_LOG
echo "Started: $(date)" >> $MERGE_LOG
echo "" >> $MERGE_LOG

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

declare -a DESCRIPTIONS=(
    "Foundation: Python project setup"
    "Foundation: Build utils core"
    "Foundation: Streamlit six tab UI"
    "File I/O: Multi-format file import"
    "CAD: CAD generation engines"
    "File I/O: File import UI preview"
    "AI: AI core Claude integration"
    "Design: Design studio UI"
    "Geospatial: Solar PV layout core"
    "Geospatial: Solar PV layout designer UI"
    "CFD: OpenFOAM integration"
    "CFD: CFD analysis wizard UI"
    "Visualization: PyVista visualization module"
    "Configuration: Agent config UI"
    "Configuration: Project history UI"
)

MERGE_COUNT=0
CONFLICT_COUNT=0

# Function to merge a single branch
merge_branch() {
    local branch=$1
    local desc=$2
    local index=$3

    echo "========================================" | tee -a $MERGE_LOG
    echo "Merging Branch $index/15: $desc" | tee -a $MERGE_LOG
    echo "Branch: $branch" | tee -a $MERGE_LOG
    echo "========================================" | tee -a $MERGE_LOG

    # Attempt merge with no-ff to preserve branch history
    if git merge $branch --no-ff -m "Merge $branch: $desc" 2>&1 | tee -a $MERGE_LOG; then
        echo "✅ SUCCESS: Branch merged successfully" | tee -a $MERGE_LOG
        MERGE_COUNT=$((MERGE_COUNT + 1))

        # Show what was added
        echo "Files added/modified:" >> $MERGE_LOG
        git diff --name-status HEAD~1 HEAD >> $MERGE_LOG
        echo "" >> $MERGE_LOG
    else
        # Check if there are conflicts
        if git status | grep -q "Unmerged paths"; then
            echo "⚠️ CONFLICT DETECTED - Attempting resolution" | tee -a $MERGE_LOG
            CONFLICT_COUNT=$((CONFLICT_COUNT + 1))

            # List conflicted files
            echo "Conflicted files:" | tee -a $MERGE_LOG
            git status --short | grep "^UU\|^AA\|^DD" | tee -a $MERGE_LOG

            # Auto-resolve by accepting incoming changes (theirs)
            echo "Auto-resolving conflicts by accepting incoming branch changes..." | tee -a $MERGE_LOG
            git checkout --theirs . 2>&1 | tee -a $MERGE_LOG
            git add . 2>&1 | tee -a $MERGE_LOG

            # Complete the merge
            if git commit -m "Merge $branch: $desc (conflicts auto-resolved)" 2>&1 | tee -a $MERGE_LOG; then
                echo "✅ RESOLVED: Conflicts resolved and merge completed" | tee -a $MERGE_LOG
                MERGE_COUNT=$((MERGE_COUNT + 1))
            else
                echo "❌ FAILED: Could not complete merge after conflict resolution" | tee -a $MERGE_LOG
                git merge --abort 2>&1 | tee -a $MERGE_LOG
                return 1
            fi
        else
            echo "❌ FAILED: Merge failed for unknown reason" | tee -a $MERGE_LOG
            return 1
        fi
    fi

    echo "" >> $MERGE_LOG
}

# Ensure we're on main branch
git checkout main

# Merge all branches
for i in "${!BRANCHES[@]}"; do
    merge_branch "${BRANCHES[$i]}" "${DESCRIPTIONS[$i]}" $((i+1))
done

# Summary
echo "========================================" | tee -a $MERGE_LOG
echo "MERGE SUMMARY" | tee -a $MERGE_LOG
echo "========================================" | tee -a $MERGE_LOG
echo "Total branches: 15" | tee -a $MERGE_LOG
echo "Successfully merged: $MERGE_COUNT" | tee -a $MERGE_LOG
echo "Conflicts encountered: $CONFLICT_COUNT" | tee -a $MERGE_LOG
echo "Completed: $(date)" | tee -a $MERGE_LOG
echo "" >> $MERGE_LOG

# Show final directory structure
echo "Final directory structure:" | tee -a $MERGE_LOG
find . -maxdepth 2 -type d -not -path "./.git/*" | sort | tee -a $MERGE_LOG

echo ""
echo "Merge process complete! Check $MERGE_LOG for details."
