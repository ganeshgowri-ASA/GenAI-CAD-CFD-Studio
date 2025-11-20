#!/bin/bash
# Automated merge script with intelligent conflict resolution

declare -a REMAINING_BRANCHES=(
    "claude/cad-generation-engines-01NPSqbUYgG87F9SQchn9g4z:CAD generation engines"
    "claude/file-import-ui-preview-01H17yM5qz3cknno4y2Zn3xg:File import UI preview"
    "claude/ai-core-claude-integration-018txitCJsd3gjqkMmzG592y:AI core Claude integration"
    "claude/design-studio-ui-01Gh7WaMPA8UGFsnUcmxhCnQ:Design studio UI"
    "claude/geospatial-solar-pv-layout-01Rz11DkZd2PHTrPkhDvJYKM:Geospatial solar PV layout"
    "claude/solar-pv-layout-designer-01MZ4Fio6GmAeMJhZfdLLbg8:Solar PV layout designer"
    "claude/cfd-openfoam-integration-01PRUcYSf42F82BpfZVyZhtw:CFD OpenFOAM integration"
    "claude/cfd-analysis-wizard-ui-017uS87TprcbnrrUMtViWPLX:CFD analysis wizard UI"
    "claude/pyvista-visualization-module-01ALvxSxy7vBpRGPMeszhif2:PyVista visualization"
    "claude/agent-config-ui-01FqhoTAuBun5gNUD9MeE8UQ:Agent config UI"
    "claude/project-history-ui-018BANKgvj5nq78gMdwETxFx:Project history UI"
)

for branch_info in "${REMAINING_BRANCHES[@]}"; do
    IFS=':' read -r branch desc <<< "$branch_info"

    echo "========================================="
    echo "Merging: $desc"
    echo "Branch: $branch"
    echo "========================================="

    # Attempt merge
    if git merge $branch --no-ff -m "Merge $branch: $desc" 2>&1; then
        echo "✅ Merged successfully (no conflicts)"
    else
        # Check for conflicts
        if git status | grep -q "Unmerged paths"; then
            echo "⚠️ Conflicts detected - resolving automatically"

            # For each conflicted file
            git status --short | grep "^UU\|^AA" | awk '{print $2}' | while read file; do
                echo "Resolving: $file"

                if [ "$file" = "requirements.txt" ]; then
                    # For requirements.txt, accept both versions and merge uniquely
                    echo "# Merging requirements from both branches" > requirements_merged.txt
                    git show HEAD:requirements.txt 2>/dev/null | grep -v "^<<<\|^===\|^>>>" >> requirements_merged.txt
                    git show $branch:requirements.txt 2>/dev/null | grep -v "^<<<\|^===\|^>>>" >> requirements_merged.txt

                    # Remove duplicates while preserving order
                    awk '!seen[$0]++' requirements_merged.txt > requirements.txt
                    rm requirements_merged.txt
                    git add requirements.txt
                else
                    # For other files, accept incoming (theirs)
                    git checkout --theirs $file 2>/dev/null || true
                    git add $file 2>/dev/null || true
                fi
            done

            # Complete merge
            git commit --no-edit
            echo "✅ Conflicts resolved and merged"
        else
            echo "❌ Merge failed for unknown reason"
            git merge --abort
            exit 1
        fi
    fi

    echo ""
done

echo "========================================="
echo "All branches merged successfully!"
echo "========================================="
