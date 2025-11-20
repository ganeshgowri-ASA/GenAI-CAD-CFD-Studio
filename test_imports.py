#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
This simulates what Streamlit Cloud will do when deploying
"""

import sys
import traceback

def test_import(module_path, description):
    """Test importing a module and report status"""
    try:
        __import__(module_path)
        print(f"✅ {description}: OK")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED")
        print(f"   Error: {e}")
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Testing GenAI CAD-CFD Studio Import Chain")
    print("=" * 60)
    print()

    all_passed = True

    # Test core modules
    print("Testing Core Modules:")
    print("-" * 60)
    all_passed &= test_import("src.ai.claude_skills", "AI: ClaudeSkills")
    all_passed &= test_import("src.ai.dimension_extractor", "AI: DimensionExtractor")
    all_passed &= test_import("src.ai.sketch_interpreter", "AI: SketchInterpreter")
    all_passed &= test_import("src.ai.prompt_templates", "AI: PromptTemplates")
    print()

    # Test UI components
    print("Testing UI Components:")
    print("-" * 60)
    all_passed &= test_import("src.ui.components.custom_css", "UI: CustomCSS")
    all_passed &= test_import("src.ui.components.sidebar", "UI: Sidebar")
    print()

    # Test UI modules
    print("Testing UI Modules:")
    print("-" * 60)
    all_passed &= test_import("src.ui.design_studio", "UI: DesignStudio")
    all_passed &= test_import("src.ui.file_import", "UI: FileImport")
    all_passed &= test_import("src.ui.layout_generator", "UI: LayoutGenerator")
    all_passed &= test_import("src.ui.cfd_analysis", "UI: CFDAnalysis")
    all_passed &= test_import("src.ui.agent_config", "UI: AgentConfig")
    all_passed &= test_import("src.ui.project_history", "UI: ProjectHistory")
    print()

    # Test main app
    print("Testing Main Application:")
    print("-" * 60)
    all_passed &= test_import("src.ui.app", "Main App")
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - App is ready to deploy!")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
    print("=" * 60)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
