#!/usr/bin/env python3
"""
CFD Analysis Studio Launcher

Run this script to start the CFD Analysis UI in Streamlit.

Usage:
    python run_cfd_analysis.py
    or
    streamlit run src/ui/cfd_analysis.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the CFD Analysis Studio."""

    # Get the path to the CFD analysis UI
    ui_path = Path(__file__).parent / "src" / "ui" / "cfd_analysis.py"

    if not ui_path.exists():
        print(f"Error: CFD Analysis UI not found at {ui_path}")
        sys.exit(1)

    print("🌊 Starting CFD Analysis Studio...")
    print(f"   UI Path: {ui_path}")
    print(f"   Python: {sys.executable}")
    print()
    print("   Access the UI at: http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print()

    # Run streamlit
    try:
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_path),
            "--server.port=8501",
            "--server.address=localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 CFD Analysis Studio stopped.")
    except Exception as e:
        print(f"\n❌ Error running Streamlit: {e}")
        print("\nMake sure Streamlit is installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
