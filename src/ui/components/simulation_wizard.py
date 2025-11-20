"""
Simulation Wizard Component
Provides a step-by-step wizard interface for CFD simulation workflow.
"""

import streamlit as st
from typing import Callable, List, Optional, Dict, Any


class SimulationWizard:
    """
    A wizard-style component for multi-step simulation workflows.

    Features:
    - Progress tracking across steps
    - Navigation controls (Back/Next/Skip)
    - State management between steps
    - Visual progress indicator
    """

    def __init__(self, steps: List[str], session_key: str = "wizard"):
        """
        Initialize the simulation wizard.

        Args:
            steps: List of step names
            session_key: Unique key for session state storage
        """
        self.steps = steps
        self.session_key = session_key
        self.total_steps = len(steps)

        # Initialize session state
        if f"{session_key}_current_step" not in st.session_state:
            st.session_state[f"{session_key}_current_step"] = 0
        if f"{session_key}_data" not in st.session_state:
            st.session_state[f"{session_key}_data"] = {}

    @property
    def current_step(self) -> int:
        """Get current step index."""
        return st.session_state[f"{self.session_key}_current_step"]

    @current_step.setter
    def current_step(self, value: int):
        """Set current step index."""
        st.session_state[f"{self.session_key}_current_step"] = max(0, min(value, self.total_steps - 1))

    @property
    def data(self) -> Dict[str, Any]:
        """Get wizard data."""
        return st.session_state[f"{self.session_key}_data"]

    def set_data(self, key: str, value: Any):
        """Store data for the wizard."""
        st.session_state[f"{self.session_key}_data"][key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the wizard."""
        return st.session_state[f"{self.session_key}_data"].get(key, default)

    def render_progress(self):
        """Render the progress indicator."""
        progress = (self.current_step + 1) / self.total_steps

        # Progress bar
        st.progress(progress)

        # Step indicator
        cols = st.columns(self.total_steps)
        for idx, (col, step_name) in enumerate(zip(cols, self.steps)):
            with col:
                if idx < self.current_step:
                    st.markdown(f"✅ **{step_name}**")
                elif idx == self.current_step:
                    st.markdown(f"🔵 **{step_name}**")
                else:
                    st.markdown(f"⚪ {step_name}")

        st.divider()

    def render_step(self, step_num: int, content_func: Callable):
        """
        Render a specific step with its content.

        Args:
            step_num: Step number (0-indexed)
            content_func: Function that renders the step content
        """
        if self.current_step == step_num:
            st.subheader(f"Step {step_num + 1}: {self.steps[step_num]}")
            content_func()

    def render_navigation(self,
                         can_proceed: bool = True,
                         show_skip: bool = False,
                         next_label: str = "Next",
                         back_label: str = "Back",
                         skip_label: str = "Skip"):
        """
        Render navigation controls.

        Args:
            can_proceed: Whether the Next button should be enabled
            show_skip: Whether to show the Skip button
            next_label: Label for the Next button
            back_label: Label for the Back button
            skip_label: Label for the Skip button
        """
        st.divider()

        cols = st.columns([1, 1, 1, 3])

        # Back button
        with cols[0]:
            if self.current_step > 0:
                if st.button(back_label, use_container_width=True):
                    self.current_step -= 1
                    st.rerun()

        # Next button
        with cols[1]:
            if self.current_step < self.total_steps - 1:
                if st.button(next_label, use_container_width=True,
                           disabled=not can_proceed, type="primary"):
                    self.current_step += 1
                    st.rerun()
            elif self.current_step == self.total_steps - 1:
                st.success("✅ Wizard Complete!")

        # Skip button (optional)
        with cols[2]:
            if show_skip and self.current_step < self.total_steps - 1:
                if st.button(skip_label, use_container_width=True):
                    self.current_step += 1
                    st.rerun()

    def reset(self):
        """Reset the wizard to the first step."""
        self.current_step = 0
        st.session_state[f"{self.session_key}_data"] = {}
        st.rerun()

    def go_to_step(self, step_num: int):
        """Navigate to a specific step."""
        if 0 <= step_num < self.total_steps:
            self.current_step = step_num
            st.rerun()
