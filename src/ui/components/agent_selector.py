"""
Agent Selector Component for Design Studio
Allows users to choose between different CAD generation engines
"""
import streamlit as st
from typing import Dict, Optional


class AgentSelector:
    """
    Component for selecting CAD generation agents/engines
    """

    # Agent configurations with capabilities and descriptions
    AGENTS = {
        "build123d": {
            "name": "Build123d",
            "icon": "🐍",
            "description": "Python-native CAD kernel",
            "capabilities": [
                "Fast execution",
                "Precise parametric modeling",
                "Native Python integration",
                "Best for programmatic designs"
            ],
            "best_for": "Complex parametric models, automation, Python workflows",
            "language": "Python",
            "speed": "⚡ Fast"
        },
        "zoo_dev": {
            "name": "Zoo.dev",
            "icon": "🦁",
            "description": "KCL (KittyCAD Language)",
            "capabilities": [
                "Engineering-grade precision",
                "Professional CAD features",
                "Cloud-based rendering",
                "Industry-standard outputs"
            ],
            "best_for": "Production-ready designs, engineering projects",
            "language": "KCL",
            "speed": "⚡⚡ Very Fast"
        },
        "adam_new": {
            "name": "Adam.new",
            "icon": "🤖",
            "description": "Natural language AI designer",
            "capabilities": [
                "Conversational interface",
                "Context-aware generation",
                "Iterative refinement",
                "Best for quick prototypes"
            ],
            "best_for": "Rapid prototyping, exploratory design, beginners",
            "language": "Natural Language",
            "speed": "⚡⚡⚡ Ultra Fast"
        }
    }

    def __init__(self, session_key: str = "selected_agent"):
        """
        Initialize the agent selector

        Args:
            session_key: Session state key for storing selected agent
        """
        self.session_key = session_key
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = "build123d"

    def render(self, show_details: bool = True) -> str:
        """
        Render the agent selector interface

        Args:
            show_details: Whether to show detailed agent information

        Returns:
            Selected agent key
        """
        st.subheader("🚀 Select CAD Engine")

        # Create radio button options with icons
        options = [
            f"{agent['icon']} {agent['name']}"
            for agent in self.AGENTS.values()
        ]
        agent_keys = list(self.AGENTS.keys())

        # Render radio buttons
        selected_display = st.radio(
            "Choose your CAD generation engine:",
            options,
            index=agent_keys.index(st.session_state[self.session_key]),
            help="Different engines have different strengths. Choose based on your needs."
        )

        # Get the selected agent key
        selected_index = options.index(selected_display)
        selected_agent = agent_keys[selected_index]
        st.session_state[self.session_key] = selected_agent

        # Show detailed information if requested
        if show_details:
            self._render_agent_details(selected_agent)

        return selected_agent

    def _render_agent_details(self, agent_key: str):
        """
        Render detailed information about the selected agent

        Args:
            agent_key: The agent key to display details for
        """
        agent = self.AGENTS[agent_key]

        with st.expander("ℹ️ Engine Details", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Description:** {agent['description']}")
                st.markdown(f"**Language:** {agent['language']}")
                st.markdown(f"**Speed:** {agent['speed']}")

            with col2:
                st.markdown(f"**Best For:** {agent['best_for']}")

            st.markdown("**Capabilities:**")
            for capability in agent['capabilities']:
                st.markdown(f"- ✓ {capability}")

    def get_selected_agent(self) -> str:
        """
        Get the currently selected agent

        Returns:
            Selected agent key
        """
        return st.session_state.get(self.session_key, "build123d")

    def get_agent_info(self, agent_key: Optional[str] = None) -> Dict:
        """
        Get information about an agent

        Args:
            agent_key: The agent to get info for (uses selected if None)

        Returns:
            Dictionary containing agent information
        """
        if agent_key is None:
            agent_key = self.get_selected_agent()

        return self.AGENTS.get(agent_key, self.AGENTS["build123d"])

    @staticmethod
    def get_available_agents() -> Dict:
        """
        Get all available agents

        Returns:
            Dictionary of all agent configurations
        """
        return AgentSelector.AGENTS
