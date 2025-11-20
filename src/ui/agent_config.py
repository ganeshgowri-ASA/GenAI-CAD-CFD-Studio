"""
Agent Configuration UI for GenAI CAD CFD Studio
Streamlit interface for managing API keys and agent settings
"""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime
import requests
import time
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.api_key_manager import get_key_manager
from src.agents.agent_registry import get_agent_registry, AgentConfig

logger = logging.getLogger(__name__)

# Initialize session state
if 'usage_stats' not in st.session_state:
    st.session_state.usage_stats = {
        'zoo_dev': {'calls': 0, 'credits': 0, 'last_call': None},
        'adam_new': {'calls': 0, 'credits': 0, 'last_call': None},
        'anthropic_claude': {'calls': 0, 'credits': 0, 'last_call': None},
        'simscale': {'calls': 0, 'credits': 0, 'last_call': None},
    }

if 'agent_status' not in st.session_state:
    st.session_state.agent_status = {}

if 'error_logs' not in st.session_state:
    st.session_state.error_logs = []


def test_api_connection(service: str, api_key: str, endpoint: str) -> tuple[bool, str]:
    """
    Test API connection for a service.

    Args:
        service: Service name
        api_key: API key to test
        endpoint: API endpoint

    Returns:
        Tuple of (success, message)
    """
    try:
        headers = {'Authorization': f'Bearer {api_key}'}

        # Different test endpoints for different services
        test_endpoints = {
            'zoo_dev': f'{endpoint}/health',
            'adam_new': f'{endpoint}/health',
            'anthropic_claude': f'{endpoint}/messages',
            'simscale': f'{endpoint}/projects'
        }

        test_url = test_endpoints.get(service, endpoint)

        # For Anthropic, use a minimal test request
        if service == 'anthropic_claude':
            response = requests.post(
                test_url,
                headers={
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                    'content-type': 'application/json'
                },
                json={
                    'model': 'claude-3-5-sonnet-20241022',
                    'max_tokens': 1,
                    'messages': [{'role': 'user', 'content': 'test'}]
                },
                timeout=10
            )
        else:
            response = requests.get(test_url, headers=headers, timeout=10)

        if response.status_code in [200, 201]:
            return True, "Connection successful!"
        else:
            return False, f"Connection failed: HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "Connection timeout"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to service"
    except Exception as e:
        return False, f"Error: {str(e)}"


def log_error(service: str, error: str):
    """Log an error for a service."""
    st.session_state.error_logs.append({
        'timestamp': datetime.now().isoformat(),
        'service': service,
        'error': error
    })
    # Keep only last 100 errors
    if len(st.session_state.error_logs) > 100:
        st.session_state.error_logs = st.session_state.error_logs[-100:]


def update_agent_status(service: str, status: bool, message: str = ""):
    """Update agent status."""
    st.session_state.agent_status[service] = {
        'status': status,
        'message': message,
        'last_checked': datetime.now().isoformat()
    }


def render_security_notice():
    """Render security notice banner."""
    st.info("""
    🔒 **Security Notice**
    - All API keys are encrypted using Fernet symmetric encryption
    - Keys are stored locally in encrypted format
    - Keys are never logged or transmitted in plain text
    - Storage location: `~/.streamlit/secrets.json` (encrypted)
    """)


def render_api_key_section():
    """Render API key management section."""
    st.subheader("🔑 API Key Management")

    key_manager = get_key_manager()
    registry = get_agent_registry()

    # Define services to manage
    services = [
        {
            'name': 'zoo_dev',
            'display': 'Zoo.dev API Key',
            'help': 'API key for Zoo.dev CAD generation service'
        },
        {
            'name': 'adam_new',
            'display': 'Adam.new API Key',
            'help': 'API key for Adam.new engineering platform'
        },
        {
            'name': 'anthropic_claude',
            'display': 'Anthropic Claude API Key',
            'help': 'API key for Claude AI assistant'
        },
        {
            'name': 'simscale',
            'display': 'SimScale API Key (Optional)',
            'help': 'API key for SimScale CFD simulations'
        }
    ]

    for service_info in services:
        service = service_info['name']
        with st.expander(f"🔐 {service_info['display']}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                # Check if key exists
                existing_key = key_manager.get_key(service)
                key_exists = existing_key is not None

                if key_exists:
                    st.success("✓ API key configured")
                    show_key = st.checkbox(f"Show {service} key", key=f"show_{service}")
                    if show_key:
                        st.code(existing_key)
                else:
                    st.warning("⚠ No API key configured")

                # Input for new/update key
                new_key = st.text_input(
                    "Enter API Key",
                    type="password",
                    key=f"input_{service}",
                    help=service_info['help']
                )

                if new_key:
                    if st.button(f"💾 Save {service} Key", key=f"save_{service}"):
                        try:
                            key_manager.store_key(service, new_key)
                            st.success(f"✓ {service} API key saved successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving key: {e}")
                            log_error(service, str(e))

            with col2:
                # Test connection button
                if key_exists:
                    if st.button(f"🔌 Test", key=f"test_{service}"):
                        with st.spinner("Testing connection..."):
                            config = registry.get_config(service)
                            if config:
                                success, message = test_api_connection(
                                    service,
                                    existing_key,
                                    config.api_endpoint
                                )
                                if success:
                                    st.success(message)
                                    update_agent_status(service, True, message)
                                else:
                                    st.error(message)
                                    update_agent_status(service, False, message)
                                    log_error(service, message)

                    # Delete key button
                    if st.button(f"🗑️ Delete", key=f"delete_{service}"):
                        key_manager.delete_key(service)
                        st.success(f"Deleted {service} API key")
                        st.rerun()


def render_agent_settings():
    """Render agent settings section."""
    st.subheader("⚙️ Agent Settings")

    registry = get_agent_registry()

    # Default agent selector
    enabled_agents = registry.list_agents(enabled_only=True)

    if enabled_agents:
        default_agent = st.selectbox(
            "Default Agent",
            options=enabled_agents,
            help="Select the default agent for CAD generation"
        )

        if 'default_agent' not in st.session_state:
            st.session_state.default_agent = default_agent

        # Agent-specific settings
        selected_config = registry.get_config(default_agent)

        if selected_config:
            with st.expander(f"⚙️ {selected_config.display_name} Settings", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    # Rate limit
                    rate_limit = st.number_input(
                        "Rate Limit (requests/min)",
                        min_value=1,
                        max_value=1000,
                        value=selected_config.rate_limit,
                        key=f"rate_limit_{default_agent}"
                    )

                    # Timeout
                    timeout = st.number_input(
                        "Timeout (seconds)",
                        min_value=1,
                        max_value=600,
                        value=selected_config.timeout,
                        key=f"timeout_{default_agent}"
                    )

                with col2:
                    # Model selection (for Claude)
                    if selected_config.agent_type == 'claude':
                        model_options = [
                            'claude-3-5-sonnet-20241022',
                            'claude-3-opus-20240229',
                            'claude-3-sonnet-20240229',
                            'claude-3-haiku-20240307'
                        ]
                        model = st.selectbox(
                            "Model",
                            options=model_options,
                            index=model_options.index(selected_config.model_name)
                            if selected_config.model_name in model_options else 0,
                            key=f"model_{default_agent}"
                        )
                    else:
                        model = st.text_input(
                            "Model Name",
                            value=selected_config.model_name or "",
                            key=f"model_{default_agent}"
                        )

                    # Enable/disable
                    enabled = st.checkbox(
                        "Enabled",
                        value=selected_config.enabled,
                        key=f"enabled_{default_agent}"
                    )

                # Custom prompt template
                st.text_area(
                    "Custom Prompt Template",
                    value=selected_config.custom_prompt_template or "",
                    help="Use {prompt} as placeholder for user input",
                    key=f"prompt_{default_agent}",
                    height=100
                )

                # Save settings button
                if st.button(f"💾 Save Settings for {selected_config.display_name}"):
                    updates = {
                        'rate_limit': rate_limit,
                        'timeout': timeout,
                        'model_name': model,
                        'enabled': enabled,
                        'custom_prompt_template': st.session_state[f"prompt_{default_agent}"]
                    }
                    if registry.update_config(default_agent, updates):
                        st.success(f"✓ Settings saved for {selected_config.display_name}!")
                        st.rerun()
                    else:
                        st.error("Failed to save settings")
    else:
        st.warning("No agents enabled. Please configure and enable agents.")


def render_usage_statistics():
    """Render usage statistics section."""
    st.subheader("📊 Usage Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Zoo.dev Calls",
            st.session_state.usage_stats['zoo_dev']['calls'],
            delta=None
        )
        st.caption(f"Credits: {st.session_state.usage_stats['zoo_dev']['credits']}")

    with col2:
        st.metric(
            "Adam.new Calls",
            st.session_state.usage_stats['adam_new']['calls'],
            delta=None
        )
        st.caption(f"Credits: {st.session_state.usage_stats['adam_new']['credits']}")

    with col3:
        st.metric(
            "Claude Calls",
            st.session_state.usage_stats['anthropic_claude']['calls'],
            delta=None
        )
        st.caption(f"Credits: {st.session_state.usage_stats['anthropic_claude']['credits']}")

    with col4:
        st.metric(
            "SimScale Calls",
            st.session_state.usage_stats['simscale']['calls'],
            delta=None
        )
        st.caption(f"Credits: {st.session_state.usage_stats['simscale']['credits']}")

    # Rate limit status
    st.divider()
    st.write("**Rate Limit Status**")

    registry = get_agent_registry()
    for agent_name in registry.list_agents(enabled_only=True):
        config = registry.get_config(agent_name)
        if config:
            progress = min(
                st.session_state.usage_stats.get(agent_name, {}).get('calls', 0) / config.rate_limit,
                1.0
            )
            st.progress(
                progress,
                text=f"{config.display_name}: {st.session_state.usage_stats.get(agent_name, {}).get('calls', 0)}/{config.rate_limit} calls/min"
            )


def render_agent_status_dashboard():
    """Render agent status dashboard."""
    st.subheader("🚦 Agent Status Dashboard")

    registry = get_agent_registry()
    agents = registry.list_agent_configs(enabled_only=False)

    if not agents:
        st.info("No agents registered")
        return

    for agent_config in agents:
        status_info = st.session_state.agent_status.get(agent_config.name, {})
        is_online = status_info.get('status', False)
        last_checked = status_info.get('last_checked', 'Never')

        col1, col2, col3, col4 = st.columns([2, 1, 2, 3])

        with col1:
            st.write(f"**{agent_config.display_name}**")

        with col2:
            if is_online:
                st.success("🟢 Online")
            else:
                st.error("🔴 Offline")

        with col3:
            if last_checked != 'Never':
                try:
                    dt = datetime.fromisoformat(last_checked)
                    st.caption(f"Last checked: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    st.caption(f"Last checked: {last_checked}")
            else:
                st.caption("Not tested yet")

        with col4:
            if not is_online and status_info.get('message'):
                st.caption(f"Error: {status_info.get('message')}")


def render_error_logs():
    """Render error logs viewer."""
    st.subheader("📋 Error Logs")

    if not st.session_state.error_logs:
        st.info("No errors logged")
        return

    # Show last 10 errors
    recent_errors = sorted(
        st.session_state.error_logs,
        key=lambda x: x['timestamp'],
        reverse=True
    )[:10]

    for error in recent_errors:
        with st.expander(
            f"🔴 {error['service']} - {error['timestamp'][:19]}",
            expanded=False
        ):
            st.code(error['error'])

    # Clear logs button
    if st.button("🗑️ Clear Error Logs"):
        st.session_state.error_logs = []
        st.rerun()


def render_custom_agent_integration():
    """Render custom agent integration section."""
    st.subheader("➕ Custom Agent Integration")

    registry = get_agent_registry()

    with st.expander("Add New Agent", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            agent_name = st.text_input(
                "Agent Name",
                placeholder="my_custom_agent",
                help="Unique identifier (lowercase, no spaces)"
            )
            display_name = st.text_input(
                "Display Name",
                placeholder="My Custom Agent",
                help="Friendly display name"
            )
            agent_type = st.selectbox(
                "Agent Type",
                options=['custom', 'build123d', 'zoo', 'adam', 'claude', 'simscale'],
                help="Select agent type"
            )

        with col2:
            api_endpoint = st.text_input(
                "API Endpoint",
                placeholder="https://api.example.com/v1",
                help="Base URL for API calls"
            )
            auth_method = st.selectbox(
                "Authentication Method",
                options=['api_key', 'bearer', 'oauth', 'none'],
                help="How to authenticate with the API"
            )
            model_name = st.text_input(
                "Model Name",
                placeholder="model-v1",
                help="Model identifier (optional)"
            )

        # Advanced settings
        col3, col4 = st.columns(2)
        with col3:
            rate_limit = st.number_input(
                "Rate Limit (req/min)",
                min_value=1,
                max_value=1000,
                value=60
            )
        with col4:
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=600,
                value=30
            )

        custom_prompt = st.text_area(
            "Custom Prompt Template",
            placeholder="Process this request: {prompt}",
            help="Use {prompt} as placeholder for user input"
        )

        if st.button("➕ Add Custom Agent"):
            if not agent_name or not display_name:
                st.error("Agent name and display name are required")
            elif agent_name in registry.list_agents():
                st.error(f"Agent '{agent_name}' already exists")
            else:
                try:
                    new_config = AgentConfig(
                        name=agent_name,
                        display_name=display_name,
                        agent_type=agent_type,
                        api_endpoint=api_endpoint or None,
                        auth_method=auth_method,
                        model_name=model_name or None,
                        rate_limit=rate_limit,
                        timeout=timeout,
                        custom_prompt_template=custom_prompt or None
                    )
                    registry.register_agent(new_config)
                    st.success(f"✓ Custom agent '{display_name}' added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding agent: {e}")

    # List existing custom agents
    st.divider()
    st.write("**Registered Custom Agents**")

    custom_agents = [
        config for config in registry.list_agent_configs()
        if config.agent_type == 'custom'
    ]

    if custom_agents:
        for agent in custom_agents:
            with st.expander(f"🤖 {agent.display_name}", expanded=False):
                st.write(f"**Name:** {agent.name}")
                st.write(f"**Endpoint:** {agent.api_endpoint}")
                st.write(f"**Auth:** {agent.auth_method}")
                st.write(f"**Rate Limit:** {agent.rate_limit} req/min")
                st.write(f"**Timeout:** {agent.timeout}s")

                if st.button(f"🗑️ Remove {agent.display_name}", key=f"remove_{agent.name}"):
                    registry.unregister_agent(agent.name)
                    st.success(f"Removed {agent.display_name}")
                    st.rerun()
    else:
        st.info("No custom agents registered yet")


def render_agent_config_page():
    """Main function to render the agent configuration page."""
    st.header('⚙️ Agent Configuration')

    # Security notice
    render_security_notice()

    st.divider()

    # API Key Management
    render_api_key_section()

    st.divider()

    # Agent Settings
    render_agent_settings()

    st.divider()

    # Usage Statistics
    render_usage_statistics()

    st.divider()

    # Agent Status Dashboard
    render_agent_status_dashboard()

    st.divider()

    # Error Logs
    render_error_logs()

    st.divider()

    # Custom Agent Integration
    render_custom_agent_integration()


def render():
    """Render function for tab integration"""
    render_agent_config_page()


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(
        page_title="Agent Configuration",
        page_icon="⚙️",
        layout="wide"
    )
    render_agent_config_page()
