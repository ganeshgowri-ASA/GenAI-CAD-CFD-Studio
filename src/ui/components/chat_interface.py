"""
Chat Interface Component for Design Studio
Handles conversational AI interactions for CAD design
"""
import streamlit as st
import time
from typing import List, Dict, Optional


class ChatInterface:
    """
    Manages the chat interface for AI-powered CAD design interactions
    """

    def __init__(self, session_key: str = "chat_messages"):
        """
        Initialize the chat interface

        Args:
            session_key: Session state key for storing messages
        """
        self.session_key = session_key
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []

    def render_messages(self, messages: Optional[List[Dict]] = None):
        """
        Render chat messages with proper formatting

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     If None, uses messages from session state
        """
        if messages is None:
            messages = st.session_state[self.session_key]

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            with st.chat_message(role):
                if role == "assistant":
                    # Add typing effect for AI responses
                    self._render_with_typing_effect(content, message.get("show_typing", False))
                else:
                    st.markdown(content)

    def _render_with_typing_effect(self, content: str, show_typing: bool = False):
        """
        Render content with optional typing effect

        Args:
            content: The message content to display
            show_typing: Whether to show typing effect
        """
        if show_typing:
            # Create a placeholder for the typing effect
            message_placeholder = st.empty()
            full_response = ""

            # Simulate typing effect
            for chunk in content.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)

            message_placeholder.markdown(content)
        else:
            st.markdown(content)

    def handle_user_input(self, prompt: str) -> Dict:
        """
        Process user input and add to message history

        Args:
            prompt: User's input text

        Returns:
            Dictionary containing the user message
        """
        if prompt:
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": time.time()
            }
            st.session_state[self.session_key].append(user_message)
            return user_message
        return {}

    def add_assistant_message(self, content: str, show_typing: bool = False):
        """
        Add an assistant message to the chat history

        Args:
            content: The assistant's response
            show_typing: Whether to show typing effect on render
        """
        assistant_message = {
            "role": "assistant",
            "content": content,
            "timestamp": time.time(),
            "show_typing": show_typing
        }
        st.session_state[self.session_key].append(assistant_message)

    def clear_messages(self):
        """Clear all messages from the chat history"""
        st.session_state[self.session_key] = []

    def get_messages(self) -> List[Dict]:
        """
        Get all messages from chat history

        Returns:
            List of message dictionaries
        """
        return st.session_state[self.session_key]

    def render_chat_input(self, placeholder: str = "Describe the 3D object you want to create...") -> Optional[str]:
        """
        Render the chat input widget

        Args:
            placeholder: Placeholder text for the input

        Returns:
            User input text or None
        """
        return st.chat_input(placeholder)
