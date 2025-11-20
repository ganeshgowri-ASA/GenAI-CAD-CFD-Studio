"""Streamlit session state management module for GenAI CAD/CFD Studio.

This module provides a convenient wrapper around Streamlit's session_state
with type hints, automatic initialization, and utility methods.
"""

from typing import Any, Optional, TypeVar, Generic, List, Callable

# Try to import streamlit, but make it optional for testing
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    # Create a mock streamlit module for testing
    class MockStreamlit:
        class SessionState:
            _state = {}

            def __contains__(self, key):
                return key in self._state

            def __getattr__(self, key):
                return self._state.get(key)

            def __setattr__(self, key, value):
                if key == '_state':
                    object.__setattr__(self, key, value)
                else:
                    self._state[key] = value

            def __delattr__(self, key):
                if key in self._state:
                    del self._state[key]

            def keys(self):
                return self._state.keys()

        session_state = SessionState()

    st = MockStreamlit()
    HAS_STREAMLIT = False


T = TypeVar('T')


class StreamlitSessionManager:
    """Manages Streamlit session state with a convenient interface.

    This class provides a wrapper around st.session_state with additional
    functionality like automatic initialization, type hints, and helper methods.

    Example:
        >>> session = StreamlitSessionManager()
        >>> session.set("username", "john_doe")
        >>> username = session.get("username")
        >>> if session.exists("username"):
        ...     print(f"User: {username}")
    """

    def __init__(self, prefix: str = ""):
        """Initialize the session manager.

        Args:
            prefix: Optional prefix for all session keys to avoid conflicts.
                Useful when managing different namespaces.
        """
        self.prefix = prefix

    def _make_key(self, key: str) -> str:
        """Create a prefixed key.

        Args:
            key: Original key name.

        Returns:
            Prefixed key name.
        """
        if self.prefix:
            return f"{self.prefix}_{key}"
        return key

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """Get a value from session state.

        Args:
            key: Session state key.
            default: Default value if key doesn't exist.

        Returns:
            Value from session state or default.

        Example:
            >>> session.get("count", 0)
            0
            >>> session.get("username", "guest")
            'guest'
        """
        full_key = self._make_key(key)
        return getattr(st.session_state, full_key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in session state.

        Args:
            key: Session state key.
            value: Value to store.

        Example:
            >>> session.set("username", "john_doe")
            >>> session.set("count", 42)
            >>> session.set("items", ["a", "b", "c"])
        """
        full_key = self._make_key(key)
        setattr(st.session_state, full_key, value)

    def exists(self, key: str) -> bool:
        """Check if a key exists in session state.

        Args:
            key: Session state key.

        Returns:
            True if key exists, False otherwise.

        Example:
            >>> session.set("username", "john")
            >>> session.exists("username")
            True
            >>> session.exists("nonexistent")
            False
        """
        full_key = self._make_key(key)
        return full_key in st.session_state

    def delete(self, key: str) -> None:
        """Delete a key from session state.

        Args:
            key: Session state key to delete.

        Example:
            >>> session.set("temp_data", "value")
            >>> session.delete("temp_data")
            >>> session.exists("temp_data")
            False
        """
        full_key = self._make_key(key)
        if full_key in st.session_state:
            delattr(st.session_state, full_key)

    def clear(self, exclude: Optional[List[str]] = None) -> None:
        """Clear all session state (or all with prefix if set).

        Args:
            exclude: List of keys to exclude from clearing.

        Example:
            >>> session.clear()  # Clear all
            >>> session.clear(exclude=["username"])  # Keep username
        """
        exclude = exclude or []
        exclude_full = [self._make_key(k) for k in exclude]

        # Get all keys to clear
        if self.prefix:
            # Clear only prefixed keys
            keys_to_clear = [
                k for k in st.session_state.keys()
                if k.startswith(f"{self.prefix}_") and k not in exclude_full
            ]
        else:
            # Clear all keys except excluded
            keys_to_clear = [
                k for k in st.session_state.keys()
                if k not in exclude_full
            ]

        for key in keys_to_clear:
            delattr(st.session_state, key)

    def get_or_init(self, key: str, init_value: T) -> T:
        """Get a value or initialize it if it doesn't exist.

        Args:
            key: Session state key.
            init_value: Value to initialize if key doesn't exist.

        Returns:
            Existing or newly initialized value.

        Example:
            >>> count = session.get_or_init("count", 0)
            >>> items = session.get_or_init("items", [])
        """
        if not self.exists(key):
            self.set(key, init_value)
        return self.get(key)

    def get_or_init_lazy(self, key: str, init_func: Callable[[], T]) -> T:
        """Get a value or lazily initialize it using a function.

        Useful for expensive initializations that should only happen once.

        Args:
            key: Session state key.
            init_func: Function to call for initialization if key doesn't exist.

        Returns:
            Existing or newly initialized value.

        Example:
            >>> def load_model():
            ...     return expensive_model_loading()
            >>> model = session.get_or_init_lazy("model", load_model)
        """
        if not self.exists(key):
            self.set(key, init_func())
        return self.get(key)

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in session state.

        Args:
            key: Session state key.
            amount: Amount to increment by. Defaults to 1.

        Returns:
            New value after increment.

        Example:
            >>> session.set("count", 5)
            >>> session.increment("count")
            6
            >>> session.increment("count", 10)
            16
        """
        current = self.get(key, 0)
        new_value = current + amount
        self.set(key, new_value)
        return new_value

    def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a numeric value in session state.

        Args:
            key: Session state key.
            amount: Amount to decrement by. Defaults to 1.

        Returns:
            New value after decrement.

        Example:
            >>> session.set("count", 10)
            >>> session.decrement("count")
            9
            >>> session.decrement("count", 5)
            4
        """
        return self.increment(key, -amount)

    def append(self, key: str, value: Any) -> List[Any]:
        """Append a value to a list in session state.

        Initializes an empty list if key doesn't exist.

        Args:
            key: Session state key.
            value: Value to append.

        Returns:
            Updated list.

        Example:
            >>> session.append("items", "apple")
            ['apple']
            >>> session.append("items", "banana")
            ['apple', 'banana']
        """
        current_list = self.get(key, [])
        if not isinstance(current_list, list):
            current_list = []
        current_list.append(value)
        self.set(key, current_list)
        return current_list

    def extend(self, key: str, values: List[Any]) -> List[Any]:
        """Extend a list in session state with multiple values.

        Args:
            key: Session state key.
            values: Values to extend the list with.

        Returns:
            Updated list.

        Example:
            >>> session.extend("items", ["apple", "banana"])
            ['apple', 'banana']
            >>> session.extend("items", ["cherry"])
            ['apple', 'banana', 'cherry']
        """
        current_list = self.get(key, [])
        if not isinstance(current_list, list):
            current_list = []
        current_list.extend(values)
        self.set(key, current_list)
        return current_list

    def toggle(self, key: str) -> bool:
        """Toggle a boolean value in session state.

        Args:
            key: Session state key.

        Returns:
            New boolean value after toggle.

        Example:
            >>> session.set("debug_mode", False)
            >>> session.toggle("debug_mode")
            True
            >>> session.toggle("debug_mode")
            False
        """
        current = self.get(key, False)
        new_value = not bool(current)
        self.set(key, new_value)
        return new_value

    def update(self, **kwargs: Any) -> None:
        """Update multiple session state values at once.

        Args:
            **kwargs: Key-value pairs to update.

        Example:
            >>> session.update(
            ...     username="john",
            ...     email="john@example.com",
            ...     count=0
            ... )
        """
        for key, value in kwargs.items():
            self.set(key, value)

    def keys(self) -> List[str]:
        """Get all session state keys (with prefix if set).

        Returns:
            List of keys in session state.

        Example:
            >>> session.set("key1", "value1")
            >>> session.set("key2", "value2")
            >>> session.keys()
            ['key1', 'key2']
        """
        if self.prefix:
            prefix_len = len(self.prefix) + 1
            return [
                k[prefix_len:] for k in st.session_state.keys()
                if k.startswith(f"{self.prefix}_")
            ]
        return list(st.session_state.keys())

    def to_dict(self) -> dict:
        """Convert session state to a dictionary.

        Returns:
            Dictionary containing all session state values.

        Example:
            >>> session.update(key1="value1", key2="value2")
            >>> session.to_dict()
            {'key1': 'value1', 'key2': 'value2'}
        """
        return {key: self.get(key) for key in self.keys()}

    def __repr__(self) -> str:
        """String representation of session manager."""
        prefix_str = f"prefix='{self.prefix}'" if self.prefix else "no prefix"
        return f"StreamlitSessionManager({prefix_str}, keys={self.keys()})"


# Create a default global session manager
session = StreamlitSessionManager()
