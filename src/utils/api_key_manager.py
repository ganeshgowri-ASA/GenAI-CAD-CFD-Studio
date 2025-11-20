"""
Secure API Key Manager for GenAI CAD CFD Studio
Handles encryption, storage, and retrieval of API keys
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from cryptography.fernet import Fernet
import base64
import hashlib


class SecureKeyManager:
    """
    Manages secure storage and retrieval of API keys using Fernet encryption.

    Keys are encrypted using Fernet symmetric encryption and stored in
    .streamlit/secrets.toml or a local encrypted file.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the SecureKeyManager.

        Args:
            storage_path: Optional path to storage file. Defaults to .streamlit/secrets.json
        """
        if storage_path is None:
            self.storage_path = Path.home() / ".streamlit" / "secrets.json"
        else:
            self.storage_path = Path(storage_path)

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or load encryption key
        self._cipher_key = self._get_or_create_cipher_key()
        self._fernet = Fernet(self._cipher_key)

        # Load existing keys
        self._keys_store = self._load_keys()

    def _get_or_create_cipher_key(self) -> bytes:
        """
        Get or create the encryption key.

        The key is derived from a combination of machine-specific info
        and stored in a secure location.
        """
        key_file = self.storage_path.parent / ".key"

        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate a new key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions (owner read/write only)
            os.chmod(key_file, 0o600)
            return key

    def _load_keys(self) -> Dict:
        """Load encrypted keys from storage."""
        if not self.storage_path.exists():
            return {}

        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_keys(self):
        """Save encrypted keys to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self._keys_store, f, indent=2)
        # Set restrictive permissions
        os.chmod(self.storage_path, 0o600)

    def encrypt_key(self, key: str) -> str:
        """
        Encrypt an API key.

        Args:
            key: The plain text API key

        Returns:
            Encrypted key as a base64 string
        """
        if not key:
            raise ValueError("Key cannot be empty")

        encrypted_bytes = self._fernet.encrypt(key.encode('utf-8'))
        return encrypted_bytes.decode('utf-8')

    def decrypt_key(self, encrypted_str: str) -> str:
        """
        Decrypt an API key.

        Args:
            encrypted_str: The encrypted key string

        Returns:
            Decrypted plain text key
        """
        if not encrypted_str:
            raise ValueError("Encrypted string cannot be empty")

        try:
            decrypted_bytes = self._fernet.decrypt(encrypted_str.encode('utf-8'))
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to decrypt key: {str(e)}")

    def store_key(self, service: str, key: str, metadata: Optional[Dict] = None):
        """
        Store an API key for a service.

        Args:
            service: Service name (e.g., 'zoo_dev', 'anthropic', 'simscale')
            key: The API key to store
            metadata: Optional metadata about the key
        """
        if not service:
            raise ValueError("Service name cannot be empty")

        encrypted_key = self.encrypt_key(key)

        self._keys_store[service] = {
            'key': encrypted_key,
            'updated_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self._save_keys()

    def get_key(self, service: str) -> Optional[str]:
        """
        Retrieve an API key for a service.

        Args:
            service: Service name

        Returns:
            Decrypted API key or None if not found
        """
        if service not in self._keys_store:
            return None

        encrypted_key = self._keys_store[service]['key']
        return self.decrypt_key(encrypted_key)

    def delete_key(self, service: str) -> bool:
        """
        Delete an API key for a service.

        Args:
            service: Service name

        Returns:
            True if key was deleted, False if not found
        """
        if service in self._keys_store:
            del self._keys_store[service]
            self._save_keys()
            return True
        return False

    def list_services(self) -> list:
        """
        List all services with stored keys.

        Returns:
            List of service names
        """
        return list(self._keys_store.keys())

    def get_metadata(self, service: str) -> Optional[Dict]:
        """
        Get metadata for a service's API key.

        Args:
            service: Service name

        Returns:
            Metadata dict or None if not found
        """
        if service not in self._keys_store:
            return None

        return self._keys_store[service].get('metadata', {})

    def update_metadata(self, service: str, metadata: Dict):
        """
        Update metadata for a service's API key.

        Args:
            service: Service name
            metadata: New metadata dict
        """
        if service in self._keys_store:
            self._keys_store[service]['metadata'] = metadata
            self._keys_store[service]['updated_at'] = datetime.now().isoformat()
            self._save_keys()

    def key_exists(self, service: str) -> bool:
        """
        Check if a key exists for a service.

        Args:
            service: Service name

        Returns:
            True if key exists, False otherwise
        """
        return service in self._keys_store


# Global instance
_key_manager_instance = None


def get_key_manager(storage_path: Optional[str] = None) -> SecureKeyManager:
    """
    Get the global SecureKeyManager instance.

    Args:
        storage_path: Optional custom storage path

    Returns:
        SecureKeyManager instance
    """
    global _key_manager_instance

    if _key_manager_instance is None:
        _key_manager_instance = SecureKeyManager(storage_path)

    return _key_manager_instance
