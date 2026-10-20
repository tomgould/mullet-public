"""Cross-agent shared key-value memory for multi-agent orchestration."""
import json
import re
import os
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

class SharedMemory:
    """Persistent key-value store for cross-agent memory."""
    def __init__(self, base_dir: str = '.mullet/state/shared-memory'):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_name(self, name: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    def _get_namespace_dir(self, namespace: str) -> Path:
        sanitized_namespace = self._sanitize_name(namespace)
        return self.base_dir / sanitized_namespace

    def _get_file_path(self, key: str, namespace: str) -> Path:
        sanitized_key = self._sanitize_name(key)
        namespace_dir = self._get_namespace_dir(namespace)
        return namespace_dir / f"{sanitized_key}.json"

    def set(self, key: str, value: Any, namespace: str = 'default', ttl: int = 0) -> None:
        """Set a key-value pair in the specified namespace with optional TTL."""
        namespace_dir = self._get_namespace_dir(namespace)
        namespace_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._get_file_path(key, namespace)

        now = datetime.now()
        expires_at = (now + timedelta(seconds=ttl)) if ttl > 0 else None

        data = {
            'key': key,
            'value': value,
            'namespace': namespace,
            'created_at': now.isoformat(),
            'updated_at': now.isoformat(),
            'ttl': ttl,
            'expires_at': expires_at.isoformat() if expires_at else None
        }

        fd, temp_file_path = tempfile.mkstemp(
            dir=str(file_path.parent),
            prefix=f".{file_path.stem}_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_file_path, file_path)  # Atomic rename
        except Exception:
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
            raise

    def get(self, key: str, namespace: str = 'default') -> Any | None:
        """Get value for a key, returns None if expired or not found."""
        file_path = self._get_file_path(key, namespace)
        if not file_path.exists():
            return None

        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            self.delete(key, namespace) # Corrupted file, delete it
            return None

        expires_at_str = data.get('expires_at')
        if expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now() > expires_at:
                self.delete(key, namespace) # Expired, delete it
                return None
        return data.get('value')

    def delete(self, key: str, namespace: str = 'default') -> bool:
        """Delete a key, returns True if key existed."""
        file_path = self._get_file_path(key, namespace)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_keys(self, namespace: str = 'default') -> list[str]:
        """List all non-expired keys in a namespace."""
        namespace_dir = self._get_namespace_dir(namespace)
        if not namespace_dir.is_dir():
            return []

        keys = []
        for file_path in namespace_dir.glob('*.json'):
            key = file_path.stem
            # Check if key is expired without fetching the full value
            try:
                with open(file_path) as f:
                    data = json.load(f)
                expires_at_str = data.get('expires_at')
                if expires_at_str:
                    expires_at = datetime.fromisoformat(expires_at_str)
                    if datetime.now() > expires_at:
                        file_path.unlink() # Delete expired file
                        continue
                keys.append(key)
            except (json.JSONDecodeError, FileNotFoundError):
                file_path.unlink(missing_ok=True) # Delete corrupted or non-existent file
                continue
        return keys

    def list_namespaces(self) -> list[str]:
        """List all existing namespace directories."""
        if not self.base_dir.is_dir():
            return []
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def clear(self, namespace: str = 'default') -> int:
        """Clear all keys in a namespace and return count deleted."""
        namespace_dir = self._get_namespace_dir(namespace)
        if not namespace_dir.is_dir():
            return 0
        
        deleted_count = 0
        for file_path in namespace_dir.glob('*.json'):
            try:
                file_path.unlink()
                deleted_count += 1
            except OSError as _exc:
                logger.warning("shared_memory: Failed to delete file %s during clear: %s", file_path, _exc)
        
        # Optionally remove the namespace directory if it's empty
        try:
            namespace_dir.rmdir()
        except OSError as _exc:
            logger.warning("shared_memory: Failed to remove namespace directory %s during clear: %s", namespace_dir, _exc)
        
        return deleted_count
