"""Simple file-based cache for API responses."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

CACHE_DIR = Path.home() / ".cache" / "asx-hedge-fund"
DEFAULT_TTL = 3600  # 1 hour


def _cache_path(key: str) -> Path:
    hashed = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{hashed}.json"


def get_cache(key: str, ttl: int = DEFAULT_TTL) -> Optional[Any]:
    """Retrieve cached data if it exists and hasn't expired."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data["timestamp"] > ttl:
            path.unlink()
            return None
        return data["value"]
    except (json.JSONDecodeError, KeyError):
        path.unlink(missing_ok=True)
        return None


def set_cache(key: str, value: Any) -> None:
    """Store data in cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    path.write_text(json.dumps({"timestamp": time.time(), "value": value}))


def clear_cache() -> None:
    """Clear all cached data."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.iterdir():
            f.unlink()
