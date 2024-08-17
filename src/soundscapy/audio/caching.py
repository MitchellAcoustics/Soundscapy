import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from error_handling import error_handler


class Cache(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any):
        pass

    @abstractmethod
    def invalidate(self, key: str):
        pass


class MemoryCache(Cache):
    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any):
        self._cache[key] = value

    def invalidate(self, key: str):
        if key in self._cache:
            del self._cache[key]


class DiskCache(Cache):
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_file(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Any]:
        cache_file = self._get_cache_file(key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                error_handler.warning(f"Failed to decode cache file: {cache_file}")
                return None
        return None

    def set(self, key: str, value: Any):
        cache_file = self._get_cache_file(key)
        try:
            with open(cache_file, "w") as f:
                json.dump(value, f)
        except (IOError, TypeError) as e:
            error_handler.error(
                f"Failed to write cache file: {cache_file}. Error: {str(e)}"
            )

    def invalidate(self, key: str):
        cache_file = self._get_cache_file(key)
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except OSError as e:
                error_handler.error(
                    f"Failed to remove cache file: {cache_file}. Error: {str(e)}"
                )


# Factory function to create the appropriate cache based on configuration
def create_cache(cache_type: str, **kwargs) -> Cache:
    if cache_type == "memory":
        return MemoryCache()
    elif cache_type == "disk":
        cache_dir = kwargs.get("cache_dir", ".cache")
        return DiskCache(cache_dir)
    else:
        error_handler.error(f"Unknown cache type: {cache_type}", "INVALID_CACHE_TYPE")
        raise ValueError(f"Unknown cache type: {cache_type}")
