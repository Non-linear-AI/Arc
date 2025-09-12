"""Performance optimizations for Arc CLI."""

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any

# Performance utilities should not depend on config to avoid circular imports


class Cache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache: dict[
            str, tuple[Any, float, int]
        ] = {}  # key -> (value, timestamp, ttl)
        self.default_ttl = default_ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self.cache:
                value, timestamp, ttl = self.cache[key]
                if time.time() - timestamp < ttl:
                    return value
                else:
                    # Expired, remove it
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with TTL."""
        with self._lock:
            ttl = ttl or self.default_ttl
            self.cache[key] = (value, time.time(), ttl)

    def invalidate(self, key: str) -> None:
        """Remove key from cache."""
        with self._lock:
            self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()

    def cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, (_, timestamp, ttl) in self.cache.items()
                if current_time - timestamp >= ttl
            ]
            for key in expired_keys:
                del self.cache[key]


class FileCache:
    """File-based cache for persistent caching across sessions."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path.home() / ".arc" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Metadata file to track cache entries
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict[str, Any]]:
        """Load cache metadata."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception:
            pass

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Any | None:
        """Get value from file cache if not expired."""
        if key not in self.metadata:
            return None

        entry = self.metadata[key]
        if time.time() - entry["timestamp"] >= entry["ttl"]:
            # Expired
            self.invalidate(key)
            return None

        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                with open(cache_path) as f:
                    return json.load(f)
        except Exception:
            self.invalidate(key)

        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:  # 1 hour default
        """Set value in file cache."""
        try:
            cache_path = self._get_cache_path(key)
            with open(cache_path, "w") as f:
                json.dump(value, f)

            self.metadata[key] = {
                "timestamp": time.time(),
                "ttl": ttl,
                "size": cache_path.stat().st_size,
            }
            self._save_metadata()
        except Exception:
            pass

    def invalidate(self, key: str) -> None:
        """Remove key from cache."""
        if key in self.metadata:
            try:
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
            except Exception:
                pass

            del self.metadata[key]
            self._save_metadata()

    def cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self.metadata.items()
            if current_time - entry["timestamp"] >= entry["ttl"]
        ]

        for key in expired_keys:
            self.invalidate(key)


class PerformanceManager:
    """Centralized performance optimization manager."""

    def __init__(self):
        self.memory_cache = Cache(default_ttl=300)  # 5 minutes
        self.file_cache = FileCache()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._cleanup_task = None

        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_response_time": 0.0,
            "total_requests": 0,
            "file_operations": 0,
            "tool_executions": 0,
        }

    def _start_cleanup_task(self):
        """Start background cache cleanup (only if event loop is running)."""
        try:

            async def cleanup_loop():
                while True:
                    await asyncio.sleep(300)  # 5 minutes
                    self.memory_cache.cleanup_expired()
                    self.file_cache.cleanup_expired()

            # Only create task if we're in an event loop
            try:
                self._cleanup_task = asyncio.create_task(cleanup_loop())
            except RuntimeError:
                # No event loop running, cleanup will be manual
                pass
        except Exception:
            # Silently ignore cleanup task creation failures
            pass

    def ensure_cleanup_running(self):
        """Ensure cleanup task is running (call this from async context)."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._start_cleanup_task()

    def cached(self, ttl: int = 300, use_file_cache: bool = False):
        """Decorator for caching function results."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create cache key
                cache_key = self._create_cache_key(func.__name__, args, kwargs)

                # Try to get from cache
                cache = self.file_cache if use_file_cache else self.memory_cache
                cached_result = cache.get(cache_key)

                if cached_result is not None:
                    self.metrics["cache_hits"] += 1
                    return cached_result

                # Cache miss - execute function
                self.metrics["cache_misses"] += 1
                start_time = time.time()

                result = await func(*args, **kwargs)

                execution_time = time.time() - start_time
                self._update_response_time(execution_time)

                # Cache the result
                cache.set(cache_key, result, ttl)

                return result

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = self._create_cache_key(func.__name__, args, kwargs)

                cache = self.file_cache if use_file_cache else self.memory_cache
                cached_result = cache.get(cache_key)

                if cached_result is not None:
                    self.metrics["cache_hits"] += 1
                    return cached_result

                self.metrics["cache_misses"] += 1
                start_time = time.time()

                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                self._update_response_time(execution_time)

                cache.set(cache_key, result, ttl)

                return result

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _create_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create a cache key from function name and arguments."""
        # Convert args and kwargs to a stable string representation
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": sorted(kwargs.items()) if kwargs else [],
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _update_response_time(self, new_time: float) -> None:
        """Update average response time metric."""
        self.metrics["total_requests"] += 1
        total_time = self.metrics["avg_response_time"] * (
            self.metrics["total_requests"] - 1
        )
        self.metrics["avg_response_time"] = (total_time + new_time) / self.metrics[
            "total_requests"
        ]

    async def parallel_execute(self, tasks: list) -> list:
        """Execute multiple async tasks in parallel."""
        if not tasks:
            return []

        return await asyncio.gather(*tasks, return_exceptions=True)

    def run_in_thread(self, func: Callable, *args, **kwargs):
        """Run blocking function in thread pool."""
        return self.thread_pool.submit(func, *args, **kwargs)

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        cache_hit_rate = 0.0
        total_cache_ops = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total_cache_ops > 0:
            cache_hit_rate = self.metrics["cache_hits"] / total_cache_ops

        return {
            **self.metrics,
            "cache_hit_rate": cache_hit_rate,
            "memory_cache_size": len(self.memory_cache.cache),
            "file_cache_size": len(self.file_cache.metadata),
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        # Note: Not clearing file cache as it's persistent

    def preload_common_operations(self):
        """Preload commonly used operations into cache."""
        # This could be expanded to preload frequently accessed files,
        # directory listings, etc.
        pass


class StreamingResponseHandler:
    """Handle streaming responses efficiently."""

    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.buffer = []
        self.callbacks = []

    def add_callback(self, callback: Callable[[str], None]):
        """Add callback for streaming chunks."""
        self.callbacks.append(callback)

    def process_chunk(self, chunk: str):
        """Process a streaming chunk."""
        self.buffer.append(chunk)

        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(chunk)
            except Exception:
                pass

        # Flush buffer if it gets too large
        if len(self.buffer) > 100:
            self.flush_buffer()

    def flush_buffer(self) -> str:
        """Flush buffer and return complete content."""
        content = "".join(self.buffer)
        self.buffer.clear()
        return content

    def get_content(self) -> str:
        """Get current buffered content."""
        return "".join(self.buffer)


class BatchProcessor:
    """Process operations in batches for better performance."""

    def __init__(self, batch_size: int = 10, delay: float = 0.1):
        self.batch_size = batch_size
        self.delay = delay
        self.queue = []
        self.processing = False

    async def add_operation(self, operation: Callable, *args, **kwargs):
        """Add operation to batch queue."""
        self.queue.append((operation, args, kwargs))

        if not self.processing:
            await self._process_batch()

    async def _process_batch(self):
        """Process queued operations in batches."""
        self.processing = True

        try:
            while self.queue:
                # Take up to batch_size operations
                batch = self.queue[: self.batch_size]
                self.queue = self.queue[self.batch_size :]

                # Execute batch in parallel
                tasks = []
                for operation, args, kwargs in batch:
                    if asyncio.iscoroutinefunction(operation):
                        tasks.append(operation(*args, **kwargs))
                    else:
                        # Wrap sync function in async
                        tasks.append(
                            asyncio.create_task(
                                asyncio.to_thread(operation, *args, **kwargs)
                            )
                        )

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Small delay between batches
                if self.queue:
                    await asyncio.sleep(self.delay)

        finally:
            self.processing = False


# Global performance manager instance
performance_manager = PerformanceManager()


# Convenience decorators
def cached(ttl: int = 300, use_file_cache: bool = False):
    """Decorator for caching function results."""
    return performance_manager.cached(ttl=ttl, use_file_cache=use_file_cache)


def timed(metric_name: str):
    """Decorator to time function execution and record metrics."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                performance_manager.metrics[metric_name] = execution_time

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                performance_manager.metrics[metric_name] = execution_time

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
