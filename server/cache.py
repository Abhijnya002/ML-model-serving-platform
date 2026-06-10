"""Redis caching layer for inference results."""

import json
import logging
import os
from typing import Optional

import redis

logger = logging.getLogger(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", 3600))


class InferenceCache:
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._enabled = True

    def connect(self) -> None:
        try:
            self._client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._client.ping()
            logger.info("Connected to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning("Redis unavailable (%s) — caching disabled", e)
            self._enabled = False

    def get(self, cache_key: str) -> Optional[dict]:
        if not self._enabled or self._client is None:
            return None
        try:
            data = self._client.get(f"inference:{cache_key}")
            if data:
                return json.loads(data)
        except redis.RedisError as e:
            logger.warning("Cache GET failed: %s", e)
        return None

    def set(self, cache_key: str, result: dict) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._client.setex(
                f"inference:{cache_key}",
                CACHE_TTL_SECONDS,
                json.dumps(result),
            )
        except redis.RedisError as e:
            logger.warning("Cache SET failed: %s", e)

    def stats(self) -> dict:
        if not self._enabled or self._client is None:
            return {"enabled": False}
        try:
            info = self._client.info("stats")
            return {
                "enabled": True,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except redis.RedisError:
            return {"enabled": True, "error": "stats unavailable"}
