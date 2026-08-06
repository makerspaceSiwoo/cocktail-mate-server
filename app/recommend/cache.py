"""Bounded process-local cache for flavor recommendation results."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock, RLock
from time import monotonic

Recommendation = list[dict[str, object]]
CacheKey = tuple[int | str, ...]
TASTE_CACHE_TTL_SECONDS = 24 * 60 * 60
TASTE_CACHE_MAX_ENTRIES = 4096

# 캐시 무효화 epoch. 캐시 키는 descriptor id 조합인데, 임베딩을 다시 적재하면 **같은
# 조합이 다른 결과를 뜻하게 된다**. TTL이 24시간이라 epoch를 올리지 않으면 낡은 결과가
# 하루 동안 계속 나간다.
#
# 규칙: `cocktails.embedding` / `preference_embedding`을 새로 적재할 때마다
# 아래 기본값을 그 적재 run id로 바꾸거나, 배포 환경에 `FLAVOR_CACHE_EPOCH`를 주입한다.
# (프로세스 로컬 캐시라 재시작만으로도 비워지지만, 롤링 배포/재적재-무재시작 상황에서는
#  epoch만이 유일한 방어선이다.)
DEFAULT_FLAVOR_CACHE_EPOCH = "run-20260806-full602-v1"
FLAVOR_CACHE_EPOCH = (
    os.getenv("FLAVOR_CACHE_EPOCH", "").strip() or DEFAULT_FLAVOR_CACHE_EPOCH
)


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float
    value: tuple[dict[str, object], ...]


class TasteRecommendationCache:
    """TTL + LRU cache with per-key single-flight computation."""

    def __init__(
        self,
        *,
        ttl_seconds: float = TASTE_CACHE_TTL_SECONDS,
        max_entries: int = TASTE_CACHE_MAX_ENTRIES,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[CacheKey, _CacheEntry] = OrderedDict()
        self._key_locks: dict[CacheKey, Lock] = {}
        self._lock = RLock()

    def clear(self) -> None:
        """Drop every cached recommendation (deploy/reload invalidation hook)."""

        with self._lock:
            self._entries.clear()

    @staticmethod
    def _copy(value: tuple[dict[str, object], ...]) -> Recommendation:
        return [dict(item) for item in value]

    def _get_unlocked(self, key: CacheKey, now: float) -> _CacheEntry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= now:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return entry

    def get_or_compute(
        self,
        key: CacheKey,
        factory: Callable[[], Recommendation],
    ) -> Recommendation:
        if not key:
            raise ValueError("empty flavor selections must bypass the cache")

        with self._lock:
            entry = self._get_unlocked(key, self._clock())
            if entry is not None:
                return self._copy(entry.value)
            key_lock = self._key_locks.setdefault(key, Lock())

        try:
            with key_lock:
                with self._lock:
                    entry = self._get_unlocked(key, self._clock())
                    if entry is not None:
                        return self._copy(entry.value)

                computed = tuple(dict(item) for item in factory())
                with self._lock:
                    now = self._clock()
                    self._entries[key] = _CacheEntry(
                        expires_at=now + self._ttl_seconds,
                        value=computed,
                    )
                    self._entries.move_to_end(key)
                    while len(self._entries) > self._max_entries:
                        self._entries.popitem(last=False)
                return self._copy(computed)
        finally:
            with self._lock:
                if self._key_locks.get(key) is key_lock:
                    del self._key_locks[key]
