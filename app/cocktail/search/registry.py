"""Process-level snapshot of the cocktail name search index.

Holds a single module-global `CocktailNameSearchIndex` built from the DB on
first use (or warmed up at app startup) and reused across requests.

IMPORTANT: this is a per-process registry. When the app runs with multiple
Uvicorn worker processes, each worker builds and holds its own copy of the
index in its own memory -- there is no cross-process sharing or invalidation.
"""

from __future__ import annotations

import threading

from app.cocktail.search.index import CocktailNameSearchIndex
from app.cocktail.search.loader import load_documents

_index: CocktailNameSearchIndex | None = None
_lock = threading.Lock()


def get_index() -> CocktailNameSearchIndex | None:
    """Return the current global index (None if not yet built)."""
    return _index


def set_index(idx: CocktailNameSearchIndex) -> None:
    """Set the global index (used by lifespan warm-up and verification)."""
    global _index
    _index = idx


def clear_index() -> None:
    """Reset the global index to None (used by verification)."""
    global _index
    _index = None


def ensure_index(db) -> CocktailNameSearchIndex:
    """Return the global index, building it from the DB on first call.

    Double-checked locking: the fast path (no lock) handles the common case
    where the index is already built; the lock only guards the rare
    first-build race so concurrent first requests build it exactly once.

    The index is built completely BEFORE being assigned to the registry, in
    a single statement -- if `load_documents`/`CocktailNameSearchIndex`
    construction raises, the registry is left unchanged (still empty rather
    than half-built).
    """
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                set_index(CocktailNameSearchIndex(load_documents(db)))
    assert _index is not None
    return _index
