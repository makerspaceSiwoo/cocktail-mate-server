"""Process-level snapshot of the cocktail name search index.

Holds a single module-global `CocktailNameSearchIndex` built from the DB on
first use (or warmed up at app startup) and reused across requests.

IMPORTANT: this is a per-process registry. When the app runs with multiple
Uvicorn worker processes, each worker builds and holds its own copy of the
index in its own memory -- there is no cross-process sharing or invalidation.
"""

from __future__ import annotations

import logging
import threading

from app.cocktail.search.index import CocktailNameSearchIndex
from app.cocktail.search.loader import load_documents

logger = logging.getLogger(__name__)

_index: CocktailNameSearchIndex | None = None
_lock = threading.Lock()


def get_index() -> CocktailNameSearchIndex | None:
    """Return the current global index (None if not yet built)."""
    return _index


def set_index(idx: CocktailNameSearchIndex) -> None:
    """Set the global index (used by lifespan warm-up and verification)."""
    global _index
    with _lock:
        _index = idx


def clear_index() -> None:
    """Reset the global index to None (used by verification)."""
    global _index
    with _lock:
        _index = None


def ensure_index(db) -> CocktailNameSearchIndex:
    """Return the global index, building it from the DB on first call.

    Double-checked locking: the fast path (no lock) handles the common case
    where the index is already built; the lock only guards the rare
    first-build race so concurrent first requests build it exactly once.

    The index is built completely BEFORE being assigned to the registry --
    if `load_documents`/`CocktailNameSearchIndex` construction raises, the
    registry is left unchanged (still empty rather than half-built).

    The global is read exactly once per code path into a local (`idx`),
    which is what's returned -- never re-reading `_index` after the lock is
    released. This closes a TOCTOU window where a concurrent `clear_index()`
    could otherwise be observed mid-function (e.g. between a not-None check
    and the return), which previously risked an `AssertionError` or a stray
    `None` return.

    A successful, non-empty build is logged at INFO with the document count
    so an empty index is visible in the logs rather than silently latching.
    If `load_documents` returns zero documents, the build is treated as
    failed: a WARNING is logged and the registry is left unset (`_index`
    stays `None`) so the next call retries against the DB instead of
    publishing an empty snapshot that would serve `items: []` forever. The
    just-built (empty) index is still returned for this call -- only
    caching it is skipped.
    """
    global _index
    idx = _index
    if idx is not None:
        return idx
    with _lock:
        idx = _index
        if idx is None:
            docs = load_documents(db)
            idx = CocktailNameSearchIndex(docs)
            if docs:
                _index = idx  # not set_index(): _lock is not reentrant
                logger.info("search index built: %d documents", len(docs))
            else:
                logger.warning(
                    "search index build returned 0 documents; leaving "
                    "index unset so the next call retries"
                )
        return idx
