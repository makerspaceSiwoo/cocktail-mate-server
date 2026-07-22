"""DB access for the cocktail name search core.

This is the ONLY module in `app.cocktail.search` allowed to touch the DB.
The rest of the package (types/normalize/matching/index) stays DB-free so it
can be built and tested from plain fixtures.
"""

from __future__ import annotations

from app.cocktail.search.index import make_document
from app.cocktail.search.types import SearchDocument


def load_documents(db) -> list[SearchDocument]:
    """Query the catalog and build search documents.

    `Cocktail` is imported here (deferred) so importing this module -- and
    anything that imports it -- never requires the DB model package to be
    configured. Only `id`, `name`, `name_en` are read; no other columns.
    """
    from cocktail_mate_db.models import Cocktail  # noqa: PLC0415

    rows = db.query(Cocktail.id, Cocktail.name, Cocktail.name_en).all()
    return [make_document(row.id, row.name, row.name_en) for row in rows]
