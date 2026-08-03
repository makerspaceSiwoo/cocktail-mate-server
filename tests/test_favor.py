from unittest.mock import Mock

from sqlalchemy.dialects import postgresql

from app.favor.repository import FavorRepository
from app.favor.service import RECENT_LIKES_LIMIT, FavorService


def test_liked_embeddings_queries_only_the_five_most_recent_likes() -> None:
    db = Mock()
    db.execute.return_value.all.return_value = [([0.1, 0.2],)]

    embeddings = FavorRepository().liked_embeddings(
        db,
        user_id=123,
        limit=RECENT_LIKES_LIMIT,
    )

    statement = db.execute.call_args.args[0]
    sql = str(
        statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert embeddings == [[0.1, 0.2]]
    assert "ORDER BY likes.created_at DESC, likes.id DESC" in sql
    assert "LIMIT 5" in sql


def test_recommend_uses_recent_like_limit_but_excludes_every_liked_cocktail() -> None:
    db = Mock()
    repository = Mock()
    repository.liked_embeddings.return_value = [[1.0, 0.0]]
    repository.liked_ids.return_value = {1, 2, 3, 4, 5, 6}
    repository.nearest_within.return_value = []

    FavorService(repository=repository).recommend(db, user_id=123)

    repository.liked_embeddings.assert_called_once_with(
        db,
        123,
        limit=RECENT_LIKES_LIMIT,
    )
    assert repository.nearest_within.call_args.args[2] == {1, 2, 3, 4, 5, 6}
