from unittest.mock import Mock

from sqlalchemy.dialects import postgresql

from app.favor.repository import FavorRepository
from app.favor.service import (
    FAVOR_LIMIT,
    RECENT_LIKES_LIMIT,
    FavorService,
)


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


def test_recommend_merges_each_recent_likes_neighbors_by_best_similarity() -> None:
    db = Mock()
    repository = Mock()
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    repository.liked_embeddings.return_value = embeddings
    repository.liked_ids.return_value = {1, 2, 3, 4, 5, 6}
    repository.nearest_within.side_effect = [
        [
            {"id": 10, "name": "A", "similarity": 0.8},
            {"id": 20, "name": "B", "similarity": 0.7},
            {"id": 30, "name": "C", "similarity": 0.65},
        ],
        [
            {"id": 20, "name": "B", "similarity": 0.9},
            {"id": 40, "name": "D", "similarity": 0.75},
            {"id": 50, "name": "E", "similarity": 0.63},
            {"id": 60, "name": "F", "similarity": 0.61},
        ],
    ]

    result = FavorService(repository=repository).recommend(db, user_id=123)

    repository.liked_embeddings.assert_called_once_with(
        db,
        123,
        limit=RECENT_LIKES_LIMIT,
    )
    assert repository.nearest_within.call_args_list == [
        ((db, embeddings[0], {1, 2, 3, 4, 5, 6}, FAVOR_LIMIT),),
        ((db, embeddings[1], {1, 2, 3, 4, 5, 6}, FAVOR_LIMIT),),
    ]
    assert [candidate["id"] for candidate in result] == [20, 10, 40, 30, 50]
    assert result[0]["similarity"] == 0.9
