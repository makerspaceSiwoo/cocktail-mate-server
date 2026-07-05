"""좋아요 실DB 동작 + 칵테일 목록 is_liked (OptionalUser) 테스트."""
from __future__ import annotations

from cocktail_mate_db.models import User
from app.core.security import hash_password


def _login(client, db, email="liker@example.com"):
    user = User(
        email=email,
        password_hash=hash_password("abcd1234!"),
        nickname=email.split("@")[0],
        provider="local",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    client.post("/auth/login", json={"email": email, "password": "abcd1234!"})
    return user


def test_like_requires_auth(client, seed_cocktail):
    cid = seed_cocktail()
    assert client.post(f"/cocktails/{cid}/like").status_code == 401
    assert client.get("/like/list").status_code == 401


def test_like_and_unlike_flow(client, db, seed_cocktail):
    cid = seed_cocktail()
    _login(client, db)

    # 좋아요
    resp = client.post(f"/cocktails/{cid}/like")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["isLiked"] is True
    assert body["likeCount"] == 1

    # 목록에 반영
    lst = client.get("/like/list").json()
    assert len(lst["cocktails"]) == 1
    assert lst["cocktails"][0]["cocktailId"] == cid

    # 중복 좋아요 idempotent
    resp2 = client.post(f"/cocktails/{cid}/like")
    assert resp2.json()["likeCount"] == 1

    # 해제
    resp3 = client.delete(f"/cocktails/{cid}/like")
    assert resp3.json()["isLiked"] is False
    assert resp3.json()["likeCount"] == 0
    assert client.get("/like/list").json()["cocktails"] == []


def test_like_nonexistent_cocktail_404(client, db):
    _login(client, db)
    assert client.post("/cocktails/999999/like").status_code == 404


def test_cocktail_list_is_liked_optional_user(client, db, seed_cocktail):
    cid = seed_cocktail()

    # 비로그인: is_liked 항상 False, 200
    anon = client.get("/list")
    assert anon.status_code == 200
    items = anon.json()["items"]
    assert any(i["id"] == cid for i in items)
    assert all(i["isLiked"] is False for i in items)

    # 로그인 + 좋아요 후: 해당 항목 is_liked True
    _login(client, db)
    client.post(f"/cocktails/{cid}/like")
    logged = client.get("/list")
    target = next(i for i in logged.json()["items"] if i["id"] == cid)
    assert target["isLiked"] is True
