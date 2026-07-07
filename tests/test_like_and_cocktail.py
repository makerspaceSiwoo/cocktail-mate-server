"""좋아요 실DB 동작 + 칵테일 목록 is_liked (OptionalUser) 테스트."""
from __future__ import annotations

from cocktail_mate_db.models import User


def _login(kakao_callback, db, provider_id="liker1", nickname="liker"):
    """카카오 소셜 로그인으로 인증 세션을 만들고 유저를 반환한다."""
    resp = kakao_callback(provider_id=provider_id, nickname=nickname)
    assert resp.status_code == 302, resp.text
    return db.query(User).filter_by(provider="kakao", provider_id=provider_id).one()


def test_like_requires_auth(client, seed_cocktail):
    cid = seed_cocktail()
    assert client.post(f"/cocktails/{cid}/like").status_code == 401
    assert client.get("/like/list").status_code == 401


def test_like_and_unlike_flow(client, db, seed_cocktail, kakao_callback):
    cid = seed_cocktail()
    _login(kakao_callback, db)

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


def test_like_nonexistent_cocktail_404(client, db, kakao_callback):
    _login(kakao_callback, db)
    assert client.post("/cocktails/999999/like").status_code == 404


def test_cocktail_list_is_liked_optional_user(client, db, seed_cocktail, kakao_callback):
    cid = seed_cocktail()

    # 비로그인: is_liked 항상 False, 200
    anon = client.get("/list")
    assert anon.status_code == 200
    items = anon.json()["items"]
    assert any(i["id"] == cid for i in items)
    assert all(i["isLiked"] is False for i in items)

    # 로그인 + 좋아요 후: 해당 항목 is_liked True
    _login(kakao_callback, db)
    client.post(f"/cocktails/{cid}/like")
    logged = client.get("/list")
    target = next(i for i in logged.json()["items"] if i["id"] == cid)
    assert target["isLiked"] is True
