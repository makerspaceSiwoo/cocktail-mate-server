"""비밀번호 찾기(forgot)/재설정(reset)/변경(change) 엔드포인트 테스트.

- forgot: 존재/미존재 이메일 모두 200(가입 여부 비노출), 존재 시 메일 발송 호출.
- reset: 정상 토큰으로 비번 변경 + 재사용 불가(1회용), 만료/변조 400, 정책위반 422,
  재설정 후 기존 refresh 토큰 삭제.
- change: 정상 200 + 새 비번 로그인, 현재 비번 틀림 400, 정책위반 422, 미인증 401,
  new==current 400.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cocktail_mate_db.models import RefreshToken, User

from app.core.security import (
    create_password_reset_token,
    hash_password,
)


def _create_local_user(db, email="reset@example.com", password="abcd1234!"):
    user = User(
        email=email,
        password_hash=hash_password(password),
        nickname=email.split("@")[0],
        provider="local",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _capture_reset_mail(monkeypatch):
    """send_password_reset_email 을 mock 하고 captured dict 반환."""
    captured = {}

    def _fake_send(email, reset_url):
        captured["email"] = email
        captured["url"] = reset_url

    monkeypatch.setattr(
        "app.auth.service.send_password_reset_email", _fake_send
    )
    return captured


# ── forgot ───────────────────────────────────────────────────
def test_forgot_existing_email_sends_mail(client, db, monkeypatch):
    _create_local_user(db)
    captured = _capture_reset_mail(monkeypatch)
    resp = client.post("/auth/password/forgot", json={"email": "reset@example.com"})
    assert resp.status_code == 200, resp.text
    assert captured.get("email") == "reset@example.com"
    assert "reset-password?token=" in captured["url"]


def test_forgot_unknown_email_still_200_no_mail(client, db, monkeypatch):
    """미존재 이메일도 200(비노출), 메일은 발송되지 않는다."""
    captured = _capture_reset_mail(monkeypatch)
    resp = client.post("/auth/password/forgot", json={"email": "nope@example.com"})
    assert resp.status_code == 200
    assert captured == {}


# ── reset ────────────────────────────────────────────────────
def _reset_token_for(user) -> str:
    return create_password_reset_token(user.id, user.password_hash)


def test_reset_happy_path_and_relogin(client, db, monkeypatch):
    user = _create_local_user(db)
    token = _reset_token_for(user)

    resp = client.post(
        "/auth/password/reset",
        json={
            "token": token,
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 200, resp.text

    # 새 비번으로 로그인 성공.
    login = client.post(
        "/auth/login", json={"email": "reset@example.com", "password": "newpass9$"}
    )
    assert login.status_code == 200, login.text


def test_reset_token_is_single_use(client, db):
    """비번이 바뀌면 password_hash 기반 서명이 무효 → 같은 토큰 재사용 실패(400)."""
    user = _create_local_user(db)
    token = _reset_token_for(user)

    first = client.post(
        "/auth/password/reset",
        json={
            "token": token,
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert first.status_code == 200, first.text

    # 같은 토큰 재사용 — password_hash 가 바뀌어 서명 검증 실패.
    reuse = client.post(
        "/auth/password/reset",
        json={
            "token": token,
            "new_password": "another8$",
            "new_password_confirm": "another8$",
        },
    )
    assert reuse.status_code == 400
    assert "만료" in reuse.json()["detail"] or "잘못" in reuse.json()["detail"]


def test_reset_deletes_existing_refresh_tokens(client, db, monkeypatch):
    """재설정 후 그 유저의 refresh 토큰이 전부 삭제되어 강제 재로그인된다."""
    user = _create_local_user(db)
    # 기존 세션(로그인) 생성.
    client.post(
        "/auth/login", json={"email": "reset@example.com", "password": "abcd1234!"}
    )
    db.expire_all()
    assert db.query(RefreshToken).filter_by(user_id=user.id).count() == 1

    token = _reset_token_for(user)
    resp = client.post(
        "/auth/password/reset",
        json={
            "token": token,
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 200, resp.text

    db.expire_all()
    assert db.query(RefreshToken).filter_by(user_id=user.id).count() == 0


def test_reset_expired_token_rejected(client, db, monkeypatch):
    """만료된 토큰 → 400."""
    user = _create_local_user(db)
    # 만료 시각을 과거로 강제한 토큰 생성.
    from jose import jwt

    from app.core import security

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "type": "password_reset",
        "iat": now - timedelta(hours=2),
        "exp": now - timedelta(hours=1),
    }
    signing_key = security.get_settings().secret_key + user.password_hash
    token = jwt.encode(payload, signing_key, algorithm=security.ALGORITHM)

    resp = client.post(
        "/auth/password/reset",
        json={
            "token": token,
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 400
    assert "만료" in resp.json()["detail"] or "잘못" in resp.json()["detail"]


def test_reset_tampered_token_rejected(client, db):
    """변조된(서명 불일치) 토큰 → 400."""
    user = _create_local_user(db)
    token = _reset_token_for(user)
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")

    resp = client.post(
        "/auth/password/reset",
        json={
            "token": tampered,
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 400


def test_reset_unknown_user_rejected(client, db):
    """존재하지 않는 유저를 가리키는 토큰 → 400(유효하지 않은 링크)."""
    # user_id 999999 는 존재하지 않음. 서명은 아무 해시로.
    token = create_password_reset_token(999999, hash_password("whatever1!"))
    resp = client.post(
        "/auth/password/reset",
        json={
            "token": token,
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 400
    assert "유효하지" in resp.json()["detail"]


def test_reset_weak_password_rejected(client, db):
    """새 비번 정책 위반 → 422."""
    user = _create_local_user(db)
    token = _reset_token_for(user)
    resp = client.post(
        "/auth/password/reset",
        json={
            "token": token,
            "new_password": "short",
            "new_password_confirm": "short",
        },
    )
    assert resp.status_code == 422


# ── change ───────────────────────────────────────────────────
def _login(client, db, email="reset@example.com", password="abcd1234!"):
    _create_local_user(db, email=email, password=password)
    resp = client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp


def test_change_happy_path_and_relogin(client, db):
    _login(client, db)
    resp = client.post(
        "/auth/password/change",
        json={
            "current_password": "abcd1234!",
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 200, resp.text

    # 새 비번으로 로그인 됨.
    login = client.post(
        "/auth/login", json={"email": "reset@example.com", "password": "newpass9$"}
    )
    assert login.status_code == 200, login.text


def test_change_wrong_current_password(client, db):
    _login(client, db)
    resp = client.post(
        "/auth/password/change",
        json={
            "current_password": "wrong123!",
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 400
    assert "현재 비밀번호" in resp.json()["detail"]


def test_change_weak_new_password(client, db):
    _login(client, db)
    resp = client.post(
        "/auth/password/change",
        json={
            "current_password": "abcd1234!",
            "new_password": "weak",
            "new_password_confirm": "weak",
        },
    )
    assert resp.status_code == 422


def test_change_requires_auth(client, db):
    """미인증(쿠키 없음) → 401."""
    resp = client.post(
        "/auth/password/change",
        json={
            "current_password": "abcd1234!",
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 401


def test_change_same_as_current_rejected(client, db):
    """새 비번 == 기존 비번 → 400."""
    _login(client, db)
    resp = client.post(
        "/auth/password/change",
        json={
            "current_password": "abcd1234!",
            "new_password": "abcd1234!",
            "new_password_confirm": "abcd1234!",
        },
    )
    assert resp.status_code == 400
    assert "동일" in resp.json()["detail"]


def test_change_social_user_rejected(client, db):
    """password_hash 가 NULL(소셜)인 유저는 400. (직접 access 쿠키 주입으로 인증.)"""
    from app.core.security import create_access_token

    social = User(
        email="social@example.com",
        password_hash=None,
        nickname="social",
        provider="kakao",
        provider_id="k-123",
        is_active=True,
    )
    db.add(social)
    db.commit()
    db.refresh(social)

    client.cookies.set("access_token", create_access_token(social.id))
    resp = client.post(
        "/auth/password/change",
        json={
            "current_password": "abcd1234!",
            "new_password": "newpass9$",
            "new_password_confirm": "newpass9$",
        },
    )
    assert resp.status_code == 400
    assert "사용할 수 없" in resp.json()["detail"]
