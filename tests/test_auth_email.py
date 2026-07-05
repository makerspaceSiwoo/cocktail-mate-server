"""이메일 매직링크 가입 흐름 + 로그인/refresh/logout/me 테스트."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


from cocktail_mate_db.models import EmailVerification, User
from app.core.security import hash_password, hash_token


def _request_verification(client, monkeypatch, email="siwoo@example.com"):
    """request-verification 호출 후 (request_id, raw_token) 반환. 메일은 mock."""
    captured = {}

    def _fake_send(to, verify_url):
        captured["url"] = verify_url

    monkeypatch.setattr("app.auth.service.send_verification_email", _fake_send)
    resp = client.post("/auth/email/request-verification", json={"email": email})
    assert resp.status_code == 200, resp.text
    request_id = resp.json()["request_id"]
    # verify_url = {FRONTEND_URL}/verify-email?token={raw}
    raw_token = captured["url"].split("token=", 1)[1]
    return request_id, raw_token


def test_signup_happy_path(client, monkeypatch, db):
    """request → verify → poll → signup → 쿠키 세팅 확인."""
    request_id, raw_token = _request_verification(client, monkeypatch)

    # 폴링: 아직 미인증
    status = client.get(
        "/auth/email/verification-status", params={"request_id": request_id}
    ).json()
    assert status == {"verified": False, "expired": False}

    # 매직링크 검증 (POST)
    resp = client.post("/auth/email/verify", json={"token": raw_token})
    assert resp.status_code == 200, resp.text

    # 폴링: 인증됨
    status = client.get(
        "/auth/email/verification-status", params={"request_id": request_id}
    ).json()
    assert status == {"verified": True, "expired": False}

    # 가입
    resp = client.post(
        "/auth/signup",
        json={
            "request_id": request_id,
            "email": "siwoo@example.com",
            "password": "abcd1234!",
            "password_confirm": "abcd1234!",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["email"] == "siwoo@example.com"
    assert body["nickname"] == "siwoo"  # local-part 기반
    assert body["provider"] == "local"

    # 가입 즉시 로그인 → 쿠키 세팅
    assert "access_token" in resp.cookies
    assert "refresh_token" in resp.cookies

    # DB: verified/consumed 기록, 유저 생성
    verification = (
        db.query(EmailVerification).filter_by(request_id=request_id).one()
    )
    assert verification.verified_at is not None
    assert verification.consumed_at is not None
    user = db.query(User).filter_by(email="siwoo@example.com").one()
    assert user.is_active is True
    assert user.password_hash is not None


def test_signup_rejects_expired_token(client, monkeypatch, db):
    """만료된 인증 요청으로 verify 시 거부."""
    request_id, raw_token = _request_verification(client, monkeypatch)
    # 만료 처리
    v = db.query(EmailVerification).filter_by(request_id=request_id).one()
    v.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    db.commit()

    resp = client.post("/auth/email/verify", json={"token": raw_token})
    assert resp.status_code == 400
    assert "만료" in resp.json()["detail"]


def test_signup_rejects_consumed_request_id(client, monkeypatch):
    """이미 소비된 request_id 로 재가입 거부."""
    request_id, raw_token = _request_verification(client, monkeypatch)
    client.post("/auth/email/verify", json={"token": raw_token})
    payload = {
        "request_id": request_id,
        "email": "siwoo@example.com",
        "password": "abcd1234!",
        "password_confirm": "abcd1234!",
    }
    assert client.post("/auth/signup", json=payload).status_code == 200
    # 두 번째 가입 시도 — 이미 소비됨
    resp = client.post("/auth/signup", json=payload)
    assert resp.status_code == 400
    assert "소비" in resp.json()["detail"] or "이미" in resp.json()["detail"]


def test_signup_rejects_email_mismatch(client, monkeypatch):
    """verify 한 이메일과 signup 이메일 불일치 거부."""
    request_id, raw_token = _request_verification(client, monkeypatch, "a@example.com")
    client.post("/auth/email/verify", json={"token": raw_token})
    resp = client.post(
        "/auth/signup",
        json={
            "request_id": request_id,
            "email": "different@example.com",
            "password": "abcd1234!",
            "password_confirm": "abcd1234!",
        },
    )
    assert resp.status_code == 400
    assert "일치" in resp.json()["detail"]


def test_signup_rejects_unverified(client, monkeypatch):
    """verify 안 한 채 signup 시 거부."""
    request_id, _ = _request_verification(client, monkeypatch)
    resp = client.post(
        "/auth/signup",
        json={
            "request_id": request_id,
            "email": "siwoo@example.com",
            "password": "abcd1234!",
            "password_confirm": "abcd1234!",
        },
    )
    assert resp.status_code == 400


def test_nickname_collision_suffix(client, monkeypatch, db):
    """같은 local-part 이메일 가입 시 닉네임에 숫자 suffix."""
    # 첫 유저: siwoo@example.com → siwoo
    r1, t1 = _request_verification(client, monkeypatch, "siwoo@example.com")
    client.post("/auth/email/verify", json={"token": t1})
    client.post(
        "/auth/signup",
        json={
            "request_id": r1,
            "email": "siwoo@example.com",
            "password": "abcd1234!",
            "password_confirm": "abcd1234!",
        },
    )
    # 둘째 유저: siwoo@other.com → siwoo1
    r2, t2 = _request_verification(client, monkeypatch, "siwoo@other.com")
    client.post("/auth/email/verify", json={"token": t2})
    resp = client.post(
        "/auth/signup",
        json={
            "request_id": r2,
            "email": "siwoo@other.com",
            "password": "abcd1234!",
            "password_confirm": "abcd1234!",
        },
    )
    assert resp.json()["nickname"] == "siwoo1"


# ── 로그인/refresh/logout/me ─────────────────────────────────
def _create_local_user(db, email="login@example.com", password="abcd1234!"):
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


def test_login_success_and_me(client, db):
    _create_local_user(db)
    resp = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    assert resp.status_code == 200, resp.text
    assert "access_token" in resp.cookies

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "login@example.com"


def test_login_wrong_password_same_message(client, db):
    """존재하는 계정, 틀린 비번 → 401 고정 메시지."""
    _create_local_user(db)
    resp = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "wrong123!"}
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "메일 또는 비밀번호가 잘못되었습니다."


def test_login_unknown_email_same_message(client, db):
    """존재하지 않는 계정 → 동일한 401 메시지 (계정 유무 노출 금지)."""
    resp = client.post(
        "/auth/login", json={"email": "nope@example.com", "password": "abcd1234!"}
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "메일 또는 비밀번호가 잘못되었습니다."


def test_me_requires_auth(client):
    assert client.get("/auth/me").status_code == 401


def test_refresh_rotation_and_logout(client, db):
    """refresh rotation → 기존 토큰 폐기 → logout 후 refresh 거부."""
    from cocktail_mate_db.models import RefreshToken

    _create_local_user(db)
    login = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    old_refresh = login.cookies["refresh_token"]

    # refresh 호출
    resp = client.post("/auth/refresh", cookies={"refresh_token": old_refresh})
    assert resp.status_code == 200, resp.text
    new_refresh = resp.cookies["refresh_token"]
    assert new_refresh != old_refresh

    # 기존(old) refresh 는 rotation 으로 폐기됨 → 재사용 거부
    reuse = client.post("/auth/refresh", cookies={"refresh_token": old_refresh})
    assert reuse.status_code == 401

    # DB: old 토큰 revoked
    old_hash = hash_token(old_refresh)
    old_row = db.query(RefreshToken).filter_by(token_hash=old_hash).one()
    assert old_row.revoked_at is not None

    # logout → new refresh 폐기
    client.post("/auth/logout", cookies={"refresh_token": new_refresh})
    after = client.post("/auth/refresh", cookies={"refresh_token": new_refresh})
    assert after.status_code == 401
