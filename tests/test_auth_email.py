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
            "nickname": "시우",
            "password": "abcd1234!",
            "password_confirm": "abcd1234!",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["email"] == "siwoo@example.com"
    assert body["nickname"] == "시우"  # 입력값 반영
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
        "nickname": "시우",
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
            "nickname": "다른유저",
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
            "nickname": "시우",
            "password": "abcd1234!",
            "password_confirm": "abcd1234!",
        },
    )
    assert resp.status_code == 400


def _signup(client, monkeypatch, email, nickname, password="abcd1234!"):
    """request→verify→signup 을 한 번에 수행하고 응답을 반환."""
    request_id, raw_token = _request_verification(client, monkeypatch, email)
    client.post("/auth/email/verify", json={"token": raw_token})
    return client.post(
        "/auth/signup",
        json={
            "request_id": request_id,
            "email": email,
            "nickname": nickname,
            "password": password,
            "password_confirm": password,
        },
    )


def test_signup_uses_input_nickname(client, monkeypatch, db):
    """입력한 닉네임이 그대로 반영된다(자동 생성 아님)."""
    resp = _signup(client, monkeypatch, "siwoo@example.com", "칵테일러123")
    assert resp.status_code == 200, resp.text
    assert resp.json()["nickname"] == "칵테일러123"


def test_signup_duplicate_nickname_conflict(client, monkeypatch, db):
    """이미 사용 중인 닉네임으로 가입 시 409."""
    first = _signup(client, monkeypatch, "a@example.com", "중복닉")
    assert first.status_code == 200, first.text
    second = _signup(client, monkeypatch, "b@example.com", "중복닉")
    assert second.status_code == 409
    assert "닉네임" in second.json()["detail"]


def test_signup_invalid_nickname_rejected(client, monkeypatch):
    """닉네임 형식 위반(1자/11자/특수문자) → 422."""
    for bad in ["a", "abcdefghijk", "닉네임!", "with space"]:
        request_id, raw_token = _request_verification(
            client, monkeypatch, "c@example.com"
        )
        client.post("/auth/email/verify", json={"token": raw_token})
        resp = client.post(
            "/auth/signup",
            json={
                "request_id": request_id,
                "email": "c@example.com",
                "nickname": bad,
                "password": "abcd1234!",
                "password_confirm": "abcd1234!",
            },
        )
        assert resp.status_code == 422, f"nickname={bad!r} → {resp.status_code}"


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


def _refresh_set_cookie_header(resp) -> str:
    """응답의 Set-Cookie 헤더들 중 refresh_token 을 세팅하는 항목을 반환."""
    for header in resp.headers.get_list("set-cookie"):
        if header.startswith("refresh_token="):
            return header
    raise AssertionError(
        f"refresh_token Set-Cookie 없음: {resp.headers.get_list('set-cookie')}"
    )


def test_login_success_and_me(client, db):
    _create_local_user(db)
    resp = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    assert resp.status_code == 200, resp.text
    assert "access_token" in resp.cookies

    # refresh 쿠키는 Path=/auth 로 스코프되어 /auth/refresh 와 /auth/logout 에 도달해야 한다.
    refresh_cookie = _refresh_set_cookie_header(resp)
    assert "Path=/auth" in refresh_cookie, refresh_cookie
    # /auth/refresh 로만 좁게 스코프되면 logout 이 폐기 못 하므로 회귀 방지.
    assert "Path=/auth/refresh" not in refresh_cookie, refresh_cookie

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


def test_refresh_rotation_reuse_rejected(client, db):
    """refresh rotation → 기존 토큰 폐기 → 재사용 거부.

    쿠키는 클라이언트 쿠키 jar 를 통해 브라우저처럼 전달한다(hand-injection 없음).
    """
    from cocktail_mate_db.models import RefreshToken

    _create_local_user(db)
    login = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    old_refresh = login.cookies["refresh_token"]

    # refresh 호출 — jar 에 담긴 refresh 쿠키가 그대로 전송됨.
    resp = client.post("/auth/refresh")
    assert resp.status_code == 200, resp.text
    new_refresh = resp.cookies["refresh_token"]
    assert new_refresh != old_refresh

    # 기존(old) refresh 는 rotation 으로 폐기됨 → 재사용 거부.
    # (jar 는 이미 new_refresh 로 갱신됐으므로 old 는 명시적으로 실어 재사용을 시뮬레이트.)
    client.cookies.set("refresh_token", old_refresh, path="/auth")
    reuse = client.post("/auth/refresh")
    assert reuse.status_code == 401

    # DB: 삭제 기반 단일 세션 — old 행은 revoked 가 아니라 삭제되어 존재하지 않는다.
    old_hash = hash_token(old_refresh)
    assert db.query(RefreshToken).filter_by(token_hash=old_hash).one_or_none() is None


def test_logout_revokes_refresh_server_side(client, db):
    """logout 은 hand-injection 없이도 서버측에서 refresh 를 폐기해야 한다.

    Critical 2 회귀 방지: refresh 쿠키가 Path=/auth 라서 브라우저(그리고 jar)가
    /auth/logout 에도 쿠키를 실어 보내고, 서버가 행을 삭제 → 이후 refresh 401.
    이전엔 Path=/auth/refresh 라 logout 에 쿠키가 도달하지 못해 토큰이 살아있었다.
    """
    from cocktail_mate_db.models import RefreshToken

    _create_local_user(db)
    login = client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    refresh = login.cookies["refresh_token"]

    # jar 에 담긴 쿠키만으로 logout — hand-injected cookies 없음.
    logout = client.post("/auth/logout")
    assert logout.status_code == 200, logout.text

    # DB: 삭제 기반 단일 세션 — 행이 revoked 가 아니라 삭제되어 존재하지 않는다.
    row = (
        db.query(RefreshToken).filter_by(token_hash=hash_token(refresh)).one_or_none()
    )
    assert row is None

    # 폐기된 토큰으로 refresh 시도 → 401 (jar 는 logout 응답의 delete-cookie 로 비워졌을 수
    # 있으므로 명시적으로 실어 서버측 폐기를 검증).
    client.cookies.set("refresh_token", refresh, path="/auth")
    after = client.post("/auth/refresh")
    assert after.status_code == 401


def _user_refresh_row_count(db, user_id: int) -> int:
    """해당 user 의 refresh_tokens 행 수를 DB 에서 직접 카운트."""
    from cocktail_mate_db.models import RefreshToken

    # 세션 캐시가 아닌 실제 DB 상태를 반영하도록 만료 후 카운트.
    db.expire_all()
    return db.query(RefreshToken).filter_by(user_id=user_id).count()


def test_single_session_invariant_one_row_per_user(client, db):
    """단일 세션 불변식: login→refresh→refresh, login→login 모두 user 당 refresh 행 == 1."""
    user = _create_local_user(db)

    # login → refresh → refresh
    client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    assert _user_refresh_row_count(db, user.id) == 1
    r1 = client.post("/auth/refresh")
    assert r1.status_code == 200, r1.text
    assert _user_refresh_row_count(db, user.id) == 1
    r2 = client.post("/auth/refresh")
    assert r2.status_code == 200, r2.text
    assert _user_refresh_row_count(db, user.id) == 1

    # 별도로: 재로그인(login → login) 도 누적 없이 1행 유지.
    client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    assert _user_refresh_row_count(db, user.id) == 1
    client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    assert _user_refresh_row_count(db, user.id) == 1


def test_logout_leaves_zero_rows(client, db):
    """logout 후 해당 user 의 refresh_tokens 행 == 0 (revoked 잔재 없음)."""
    user = _create_local_user(db)
    client.post(
        "/auth/login", json={"email": "login@example.com", "password": "abcd1234!"}
    )
    assert _user_refresh_row_count(db, user.id) == 1

    logout = client.post("/auth/logout")
    assert logout.status_code == 200, logout.text
    assert _user_refresh_row_count(db, user.id) == 0
