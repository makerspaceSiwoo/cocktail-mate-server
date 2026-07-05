"""비밀번호 정책 검증 (signup 422) 테스트.

정책: 최소 8자, 영문+숫자+특수문자(!@#$) 각 1개 이상, 그 외 문자 금지.
"""
from __future__ import annotations

import pytest


def _signup(client, monkeypatch, password, email="pw@example.com"):
    """verify 까지 마친 뒤 주어진 비번으로 signup 시도 → 응답 반환."""
    captured = {}

    def _fake_send(to, verify_url):
        captured["url"] = verify_url

    monkeypatch.setattr("app.auth.service.send_verification_email", _fake_send)
    r = client.post("/auth/email/request-verification", json={"email": email})
    request_id = r.json()["request_id"]
    raw_token = captured["url"].split("token=", 1)[1]
    client.post("/auth/email/verify", json={"token": raw_token})
    return client.post(
        "/auth/signup",
        json={
            "request_id": request_id,
            "email": email,
            "nickname": "테스터",
            "password": password,
            "password_confirm": password,
        },
    )


@pytest.mark.parametrize(
    "password",
    [
        "ab1!",          # 8자 미만
        "abcdefgh!",     # 숫자 없음
        "12345678!",     # 영문 없음
        "abcd12345",     # 특수문자 없음
        "abcd1234%",     # 허용 외 문자(%)
        "abcd 1234!",    # 공백(허용 외)
    ],
)
def test_password_policy_rejected(client, monkeypatch, password):
    resp = _signup(client, monkeypatch, password)
    assert resp.status_code == 422, f"{password!r} → {resp.status_code}"


def test_password_mismatch_rejected(client, monkeypatch):
    """password != password_confirm → 422."""
    captured = {}

    def _fake_send(to, verify_url):
        captured["url"] = verify_url

    monkeypatch.setattr("app.auth.service.send_verification_email", _fake_send)
    r = client.post(
        "/auth/email/request-verification", json={"email": "mm@example.com"}
    )
    request_id = r.json()["request_id"]
    raw_token = captured["url"].split("token=", 1)[1]
    client.post("/auth/email/verify", json={"token": raw_token})
    resp = client.post(
        "/auth/signup",
        json={
            "request_id": request_id,
            "email": "mm@example.com",
            "nickname": "테스터",
            "password": "abcd1234!",
            "password_confirm": "abcd9999!",
        },
    )
    assert resp.status_code == 422


def test_valid_password_accepted(client, monkeypatch):
    resp = _signup(client, monkeypatch, "abcd1234!", email="ok@example.com")
    assert resp.status_code == 200, resp.text
