"""Rate limiting 429 테스트 (client_rl 픽스처로 limiter 활성)."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _mock_mail(monkeypatch):
    """메일 발송은 항상 mock (rate limit 테스트는 발송 자체가 관심사가 아님)."""
    monkeypatch.setattr(
        "app.auth.service.send_verification_email", lambda *a, **k: None
    )


def test_verification_status_ip_rate_limit(client_rl):
    """GET /auth/email/verification-status: IP 30회/분 초과 시 429."""
    # 존재하지 않는 request_id — 200(만료 응답) 반환되지만 rate limit 은 걸림.
    codes = []
    for _ in range(35):
        r = client_rl.get(
            "/auth/email/verification-status", params={"request_id": "nope"}
        )
        codes.append(r.status_code)
    assert 429 in codes, codes
    # 처음 몇 개는 200
    assert codes[0] == 200


def test_request_verification_same_email_rate_limited(client_rl):
    """POST request-verification: 동일 이메일 1회/분 → 2번째 429."""
    first = client_rl.post(
        "/auth/email/request-verification", json={"email": "rl@example.com"}
    )
    assert first.status_code == 200
    second = client_rl.post(
        "/auth/email/request-verification", json={"email": "rl@example.com"}
    )
    assert second.status_code == 429


def test_request_verification_different_emails_do_not_collide(client_rl):
    """서로 다른 이메일은 (같은 IP 라도) 이메일 기준 버킷을 공유하지 않는다.

    Critical 1 회귀 방지: 예전에는 email key 가 항상 None → IP 폴백이라
    한 IP 에서 두 번째 이메일 요청이 첫 이메일의 버킷과 충돌해 429 였다.
    """
    first = client_rl.post(
        "/auth/email/request-verification", json={"email": "a@example.com"}
    )
    assert first.status_code == 200, first.text
    # 다른 이메일 — 이메일 1/분 버킷은 별개이므로 200 이어야 한다.
    second = client_rl.post(
        "/auth/email/request-verification", json={"email": "b@example.com"}
    )
    assert second.status_code == 200, second.text


def test_request_verification_ip_limit_independent_of_email(client_rl):
    """IP 10회/시간 은 이메일과 독립적으로 적용된다.

    매 요청마다 이메일을 바꿔 이메일 1/분·5/시간 버킷을 피하면,
    IP 10/시간 버킷만 소진되어 11번째부터 429 가 나와야 한다.
    """
    codes = []
    for i in range(12):
        r = client_rl.post(
            "/auth/email/request-verification",
            json={"email": f"ip{i}@example.com"},
        )
        codes.append(r.status_code)
    # 앞 10개는 200, 이후 IP 한도로 429.
    assert codes[:10] == [200] * 10, codes
    assert 429 in codes[10:], codes


def test_login_same_email_rate_limited(client_rl):
    """POST /auth/login: 동일 이메일 5회/5분 초과 시 429 (계정 없어도 rate limit 우선)."""
    codes = []
    for _ in range(7):
        r = client_rl.post(
            "/auth/login",
            json={"email": "loginrl@example.com", "password": "abcd1234!"},
        )
        codes.append(r.status_code)
    # 처음 5개는 인증 실패(401), 6번째부터 이메일 한도 429.
    assert codes[:5] == [401] * 5, codes
    assert 429 in codes[5:], codes


def test_login_different_emails_do_not_collide(client_rl):
    """login 도 서로 다른 이메일은 이메일 기준 버킷을 공유하지 않는다."""
    r1 = client_rl.post(
        "/auth/login", json={"email": "one@example.com", "password": "abcd1234!"}
    )
    r2 = client_rl.post(
        "/auth/login", json={"email": "two@example.com", "password": "abcd1234!"}
    )
    # 둘 다 계정 없음 → 401 이지만, 429(이메일 충돌)는 아니어야 한다.
    assert r1.status_code == 401, r1.text
    assert r2.status_code == 401, r2.text
