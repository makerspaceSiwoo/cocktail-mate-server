"""Rate limiting 429 테스트 (client_rl 픽스처로 limiter 활성)."""
from __future__ import annotations



def test_verification_status_ip_rate_limit(client_rl, monkeypatch):
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


def test_request_verification_email_rate_limit(client_rl, monkeypatch):
    """POST request-verification: 동일 이메일 1회/분 → 2번째 429."""
    monkeypatch.setattr(
        "app.auth.service.send_verification_email", lambda *a, **k: None
    )
    first = client_rl.post(
        "/auth/email/request-verification", json={"email": "rl@example.com"}
    )
    assert first.status_code == 200
    second = client_rl.post(
        "/auth/email/request-verification", json={"email": "rl@example.com"}
    )
    assert second.status_code == 429
