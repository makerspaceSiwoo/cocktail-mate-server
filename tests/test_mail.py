"""매직 링크 메일 발송 백엔드 테스트.

- MAIL_API_KEY 설정 시 Resend HTTP API(httpx.post) 로 발송 — 실제 네트워크 미접속(mock).
- 아무것도 미설정이면 콘솔 백엔드 — Resend 호출 없이 예외 없이 종료.
- Resend 발송 실패는 예외를 던지지 않고 삼킨다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.auth import mail
from app.core.config import Settings


def _settings(**overrides) -> Settings:
    """mail_* 필드를 명시적으로 넣은 Settings 를 구성한다."""
    base = dict(
        mail_api_key="",
        mail_from="sender@example.com",
    )
    base.update(overrides)
    return Settings(**base)


def test_resend_backend_sends_via_httpx(monkeypatch):
    """MAIL_API_KEY 설정 시 Resend HTTP API 로 발송하고 페이로드가 올바르다."""
    settings = _settings(mail_api_key="re_test_key")
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    verify_url = "https://front.example/verify?token=abc123"
    recipient = "user@example.com"

    with patch("app.auth.mail.httpx.post") as httpx_post:
        httpx_post.return_value = MagicMock(status_code=200)
        mail.send_verification_email(recipient, verify_url)

    httpx_post.assert_called_once()
    kwargs = httpx_post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer re_test_key"
    payload = kwargs["json"]
    assert payload["to"] == [recipient]
    assert payload["from"] == "sender@example.com"
    assert payload["subject"] == "칵테일메이트 이메일 인증"
    assert verify_url in payload["html"]


def test_console_backend_when_nothing_configured(monkeypatch):
    """Resend 미설정 시 어떤 발송도 하지 않고 예외 없이 종료한다."""
    settings = _settings()
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    with patch("app.auth.mail.httpx.post") as httpx_post:
        mail.send_verification_email("user@example.com", "https://front/verify?t=x")

    httpx_post.assert_not_called()


def test_password_reset_resend_backend(monkeypatch):
    """비밀번호 재설정 메일도 Resend 로 발송되며 제목/링크가 올바르다."""
    settings = _settings(mail_api_key="re_test_key")
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    reset_url = "https://front.example/reset-password?token=abc123"
    with patch("app.auth.mail.httpx.post") as httpx_post:
        httpx_post.return_value = MagicMock(status_code=200)
        mail.send_password_reset_email("user@example.com", reset_url)

    httpx_post.assert_called_once()
    payload = httpx_post.call_args.kwargs["json"]
    assert payload["subject"] == "칵테일메이트 비밀번호 재설정"
    assert reset_url in payload["html"]


def test_resend_failure_is_swallowed(monkeypatch):
    """Resend 발송 중 예외가 나도 send_verification_email 은 예외를 전파하지 않는다."""
    settings = _settings(mail_api_key="re_test_key")
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    failing = MagicMock(side_effect=OSError("connection refused"))
    with patch("app.auth.mail.httpx.post", failing):
        # 예외가 밖으로 새면 이 호출에서 raise 됨 → 테스트 실패
        mail.send_verification_email("user@example.com", "https://front/verify?t=x")
