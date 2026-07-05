"""매직 링크 메일 발송 백엔드 테스트.

- SMTP 설정 시 smtplib 로 발송(starttls/login/send_message 호출) — 실제 네트워크 미접속(mock).
- 아무것도 미설정이면 콘솔 백엔드 — SMTP/Resend 호출 없이 예외 없이 종료.
- SMTP 발송 실패는 예외를 던지지 않고 삼킨다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.auth import mail
from app.core.config import Settings


def _settings(**overrides) -> Settings:
    """smtp_* 필드를 명시적으로 넣은 Settings 를 구성한다."""
    base = dict(
        mail_api_key="",
        mail_from="sender@gmail.com",
        smtp_host="",
        smtp_port=587,
        smtp_user="",
        smtp_password="",
    )
    base.update(overrides)
    return Settings(**base)


def test_smtp_backend_sends_via_smtplib(monkeypatch):
    """SMTP 설정 시 starttls/login/send_message 를 호출하고 메일 내용이 올바르다."""
    settings = _settings(
        smtp_host="smtp.gmail.com",
        smtp_user="me@gmail.com",
        smtp_password="app-password-16",
    )
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    verify_url = "https://front.example/verify?token=abc123"
    recipient = "user@example.com"

    with patch("app.auth.mail.smtplib.SMTP") as smtp_cls:
        smtp = smtp_cls.return_value.__enter__.return_value
        mail.send_verification_email(recipient, verify_url)

    smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=10)
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("me@gmail.com", "app-password-16")
    smtp.send_message.assert_called_once()

    sent_msg = smtp.send_message.call_args.args[0]
    assert sent_msg["To"] == recipient
    assert sent_msg["From"] == "sender@gmail.com"
    assert sent_msg["Subject"] == "칵테일메이트 이메일 인증"
    text_part = sent_msg.get_body(preferencelist=("plain",))  # plain-text fallback
    assert text_part is not None
    assert verify_url in text_part.get_content()
    html_part = sent_msg.get_body(preferencelist=("html",))
    assert html_part is not None
    assert verify_url in html_part.get_content()


def test_console_backend_when_nothing_configured(monkeypatch):
    """SMTP/Resend 미설정 시 어떤 발송도 하지 않고 예외 없이 종료한다."""
    settings = _settings()
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    with (
        patch("app.auth.mail.smtplib.SMTP") as smtp_cls,
        patch("app.auth.mail.httpx.post") as httpx_post,
    ):
        mail.send_verification_email("user@example.com", "https://front/verify?t=x")

    smtp_cls.assert_not_called()
    httpx_post.assert_not_called()


def test_password_reset_smtp_backend(monkeypatch):
    """비밀번호 재설정 메일도 SMTP 로 발송되며 제목/링크가 올바르다."""
    settings = _settings(
        smtp_host="smtp.gmail.com",
        smtp_user="me@gmail.com",
        smtp_password="app-password-16",
    )
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    reset_url = "https://front.example/reset-password?token=abc123"
    with patch("app.auth.mail.smtplib.SMTP") as smtp_cls:
        smtp = smtp_cls.return_value.__enter__.return_value
        mail.send_password_reset_email("user@example.com", reset_url)

    smtp.send_message.assert_called_once()
    sent_msg = smtp.send_message.call_args.args[0]
    assert sent_msg["Subject"] == "칵테일메이트 비밀번호 재설정"
    html_part = sent_msg.get_body(preferencelist=("html",))
    assert reset_url in html_part.get_content()


def test_smtp_failure_is_swallowed(monkeypatch):
    """SMTP 발송 중 예외가 나도 send_verification_email 은 예외를 전파하지 않는다."""
    settings = _settings(
        smtp_host="smtp.gmail.com",
        smtp_user="me@gmail.com",
        smtp_password="app-password-16",
    )
    monkeypatch.setattr(mail, "get_settings", lambda: settings)

    failing = MagicMock(side_effect=OSError("connection refused"))
    with patch("app.auth.mail.smtplib.SMTP", failing):
        # 예외가 밖으로 새면 이 호출에서 raise 됨 → 테스트 실패
        mail.send_verification_email("user@example.com", "https://front/verify?t=x")
