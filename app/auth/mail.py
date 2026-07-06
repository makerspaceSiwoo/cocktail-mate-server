"""매직 링크 메일 발송.

발송 백엔드는 설정에 따라 두 가지 중 하나로 선택된다:
- Resend: MAIL_API_KEY 가 설정되면 SDK 없이 httpx 로 Resend HTTP API 를 호출한다.
- 콘솔: 비어 있으면 개발용 콘솔 백엔드로 동작 — 링크를 로그로만 출력한다.
"""
from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger("app.auth.mail")

RESEND_API_URL = "https://api.resend.com/emails"

MAIL_SUBJECT = "칵테일메이트 이메일 인증"
RESET_MAIL_SUBJECT = "칵테일메이트 비밀번호 재설정"


def _build_html(verify_url: str) -> str:
    """인증 메일 HTML 본문을 생성한다."""
    return (
        "<p>칵테일메이트 이메일 인증</p>"
        f'<p>아래 링크를 눌러 인증을 완료하세요:</p>'
        f'<p><a href="{verify_url}">{verify_url}</a></p>'
        "<p>본인이 요청하지 않았다면 이 메일을 무시하세요.</p>"
    )


def _build_reset_html(reset_url: str) -> str:
    """비밀번호 재설정 메일 HTML 본문을 생성한다."""
    return (
        "<p>칵테일메이트 비밀번호 재설정</p>"
        f'<p>아래 링크를 눌러 비밀번호를 재설정하세요:</p>'
        f'<p><a href="{reset_url}">{reset_url}</a></p>'
        "<p>본인이 요청하지 않았다면 이 메일을 무시하세요.</p>"
    )


def send_verification_email(email: str, verify_url: str) -> None:
    """가입 인증용 매직 링크 메일을 발송한다.

    백엔드 선택 우선순위:
      1. Resend HTTP API (MAIL_API_KEY 설정) — httpx 로 발송.
      2. 콘솔 백엔드 (미설정) — 링크를 로그로 출력하고 종료.

    발송 실패 시 예외를 던지지 않고 로그만 남긴다
    (호출부가 request_id 는 이미 발급했고, 사용자는 재요청 가능하므로).
    """
    _send_email(email, MAIL_SUBJECT, verify_url, _build_html(verify_url))


def send_password_reset_email(email: str, reset_url: str) -> None:
    """비밀번호 재설정 링크 메일을 발송한다.

    백엔드 선택/실패 처리는 send_verification_email 과 동일 (Resend>콘솔,
    실패 시 예외 대신 로그).
    """
    _send_email(email, RESET_MAIL_SUBJECT, reset_url, _build_reset_html(reset_url))


def _send_email(email: str, subject: str, url: str, html: str) -> None:
    """설정된 백엔드(Resend>콘솔)로 메일을 발송한다.

    발송 실패 시 예외를 던지지 않고 로그만 남긴다.
    """
    settings = get_settings()

    if settings.mail_api_key:
        _send_via_resend(settings, email, subject, html)
        return

    logger.info("[MAIL:console] to=%s subject=%s url=%s", email, subject, url)


def _send_via_resend(settings, email: str, subject: str, html: str) -> None:
    """Resend HTTP API 로 메일을 발송한다."""
    try:
        resp = httpx.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.mail_api_key}"},
            json={
                "from": settings.mail_from,
                "to": [email],
                "subject": subject,
                "html": html,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 - 발송 실패는 로깅만 (요청 흐름 차단 안 함)
        logger.exception("email 발송 실패 to=%s subject=%s", email, subject)
