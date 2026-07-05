"""매직 링크 메일 발송.

발송 백엔드는 설정에 따라 세 가지 중 하나로 선택된다:
- SMTP: SMTP_HOST/USER/PASSWORD 가 모두 설정되면 stdlib smtplib 로 발송 (Gmail SMTP 등).
- Resend: MAIL_API_KEY 가 설정되면 SDK 없이 httpx 로 Resend HTTP API 를 호출한다.
- 콘솔: 위 둘 다 비어 있으면 개발용 콘솔 백엔드로 동작 — 링크를 로그로만 출력한다.
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

import httpx

from app.core.config import get_settings

logger = logging.getLogger("app.auth.mail")

RESEND_API_URL = "https://api.resend.com/emails"

MAIL_SUBJECT = "칵테일메이트 이메일 인증"


def _build_html(verify_url: str) -> str:
    """인증 메일 HTML 본문을 생성한다."""
    return (
        "<p>칵테일메이트 이메일 인증</p>"
        f'<p>아래 링크를 눌러 인증을 완료하세요:</p>'
        f'<p><a href="{verify_url}">{verify_url}</a></p>'
        "<p>본인이 요청하지 않았다면 이 메일을 무시하세요.</p>"
    )


def send_verification_email(email: str, verify_url: str) -> None:
    """가입 인증용 매직 링크 메일을 발송한다.

    백엔드 선택 우선순위:
      1. SMTP (SMTP_HOST/USER/PASSWORD 모두 설정) — smtplib 로 발송.
      2. Resend HTTP API (MAIL_API_KEY 설정) — httpx 로 발송.
      3. 콘솔 백엔드 (둘 다 미설정) — 링크를 로그로 출력하고 종료.

    발송 실패 시 예외를 던지지 않고 로그만 남긴다
    (호출부가 request_id 는 이미 발급했고, 사용자는 재요청 가능하므로).
    """
    settings = get_settings()

    if settings.smtp_host and settings.smtp_user and settings.smtp_password:
        _send_via_smtp(settings, email, verify_url)
        return

    if settings.mail_api_key:
        _send_via_resend(settings, email, verify_url)
        return

    logger.info("[MAIL:console] to=%s verify_url=%s", email, verify_url)


def _send_via_smtp(settings, email: str, verify_url: str) -> None:
    """SMTP(STARTTLS) 로 인증 메일을 발송한다 (Gmail SMTP 등).

    Gmail 은 인증한 계정으로 From 헤더를 덮어쓰므로 MAIL_FROM 은 Gmail 주소
    (또는 "이름 <addr@gmail.com>") 로 두는 것을 권장한다.
    """
    msg = EmailMessage()
    msg["From"] = settings.mail_from
    msg["To"] = email
    msg["Subject"] = MAIL_SUBJECT
    msg.set_content(verify_url)  # plain-text fallback
    msg.add_alternative(_build_html(verify_url), subtype="html")

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
    except Exception:  # noqa: BLE001 - 발송 실패는 로깅만 (요청 흐름 차단 안 함)
        logger.exception("verification email(SMTP) 발송 실패 to=%s", email)


def _send_via_resend(settings, email: str, verify_url: str) -> None:
    """Resend HTTP API 로 인증 메일을 발송한다."""
    try:
        resp = httpx.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.mail_api_key}"},
            json={
                "from": settings.mail_from,
                "to": [email],
                "subject": MAIL_SUBJECT,
                "html": _build_html(verify_url),
            },
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 - 발송 실패는 로깅만 (요청 흐름 차단 안 함)
        logger.exception("verification email 발송 실패 to=%s", email)
