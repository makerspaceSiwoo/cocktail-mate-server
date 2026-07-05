"""매직 링크 메일 발송.

SDK 없이 httpx 로 Resend HTTP API 를 호출한다.
MAIL_API_KEY 가 비어 있으면 개발용 콘솔 백엔드로 동작 — 링크를 로그로만 출력한다.
"""
from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger("app.auth.mail")

RESEND_API_URL = "https://api.resend.com/emails"


def send_verification_email(email: str, verify_url: str) -> None:
    """가입 인증용 매직 링크 메일을 발송한다.

    - 개발(콘솔 백엔드): MAIL_API_KEY 미설정 시 링크를 로그로 출력하고 종료.
    - 운영: Resend HTTP API 로 발송. 실패 시 예외를 던지지 않고 로그만 남긴다
      (호출부가 request_id 는 이미 발급했고, 사용자는 재요청 가능하므로).
    """
    settings = get_settings()

    if not settings.mail_api_key:
        logger.info("[MAIL:console] to=%s verify_url=%s", email, verify_url)
        return

    html = (
        "<p>칵테일메이트 이메일 인증</p>"
        f'<p>아래 링크를 눌러 인증을 완료하세요:</p>'
        f'<p><a href="{verify_url}">{verify_url}</a></p>'
        "<p>본인이 요청하지 않았다면 이 메일을 무시하세요.</p>"
    )
    try:
        resp = httpx.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {settings.mail_api_key}"},
            json={
                "from": settings.mail_from,
                "to": [email],
                "subject": "칵테일메이트 이메일 인증",
                "html": html,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
    except Exception:  # noqa: BLE001 - 발송 실패는 로깅만 (요청 흐름 차단 안 함)
        logger.exception("verification email 발송 실패 to=%s", email)
