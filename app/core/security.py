"""인증 보안 유틸 — JWT 발급/검증, 고엔트로피 토큰 생성/해시.

- JWT: python-jose(HS256). access/refresh 를 payload의 `type`으로 구분한다.
- refresh 토큰: `secrets.token_urlsafe`(고엔트로피) → 저장은 sha256 해시로만.
  (고엔트로피 토큰은 argon2 불필요 — 사전공격이 불가하므로 sha256으로 충분.)

소셜 로그인 전용이므로 비밀번호 해시/검증은 없다.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.core.config import get_settings

ALGORITHM = "HS256"


def _create_token(user_id: int, token_type: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
        # jti: 같은 초에 재발급해도 토큰(및 그 sha256 해시)이 항상 달라지게 하는 nonce.
        # (refresh rotation 시 동일 토큰/해시 충돌 방지)
        "jti": secrets.token_urlsafe(8),
    }
    settings = get_settings()
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def create_access_token(user_id: int) -> str:
    """access 토큰(기본 30분) 발급."""
    settings = get_settings()
    return _create_token(
        user_id, "access", timedelta(minutes=settings.access_token_expire_minutes)
    )


def create_refresh_token(user_id: int) -> str:
    """refresh 토큰(기본 14일) 발급. 본문은 JWT지만 rotation/폐기는 DB 해시로 관리한다."""
    settings = get_settings()
    return _create_token(
        user_id, "refresh", timedelta(days=settings.refresh_token_expire_days)
    )


def decode_token(token: str) -> dict:
    """JWT 검증/디코드. 만료·서명 오류 시 jose.JWTError 를 던진다."""
    settings = get_settings()
    return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])


def generate_random_token() -> str:
    """추측 불가한 고엔트로피 토큰(refresh 원본)."""
    return secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    """고엔트로피 토큰의 DB 저장용 sha256 해시(hex)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# JWTError 를 밖에서 잡을 수 있게 재노출.
__all__ = [
    "JWTError",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "generate_random_token",
    "hash_token",
]
