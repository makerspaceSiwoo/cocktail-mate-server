"""인증 보안 유틸 — 비밀번호 해시, JWT 발급/검증, 고엔트로피 토큰 생성/해시.

- 비밀번호: pwdlib(argon2) — 사람이 정한 저엔트로피 값이라 느린 KDF 필요.
- JWT: python-jose(HS256). access/refresh 를 payload의 `type`으로 구분한다.
- 매직링크/refresh 토큰: `secrets.token_urlsafe`(고엔트로피) → 저장은 sha256 해시로만.
  (고엔트로피 토큰은 argon2 불필요 — 사전공격이 불가하므로 sha256으로 충분.)
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from pwdlib import PasswordHash

from app.core.config import get_settings

ALGORITHM = "HS256"

# argon2 권장 프리셋 — pwdlib 기본 팩토리.
_password_hash = PasswordHash.recommended()

# 타이밍 공격 방어용 더미 해시: 유저가 없어도 verify_password 를 실제로 한 번 돌려
# 응답 시간을 일정하게 만든다. (아래 verify_dummy_password 참고)
_DUMMY_PASSWORD_HASH = _password_hash.hash("cocktail-mate-dummy-timing-guard")


def hash_password(password: str) -> str:
    """평문 비밀번호를 argon2 해시로 변환한다."""
    return _password_hash.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """평문 비밀번호와 저장된 해시를 대조한다."""
    return _password_hash.verify(password, password_hash)


def verify_dummy_password(password: str) -> None:
    """타이밍 공격 방어: (provider,email) 유저가 없을 때도 argon2 검증을 수행한다.

    반환값은 항상 무시 — 목적은 오직 '유저 존재 여부와 무관하게 동일한 지연'.
    """
    _password_hash.verify(password, _DUMMY_PASSWORD_HASH)


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


# JWTError 를 밖에서 잡을 수 있게 재노출.
__all__ = [
    "JWTError",
    "hash_password",
    "verify_password",
    "verify_dummy_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "generate_random_token",
    "hash_token",
]


def generate_random_token() -> str:
    """추측 불가한 고엔트로피 토큰(매직링크/refresh 원본)."""
    return secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    """고엔트로피 토큰의 DB 저장용 sha256 해시(hex)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
