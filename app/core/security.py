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


# ── 비밀번호 재설정 토큰 (DB 저장 없음) ─────────────────────────
# 서명키에 유저의 현재 password_hash 를 섞는다 → 비번이 바뀌면 서명이 자동 무효(1회용).
def _reset_signing_key(password_hash: str) -> str:
    settings = get_settings()
    return settings.secret_key + password_hash


def create_password_reset_token(user_id: int, password_hash: str) -> str:
    """비밀번호 재설정 JWT 발급.

    서명키 = secret_key + 현재 password_hash. 재설정으로 password_hash 가 바뀌면
    같은 토큰의 서명이 더 이상 검증되지 않으므로 자연스럽게 1회용이 된다.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "password_reset",
        "iat": now,
        "exp": now + timedelta(minutes=settings.reset_token_expire_minutes),
    }
    return jwt.encode(
        payload, _reset_signing_key(password_hash), algorithm=ALGORITHM
    )


def extract_reset_token_subject(token: str) -> int:
    """서명 검증 없이 sub(user_id)만 추출한다.

    password_hash 를 알아야 정식 검증이 가능한데, 그 전에 어떤 유저인지 먼저 알아야 하므로
    sub 만 우선 꺼낸다. 반드시 이후 decode_password_reset_token 으로 정식 검증할 것.
    실패 시 jose.JWTError 를 던진다.
    """
    # 서명/만료 모두 여기서는 검증하지 않는다 — sub 만 추출하는 것이 목적.
    # (정식 검증은 password_hash 를 아는 decode_password_reset_token 에서 수행.)
    payload = jwt.decode(
        token,
        key="",
        options={"verify_signature": False, "verify_exp": False},
    )
    sub = payload.get("sub")
    if sub is None:
        raise JWTError("sub 없음")
    return int(sub)


def decode_password_reset_token(token: str, password_hash: str) -> int:
    """secret_key+password_hash 키로 재설정 JWT 를 정식 검증한다.

    서명 불일치/만료/type≠password_reset 이면 jose.JWTError 를 던진다.
    성공 시 user_id(int) 반환.
    """
    payload = jwt.decode(
        token, _reset_signing_key(password_hash), algorithms=[ALGORITHM]
    )
    if payload.get("type") != "password_reset":
        raise JWTError("잘못된 토큰 타입")
    sub = payload.get("sub")
    if sub is None:
        raise JWTError("sub 없음")
    return int(sub)


# JWTError 를 밖에서 잡을 수 있게 재노출.
__all__ = [
    "JWTError",
    "hash_password",
    "verify_password",
    "verify_dummy_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "create_password_reset_token",
    "extract_reset_token_subject",
    "decode_password_reset_token",
    "generate_random_token",
    "hash_token",
]


def generate_random_token() -> str:
    """추측 불가한 고엔트로피 토큰(매직링크/refresh 원본)."""
    return secrets.token_urlsafe(32)


def hash_token(raw: str) -> str:
    """고엔트로피 토큰의 DB 저장용 sha256 해시(hex)."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
