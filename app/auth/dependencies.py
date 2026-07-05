"""인증 의존성 — HttpOnly access_token 쿠키에서 현재 유저를 추출한다.

OAuth2PasswordBearer(Authorization 헤더)가 아니라 쿠키에서 읽는다.
- get_current_user: 미인증/무효 시 401.
- get_optional_user: 미인증/무효 시 None (비로그인 허용 엔드포인트용).
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, status
from sqlalchemy.orm import Session

from cocktail_mate_db.models import User

from app.auth.repository import AuthRepository
from app.core.database import get_db
from app.core.security import JWTError, decode_token

_repository = AuthRepository()

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="인증이 필요합니다.",
)


def _resolve_user(token: str | None, db: Session) -> User | None:
    """access 쿠키 → 유저. 실패 시 None."""
    if not token:
        return None
    try:
        payload = decode_token(token)
    except JWTError:
        return None
    if payload.get("type") != "access":
        return None
    sub = payload.get("sub")
    if sub is None:
        return None
    try:
        user_id = int(sub)
    except (TypeError, ValueError):
        return None

    user = _repository.get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        return None
    return user


def get_current_user(
    access_token: Annotated[str | None, Cookie()] = None,
    db: Session = Depends(get_db),
) -> User:
    user = _resolve_user(access_token, db)
    if user is None:
        raise _CREDENTIALS_EXCEPTION
    return user


def get_optional_user(
    access_token: Annotated[str | None, Cookie()] = None,
    db: Session = Depends(get_db),
) -> User | None:
    return _resolve_user(access_token, db)


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
