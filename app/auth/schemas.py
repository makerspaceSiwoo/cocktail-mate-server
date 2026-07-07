"""인증 요청/응답 스키마.

소셜 로그인 전용 — 이메일/비밀번호 입력 스키마는 없다.
"""
from __future__ import annotations

from pydantic import BaseModel


class UserResponse(BaseModel):
    """인증된 유저 공개 정보."""

    id: int
    email: str | None = None  # 소셜 프로필에 이메일이 없을 수 있음
    nickname: str
    provider: str
    profile_image_url: str | None = None


class MessageResponse(BaseModel):
    message: str
