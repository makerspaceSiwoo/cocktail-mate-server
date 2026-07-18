"""인증 요청/응답 스키마.

소셜 로그인 전용 — 이메일/비밀번호 입력 스키마는 없다.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator


class UserResponse(BaseModel):
    """인증된 유저 공개 정보."""

    id: int
    email: str | None = None  # 소셜 프로필에 이메일이 없을 수 있음
    nickname: str
    provider: str
    profile_image_url: str | None = None


class MessageResponse(BaseModel):
    message: str


class NicknameChangeRequest(BaseModel):
    """닉네임 변경 요청 스키마."""

    nickname: str

    @field_validator("nickname")
    @classmethod
    def validate_nickname(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2 or len(value) > 10:
            raise ValueError("닉네임은 2~10자여야 합니다.")
        if (
            re.fullmatch(
                r"^[가-힣a-zA-Z0-9]+([-_][가-힣a-zA-Z0-9]+)*$",
                value,
            )
            is None
        ):
            raise ValueError(
                "닉네임은 2~10자의 한글, 영문, 숫자와 하이픈(-), 및줄(_)만 사용할 수 있으며, "
                "하이픈과 밑줄은 처음, 끝 또는 연속해서 사용할 수 없습니다."
            )
        return value
