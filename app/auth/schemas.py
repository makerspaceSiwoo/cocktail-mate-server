"""인증 요청/응답 스키마.

비밀번호 정책은 여기(pydantic validator)에서 강제한다:
- 최소 8자
- 영문/숫자/특수문자 각 1개 이상
- 특수문자는 `!@#$` 4종만 허용
- 허용 문자 화이트리스트 `^[A-Za-z0-9!@#$]{8,}$` (그 외 문자 포함 시 거부)
"""
from __future__ import annotations

import re

from pydantic import BaseModel, EmailStr, field_validator, model_validator

# 허용 문자 전체(길이 포함) 화이트리스트.
_PASSWORD_ALLOWED_RE = re.compile(r"^[A-Za-z0-9!@#$]{8,}$")
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")
_HAS_DIGIT_RE = re.compile(r"[0-9]")
_HAS_SPECIAL_RE = re.compile(r"[!@#$]")


def validate_password_policy(password: str) -> str:
    """비밀번호 정책을 검증하고, 통과하면 원본을 그대로 반환한다."""
    if len(password) < 8:
        raise ValueError("비밀번호는 최소 8자 이상이어야 합니다.")
    if not _PASSWORD_ALLOWED_RE.match(password):
        # 길이는 위에서 이미 검사 → 여기 걸리면 허용 외 문자 포함.
        raise ValueError("비밀번호에 허용되지 않은 문자가 포함되어 있습니다. (영문/숫자/!@#$ 만 허용)")
    if not _HAS_LETTER_RE.search(password):
        raise ValueError("비밀번호에는 영문이 1자 이상 포함되어야 합니다.")
    if not _HAS_DIGIT_RE.search(password):
        raise ValueError("비밀번호에는 숫자가 1자 이상 포함되어야 합니다.")
    if not _HAS_SPECIAL_RE.search(password):
        raise ValueError("비밀번호에는 특수문자(!@#$)가 1자 이상 포함되어야 합니다.")
    return password


# ── 이메일 인증 ─────────────────────────────────────────────
class RequestVerificationRequest(BaseModel):
    email: EmailStr


class RequestVerificationResponse(BaseModel):
    request_id: str


class VerificationStatusResponse(BaseModel):
    verified: bool
    expired: bool


class VerifyEmailRequest(BaseModel):
    token: str


class VerifyEmailResponse(BaseModel):
    message: str


# ── 가입/로그인 ─────────────────────────────────────────────
class SignupRequest(BaseModel):
    request_id: str
    email: EmailStr
    password: str
    password_confirm: str

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str) -> str:
        return validate_password_policy(v)

    @model_validator(mode="after")
    def _check_confirm(self) -> "SignupRequest":
        if self.password != self.password_confirm:
            raise ValueError("비밀번호가 일치하지 않습니다.")
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """인증된 유저 공개 정보(비밀번호 해시 등 민감 필드 제외)."""

    id: int
    email: str
    nickname: str
    provider: str
    profile_image_url: str | None = None


class MessageResponse(BaseModel):
    message: str
