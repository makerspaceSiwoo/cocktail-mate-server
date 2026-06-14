"""유저/인증 응답 스키마."""
from pydantic import BaseModel


class SignupResponse(BaseModel):
    userId: int
    nickname: str
    email: str
    message: str


class LoginResponse(BaseModel):
    userId: int
    nickname: str
    email: str
    accessToken: str
    message: str


class LogoutResponse(BaseModel):
    message: str
    profileImageUrl: str


class MyInfoResponse(BaseModel):
    userId: int
    nickname: str
    email: str
    profileImageUrl: str
