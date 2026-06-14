"""유저/인증 라우터. 경로/응답은 기존 구현과 동일 (prefix 없음)."""
from fastapi import APIRouter

from app.user.schemas import (
    LoginResponse,
    LogoutResponse,
    MyInfoResponse,
    SignupResponse,
)
from app.user.service import UserService

router = APIRouter(tags=["user"])
service = UserService()


@router.post("/signup", response_model=SignupResponse)
def signup():
    return service.signup()


@router.post("/login", response_model=LoginResponse)
def login():
    return service.login()


@router.post("/logout", response_model=LogoutResponse)
def logout():
    return service.logout()


@router.get("/my/info", response_model=MyInfoResponse)
def get_my_info():
    return service.my_info()
