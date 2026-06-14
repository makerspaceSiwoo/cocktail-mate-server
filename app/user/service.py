"""유저/인증 비즈니스 로직. 응답은 기존 mock과 동일하게 유지."""
from app.user.repository import UserRepository


class UserService:
    def __init__(self, repository: UserRepository | None = None) -> None:
        self.repository = repository or UserRepository()

    def signup(self) -> dict:
        return {
            "userId": 1,
            "nickname": "mockUser",
            "email": "mock@example.com",
            "message": "회원가입 성공",
        }

    def login(self) -> dict:
        return {
            "userId": 1,
            "nickname": "mockUser",
            "email": "mock@example.com",
            "accessToken": "mock-access-token",
            "message": "로그인 성공",
        }

    def logout(self) -> dict:
        return {
            "message": "로그아웃 성공",
            "profileImageUrl": "https://fastly.picsum.photos/id/64/200/200.jpg",
        }

    def my_info(self) -> dict:
        return {
            "userId": 1,
            "nickname": "mockUser",
            "email": "mock@example.com",
            "profileImageUrl": "https://fastly.picsum.photos/id/64/200/200.jpg",
        }
