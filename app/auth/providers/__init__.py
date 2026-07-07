"""소셜 provider 레지스트리.

새 provider 추가: 클래스 구현 후 아래 `_PROVIDERS` 에 인스턴스 한 줄만 등록하면
라우트(/auth/{provider}/...)가 자동으로 인식한다.
"""
from __future__ import annotations

from .base import SocialAuthError, SocialProfile, SocialProvider
from .kakao import KakaoProvider

_PROVIDERS: dict[str, SocialProvider] = {
    "kakao": KakaoProvider(),
    # 구글 확장: providers/google.py 의 GoogleProvider + config 의 google_* env 를 채운 뒤 아래 주석 해제.
    # "google": GoogleProvider(),
}


def get_provider(name: str) -> SocialProvider | None:
    """이름으로 등록된 provider 조회 (없으면 None)."""
    return _PROVIDERS.get(name)


__all__ = [
    "SocialAuthError",
    "SocialProfile",
    "SocialProvider",
    "get_provider",
]
