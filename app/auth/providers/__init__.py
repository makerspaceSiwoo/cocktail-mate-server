"""소셜 provider 레지스트리.

새 provider 추가: 클래스 구현 후 아래 `_PROVIDERS` 에 인스턴스 한 줄만 등록하면
라우트(/auth/{provider}/...)가 자동으로 인식한다.
"""
from __future__ import annotations

from .base import EmailConsentRequired, SocialAuthError, SocialProfile, SocialProvider
from .kakao import KakaoProvider

_PROVIDERS: dict[str, SocialProvider] = {
    "kakao": KakaoProvider(),
    # "google": GoogleProvider(),  # 확장 예시 — 클래스 + config env 추가만으로 활성화
}


def get_provider(name: str) -> SocialProvider | None:
    """이름으로 등록된 provider 조회 (없으면 None)."""
    return _PROVIDERS.get(name)


__all__ = [
    "EmailConsentRequired",
    "SocialAuthError",
    "SocialProfile",
    "SocialProvider",
    "get_provider",
]
