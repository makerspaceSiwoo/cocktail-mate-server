"""소셜 로그인 provider 추상화.

새 provider(예: Google) 추가는 이 파일 상속 클래스 1개 + config env 만으로 끝난다:
1. `SocialProvider` 를 상속해 authorize_url / exchange_code / fetch_profile 구현
2. config.py 에 해당 provider env 추가
3. `providers/__init__.py` 레지스트리에 인스턴스 등록

`SocialProfile` 은 provider 별 응답을 서버 공통 표현으로 정규화한 값이다.
이메일은 provider 가 제공하지 않을 수 있어 nullable 이다 (이메일 없이도 가입 허용).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


class SocialAuthError(Exception):
    """소셜 인증 처리 중 일반 오류 (토큰 교환 실패 등)."""


@dataclass
class SocialProfile:
    """provider 응답을 정규화한 서버 공통 프로필."""

    provider: str
    provider_id: str  # provider 고유 유저 id (문자열) — 유저 식별자
    email: str | None = None  # provider 가 제공하지 않을 수 있음
    nickname: str | None = None
    profile_image_url: str | None = None


class SocialProvider(ABC):
    """소셜 로그인 provider 인터페이스."""

    name: str

    @abstractmethod
    def authorize_url(self, state: str) -> str:
        """유저를 보낼 provider 인가(authorize) URL."""

    @abstractmethod
    async def exchange_code(self, code: str, client: httpx.AsyncClient) -> str:
        """authorization code → provider access token 교환."""

    @abstractmethod
    async def fetch_profile(
        self, access_token: str, client: httpx.AsyncClient
    ) -> SocialProfile:
        """access token 으로 유저 프로필 조회 → 정규화."""
