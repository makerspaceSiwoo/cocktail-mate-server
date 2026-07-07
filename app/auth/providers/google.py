"""구글 OAuth provider (확장용 placeholder).

기본적으로 레지스트리(providers/__init__.py)에 등록되어 있지 않아 비활성이다.
활성화 방법:
1. .env 에 GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REDIRECT_URI 채우기
2. providers/__init__.py 의 `# "google": GoogleProvider()` 주석 해제
3. 프론트 로그인 페이지에 구글 버튼 1개 추가 (`/auth/google/login`)

- 인가:   https://accounts.google.com/o/oauth2/v2/auth
- 토큰:   https://oauth2.googleapis.com/token
- 유저:   https://openidconnect.googleapis.com/v1/userinfo
provider_id = 응답 `sub`. 닉네임 = `name`, 프로필사진 = `picture`, 이메일 = `email`.
"""
from __future__ import annotations

import httpx

from app.core.config import get_settings

from .base import SocialAuthError, SocialProfile, SocialProvider

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
PROFILE_URL = "https://openidconnect.googleapis.com/v1/userinfo"


class GoogleProvider(SocialProvider):
    name = "google"

    def authorize_url(self, state: str) -> str:
        settings = get_settings()
        params = httpx.QueryParams(
            {
                "client_id": settings.google_client_id,
                "redirect_uri": settings.google_redirect_uri,
                "response_type": "code",
                "state": state,
                "scope": "openid profile email",
            }
        )
        return f"{AUTHORIZE_URL}?{params}"

    async def exchange_code(self, code: str, client: httpx.AsyncClient) -> str:
        settings = get_settings()
        resp = await client.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "code": code,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise SocialAuthError("구글 토큰 교환에 실패했습니다.")
        access_token = resp.json().get("access_token")
        if not access_token:
            raise SocialAuthError("구글 access token 을 받지 못했습니다.")
        return access_token

    async def fetch_profile(
        self, access_token: str, client: httpx.AsyncClient
    ) -> SocialProfile:
        resp = await client.get(
            PROFILE_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise SocialAuthError("구글 유저 정보 조회에 실패했습니다.")
        body = resp.json()

        provider_id = str(body.get("sub", "")) or None
        if not provider_id:
            raise SocialAuthError("구글 유저 id 를 받지 못했습니다.")

        return SocialProfile(
            provider=self.name,
            provider_id=provider_id,
            email=body.get("email"),
            nickname=body.get("name"),
            profile_image_url=body.get("picture"),
        )
