"""카카오 OAuth provider.

- 인가:   https://kauth.kakao.com/oauth/authorize
- 토큰:   https://kauth.kakao.com/oauth/token
- 유저:   https://kapi.kakao.com/v2/user/me
- 해제:   https://kapi.kakao.com/v1/user/unlink

provider_id = 응답 `id`(숫자 → str), 이메일 = `kakao_account.email`.
이메일 미동의(제공 거부) 시 EmailConsentRequired 를 던진다 (라우터가 unlink 처리).
"""
from __future__ import annotations

import httpx

from app.core.config import get_settings

from .base import EmailConsentRequired, SocialAuthError, SocialProfile, SocialProvider

AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
TOKEN_URL = "https://kauth.kakao.com/oauth/token"
PROFILE_URL = "https://kapi.kakao.com/v2/user/me"
UNLINK_URL = "https://kapi.kakao.com/v1/user/unlink"


class KakaoProvider(SocialProvider):
    name = "kakao"

    def authorize_url(self, state: str) -> str:
        settings = get_settings()
        params = httpx.QueryParams(
            {
                "client_id": settings.kakao_client_id,
                "redirect_uri": settings.kakao_redirect_uri,
                "response_type": "code",
                "state": state,
                # 이메일은 서비스 필수 → 동의 항목으로 요청.
                "scope": "account_email",
            }
        )
        return f"{AUTHORIZE_URL}?{params}"

    async def exchange_code(self, code: str, client: httpx.AsyncClient) -> str:
        settings = get_settings()
        data = {
            "grant_type": "authorization_code",
            "client_id": settings.kakao_client_id,
            "redirect_uri": settings.kakao_redirect_uri,
            "code": code,
        }
        if settings.kakao_client_secret:
            data["client_secret"] = settings.kakao_client_secret

        resp = await client.post(
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            raise SocialAuthError("카카오 토큰 교환에 실패했습니다.")
        access_token = resp.json().get("access_token")
        if not access_token:
            raise SocialAuthError("카카오 access token 을 받지 못했습니다.")
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
            raise SocialAuthError("카카오 유저 정보 조회에 실패했습니다.")
        body = resp.json()

        provider_id = str(body.get("id", "")) or None
        if not provider_id:
            raise SocialAuthError("카카오 유저 id 를 받지 못했습니다.")

        account = body.get("kakao_account") or {}
        email = account.get("email")
        # 이메일 동의 안 함 / 미검증 → 가입 거부 대상.
        if not email or account.get("email_needs_agreement") is True:
            raise EmailConsentRequired(access_token)

        profile = account.get("profile") or {}
        nickname = profile.get("nickname")
        image_url = profile.get("profile_image_url")

        return SocialProfile(
            provider=self.name,
            provider_id=provider_id,
            email=email,
            nickname=nickname,
            profile_image_url=image_url,
        )

    async def unlink(self, access_token: str, client: httpx.AsyncClient) -> None:
        await client.post(
            UNLINK_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
