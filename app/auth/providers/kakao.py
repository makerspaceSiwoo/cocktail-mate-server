"""카카오 OAuth provider.

- 인가:   https://kauth.kakao.com/oauth/authorize
- 토큰:   https://kauth.kakao.com/oauth/token
- 유저:   https://kapi.kakao.com/v2/user/me

provider_id = 응답 `id`(숫자 → str). 닉네임만 받는다 (프로필사진·이메일 scope 미사용).
이메일 동의항목(account_email)은 비즈앱 심사가 필요하므로 요청하지 않는다 —
프로필에 이메일이 있으면 저장하되, 없으면 NULL 로 가입한다.
닉네임은 콘솔에서 '선택 동의'로 두어 유저가 거부하면 서버가 기본 닉네임을 부여한다.
"""

from __future__ import annotations

import httpx

from app.core.config import get_settings

from .base import SocialAuthError, SocialProfile, SocialProvider

AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
TOKEN_URL = "https://kauth.kakao.com/oauth/token"
PROFILE_URL = "https://kapi.kakao.com/v2/user/me"


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
                # 닉네임만 요청 (프로필사진·이메일 미요청 → 동의항목 최소, 개인 앱으로 바로 동작).
                "scope": "profile_nickname",
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
        profile = account.get("profile") or {}
        nickname = profile.get("nickname")
        # 프로필사진·이메일은 scope 미요청 → 받지 않는다(profile_image_url/email = NULL).
        # 단, 카카오가 값을 실어 보내면 이메일만 그대로 저장(닉네임 없으면 service 가 기본값 부여).
        email = account.get("email")

        return SocialProfile(
            provider=self.name,
            provider_id=provider_id,
            email=email,
            nickname=nickname,
        )
