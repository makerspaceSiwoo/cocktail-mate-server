"""카카오 OAuth provider.

- 인가:   https://kauth.kakao.com/oauth/authorize
- 토큰:   https://kauth.kakao.com/oauth/token
- 유저:   https://kapi.kakao.com/v2/user/me

provider_id = 응답 `id`(숫자 → str)만 사용한다. 동의항목(닉네임/프로필사진/이메일)은
요청하지 않는다 — 카카오 프로필 정보를 저장하지 않고, 닉네임은 서버가 기본값을 부여한다.
(카카오 닉네임이 실명일 수 있어 받지 않는다. 콘솔의 닉네임 동의항목은 '사용 안 함' 권장.)
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
                # 동의항목 미요청 — 유저 식별자(id)만 받는다. 닉네임은 서버 기본값 사용.
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

        # 동의항목을 요청하지 않으므로 카카오 프로필(닉네임/사진/이메일)은 저장하지 않는다.
        # 유저 식별자(provider_id)만 사용하고, 닉네임은 service 가 기본값을 부여한다.
        return SocialProfile(provider=self.name, provider_id=provider_id)
