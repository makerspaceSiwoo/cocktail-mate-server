"""애플리케이션 설정.

모든 비밀값/환경값은 `.env`에서 주입한다 (커밋 금지). 예시는 `.env.example` 참고.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- App ----
    app_env: str = "local"

    # ---- Database ----
    # DB 접속(DATABASE_URL)은 app/core/db_settings.py(서버 연결 전용, 수정 금지)에서 관리.

    # ---- Object Storage (S3 compatible: 로컬 MinIO / 프로덕션 Oracle Object Storage) ----
    storage_endpoint: str = "http://minio:9000"
    storage_region: str = "ap-chuncheon-1"
    storage_access_key: str = "minioadmin"
    storage_secret_key: str = "minioadmin"
    storage_bucket: str = "cocktail-images"
    storage_public_base_url: str = "http://localhost:9000/cocktail-images"

    # ---- CORS ----
    # production 에서 허용할 프론트 origin 목록(콤마 구분). 예) "https://foo.vercel.app"
    # 개발 환경(app_env != production)에서는 localhost 모든 포트가 자동 허용된다.
    cors_origins: str = ""

    # ---- Auth (JWT 쿠키) ----
    # JWT 서명 키 — `openssl rand -hex 32` 로 생성한 32바이트 이상 랜덤값을 .env에 주입.
    # (기본값은 .env 미설정 시 import만 되게 하는 placeholder — 실제 배포는 반드시 .env로 덮어쓸 것)
    secret_key: str = "dev-insecure-secret-change-me"  # noqa: S105
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14
    email_verify_expire_hours: int = 24

    # 매직 링크가 여는 프론트 주소 (verify-email 페이지 및 소셜 콜백 후 리다이렉트 대상)
    frontend_url: str = "http://localhost:3000"

    # ---- Mail (Resend HTTP API; SDK 없이 httpx) ----
    # MAIL_API_KEY 가 비어 있으면 개발용 콘솔 백엔드로 동작 — 매직 링크를 로그로만 출력한다.
    mail_api_key: str = ""
    mail_from: str = "no-reply@cocktail-mate.local"

    # ---- 소셜 로그인 (카카오; provider 추가는 KakaoProvider 참고) ----
    kakao_client_id: str = ""
    kakao_client_secret: str = ""
    kakao_redirect_uri: str = "http://localhost:8000/auth/kakao/callback"

    # ---- 쿠키 ----
    # 프로덕션(HTTPS)에서는 반드시 true. 로컬 http 개발에서는 false 가능.
    cookie_secure: bool = True
    cookie_samesite: str = "lax"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """프로세스 수명 동안 설정을 한 번만 로드한다."""
    return Settings()
