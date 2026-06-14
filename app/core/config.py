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
