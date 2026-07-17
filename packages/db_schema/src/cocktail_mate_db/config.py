"""중앙 설정 — .env(또는 환경변수)에서 읽어 전역 `settings`로 제공한다.

Next.js의 process.env처럼, 어디서든:

    from cocktail_mate_db.config import settings
    settings.DATABASE_URL
    settings.CM_APP_PASSWORD

운영 도구(alembic, scripts/)에서만 쓴다. 모델 import 경로
(`cocktail_mate_db.__init__`)에는 들어가지 않으므로, api-server가 패키지를
설치해 모델만 import할 때는 pydantic-settings가 필요 없다([migrations] extra).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",            # 실행 디렉터리의 .env를 읽음 (env 변수가 우선)
        env_file_encoding="utf-8",
        extra="ignore",             # POSTGRES_* 등 docker-compose 전용 변수는 무시
    )

    # cm_admin 접속 문자열 (alembic / 스키마 생성에 필수)
    DATABASE_URL: str
    # 계정 생성(scripts/apply_roles.py) 시에만 필요 — 없으면 None
    CM_APP_PASSWORD: str | None = None
    CM_DML_PASSWORD: str | None = None

    @property
    def psycopg_dsn(self) -> str:
        """psycopg2.connect용 DSN — SQLAlchemy URL의 '+psycopg2'를 제거."""
        return self.DATABASE_URL.replace("+psycopg2", "", 1)


# 전역 인스턴스 — import 시점에 .env를 한 번 읽는다.
settings = Settings()
