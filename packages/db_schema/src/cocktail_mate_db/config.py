"""중앙 설정 — .env(또는 환경변수)에서 읽어 전역 `settings`로 제공한다.

Next.js의 process.env처럼, 어디서든:

    from cocktail_mate_db.config import settings
    settings.admin_url          # alembic / apply_roles 전용 (cm_admin, DDL 권한)
    settings.CM_APP_PASSWORD

운영 도구(alembic, scripts/)에서만 쓴다. 모델 import 경로
(`cocktail_mate_db.__init__`)에는 들어가지 않으므로, api-server가 패키지를
설치해 모델만 import할 때는 pydantic-settings가 필요 없다([migrations] extra).

URL 우선순위:
  ALEMBIC_DATABASE_URL  — alembic / apply_roles 전용 (cm_admin, DDL 권한)
  DATABASE_URL          — 로컬 단일 URL 폴백 (없으면 admin_url 접근 시 에러)
  런타임 API의 DATABASE_URL(cm_app)은 서버 .env에만 존재하고 여기에 없다.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",            # 실행 디렉터리의 .env를 읽음 (env 변수가 우선)
        env_file_encoding="utf-8",
        extra="ignore",             # POSTGRES_* 등 docker-compose 전용 변수는 무시
    )

    # cm_admin 접속 문자열 — alembic / 스키마 생성 / apply_roles 전용 (DDL 권한).
    # 런타임 DATABASE_URL(cm_app)과 이름이 겹치지 않도록 별도 키를 쓴다.
    ALEMBIC_DATABASE_URL: str | None = None
    # (선택) 로컬에서 단일 URL만 쓸 때의 폴백 — 상용 API는 cm_app URL을 여기 둔다.
    DATABASE_URL: str | None = None
    # 계정 생성(scripts/apply_roles.py) 시에만 필요 — 없으면 None
    CM_APP_PASSWORD: str | None = None
    CM_DML_PASSWORD: str | None = None

    @property
    def admin_url(self) -> str:
        """운영 도구(alembic/apply_roles)가 쓰는 cm_admin URL. 없으면 명확히 에러."""
        url = self.ALEMBIC_DATABASE_URL or self.DATABASE_URL
        if not url:
            raise RuntimeError(
                "ALEMBIC_DATABASE_URL (또는 폴백 DATABASE_URL)가 설정되어야 마이그레이션을 실행할 수 있습니다."
            )
        return url

    @property
    def psycopg_dsn(self) -> str:
        """psycopg2.connect용 DSN — SQLAlchemy URL의 '+psycopg2'를 제거."""
        return self.admin_url.replace("+psycopg2", "", 1)


# 전역 인스턴스 — import 시점에 .env를 한 번 읽는다.
settings = Settings()
