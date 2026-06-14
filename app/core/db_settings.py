# 서버 연결 관련 설정 파일 - 수정 금지
"""DB(서버) 연결 설정 — `.env`에서 읽어 전역 `db_settings`로 제공한다.

cocktail-mate-db 레포의 config.py와 동일한 방식(pydantic-settings, **대문자 필드**).
연결값(`DATABASE_URL` — host·user·password 포함)은 `.env`에만 두고 코드/깃엔 남기지 않는다.

- 로컬 개발: `.env`의 `DATABASE_URL`을 OCI 공인 IP 개발 DB(cm_app)로 지정 → 로컬 api가
  실제 원격 DB 데이터를 사용한다.
- 프로덕션(OCI 인스턴스): 호스트만 `host.docker.internal:5432` (자기 공인 IP hairpin 불가).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # cm_app 런타임 접속 문자열 — 실제 값은 .env의 DATABASE_URL에서 주입.
    # (기본값은 .env 미설정 시 import만 되게 하는 placeholder — 실제 접속은 .env 필요)
    DATABASE_URL: str = "postgresql+psycopg2://cm_app:change_me@localhost:5432/cocktail_mate"


# 전역 인스턴스 — import 시점에 .env를 한 번 읽는다.
db_settings = DBSettings()
