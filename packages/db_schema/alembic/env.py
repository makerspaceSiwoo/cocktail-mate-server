"""Alembic 마이그레이션 환경.

- DB 접속: 전역 settings.admin_url (ALEMBIC_DATABASE_URL, 또는 폴백 DATABASE_URL) — alembic.ini에는 커밋하지 않는다.
- target_metadata: cocktail_mate_db 패키지 import만으로 전 모델이 등록된다.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from cocktail_mate_db import Base
from cocktail_mate_db.config import settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", settings.admin_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """offline 모드: DB 연결 없이 SQL 스크립트만 생성."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """online 모드: 실제 DB에 연결해 마이그레이션 실행."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
