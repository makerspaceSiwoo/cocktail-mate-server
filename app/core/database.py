"""SQLAlchemy 데이터베이스 연결.

ERD 제공 전까지 실제 모델/테이블은 정의하지 않는다. 엔진/세션/Base와
요청 스코프 세션 의존성(`get_db`)만 제공한다.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # 유휴 커넥션 끊김 방지
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """모든 ORM 모델의 공통 베이스 (ERD 확정 후 모델이 상속)."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI 의존성: 요청당 세션을 열고 종료 시 닫는다."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
