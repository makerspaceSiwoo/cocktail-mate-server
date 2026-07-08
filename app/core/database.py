# 서버 연결 관련 설정 파일 - 수정 금지
"""SQLAlchemy 데이터베이스 연결.

모델/Base는 cocktail-mate-db 패키지(private 레포)가 canonical로 관리한다.
여기서는 엔진/세션/세션 의존성(`get_db`)과 Base re-export만 제공한다.
접속 문자열은 `app.core.db_settings.db_settings.DATABASE_URL`(.env에서 주입).
"""

from collections.abc import Generator

from cocktail_mate_db.base import Base  # noqa: F401 — 기존 import 경로 호환용 re-export
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.db_settings import db_settings

engine = create_engine(
    db_settings.DATABASE_URL,
    pool_pre_ping=True,  # 유휴 커넥션 끊김 방지
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 의존성: 요청당 세션을 열고 종료 시 닫는다."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
