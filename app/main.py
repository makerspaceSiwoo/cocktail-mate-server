"""FastAPI 애플리케이션 진입점.

도메인 라우터(cocktail/user/like)를 등록하고, 배포 검증용 /health를 제공한다.
기존 엔드포인트 경로/응답은 도메인 패키지로 이동했을 뿐 그대로 유지된다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app import cocktail
from app.auth.router import router as auth_router
from app.cocktail.router import router as cocktail_router
from app.core.config import get_settings
from app.core.database import engine
from app.core.rate_limit import limiter
from app.core.storage import check_bucket
from app.like.router import router as like_router

from slowapi import _rate_limit_exceeded_handler

from cocktail_mate_db.models.cocktail import Cocktail
from app.core.database import SessionLocal


def create_app() -> FastAPI:
    settings = get_settings()

    # 토이 프로젝트 — 개발 편의 우선이라 production에서도 /docs·/redoc·/openapi.json 공개.
    # (민감 데이터 없음. 추후 필요하면 Basic Auth 등으로 보호 가능.)
    # @TODO 개발 완료 후 production에서 docs 접근 auth 추가
    app = FastAPI(title="cocktail-mate-server")

    # ── Rate limiting (slowapi) ──
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    cors_kwargs: dict = {
        "allow_origins": settings.cors_origin_list,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    if not settings.is_production:
        # 개발 환경: localhost / 127.0.0.1 의 모든 포트(3000, 6006 등)를 자동 허용
        cors_kwargs["allow_origin_regex"] = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    app.add_middleware(CORSMiddleware, **cors_kwargs)

    app.include_router(cocktail_router)
    app.include_router(auth_router)
    app.include_router(like_router)

    @app.get("/health", tags=["infra"])
    def health():
        """DB 연결 / pgvector 확장 / 오브젝트 스토리지 접근 상태를 점검한다."""
        status = {"db": "fail", "vector": "fail", "storage": "fail"}

        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                status["db"] = "ok"
                has_vector = conn.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                ).first()
                status["vector"] = "ok" if has_vector else "missing"
        except Exception as exc:  # noqa: BLE001 - 헬스체크는 원인 문자열만 노출
            status["db"] = f"fail: {type(exc).__name__}"

        try:
            check_bucket()
            status["storage"] = "ok"
        except Exception as exc:  # noqa: BLE001
            status["storage"] = f"fail: {type(exc).__name__}"

        return status

    # ── 임시 진단용 (db-check): OCI db 데이터 조회 확인. 검증 끝나면 제거할 것. ──
    @app.get("/db-check", tags=["infra"])
    def db_check():
        """배포된 api ↔ OCI db 연결 + 데이터 조회 확인용.

        users 테이블을 **안전 필드만**(password_hash 제외) 반환한다.
        ⚠️ 임시 엔드포인트 — 실사용자 데이터가 들어오기 전에 삭제.
        """
        from cocktail_mate_db.models import User

        from app.core.database import SessionLocal

        with SessionLocal() as db:
            users = db.query(User).order_by(User.id).all()
            return {
                "count": len(users),
                "users": [
                    {
                        "id": u.id,
                        "email": u.email,
                        "nickname": u.nickname,
                        "is_active": u.is_active,
                        "created_at": u.created_at.isoformat() if u.created_at else None,
                    }
                    for u in users
                ],
            }
    

    # ── 임시 진단용 (db-test): cocktails 테이블 상위 10개 조회 확인 ──
    @app.get("/db-test", tags=["infra"])
    def db_test():
        """배포된 api ↔ DB 연결 + cocktails 데이터 조회 확인용.

        cocktails 테이블에서 id 기준 상위 10개의 id/name만 반환한다.
        """
        from cocktail_mate_db.models.cocktail import Cocktail

        from app.core.database import SessionLocal

        with SessionLocal() as db:
            cocktails = db.query(Cocktail).order_by(Cocktail.id).limit(10).all()
            return {
                "count": len(cocktails),
                "cocktails": [
                    {
                        "id": c.id,
                        "name": c.name,
                    }
                    for c in cocktails
                ],
            }

    return app

app = create_app()
