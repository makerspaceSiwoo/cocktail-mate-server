"""FastAPI 애플리케이션 진입점.

도메인 라우터(cocktail/user/like)를 등록하고, 배포 검증용 /health를 제공한다.
기존 엔드포인트 경로/응답은 도메인 패키지로 이동했을 뿐 그대로 유지된다.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.cocktail.router import router as cocktail_router
from app.core.database import engine
from app.core.storage import check_bucket
from app.like.router import router as like_router
from app.user.router import router as user_router


def create_app() -> FastAPI:
    app = FastAPI(title="cocktail-mate-server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(cocktail_router)
    app.include_router(user_router)
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

    return app


app = create_app()
