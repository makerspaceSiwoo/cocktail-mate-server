from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routers import health

app = FastAPI(title="Cocktail Mate Server", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    # TODO: tighten this when Vercel project slug is finalized.
    # Example: r"https://cocktail-mate(-[a-z0-9-]+)?\.vercel\.app"
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
