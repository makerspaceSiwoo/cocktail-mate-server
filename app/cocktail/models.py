"""칵테일 ORM 모델 placeholder.

ERD 확정 후 실제 테이블/컬럼(예: pgvector 임베딩 컬럼 포함)을 여기에 정의한다.
"""
# from sqlalchemy.orm import Mapped, mapped_column
# from pgvector.sqlalchemy import Vector
# from app.core.database import Base
#
# class Cocktail(Base):
#     __tablename__ = "cocktails"
#     id: Mapped[int] = mapped_column(primary_key=True)
#     ...
#     embedding: Mapped[list[float]] = mapped_column(Vector(1536))
