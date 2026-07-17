-- pgvector 확장 활성화 (벡터 유사도 검색용).
-- 실제 테이블/임베딩 컬럼은 ERD 확정 후 마이그레이션으로 추가한다.
CREATE EXTENSION IF NOT EXISTS vector;
