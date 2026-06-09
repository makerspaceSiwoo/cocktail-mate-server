-- cocktail-mate 테이블 스키마 (ERD 기반). 01-init.sql(vector 확장 활성화) 이후 실행된다.
-- 결정사항:
--   - users.LoginID = 이메일(email)로 사용. profile_image_url(nullable) 추가.
--   - 비밀번호는 평문 금지 → password_hash(해시) 컬럼.
--   - cocktails.embedding 차원 미정 → 우선 1536(OpenAI text-embedding-3-small 기준). 추후 변경 가능.
--   - 조인테이블 'Field(순서)' → amount(FLOAT) + unit(VARCHAR, 예: ml/oz/dash)로 모델링.
--   - likes/조인은 단일 id PK + 자연키 UNIQUE 제약으로 중복 방지(ERD의 복합 PK 과잉 정리).
-- ※ 아직 완성본이 아니므로 컬럼 추가/변경 가능. 스키마 변경이 잦아지면 Alembic 도입 권장.

-- 사용자
CREATE TABLE IF NOT EXISTS users (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email             VARCHAR(255) NOT NULL UNIQUE,   -- 로그인 식별자 = 이메일
    password_hash     VARCHAR(255) NOT NULL,          -- bcrypt 등 해시 저장
    nickname          VARCHAR(255) NOT NULL UNIQUE,
    is_active         BOOLEAN      NOT NULL DEFAULT FALSE,
    profile_image_url TEXT,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 칵테일
CREATE TABLE IF NOT EXISTS cocktails (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        VARCHAR(255) NOT NULL UNIQUE,
    image_url   TEXT,
    glass       VARCHAR(255),
    abv         REAL,
    num_like    INTEGER      NOT NULL DEFAULT 0,
    recipe      TEXT,
    description TEXT,
    base_tag    VARCHAR(50),                 -- 베이스 술타입 (gin/rum/tequila ...)
    embedding   vector(1536),                -- 임베딩 벡터 (semantic search용, 차원 추후 확정)
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- 재료
CREATE TABLE IF NOT EXISTS ingredients (
    id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE
);

-- 칵테일-재료별 설명 (조인 + 용량/표기)
CREATE TABLE IF NOT EXISTS cocktail_ingredients (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cocktail_id   BIGINT NOT NULL REFERENCES cocktails(id)   ON DELETE CASCADE,
    ingredient_id BIGINT NOT NULL REFERENCES ingredients(id) ON DELETE RESTRICT,
    amount        REAL,
    unit          VARCHAR(50),               -- ERD 'Field'(순서/표기): ml/oz/dash 등
    UNIQUE (cocktail_id, ingredient_id)
);

-- 좋아요 (user-cocktail 다대다, 중복 방지)
CREATE TABLE IF NOT EXISTS likes (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id)     ON DELETE CASCADE,
    cocktail_id BIGINT NOT NULL REFERENCES cocktails(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, cocktail_id)
);

-- 조회 인덱스
CREATE INDEX IF NOT EXISTS idx_cocktails_base_tag            ON cocktails(base_tag);
CREATE INDEX IF NOT EXISTS idx_cocktail_ingredients_cocktail ON cocktail_ingredients(cocktail_id);
CREATE INDEX IF NOT EXISTS idx_cocktail_ingredients_ingr     ON cocktail_ingredients(ingredient_id);
CREATE INDEX IF NOT EXISTS idx_likes_user                    ON likes(user_id);
CREATE INDEX IF NOT EXISTS idx_likes_cocktail                ON likes(cocktail_id);
