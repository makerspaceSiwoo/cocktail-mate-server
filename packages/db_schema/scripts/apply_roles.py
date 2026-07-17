"""DB 계정(cm_app/cm_dml) 생성 + 권한 부여.

비밀번호는 전역 settings(.env)에서 읽는다 — 셸 변수 주입(-v) 없음.
반드시 **스키마 생성(alembic) 전에** 실행: cm_admin이 만들 테이블에
default privileges가 자동 적용되려면 권한 설정이 먼저여야 한다.

카탈로그 테이블(cocktails / ingredients / cocktail_ingredients)은 사람용 계정
cm_dml에서 쓰기 권한을 회수해 **읽기 전용**으로 둔다(아래 GRANTS의 DO 블록).
이 회수는 테이블이 존재할 때만 적용되므로, **신규 DB는 migrate-up 후 한 번 더**
이 스크립트를 돌려야 카탈로그 읽기 전용이 실제로 걸린다(기존 DB는 1회 실행이면 됨).

실행 (ALEMBIC_DATABASE_URL이 cm_admin 계정이어야 함; 없으면 DATABASE_URL 로 폴백):
    python scripts/apply_roles.py
재실행해도 안전(계정이 있으면 비밀번호만 갱신, 권한은 멱등).
"""

import sys

import psycopg2
from psycopg2 import sql

from cocktail_mate_db.config import settings

# 비밀번호 없는 정적 권한 — 그대로 커밋 (repo == disk)
GRANTS = """
GRANT CONNECT ON DATABASE cocktail_mate TO cm_app, cm_dml;
GRANT USAGE ON SCHEMA public TO cm_app, cm_dml;
REVOKE CREATE ON SCHEMA public FROM cm_app, cm_dml;          -- DDL 차단
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cm_app, cm_dml;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cm_app, cm_dml;
-- 앞으로 cm_admin이 만드는 새 객체에 자동 부여 (스키마 생성 전 실행이 핵심)
ALTER DEFAULT PRIVILEGES FOR ROLE cm_admin IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cm_app, cm_dml;
ALTER DEFAULT PRIVILEGES FOR ROLE cm_admin IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO cm_app, cm_dml;

-- 카탈로그(큐레이션) 테이블은 cm_dml에서 쓰기 권한 회수 → SELECT만 남긴다.
--   · cm_dml(사람/DBeaver): cocktails·ingredients·cocktail_ingredients 는 읽기 전용,
--     users·likes 등 나머지는 그대로 INSERT/UPDATE/DELETE 가능.
--   · cm_app(런타임)은 건드리지 않는다. 카탈로그 적재는 Alembic 시드 / AI 파이프라인(cm_admin).
-- 위 GRANT(ALL TABLES) 뒤에 와야 net 결과가 '읽기 전용'으로 수렴한다(재실행 안전).
-- 테이블이 아직 없을 수 있으므로(스키마 생성 전 실행 가능) 존재할 때만 회수한다.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['cocktails', 'ingredients', 'cocktail_ingredients'] LOOP
    IF to_regclass('public.' || t) IS NOT NULL THEN
      EXECUTE format('REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.%I FROM cm_dml', t);
    END IF;
  END LOOP;
END $$;
"""


def main() -> None:
    if not (settings.CM_APP_PASSWORD and settings.CM_DML_PASSWORD):
        sys.exit("❌ .env에 CM_APP_PASSWORD / CM_DML_PASSWORD 가 필요합니다.")

    conn = psycopg2.connect(settings.psycopg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        for role, pw in (
            ("cm_app", settings.CM_APP_PASSWORD),
            ("cm_dml", settings.CM_DML_PASSWORD),
        ):
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            exists = cur.fetchone() is not None
            verb = (
                "ALTER ROLE {} WITH LOGIN PASSWORD {}"
                if exists
                else "CREATE ROLE {} LOGIN PASSWORD {}"
            )
            cur.execute(sql.SQL(verb).format(sql.Identifier(role), sql.Literal(pw)))
            print(f"  {'updated' if exists else 'created'} role {role}")
        cur.execute(GRANTS)
    conn.close()
    print("✅ 계정/권한 적용 완료 (cm_app, cm_dml)")
    print(
        "   ↳ cm_dml: cocktails/ingredients/cocktail_ingredients 는 읽기 전용(존재 시)"
    )


if __name__ == "__main__":
    main()
