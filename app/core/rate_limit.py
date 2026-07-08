"""slowapi 기반 rate limiting 공용 설정.

전역 `limiter` 인스턴스를 main.py(state 등록 + 429 핸들러)와 각 라우터가 공유한다.
- IP 기준: 기본 key = remote address.
- 이메일 기준 제한이 필요한 엔드포인트는 요청 본문에서 email 을 뽑아 key로 쓰는
  별도 데코레이터(`email_key_func` 조합)를 라우터에서 지정한다.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

# 기본 key: 클라이언트 IP.
limiter = Limiter(key_func=get_remote_address)
