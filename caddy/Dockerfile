# 공식 caddy:2 이미지에는 rate_limit 모듈이 없어 xcaddy로 포함해 빌드한다.
# 모듈: github.com/mholt/caddy-ratelimit (IP당 요청 수 제한 → 429 응답)
FROM caddy:2-builder AS builder

RUN xcaddy build --with github.com/mholt/caddy-ratelimit

FROM caddy:2

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
