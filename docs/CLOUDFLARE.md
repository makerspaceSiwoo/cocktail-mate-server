# Cloudflare 도입 가이드 (DDoS 방어 + WAF)

현재 구조는 Oracle 인스턴스가 모든 트래픽을 직접 받는다.
Cloudflare 무료 플랜을 앞단에 두면 **L3/4 DDoS 자동 차단, WAF, 봇 차단, 캐싱**을
서버 자원 소모 없이 얻는다. 작은 단일 인스턴스에는 가장 효과적인 방어다.

```
지금:   브라우저 ──────────────────────→ Oracle(Caddy:443)
이후:   브라우저 ──→ Cloudflare(프록시) ──→ Oracle(Caddy:443, CF IP만 허용)
```

## 선행 조건 (사용자 직접 작업)

1. **실제 도메인 1개** — nip.io 같은 공용 와일드카드 도메인은 Cloudflare에 등록할 수 없다.
   - 구매처 예: Cloudflare Registrar(원가), 가비아, Namecheap. `.com` 연 1~2만원 수준.
2. **Cloudflare 무료 계정** — https://dash.cloudflare.com/sign-up

## 설정 순서

### 1. 사이트 추가 + 네임서버 변경
- Cloudflare 대시보드 → Add a Site → 도메인 입력 → Free 플랜 선택.
- 안내된 네임서버 2개를 도메인 등록처(가비아 등)에서 교체. 전파까지 수 분~수 시간.

### 2. DNS 레코드
| Type | Name | Content | Proxy |
|---|---|---|---|
| A | `api` (→ api.도메인) | `152.69.225.226` | **Proxied(주황 구름) 켜기** |

Proxy를 켜야 DDoS 방어가 동작한다(끄면 그냥 DNS).

### 3. SSL/TLS 모드 = **Full (strict)**
- 대시보드 → SSL/TLS → Overview → Full (strict).
- Caddy가 origin에서 Let's Encrypt 인증서를 그대로 발급/갱신하므로 추가 작업 없음.
  - 발급이 막히면(드묾) 대안: SSL/TLS → Origin Server에서 **Origin CA 인증서** 발급 후
    Caddyfile에 `tls cert.pem key.pem` 지정.
- Edge → "Always Use HTTPS" 켜기.

### 4. Caddyfile 수정 (이 레포)
```caddyfile
api.<도메인> {            # ← 152-69-225-226.nip.io 를 교체
	rate_limit {
		zone per_ip {
			key {http.request.header.CF-Connecting-IP}   # ← 프록시 뒤에선 실제 클라이언트 IP
			events 100
			window 1m
		}
	}
	reverse_proxy api:8000
}
```
서버 반영: `git pull && sudo docker compose -f docker-compose.prod.yml up -d --build caddy`

### 5. Origin 잠그기 — 80/443을 Cloudflare IP만 허용
프록시를 우회해 origin IP로 직접 때리는 공격을 막는 핵심 단계.
- Cloudflare 공식 IP 대역: https://www.cloudflare.com/ips-v4 , /ips-v6
- OCI 콘솔 → VCN → Security List에서 80/443 ingress의 `0.0.0.0/0`을 위 대역들로 교체
  (terraform 관리 시 `network.tf`의 ingress 규칙을 대역 목록 변수로 전환).
- 주의: Let's Encrypt HTTP-01 챌린지도 CF 프록시를 통과하므로 문제 없음.

### 6. 프론트/CORS 갱신
- Vercel 환경변수 + 로컬 `.env.local`: `NEXT_PUBLIC_API_URL=https://api.<도메인>`
- 서버 `.env`의 `CORS_ORIGINS`는 **프론트 도메인 기준**이라 변경 불필요.

### 7. 검증
```bash
curl -sSI https://api.<도메인>/health | grep -i -E 'cf-ray|server'
# cf-ray 헤더가 보이면 Cloudflare 경유 성공
curl -s https://api.<도메인>/health    # {"db":"ok",...}
```

## 추가 권장 (무료 플랜 범위)
- **Security → Settings → Bot Fight Mode** 켜기.
- **Security → WAF → Rate limiting rules**: 무료 1개 제공 — 추후 `/login` 등 민감 경로에 별도 제한.
- 공격 징후 시 **Under Attack Mode**(JS 챌린지) 일시 활성화.
- nip.io 주소는 이전 완료 후 Caddyfile에서 제거(원본 IP 노출 경로 차단).
