#!/usr/bin/env bash
# Oracle 인스턴스 1회 하드닝 스크립트.
# 보안 점검(2026-06-10)에서 나온 권장 사항을 한 번에 적용한다.
# 실행: 서버에 ssh 접속 후  bash ~/cocktail-mate-server/scripts/harden-server.sh
set -euo pipefail

echo "[1/4] fail2ban 설치 — SSH 무차별 대입 자동 차단"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y fail2ban
sudo systemctl enable --now fail2ban

echo "[2/4] rpcbind 제거 — 미사용 서비스 + UDP 증폭 DDoS 매개 제거"
sudo systemctl disable --now rpcbind.socket rpcbind 2>/dev/null || true
sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y rpcbind 2>/dev/null || true

echo "[3/4] 보류 중 OS 보안 업데이트 적용"
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

echo "[4/4] root SSH 로그인 차단 (ubuntu 계정만 사용)"
echo 'PermitRootLogin no' | sudo tee /etc/ssh/sshd_config.d/99-no-root.conf >/dev/null
sudo sshd -t                      # 문법 오류 시 여기서 중단 → 잠금 사고 방지
sudo systemctl reload ssh

echo "---- 결과 확인 ----"
sudo fail2ban-client status sshd || true
echo "하드닝 완료. 재부팅이 필요한 커널 업데이트가 있으면: sudo reboot"
