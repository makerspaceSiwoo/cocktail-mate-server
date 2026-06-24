#!/usr/bin/env bash
# make up 직전 preflight.
# cocktail-mate-db(private 레포)는 빌드 중 git+ssh로 설치되므로,
# 호스트 ssh-agent에 GitHub 인증 키가 있어야 한다. 없으면 빌드가
# "Permission denied (publickey)"로 깨진다 — cryptic하게 깨지기 전에 여기서 잡는다.
set -euo pipefail

# GitHub 인증 자체를 확인한다(= agent에 키 로드됨 + 그 키가 계정에 등록됨).
# ssh -T git@github.com 은 인증 성공해도 exit 1이라(셸 미제공) 메시지로 판정한다.
# pipefail 환경에서 ssh의 exit 1이 파이프를 오염시키지 않게 출력을 먼저 받는다.
auth_output=$(ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com 2>&1 || true)
if printf '%s' "$auth_output" | grep -q "successfully authenticated"; then
  exit 0
fi

cat >&2 <<'MSG'
✗ GitHub SSH 인증 실패 — cocktail-mate-db(private) 설치가 빌드 중 깨집니다.

원인은 보통 셋 중 하나입니다:
  1) cocktail-mate-db 레포에 collaborator로 초대되지 않음
       → 팀 관리자에게 본인 GitHub 계정 초대 요청
  2) GitHub SSH 키가 ssh-agent에 안 올라가 있음
       → ssh-add -l 로 확인. 비어 있으면:
         ssh-add --apple-use-keychain ~/.ssh/<github-key>   (macOS, 재부팅 후에도 유지)
         ssh-add ~/.ssh/<github-key>                        (그 외)
  3) GitHub에 SSH 키 자체가 등록 안 됨
       → https://github.com/settings/keys

확인:  ssh -T git@github.com   → "successfully authenticated" 가 떠야 정상.
자세히: docs/ONBOARDING.md
MSG
exit 1
