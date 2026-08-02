# 칵테일 이미지 프롬프트·Gemini Batch·업로드 운영

Gemini는 칵테일 이름과 레시피로 음료 내용물의 외양만 설명한다. 서버는 이
설명에 재사용 가능한 잔·베이스 배경·공통 구도를 이어 붙여 CSV로 내보낸다.
완성된 CSV 프롬프트는 유료 `gemini-3.1-flash-lite-image` Batch API에 전달하고
결과 이미지를 로컬 폴더에 저장한다.

Batch API는 `4:3`, `1K` 이미지를 생성한다. 서버 업로드 API는 원본을 4:3으로
검증·정리한 뒤 다음 두 WebP를 저장한다.

- 메인: `400x300`, Unsharp Mask 적용, 품질 92
- 썸네일: `128x96`, Unsharp Mask 적용, 품질 88

한 요청의 모든 파일과 DB ID를 먼저 검증하고, 파일 저장이 모두 끝난 뒤 한
트랜잭션으로 `cocktails.image_url`을 갱신한다.

## 프롬프트 구성

최종 프롬프트는 다음 네 부분을 순서대로 단순 결합한다.

1. 칵테일 내용물: 이름과 레시피로 `gemini-3.5-flash-lite`가 생성
2. 잔 모양: `scripts/prompts/glass_prompts.csv`
3. 배경: `base_tag`에 따라 `scripts/prompts/base_backgrounds.csv`에서 선택
4. 구도와 품질: `scripts/prompts/composition.txt`

공통 구도는 고정 85mm 렌즈와 약 10도 상단 시점을 사용해 잔의 전체 림과
내용물 표면이 보이게 한다. 밝은 좌상단 스튜디오 조명 아래 잔 바닥에서 2시
방향으로 이어지는 짧고 부드러운 그림자를 고정하며, 바닥면·테이블·수평선은
표현하지 않는다.

모델 출력부터 최종 파일까지 4:3이므로 큰 중앙 크롭 없이 잔 전체 여백을
보존한다.

배경 CSV는 프론트의 `BASE_TAG_MAP` 9종 팔레트를 사용하며, `other`는
`#8f949b`이다. 잔 CSV에는 현재 프로덕션 DB의 잔 14종이 모두 들어 있다.

실제로 사용한 네 조각과 최종 프롬프트는 칵테일별 상태 JSON에 저장된다. 음료
설명 fingerprint는 이미지 provider 설정과 분리되어 있으므로 이미지 모델이나
배경이 바뀌어도 기존 Gemini 음료 설명을 재사용한다.

## 환경변수

로컬은 `.env.local`, 프로덕션 서버는 `.env`를 사용한다. CLI는 `.env.local`이
있으면 이를 우선한다.

```dotenv
GEMINI_API_KEY=<Google AI Studio key>
GEMINI_TEXT_MODEL=gemini-3.5-flash-lite
```

API 키는 로그나 Git에 남기지 않는다. 공통 구도 프롬프트로 카탈로그 전반의
카메라 위치와 피사체 배치를 최대한 일정하게 유지한다.

## 1. 프롬프트 CSV 만들기

```bash
make image-export-prompts
```

기본 출력은 `image-upload/cocktail-image-prompts.csv`이다. 컬럼은 다음과 같다.

- `id`
- `cocktail_name`
- `cocktail_name_en`
- `image_filename`: `cocktail-{id}.png`
- `final_image_prompt`

기존 상태 JSON에 같은 Gemini 음료 설명이 있으면 API를 다시 호출하지 않는다.
없는 항목만 Gemini를 호출하고 성공 건마다 상태를 저장하므로 중단 후 같은
명령을 다시 실행할 수 있다. 로컬 기본 캐시 위치는 `image-upload/state/`이다.

소량 또는 캐시만 확인하려면 다음 옵션을 사용한다.

```bash
python3 -m scripts.export_cocktail_image_prompts --limit 10
python3 -m scripts.export_cocktail_image_prompts --cocktail-id 1
python3 -m scripts.export_cocktail_image_prompts --cached-only
```

## 2. 유료 Batch 입력 준비

아래 명령은 API를 호출하거나 과금하지 않는다. CSV에서 ID 35 이상을 고르고,
`image-upload/images/`에 PNG/JPEG/WebP가 이미 있는 ID는 제외한다. 현재 기준으로
ID 1~34의 JPEG를 제외한 568건이 만들어진다.

```bash
make image-batch-prepare
```

생성 파일은 다음과 같다.

- `image-upload/batch/requests.jsonl`: Gemini Batch 입력
- `image-upload/batch/manifest.json`: ID·파일명·해시·예상 이미지 출력 비용

`prepare` 출력의 요청 수와 비용을 반드시 확인한다. 현재 고정 모델 가격 기준
568건의 이미지 출력 예상가는 `$9.5424`이며 입력·생각 토큰 비용은 별도다.

## 3. 유료 Batch 제출

`GEMINI_API_KEY`가 연결된 프로젝트의 Prepay 충전, auto-reload 비활성화,
프로젝트 spend cap 설정을 먼저 끝낸다. 유료 제출은 실수 방지를 위해 요청 수,
이미지 출력 비용 상한, 확인 플래그가 모두 필요하다.

```bash
.venv/bin/python -m scripts.generate_cocktail_images_batch submit \
  --max-requests 568 \
  --max-image-output-cost-usd 9.55 \
  --confirm-paid-batch
```

이 명령만 실제 유료 Batch 작업을 만든다. 자동 재시도는 없으며 작업 정보는
`image-upload/batch/job.json`에 저장된다. 동일 파일을 실수로 재제출하지 않도록
기존 job 파일이 있으면 기본적으로 거부한다.

## 4. 상태 확인·이미지 저장

상태만 한 번 확인하려면 다음 명령을 사용한다.

```bash
make image-batch-status
```

완료까지 기다렸다가 결과를 바로 저장하려면 다음 명령을 사용한다.

```bash
make image-batch-wait
```

성공한 결과는 `image-upload/images/cocktail-{id}.png`로 원자 저장된다. 4:3이
아니거나 이미지가 없거나 안전 차단된 항목은 저장하지 않고
`image-upload/batch/failed.csv`에 기록한다. 성공 작업의 결과 파일은 나중에
다음 명령만 실행해도 다시 내려받을 수 있다.

```bash
make image-batch-download
```

## 5. 수동 이미지 추가

CSV의 각 `final_image_prompt`로 이미지를 만들고 `image_filename` 그대로
저장한다. 모든 파일은 다음 전용 폴더에 넣는다.

```bash
image-upload/images/
├── cocktail-1.png
├── cocktail-2.png
└── cocktail-3.png
```

PNG, JPEG, WebP를 받을 수 있으며 CSV 기본 파일명은 PNG다. 다른 확장자를
사용해도 클라이언트가 같은 `cocktail-{id}` 파일을 찾아낸다. 반드시
`cocktail-{id}.확장자` 형식은 유지한다.

## 6. 반영 대상 미리 확인

파일이나 DB를 변경하지 않고 현재 폴더에서 찾은 파일과 배치 크기를 확인한다.

```bash
python3 -m scripts.upload_cocktail_images --dry-run
```

CSV에 아직 생성하지 않은 이미지가 있어도 기본적으로 건너뛴다. 하나라도
누락되면 중단하려면 `--require-all`을 추가한다.

## 7. 서버 내부에서 이미지 반영

외부 업로드 API는 제공하지 않는다. 로컬에서 생성한 CSV와 이미지 폴더를 SSH로
운영 서버의 비공개 입력 경로에 복사한다.

```bash
rsync -av image-upload/ ubuntu@<OCI_HOST>:/srv/cocktail-mate/image-upload/
```

운영 서버에서 API 컨테이너 내부의 배치 스크립트를 실행한다. 입력 경로는 읽기
전용으로, 최종 이미지 경로는 쓰기 가능 볼륨으로 마운트돼 있다.

```bash
cd ~/cocktail-mate-server
sudo docker compose -f docker-compose.prod.yml exec api \
  python -m scripts.upload_cocktail_images --require-all
```

로컬 또는 별도 내부 작업 환경에서 DB와 출력 디렉터리를 직접 설정했다면 다음
명령도 사용할 수 있다.

```bash
make image-upload-batch
```

스크립트가 CSV 순서대로 파일을 최대 10개씩 처리한다. 파일명에서 칵테일 ID를 읽고, 원본을
`400x300`/`128x96` WebP로 변환해 서버 스토리지에 저장한 뒤 DB URL을 갱신한다.

`/admin/cocktail-images/*`는 Caddy와 FastAPI 양쪽에서 공개하지 않는다.

같은 파일을 다시 올리면 같은 콘텐츠 해시 URL이 만들어지므로 재실행해도
안전하다. 다른 이미지를 올리면 새 해시 URL이 생기고 기존 파일은 장기 캐시
안전성을 위해 즉시 삭제하지 않는다.
