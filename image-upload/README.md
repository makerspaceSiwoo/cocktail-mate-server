# 수동 생성 이미지 작업 폴더

`cocktail-image-prompts.csv`의 `image_filename` 값 그대로 이미지를 저장해
`image-upload/images/`에 넣는다. 기본 파일명은 `cocktail-{id}.png`이다.
`image-upload/state/`에는 Gemini 설명 캐시가 저장되며 둘 다 Git에서 제외된다.

유료 Gemini Batch 이미지 작업 파일과 작업 ID는 `image-upload/batch/`에 저장된다.
ID 1~34처럼 기존 JPEG가 있으면 `prepare`가 자동으로 제외한다.

```bash
make image-batch-prepare
.venv/bin/python -m scripts.generate_cocktail_images_batch submit \
  --max-requests 568 \
  --max-image-output-cost-usd 9.55 \
  --confirm-paid-batch
make image-batch-wait
```

Batch 다운로드 또는 수동 저장이 끝난 뒤 프로젝트 루트에서 다음 명령을
실행한다.

```bash
make image-upload-batch
```

클라이언트는 CSV 순서대로 파일을 읽어 서버의 비공개 관리 API에 요청당 최대
10개씩 업로드한다. 성공한 배치는 서버에서 `400x300` 메인과 `128x96` 썸네일로
변환되고 해당 칵테일의 DB URL이 갱신된다.
