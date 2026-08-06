# Sensory Vertex Batch → graph handoff

이 문서는 이전 대화 컨텍스트 없이 작업을 이어받기 위한 운영 인계서다.
기준 시각은 2026-08-06이며, 모든 경로는 별도 설명이 없으면 로컬
파일시스템의 절대 경로다.

## 1. 현재 코드 기준점

- 저장소 worktree:
  `/Users/jungsiwoo/GitHub/cocktail-mate/.worktrees/cocktail-mate-server-sensory-graph-modules`
- 브랜치: `feat/sensory-graph-modules`
- base: `origin/main`의 `6316f92`
- 문서 작성 직전 HEAD: `db55535`
- 관련 커밋:
  - `4332d38`: sensory teacher, exact similarity, S² graph 모듈
  - `3f69848`: 초기 Vertex 차단 기록
  - `1e0592a`: 실제 칵테일 pilot 경계
  - `25300c6`: 승인된 pilot manifest hash pin
  - `60fa834`, `4f89aaf`: 검증된 전용 bucket 계약
  - `8679f35`: project 범위 `storage.buckets.list`와 bucket 범위
    `iam/testPermissions` 분리
  - `db55535`: Vertex가 반환하는 정확한 숫자 project resource name 허용

현재 worktree는 상용 DB를 읽거나 수정하지 않았다. 이후에도 사용자가 별도로
DB 반영을 승인하기 전까지 DB 연결, ORM 실행, migration, `UPDATE`, `INSERT`,
`DELETE`, pgvector 적재를 모두 금지한다.

## 2. 비밀 및 인증 안전 규칙

- API key 값, 서비스 계정 JSON 내용, 실제 bucket 이름, 실제 Vertex job
  resource name을 로그·문서·커밋에 남기지 않는다.
- 서비스 계정 JSON 파일을 `cat`, `jq`, Python `json.load`, 에디터 등으로 직접
  읽지 않는다. `google.auth.default()`를 통한 공식 ADC만 사용한다.
- 환경 파일을 로드할 때 값을 출력하지 않는다. 인자 없는 `set`/`env`와 shell
  tracing (`set -x`)을 사용하지 않는다.
- live adapter는 인증 작업 동안 `GEMINI_API_KEY`와 `GOOGLE_API_KEY`를
  프로세스 환경에서 제거한다. Vertex 호출은 ADC만 사용한다.
- GCS cleanup은 금지되어 있다. 현재 결과와 감사 ledger를 보존한다.
- `/private/tmp`은 임시 저장소다. 재부팅·정리 전에 필요한 결과를 안전한 로컬
  보관 위치로 복사하되, 상용 DB에는 넣지 않는다.

## 3. 구현된 모듈

### 3.1 `app/sensory_embedding`

- 48개 감각축 × A–E 5단계 teacher soft-label, 즉 `Raw240`
- category-balanced unit-L2 `Graph48`
- 사용자 선택 masking과 category balancing을 적용한 unnormalized
  `Preference48`
- 사용자 query → cocktail은 `Preference48` maximum inner product search
  계약
- Vertex Batch request 생성, 응답 logprobs 파싱, quarantine, projection
- service-account ADC 기반 live create/status/download 경계

### 3.2 `app/vector_similarity`

- cocktail → cocktail 추천과 시각화 topology의 단일 source of truth
- `Graph48` exact cosine directed top-5
- A가 B를 top-5로 고르거나 B가 A를 top-5로 고르면 연결하는 undirected
  union edge
- 현재 602개 규모에서는 exact all-pairs가 기준 구현
- ANN은 interface marker만 있고 실제 backend adapter는 아직 없음

### 3.3 `app/spherical_graph`

- 고차원 벡터를 2D/3D로 축소하지 않는다.
- 확정된 cocktail node와 cosine union edge topology를 graph-only force
  layout으로 S² 단위 구면에 배치한다.
- cluster hub는 layout 내부 전용이며 public JSON에 node/edge로 노출되지 않는다.
- 추천 directed top-5와 graph union edge는 module 2의 같은 canonical
  artifact를 사용한다.

## 4. 완료된 10-cocktail live pilot

### 4.1 입력과 감사 ledger

| 파일 | SHA-256 |
|---|---|
| `/private/tmp/cocktail-mate-vertex-real-pilot-10-prep-v2/pilot-manifest.json` | `06f7a1398537812bf5e31daecba9be7dfaa495ad54149003b6008034d059f396` |
| `/private/tmp/cocktail-mate-vertex-real-pilot-10-prep-v2/live-ledger-dedicated-v2.json` | `f05aba66dfc1c14a410a92b8ff7625e29f1e51a4c11d0b4ef66717e62be1ae4f` |

pilot 구성은 10 cocktails × 48 axes = 480 requests, 8 shards × 60
requests다. 8개 job 모두 `JOB_STATE_SUCCEEDED`, create attempt는 shard당
정확히 1회, retry/fallback은 0회다. 8개 출력도 모두 hash-verified 상태다.

### 4.2 결과

결과 root:
`/private/tmp/cocktail-mate-vertex-real-pilot-10-results-v1`

| 파일 | SHA-256 |
|---|---|
| `parse-summary.json` | `2bc01e01d6ab91b429feb4554d0a5a929b14d538781fd28c63025107bc876ef0` |
| `parsed.jsonl` | `a158577b75282df249f0d3f25958eb656f240f0edd37b330bb1ec0089d31f669` |
| `quarantine.jsonl` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

검증 결과:

- raw 480, parsed 480, quarantine 0
- missing/duplicate/unexpected key 모두 0
- 모든 record에 ordered A–E 5확률 존재
- 확률합 최대 오차 `4.440892098500626e-16`
- prompt tokens 167,970, output tokens 480, 합계 168,450
- 측정 단가 기반 추정 비용 `$0.0257955`
- 보수적 pilot 상한 `$0.08004` 이내
- prompt tokens/request: min 256, mean 349.9375, max 425

## 5. 준비된 602-cocktail full candidate

준비 root:
`/private/tmp/cocktail-mate-vertex-full-602-prep-v1`

이 디렉터리는 현재 **제출 금지 상태의 candidate**다. 아래 blocker가 수정되면
manifest와 bucket contract를 재생성하고 모든 hash를 다시 기록해야 할 수 있다.

### 5.1 규모와 비용

- cohort: 정확히 현재 DB cohort와 대응하는 602 IDs
- cohort ID-set SHA-256:
  `56e77646b60ad9b45cbdcd43f4807dde994ef40b1d5e4461dbfa41ca2d59c05f`
- cohort source SHA-256:
  `8755a91cfd2709b87fad3a05e5daef158d7ea589cb08e6c3f09ab4ecabd4ab6f`
- 602 × 48 = 28,896 requests
- 8 shards × 3,612 requests
- model: `gemini-2.5-flash`, location: `global`
- conservative model cost: `$4.818408`
- historical reserve: `$0.50`
- conservative total: `$5.318408`
- actual pilot usage extrapolation: 약 `$1.5528891`
- soft stop: `$7.50`
- hard block: `$10.00`

### 5.2 현재 candidate hashes

| 파일 | SHA-256 |
|---|---|
| `manifest.json` | `48381f029a85e7d611a717da81ed1d9151aa99f258bb3d72ef543c6a107f66f7` |
| `full-production-review.json` | `91c6693a975afa634d1219a96d703804bbe86cbade17fcfff2ada71fcafde948` |
| `cost-ledger.json` | `0818681bb1c2802e25de63c4aff17aeec08cb6fd2b9cc05eaa367d0c0e195ec3` |
| `frozen-cocktails.csv` | `4a51835460938ddebc11507d34da2835796e4f73179b15279ddd94253523560b` |
| `cohort-source.csv` | `8755a91cfd2709b87fad3a05e5daef158d7ea589cb08e6c3f09ab4ecabd4ab6f` |
| `requests-00.jsonl` | `2f2ae642421768cabc6593f94f5af536ad2e3b93e462e3661ff6185d8c417889` |
| `requests-01.jsonl` | `96bacc695a0960ce0e326e841397e4572d735941a403dda2182512d5ae431d88` |
| `requests-02.jsonl` | `c4f3e1b904eb409cfd9aa6b33d7a1d77a9ff8e932c438b21082c028b22c4c294` |
| `requests-03.jsonl` | `b7c3d23a45f70d1d17cc711bd6f4aa9ea44dc45f04245343f7e53b0223a568a4` |
| `requests-04.jsonl` | `4d72507ff0638376081a819954aabc0986f62b69c941f05a90b8b4bb122094fc` |
| `requests-05.jsonl` | `c721155075a38944dcd3c8124be857fca5272c3601bc0e8f857adc9d81b5d24c` |
| `requests-06.jsonl` | `c9519cfa1c0444e19e8c43191d90fe6deb14a804c5689b17210c189849a7a6ec` |
| `requests-07.jsonl` | `f3fbba884e4a054e9ef349fb90f7dc255affd081a0ce3df3946b8c1f5f93f0b5` |

## 6. 현재 blocker와 진행 중 수정

독립 verifier가 현재 HEAD에서 full 제출을 차단했다.

1. full manifest의 전체 SHA-256가 live code에 pin되지 않았다. 현재는 내부
   field와 shard hash를 함께 바꾸는 self-consistent tamper를 전체 manifest
   identity로 차단하지 못한다.
2. full candidate의 전용 bucket contract가 pilot manifest identity에 묶인
   기존 계약이다.
3. full `create` CLI는 전용 bucket contract 파일을 받지 않고
   `EXISTING_SHARED` bucket 탐색 경로를 사용한다.

다른 구현 에이전트가 다음 수정 작업을 진행 중이다.

- 승인된 full manifest 전체 digest pin
- full manifest에 결속된 전용 bucket contract 검증
- full create 시 `--dedicated-bucket-file` 필수화
- pilot과 full manifest/bucket 교차 사용 차단
- self-consistent tamper, 잘못된 bucket binding, shared-bucket fallback 회귀 테스트

**이 수정 커밋과 독립 verifier PASS 전에는 아래 live full command를 절대 실행하지
않는다.** 수정으로 manifest가 바뀌면 이 문서의 candidate hash도 갱신한다.

## 7. Full live 실행 절차

이 절차는 blocker 수정과 독립 검증이 끝난 뒤에만 유효하다. DB에는 접근하지
않는다. 실제 환경 파일은 값 비출력 방식으로 로드하고, 아래
`<SERVER_ENV_FILE>`과 `<PYTHON>` placeholder를 로컬의 승인된 값으로 치환한다.

### 7.1 시작 전 gate

1. `git status`가 예상한 변경만 포함하는지 확인한다.
2. blocker 수정 commit SHA와 full manifest SHA를 기록한다.
3. manifest가 602/28,896/8×3,612, 올바른 cohort/prompt/config/registry hash,
   full token review passed, `$5.318408 < $7.50 < $10`을 만족하는지 로컬
   검증한다.
4. full 전용 bucket contract가 **현재 full manifest SHA**에 묶였는지
   검사한다. contract 내용을 출력하지 않는다.
5. 새로운 create-only ledger 경로를 정하고 파일이 존재하지 않음을 확인한다.
6. production DB 연결 또는 수정 코드가 실행 경로에 없음을 다시 확인한다.

### 7.2 순차 create/status

blocker fix 후 CLI는 다음 형태여야 한다.

```bash
set -a
source <SERVER_ENV_FILE>
set +a
env PYTHONPATH=. <PYTHON> scripts/sensory_vertex_live.py create \
  --execute-live \
  --manifest /private/tmp/cocktail-mate-vertex-full-602-prep-v1/manifest.json \
  --dedicated-bucket-file /private/tmp/cocktail-mate-vertex-full-602-prep-v1/dedicated-bucket.json \
  --ledger /private/tmp/cocktail-mate-vertex-full-602-prep-v1/live-ledger-full-v1.json \
  --shard-index 0
```

한 shard의 create는 정확히 한 번만 호출한다. 이후 terminal 상태까지 read-only
status만 호출한다.

```bash
env PYTHONPATH=. <PYTHON> scripts/sensory_vertex_live.py status \
  --execute-live \
  --ledger /private/tmp/cocktail-mate-vertex-full-602-prep-v1/live-ledger-full-v1.json \
  --shard-index 0
```

`JOB_STATE_SUCCEEDED`인 경우에만 다음 shard를 `1, 2, ..., 7` 순서로 생성한다.
한 번에 둘 이상의 active job을 만들지 않는다.

### 7.3 실패·모호 상태 규칙

- retry, fallback, resubmit을 하지 않는다.
- create가 실패하거나 `UNKNOWN_REMOTE_STATE`가 되면 즉시 이후 shard를
  중단한다.
- 원격 create 여부가 모호하면 새 job을 만들지 않는다.
- reconciliation은 read-only list/get으로 기존 job을 유일하게 식별할 때만
  허용한다. display name, model, 정확한 source URI, 정확한 destination URI,
  project/location, 시간 범위를 모두 대조한다.
- 후보가 0개 또는 2개 이상이면 모호 상태를 유지하고 사용자에게 보고한다.
- reconciliation은 기존 local ledger에 감사 event를 추가할 뿐 원격 mutation을
  하지 않아야 한다.
- 숫자 project resource segment는 코드에 고정된 정확한 reviewed project
  number만 허용한다.

### 7.4 Download

8/8 job 성공 후, create-only 결과 root가 존재하지 않는지 확인한다.

```bash
env PYTHONPATH=. <PYTHON> scripts/sensory_vertex_live.py download \
  --execute-live \
  --ledger /private/tmp/cocktail-mate-vertex-full-602-prep-v1/live-ledger-full-v1.json \
  --shard-index 0 \
  --output-dir /private/tmp/cocktail-mate-vertex-full-602-results-v1
```

shard 0부터 7까지 순차 실행한다. 각 shard는 object count, size, generation,
local SHA-256 검증이 완료되어야 한다. 다운로드가 끝나도 cleanup은 실행하지
않는다.

## 8. Offline parse → project → artifacts

아래 단계에는 network와 DB가 없어야 한다.

### 8.1 Parse

각 `<RESPONSE_SHARD_NN>`은 download ledger에 기록된 해당 shard의
`predictions.jsonl` 절대경로다.

```bash
env PYTHONPATH=. <PYTHON> scripts/sensory_vertex_batch.py parse \
  --manifest /private/tmp/cocktail-mate-vertex-full-602-prep-v1/manifest.json \
  --response <RESPONSE_SHARD_00> \
  --response <RESPONSE_SHARD_01> \
  --response <RESPONSE_SHARD_02> \
  --response <RESPONSE_SHARD_03> \
  --response <RESPONSE_SHARD_04> \
  --response <RESPONSE_SHARD_05> \
  --response <RESPONSE_SHARD_06> \
  --response <RESPONSE_SHARD_07> \
  --output /private/tmp/cocktail-mate-vertex-full-602-results-v1/parsed.jsonl \
  --quarantine /private/tmp/cocktail-mate-vertex-full-602-results-v1/quarantine.jsonl \
  --summary /private/tmp/cocktail-mate-vertex-full-602-results-v1/parse-summary.json
```

필수 gate: expected/accepted `28,896/28,896`, quarantine 0, missing 0,
duplicate 0, unexpected 0, 모든 A–E finite/nonnegative, 각 확률합 오차 허용치
내부.

### 8.2 Project

```bash
env PYTHONPATH=. <PYTHON> scripts/sensory_vertex_batch.py project \
  --input /private/tmp/cocktail-mate-vertex-full-602-results-v1/parsed.jsonl \
  --output /private/tmp/cocktail-mate-vertex-full-602-results-v1/projection-ready.jsonl
```

필수 gate: 정확히 602 records, 각 record에 registry 순서의 48축과 각 축의
A–E 5확률, lineage hash가 존재해야 한다. `--allow-partial`을 사용하지 않는다.

### 8.3 Raw240/Graph48/Preference48/top-5/S² build

먼저 overwrite를 막기 위해 새 output directory를 고른다.

```bash
env PYTHONPATH=. <PYTHON> scripts/build_sensory_artifacts.py \
  --input /private/tmp/cocktail-mate-vertex-full-602-results-v1/projection-ready.jsonl \
  --output-dir /private/tmp/cocktail-mate-sensory-graph-602-v1 \
  --run-id sensory-602-v1 \
  --clusters 7 \
  --seed 20260806 \
  --iterations 450 \
  --multistarts 16 \
  --report-only
```

먼저 `--report-only` 결과를 검토한다. 모든 quality gate가 통과한 동일 입력과
설정에 대해서만 새 create-only output directory를 사용해
`--enforce-quality`로 재구축한다.

## 9. 품질 gate

### Teacher와 vector

- 602 unique integer cocktail IDs와 pinned cohort hash
- 602 × 48 = 28,896 complete distributions
- quarantine/error/missing/duplicate/unexpected 모두 0
- 각 축 ordered labels A–E, finite/nonnegative 확률, 합 1
- `Raw240` dimension 240
- `Graph48` dimension 48, category-balanced, finite, nonzero, unit-L2
- `Preference48` dimension 48, unnormalized MIPS 계약 유지
- 모든 source/response/projection/vector lineage SHA 검증

### Recommendation과 edge

- exact cosine directed rows: `602 × 5 = 3,010`
- source별 rank 1..5, self edge 없음, ID tie-break 결정적
- graph edge는 directed top-5의 either-direction union과 정확히 동일
- cocktail → cocktail 추천과 시각화가 같은 canonical artifact를 사용
- hub node/edge가 public JSON에 0개

### S²

- 모든 좌표 unit norm, 최대 norm error `≤ 1e-12`
- mean coordinate Recall@5 `≥ 0.60`
- top-5 neighbor coverage `≥ 0.90`
- cosine bottom-decile nonneighbor false-close count `0`
- union-edge angular target RMSE `≤ 0.40` radians
- seed, multistart objective, selected start, coordinate SHA 기록

legacy32로 만든 과거 diagnostic S²는 Recall@5 약 `0.010963`, coverage 약
`0.053156`, false-close `6,742`, RMSE 약 `0.264759`로 promotion 실패다.
새 sensory teacher 결과의 품질을 별도로 측정해야 하며, legacy32 성공으로
간주하면 안 된다.

## 10. 구현 완료와 미구현

완료:

- sensory 48-axis registry와 Raw240/Graph48/Preference48 계약
- user selection masking + Preference48 exact MIPS
- exact Graph48 cosine top-5와 union edge
- hidden-hub graph-only S² layout과 public artifact
- Vertex request/build/parse/project offline pipeline
- ADC 기반 sequential live pilot boundary
- 10×48 실제 recipe pilot와 logprobs parser 검증
- 602 full candidate의 local-only 준비와 비용 review

미구현 또는 production 미연결:

- pgvector ANN backend adapter
- sensory recommendation API endpoint/service wiring
- cocktail-to-cocktail production recommendation 교체
- DB schema/migration/vector 적재
- frontend S² renderer와 interaction
- production artifact serving/version switch/rollback
- weighted Wasserstein distance 또는 soft-label reranker

## 11. 테스트

worktree root에서 실행한다.

```bash
env PYTHONPATH=. <PYTHON> -m pytest \
  tests/test_sensory_embedding.py \
  tests/test_sensory_embedding_v2.py \
  tests/test_sensory_teacher_projection.py \
  tests/test_sensory_vertex_batch.py \
  tests/test_sensory_vertex_batch_blackbox.py \
  tests/test_sensory_vertex_live.py \
  tests/test_vector_similarity.py \
  tests/test_vector_similarity_v2.py \
  tests/test_spherical_graph.py \
  tests/test_spherical_graph_v2.py \
  tests/test_sensory_module_integration.py \
  tests/test_sensory_embedding_similarity_blackbox.py \
  tests/test_build_sensory_artifacts.py

<PYTHON> -m ruff format --check \
  app/sensory_embedding app/vector_similarity app/spherical_graph \
  scripts/sensory_vertex_batch.py scripts/sensory_vertex_live.py \
  scripts/build_sensory_artifacts.py tests

<PYTHON> -m ruff check \
  app/sensory_embedding app/vector_similarity app/spherical_graph \
  scripts/sensory_vertex_batch.py scripts/sensory_vertex_live.py \
  scripts/build_sensory_artifacts.py tests
```

repository에 설치된 mypy 버전과 기존 strict command가 있으면 세 모듈과 세
pipeline script에도 실행한다. live 호출 test에서는 실제 network를 사용하지
않고 fake gateway를 사용해야 한다.

## 12. 완료 정의와 DB 승인 조건

이번 로컬 실험의 완료 조건:

1. full blocker 수정 + 독립 verifier PASS
2. 8/8 full jobs success, create 각 1회, retry/fallback 0
3. 28,896/28,896 parse, quarantine/error/missing/duplicate 0
4. projection-ready 602 records
5. Raw240/Graph48/Preference48와 exact top-5/union artifact 생성
6. S² 모든 promotion quality gate 통과
7. 실제 usage/cost와 `$10` budget 준수 보고
8. 모든 artifact/ledger/manifest SHA와 재현 명령 기록
9. production DB write 0, cleanup 0

위 조건이 충족되어도 DB 수정 권한이 자동으로 생기지 않는다. 사용자에게 결과,
품질 지표, 비용, schema/API 전환안, rollback 계획을 먼저 보고해야 한다.
그 후 사용자가 **별도로 production DB 반영을 명시 승인**한 경우에만 새
브랜치/마이그레이션/백업 검증을 거쳐 DB 작업을 시작한다. 모호한 “계속”이나
Vertex Batch 승인은 DB 수정 승인으로 해석하지 않는다.
