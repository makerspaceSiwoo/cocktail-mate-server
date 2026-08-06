# Sensory Vertex Batch live boundary

`scripts.sensory_vertex_live` is the reviewed cloud adapter for artifacts built
offline. It never reads the database and does not build or alter prompts.

## Frozen live cohort

Live creation accepts only the frozen 602-cocktail cohort:

- 602 cocktails × 48 axes = 28,896 requests
- eight ordered shards of 3,612 requests
- ID-set SHA-256
  `56e77646b60ad9b45cbdcd43f4807dde994ef40b1d5e4461dbfa41ca2d59c05f`
- cohort source-file SHA-256
  `8755a91cfd2709b87fad3a05e5daef158d7ea589cb08e6c3f09ab4ecabd4ab6f`

The live validator requires `cohort_source_sha256`,
`cohort_id_set_sha256`, `cohort_row_count`, and both manifest ID-set fields to
equal the pinned cohort. It reuses the offline full-manifest validator,
including all 28,896 request identities, and rejects the 622-row superset. The
manifest must also bind the exact reviewed project
`gen-lang-client-0477982146`, Vertex location `global`, model
`gemini-2.5-flash`, prompt/config/registry hashes, a passing token pilot, the
one-day lifecycle contract, and the $7.50 soft/$10 hard budget controls.

## Authentication and preflight

Every remote command requires `--execute-live`. Authentication is
service-account Application Default Credentials loaded only by `google-auth`.
The adapter does not open or parse a credential JSON file. `GEMINI_API_KEY` and
`GOOGLE_API_KEY` are removed from the process environment for the complete
authenticated operation and restored afterward.

Before the first mutation, a read-only project IAM preflight requires the
reviewed GCS bucket and object permissions, including `storage.buckets.list`.
A failed or ambiguous preflight is written to the atomic live ledger and is not
automatically repeated. Correct the external IAM state and reconcile the ledger
under review rather than resubmitting the same operation.

### 2026-08-06 capability attempt

The approved one-record synthetic Batch capability path was invoked with
service-account ADC before, and once again after the user reported granting
Storage Admin. Both attempts stopped at the first Cloud Storage operation:

```text
status: LIVE_BLOCKED
phase: bucket_list
missing permission: storage.buckets.list
```

No request JSONL was uploaded, no Vertex Batch job was created, no job ID or
logprobs result exists, and no Vertex model charge was incurred. Do not repeat
the same call until IAM state changes.

The post-grant read-only bucket-list check also returned HTTP 403. ADC resolved
to project `gen-lang-client-0477982146` and service account
`cocktail-mate-logprobs@gen-lang-client-0477982146.iam.gserviceaccount.com`.
Therefore the project-level binding for that exact principal must be confirmed;
a bucket-level grant does not provide project bucket-list/create access.

For the dedicated create/use/delete bucket workflow in this adapter, the
simplest temporary project-level grant is `roles/storage.admin` on
`gen-lang-client-0477982146` for the same service account that already has
Vertex AI User. Revoke it after verified download and cleanup. A narrower
alternative is a reviewed pre-existing bucket plus the required bucket/object
permissions, but that changes the dedicated-bucket workflow and must be
configured explicitly.

## Sequential operation

Create exactly one shard job:

```bash
python -m scripts.sensory_vertex_live create \
  --execute-live \
  --manifest sensory-batch/RUN/offline/manifest.json \
  --ledger sensory-batch/RUN/live-ledger.json \
  --shard-index 0
```

The command creates one dedicated run bucket in the valid GCS region
`ASIA-NORTHEAST3`, separately from the `global` Vertex endpoint, with uniform
bucket-level access, public access prevention, and a one-day deletion
lifecycle. It uploads the input JSONL with `ifGenerationMatch=0`, then records
`CREATION_DISPATCHING` before making one SDK Batch-create call. SDK retries are
configured to one total attempt. Any ambiguous result becomes
`UNKNOWN_REMOTE_STATE`; there is no resubmission or Standard-model fallback.

Only a recorded terminal state permits the next shard:

```bash
python -m scripts.sensory_vertex_live status \
  --execute-live \
  --ledger sensory-batch/RUN/live-ledger.json \
  --shard-index 0
```

`status` performs one read-only remote lookup. Repeat it deliberately as needed;
it never creates, cancels, or updates a remote job. A failed status read records
an unknown state and blocks new creation.

After a recorded `JOB_STATE_SUCCEEDED`, download every object under that job's
output prefix:

```bash
python -m scripts.sensory_vertex_live download \
  --execute-live \
  --ledger sensory-batch/RUN/live-ledger.json \
  --shard-index 0 \
  --output-dir sensory-batch/RUN/recorded-output
```

Local output files are create-only. The ledger records object generations,
sizes, paths, and SHA-256 hashes only after re-reading and verifying every
created file.

After all eight jobs succeeded and all eight downloads are verified, explicitly
delete every object, verify the bucket is empty, and delete the dedicated
bucket:

```bash
python -m scripts.sensory_vertex_live cleanup \
  --execute-live \
  --ledger sensory-batch/RUN/live-ledger.json
```

The ledger is atomically replaced with mode `0600`; individual minimized event
artifacts are atomically create-only. Event files store hashes of remote names.
The live ledger retains exact bucket and job resource names only where they are
required for status, download, and cleanup.
