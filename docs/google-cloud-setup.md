# Google Cloud Setup Checklist (RPS Agent Lab)

This checklist matches the current codebase after Step 3 scaffolding.

## 1) Confirm project and region

- Project: `directed-sonar-429119-u2`
- Preferred region: `us-west1` (already used by your App Engine app)

Keep Cloud Tasks queue and storage bucket in the same region where practical.

## 2) Enable required APIs

In Google Cloud Console, open **APIs & Services > Library** and enable:

- `Cloud Tasks API`
- `Cloud Storage API`
- `App Engine Admin API` (already enabled for deploys, but confirm)
- `Cloud Build API` (already used by deploy, but confirm)
- `Artifact Registry API` (recommended for future split worker image workflows)

## 3) Create a durable storage bucket

In **Cloud Storage > Buckets**, create:

- Bucket name: `directed-sonar-429119-u2-rps-data` (or your preferred unique name)
- Location type: `Region`
- Region: `us-west1`
- Storage class: `Standard`
- Public access: **prevent public access**
- Access control: `Uniform`

No manual folders are required, but these prefixes will be used by app config:

- `events/`
- `models/`
- `exports/`

## 4) Create Cloud Tasks queue for training jobs

In **Cloud Tasks > Queues**, create queue:

- Queue ID: `rps-training`
- Region: `us-west1`
- Dispatch type: `HTTP`
- Max dispatches per second: `1` (safe starting point)
- Max concurrent dispatches: `1` (safe starting point)
- Retries: keep defaults for now

## 5) IAM roles for runtime service account

Runtime service account in your deployment:

- `circa-orbit-sa@directed-sonar-429119-u2.iam.gserviceaccount.com`

Grant these roles at project scope (or narrower resource scope if preferred):

- `Cloud Tasks Enqueuer` (`roles/cloudtasks.enqueuer`)
- `Storage Object Admin` (`roles/storage.objectAdmin`) on the RPS data bucket

Optional hardening later:

- replace broad object admin with finer-grained bucket IAM

## 6) Choose worker auth mode

Current code supports a shared header token (`X-Worker-Token`) for worker endpoint auth.

Generate a long random token (at least 32 chars) and store it for:

- `INTERNAL_WORKER_TOKEN`

## 7) Update `app.yaml` env vars for production mode

Set these values:

- `EVENTS_DIR: gs://directed-sonar-429119-u2-rps-data/events`
- `MODELS_DIR: gs://directed-sonar-429119-u2-rps-data/models`
- `EXPORTS_DIR: gs://directed-sonar-429119-u2-rps-data/exports`
- `TRAINING_EXECUTION_MODE: task_queue`
- `TASKS_PROJECT_ID: directed-sonar-429119-u2`
- `TASKS_LOCATION: us-west1`
- `TASKS_QUEUE: rps-training`
- `TRAINING_WORKER_URL: https://directed-sonar-429119-u2.uw.r.appspot.com`
- `INTERNAL_WORKER_TOKEN: <your-random-token>`

`DB_PATH` remains SQLite-based in this phase. It is still temporary on App Engine when set to `/tmp`.

## 8) Deploy

Deploy after editing `app.yaml`:

```powershell
gcloud app deploy app.yaml
```

## 9) Verify end-to-end

1. Open `/training`.
2. Submit a training job.
3. Confirm job enters `queued`, then `running`, then `completed`.
4. Confirm model artifacts appear under your bucket `models/` prefix.
5. Play with `active_model` on `/play`.

## 10) Known current limitation

- Database metadata (`games`, `rounds`, `jobs`, `models`) is still SQLite-based in this phase.
- For durable production metadata, next step is Cloud SQL migration (or equivalent managed DB).
