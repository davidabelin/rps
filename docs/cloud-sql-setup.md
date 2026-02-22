# Cloud SQL Migration (PostgreSQL) for RPS Agent Lab

This document covers the next step after bucket + Cloud Tasks setup.

## What changed in code

- `rps_storage/repository.py` now supports:
  - local SQLite path (existing behavior), or
  - `DATABASE_URL` for SQL databases (PostgreSQL-ready).
- `rps_web/__init__.py` now prefers `DATABASE_URL` when set.

## Console setup

## 1) Enable API

Enable **Cloud SQL Admin API** in:

- APIs & Services -> Library

## 2) Create PostgreSQL instance

Cloud SQL -> Instances -> Create instance -> PostgreSQL

- Instance ID: `rps-sql`
- Region: same as app if possible (for you, likely `us-west1` or your active App Engine region)
- PostgreSQL version: latest stable offered
- Machine preset: start small (shared-core) for early stage
- Availability: Zonal for dev/stage
- Backups: enabled

## 3) Create DB and user

Inside the new instance:

- Databases -> Create database: `rps`
- Users -> Add user:
  - username: `rps_app`
  - password: strong random password

## 4) Configure connectivity

For App Engine standard, use Cloud SQL Auth Proxy unix socket path via SQLAlchemy URL:

`postgresql+pg8000://rps_app:<PASSWORD>@/rps?unix_sock=/cloudsql/<PROJECT_ID>:<REGION>:<INSTANCE_ID>/.s.PGSQL.5432`

Example (shape only):

`postgresql+pg8000://rps_app:***@/rps?unix_sock=/cloudsql/directed-sonar-429119-u2:us-west1:rps-sql/.s.PGSQL.5432`

## 5) Grant IAM to service account

For `circa-orbit-sa@directed-sonar-429119-u2.iam.gserviceaccount.com`, grant:

- `Cloud SQL Client` (`roles/cloudsql.client`)

## 6) Update `app.yaml`

Set:

- `DATABASE_URL` to the PostgreSQL URL above

Keep `DB_PATH` temporarily as fallback value but it will be ignored when `DATABASE_URL` is non-empty.

## 7) Deploy

`gcloud app deploy app.yaml`

## 8) Verify

1. Open `/healthz`.
2. Create a new game in `/play`.
3. Play a few rounds.
4. Open `/training`, create a training job, verify job progresses.
5. Confirm new rows appear in Cloud SQL tables (`games`, `rounds`, `training_jobs`, `models`, `rl_jobs`).

## 9) Cutover note

- Existing historical SQLite data does not auto-migrate.
- If needed, we can run a one-time export/import script from SQLite -> Cloud SQL.
