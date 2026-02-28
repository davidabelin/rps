# Training Worker Contract

This project uses a split-worker architecture:

- Web/API service: Flask app (`run:app`)
- Worker service: asynchronous training execution (`rps_training.jobs.TrainingJobManager`)

## Contract payload

`POST /api/v1/training/jobs`

```json
{
  "model_type": "decision_tree | mlp | frequency",
  "lookback": 5,
  "test_size": 0.2,
  "learning_rate": 0.001,
  "hidden_layer_sizes": [64, 32],
  "epochs": 200,
  "batch_size": "auto",
  "random_state": 42
}
```

## Worker trigger endpoint

`POST /api/v1/internal/training/jobs/{job_id}/run`

- Header: `X-Worker-Token: <INTERNAL_WORKER_TOKEN>`
- Intended caller: Cloud Tasks HTTP task
- Behavior: loads job params from DB and executes training for `job_id`

## Job lifecycle

- `queued`
- `running`
- `completed`
- `failed`

## Future Cloud Run worker handoff

For production, the web service should enqueue the payload to a task queue and the worker service should:

1. load round logs from shared storage
2. train model artifact
3. persist model metadata
4. update `training_jobs` status
