"""Dataset/event export helpers for gameplay-derived training corpora."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rps_storage.object_store import is_gcs_uri, join_storage_path, write_bytes, write_text


def append_round_event(event: dict, events_dir: str) -> str:
    """Append one round event to local JSONL or per-object GCS path."""

    timestamp = datetime.now(UTC)
    payload = json.dumps(event, sort_keys=True) + "\n"
    if is_gcs_uri(events_dir):
        day_key = timestamp.strftime("%Y%m%d")
        object_name = f"{timestamp.strftime('%H%M%S%f')}_{uuid4().hex}.json"
        destination = join_storage_path(events_dir, day_key, object_name)
        return write_text(destination, payload, content_type="application/json; charset=utf-8")
    root = Path(events_dir)
    root.mkdir(parents=True, exist_ok=True)
    filename = timestamp.strftime("%Y%m%d") + ".jsonl"
    path = root / filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
    return str(path)


def export_rounds_to_jsonl(rounds: list[dict], output_path: str) -> str:
    """Export round rows as newline-delimited JSON."""

    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rounds)
    if is_gcs_uri(output_path):
        return write_text(output_path, payload, content_type="application/x-ndjson; charset=utf-8")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return str(path)


def export_rounds_to_parquet(rounds: list[dict], output_path: str) -> str:
    """Export round rows as parquet file/object.

    Raises
    ------
    RuntimeError
        If pandas/parquet backend dependencies are unavailable.
    """

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Parquet export requires pandas and pyarrow/fastparquet") from exc
    frame = pd.DataFrame(rounds)
    if is_gcs_uri(output_path):
        from io import BytesIO

        buffer = BytesIO()
        frame.to_parquet(buffer, index=False)
        return write_bytes(output_path, buffer.getvalue(), content_type="application/octet-stream")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return str(path)
