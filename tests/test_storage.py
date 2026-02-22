from pathlib import Path

from rps_storage.object_store import is_gcs_uri, join_storage_path, read_bytes, write_bytes, write_text


def test_storage_path_helpers():
    assert is_gcs_uri("gs://bucket/models")
    assert not is_gcs_uri("data/models")
    assert join_storage_path("data/models", "a.pkl").endswith(str(Path("data/models") / "a.pkl"))
    assert join_storage_path("gs://bucket/models", "a.pkl") == "gs://bucket/models/a.pkl"


def test_local_object_store_round_trip(tmp_path: Path):
    target = tmp_path / "models" / "artifact.bin"
    write_bytes(str(target), b"abc123")
    assert read_bytes(str(target)) == b"abc123"

    text_target = tmp_path / "events" / "sample.jsonl"
    write_text(str(text_target), '{"ok": true}\n')
    assert text_target.read_text(encoding="utf-8") == '{"ok": true}\n'
