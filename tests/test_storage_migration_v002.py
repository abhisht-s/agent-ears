from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from music_locator.storage import MIGRATION_001, Database


def test_v001_database_migrates_midi_targets_and_allows_null_audio_ppq(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "locator.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(MIGRATION_001)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
            ("2026-08-08T00:00:00+00:00",),
        )
        connection.execute(
            """
            INSERT INTO assets(asset_id, kind, sha256, media_type, original_name,
                               byte_length, blob_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "asset-midi",
                "target_midi",
                "a" * 64,
                "audio/midi",
                "oracle.mid",
                1,
                "blobs/sha256/aa/oracle",
                "2026-08-08T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO artifacts(artifact_id, analysis_run_id, artifact_type,
                                  schema_version, sha256, media_type, byte_length,
                                  blob_path, created_at)
            VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "artifact-midi",
                "target.timed_notes",
                "0.0.1",
                "b" * 64,
                "application/json",
                1,
                "blobs/sha256/bb/oracle",
                "2026-08-08T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO targets(target_id, name, kind, source_asset_id, artifact_id,
                                ticks_per_beat, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "target-midi",
                "oracle",
                "midi",
                "asset-midi",
                "artifact-midi",
                480,
                "2026-08-08T00:00:00+00:00",
            ),
        )

    database = Database(database_path)
    database.initialize()

    midi = database.fetchone("SELECT * FROM targets WHERE target_id = ?", ("target-midi",))
    assert midi is not None
    assert midi["kind"] == "midi"
    assert midi["ticks_per_beat"] == 480
    assert [
        row["version"]
        for row in database.fetchall("SELECT version FROM schema_migrations ORDER BY version")
    ] == [1, 2]

    ticks_column = next(
        row
        for row in database.fetchall("PRAGMA table_info(targets)")
        if row["name"] == "ticks_per_beat"
    )
    assert ticks_column["notnull"] == 0

    database.execute(
        """
        INSERT INTO targets(target_id, name, kind, source_asset_id, artifact_id,
                            ticks_per_beat, created_at)
        VALUES (?, ?, ?, ?, ?, NULL, ?)
        """,
        (
            "target-audio",
            "synthetic audio",
            "audio",
            "asset-midi",
            "artifact-midi",
            "2026-08-08T00:00:01+00:00",
        ),
    )
    audio = database.fetchone("SELECT * FROM targets WHERE target_id = ?", ("target-audio",))
    assert audio is not None
    assert audio["ticks_per_beat"] is None

    # Startup after migration must be idempotent and retain both target kinds.
    database.initialize()
    assert database.fetchone("SELECT COUNT(*) AS count FROM targets")["count"] == 2


def test_v002_migration_validation_failure_rolls_back_schema_and_marker(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "locator.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(MIGRATION_001)
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (1, ?)",
            ("2026-08-08T00:00:00+00:00",),
        )
        connection.commit()
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO targets(target_id, name, kind, source_asset_id, artifact_id,
                                ticks_per_beat, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "target-dangling",
                "dangling",
                "midi",
                "missing-asset",
                "missing-artifact",
                480,
                "2026-08-08T00:00:01+00:00",
            ),
        )

    database = Database(database_path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="foreign-key violation"):
            database.initialize()

        assert [
            row["version"]
            for row in database.fetchall("SELECT version FROM schema_migrations ORDER BY version")
        ] == [1]
        ticks_column = next(
            row
            for row in database.fetchall("PRAGMA table_info(targets)")
            if row["name"] == "ticks_per_beat"
        )
        assert ticks_column["notnull"] == 1
        assert (
            database.fetchone(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'target_decodes'"
            )
            is None
        )
