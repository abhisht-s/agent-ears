"""Immutable SHA-256 blob storage and the SQLite evidence graph."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from music_locator.models import AssetKind, AssetRecord, JobRecord, JobStatus


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class BlobStore:
    """Content-addressed immutable byte storage."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for_sha256(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid SHA-256 digest")
        return self.root / "sha256" / digest[:2] / digest[2:]

    def put_bytes(self, data: bytes) -> tuple[str, Path]:
        digest = hashlib.sha256(data).hexdigest()
        destination = self.path_for_sha256(digest)
        if destination.exists():
            if destination.read_bytes() != data:
                raise RuntimeError("content-address collision or corrupt blob")
            return digest, destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix="incoming-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, 0o444)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return digest, destination

    def put_file(self, source: Path) -> tuple[str, Path]:
        return self.put_bytes(source.read_bytes())

    def get_bytes(self, digest: str) -> bytes:
        return self.path_for_sha256(digest).read_bytes()


MIGRATION_001 = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    media_type TEXT NOT NULL,
    original_name TEXT NOT NULL,
    byte_length INTEGER NOT NULL CHECK (byte_length >= 0),
    blob_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(kind, sha256, original_name)
);
CREATE INDEX IF NOT EXISTS assets_sha256_idx ON assets(sha256);

CREATE TABLE IF NOT EXISTS audio_streams (
    stream_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
    channels INTEGER NOT NULL CHECK (channels > 0),
    frame_count INTEGER NOT NULL CHECK (frame_count >= 0),
    sample_format TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS time_maps (
    time_map_id TEXT PRIMARY KEY,
    source_stream_id TEXT NOT NULL REFERENCES audio_streams(stream_id),
    derived_stream_id TEXT NOT NULL REFERENCES audio_streams(stream_id),
    artifact_id TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(source_stream_id, derived_stream_id)
);

CREATE TABLE IF NOT EXISTS derivatives (
    derivative_id TEXT PRIMARY KEY,
    source_stream_id TEXT NOT NULL REFERENCES audio_streams(stream_id),
    derived_stream_id TEXT NOT NULL REFERENCES audio_streams(stream_id),
    operation TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    time_map_id TEXT NOT NULL REFERENCES time_maps(time_map_id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    analysis_run_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    code_version TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    analysis_run_id TEXT REFERENCES analysis_runs(analysis_run_id),
    artifact_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    media_type TEXT NOT NULL,
    byte_length INTEGER NOT NULL CHECK (byte_length >= 0),
    blob_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_sha256_idx ON artifacts(sha256);

CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    original_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    original_stream_id TEXT NOT NULL REFERENCES audio_streams(stream_id),
    analysis_artifact_id TEXT REFERENCES artifacts(artifact_id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interpretations (
    interpretation_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(query_id),
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    rank INTEGER NOT NULL CHECK (rank BETWEEN 1 AND 4),
    selected INTEGER NOT NULL CHECK (selected IN (0, 1)),
    score REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(query_id, rank)
);

CREATE TABLE IF NOT EXISTS targets (
    target_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    ticks_per_beat INTEGER NOT NULL CHECK (ticks_per_beat > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexes (
    index_id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    parameters_json TEXT NOT NULL,
    target_set_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS searches (
    search_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(query_id),
    interpretation_id TEXT REFERENCES interpretations(interpretation_id),
    index_id TEXT REFERENCES indexes(index_id),
    analysis_run_id TEXT REFERENCES analysis_runs(analysis_run_id),
    result_artifact_id TEXT REFERENCES artifacts(artifact_id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alignments (
    alignment_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL REFERENCES searches(search_id),
    target_id TEXT NOT NULL REFERENCES targets(target_id),
    rank INTEGER NOT NULL CHECK (rank > 0),
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    uncalibrated_score REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(search_id, rank)
);

CREATE TABLE IF NOT EXISTS corrections (
    correction_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL REFERENCES searches(search_id),
    supersedes_correction_id TEXT REFERENCES corrections(correction_id),
    correction_type TEXT NOT NULL,
    payload_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    result_entity_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


MIGRATION_002 = """
PRAGMA foreign_keys = OFF;
BEGIN IMMEDIATE;

ALTER TABLE alignments RENAME TO alignments_v001;
ALTER TABLE targets RENAME TO targets_v001;

CREATE TABLE targets (
    target_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    source_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    ticks_per_beat INTEGER,
    created_at TEXT NOT NULL,
    CHECK (
        (kind = 'audio' AND ticks_per_beat IS NULL)
        OR (kind IN ('midi', 'timed_notes') AND ticks_per_beat > 0)
    )
);

CREATE TABLE alignments (
    alignment_id TEXT PRIMARY KEY,
    search_id TEXT NOT NULL REFERENCES searches(search_id),
    target_id TEXT NOT NULL REFERENCES targets(target_id),
    rank INTEGER NOT NULL CHECK (rank > 0),
    artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    uncalibrated_score REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(search_id, rank)
);

INSERT INTO targets(target_id, name, kind, source_asset_id, artifact_id,
                    ticks_per_beat, created_at)
SELECT target_id, name, kind, source_asset_id, artifact_id,
       ticks_per_beat, created_at
FROM targets_v001;

INSERT INTO alignments(alignment_id, search_id, target_id, rank, artifact_id,
                       uncalibrated_score, created_at)
SELECT alignment_id, search_id, target_id, rank, artifact_id,
       uncalibrated_score, created_at
FROM alignments_v001;

DROP TABLE alignments_v001;
DROP TABLE targets_v001;

CREATE TABLE target_decodes (
    decode_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL REFERENCES targets(target_id),
    cache_key TEXT NOT NULL UNIQUE,
    analysis_run_id TEXT NOT NULL REFERENCES analysis_runs(analysis_run_id),
    manifest_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    time_map_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    decoded_asset_id TEXT NOT NULL REFERENCES assets(asset_id),
    decoded_stream_id TEXT NOT NULL REFERENCES audio_streams(stream_id),
    created_at TEXT NOT NULL
);
CREATE INDEX target_decodes_target_idx ON target_decodes(target_id, created_at);

CREATE TABLE target_analyses (
    target_analysis_id TEXT PRIMARY KEY,
    target_id TEXT NOT NULL REFERENCES targets(target_id),
    decode_id TEXT NOT NULL REFERENCES target_decodes(decode_id),
    cache_key TEXT NOT NULL UNIQUE,
    analysis_run_id TEXT NOT NULL REFERENCES analysis_runs(analysis_run_id),
    bundle_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    created_at TEXT NOT NULL
);
CREATE INDEX target_analyses_target_idx ON target_analyses(target_id, created_at);

CREATE TABLE target_views (
    view_id TEXT PRIMARY KEY,
    target_analysis_id TEXT NOT NULL REFERENCES target_analyses(target_analysis_id),
    target_id TEXT NOT NULL REFERENCES targets(target_id),
    representation_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
    source_role TEXT NOT NULL,
    backend TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX target_views_target_idx ON target_views(target_id, target_analysis_id);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(MIGRATION_001)
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (1, utc_now()),
            )
            migration = connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = 2"
            ).fetchone()
            if migration is None:
                connection.executescript(MIGRATION_002)
                violations = connection.execute("PRAGMA foreign_key_check").fetchall()
                if violations:
                    raise RuntimeError(
                        f"schema migration 2 produced {len(violations)} foreign-key violation(s)"
                    )
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (2, utc_now()),
                )
                connection.commit()
                connection.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> None:
        with self.connect() as connection:
            connection.execute(sql, parameters)

    def fetchone(self, sql: str, parameters: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self.connect() as connection:
            return cast(sqlite3.Row | None, connection.execute(sql, parameters).fetchone())

    def fetchall(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(sql, parameters).fetchall())


class Workspace:
    """Owns durable state; higher layers never write ad-hoc files into it."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.database = Database(self.root / "locator.sqlite3")
        self.blobs = BlobStore(self.root / "blobs")

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.database.initialize()

    def store_asset(
        self,
        source: Path | bytes,
        *,
        kind: AssetKind,
        media_type: str,
        original_name: str,
    ) -> AssetRecord:
        data = source.read_bytes() if isinstance(source, Path) else source
        digest, path = self.blobs.put_bytes(data)
        existing = self.database.fetchone(
            "SELECT * FROM assets WHERE kind = ? AND sha256 = ? AND original_name = ?",
            (kind.value, digest, original_name),
        )
        if existing is not None:
            return AssetRecord.model_validate(dict(existing))
        record = AssetRecord(
            asset_id=new_id("asset"),
            kind=kind,
            sha256=digest,
            media_type=media_type,
            original_name=original_name,
            byte_length=len(data),
            blob_path=str(path.relative_to(self.root)),
            created_at=utc_now(),
        )
        self.database.execute(
            """
            INSERT INTO assets(asset_id, kind, sha256, media_type, original_name,
                               byte_length, blob_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.asset_id,
                record.kind.value,
                record.sha256,
                record.media_type,
                record.original_name,
                record.byte_length,
                record.blob_path,
                record.created_at,
            ),
        )
        return record

    def store_json_artifact(
        self,
        payload: BaseModel | dict[str, Any] | list[Any],
        *,
        artifact_type: str,
        analysis_run_id: str | None = None,
    ) -> str:
        if isinstance(payload, BaseModel):
            value: Any = payload.model_dump(mode="json")
        else:
            value = payload
        data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        schema_version = "0.0.1"
        if isinstance(value, dict) and isinstance(value.get("schema_version"), str):
            schema_version = str(value["schema_version"])
        return self.store_artifact_bytes(
            data,
            artifact_type=artifact_type,
            media_type="application/json",
            schema_version=schema_version,
            analysis_run_id=analysis_run_id,
        )

    def store_artifact_bytes(
        self,
        data: bytes,
        *,
        artifact_type: str,
        media_type: str,
        schema_version: str = "0.0.2",
        analysis_run_id: str | None = None,
    ) -> str:
        """Store an immutable non-asset artifact such as a compressed evidence grid."""

        digest, path = self.blobs.put_bytes(data)
        artifact_id = new_id("artifact")
        self.database.execute(
            """
            INSERT INTO artifacts(artifact_id, analysis_run_id, artifact_type, schema_version,
                                  sha256, media_type, byte_length, blob_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                analysis_run_id,
                artifact_type,
                schema_version,
                digest,
                media_type,
                len(data),
                str(path.relative_to(self.root)),
                utc_now(),
            ),
        )
        return artifact_id

    def load_artifact_json(self, artifact_id: str) -> Any:
        row = self.database.fetchone(
            "SELECT sha256 FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return json.loads(self.blobs.get_bytes(str(row["sha256"])))

    def asset_path(self, asset_id: str) -> Path:
        row = self.database.fetchone("SELECT sha256 FROM assets WHERE asset_id = ?", (asset_id,))
        if row is None:
            raise KeyError(f"unknown asset: {asset_id}")
        return self.blobs.path_for_sha256(str(row["sha256"]))

    def create_analysis_run(self, operation: str, parameters: dict[str, Any]) -> str:
        run_id = new_id("run")
        self.database.execute(
            """
            INSERT INTO analysis_runs(analysis_run_id, operation, parameters_json, code_version,
                                      status, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                operation,
                json.dumps(parameters, sort_keys=True),
                "0.0.2",
                "running",
                utc_now(),
            ),
        )
        return run_id

    def finish_analysis_run(self, run_id: str, *, error: str | None = None) -> None:
        self.database.execute(
            """
            UPDATE analysis_runs SET status = ?, completed_at = ?, error = ?
            WHERE analysis_run_id = ?
            """,
            ("failed" if error else "succeeded", utc_now(), error, run_id),
        )

    def artifact_row(self, artifact_id: str) -> sqlite3.Row:
        row = self.database.fetchone(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return row

    def asset_row(self, asset_id: str) -> sqlite3.Row:
        row = self.database.fetchone("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))
        if row is None:
            raise KeyError(f"unknown asset: {asset_id}")
        return row

    def create_job(
        self, operation: str, arguments: dict[str, str | int | float | bool]
    ) -> JobRecord:
        job = JobRecord(
            job_id=new_id("job"),
            operation=operation,
            status=JobStatus.RUNNING,
            arguments=arguments,
            result_entity_id=None,
            error=None,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        self.database.execute(
            """
            INSERT INTO jobs(job_id, operation, status, arguments_json, result_entity_id,
                             error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                job.operation,
                job.status.value,
                json.dumps(job.arguments, sort_keys=True),
                job.result_entity_id,
                job.error,
                job.created_at,
                job.updated_at,
            ),
        )
        return job

    def finish_job(
        self, job_id: str, *, result_entity_id: str | None = None, error: str | None = None
    ) -> None:
        self.database.execute(
            """
            UPDATE jobs SET status = ?, result_entity_id = ?, error = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (
                JobStatus.FAILED.value if error else JobStatus.SUCCEEDED.value,
                result_entity_id,
                error,
                utc_now(),
                job_id,
            ),
        )
