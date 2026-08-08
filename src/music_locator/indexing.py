"""Deterministic transparent shingle indexes for small local corpora."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from music_locator.models import (
    IndexSnapshot,
    TargetIndexEntry,
    TargetTrackRepresentation,
)
from music_locator.representation import (
    phrase_shingle_keys,
    shingle_keys,
    target_feature_views,
)
from music_locator.storage import Workspace, new_id, utc_now
from music_locator.targets import list_targets
from music_locator.voice_extraction import EXTRACTION_VERSION, extract_monophonic_voices

DEFAULT_SHINGLE_SIZE = 3


def build_index(workspace: Workspace, *, shingle_size: int = DEFAULT_SHINGLE_SIZE) -> str:
    workspace.initialize()
    index_id = new_id("index")
    entries: list[TargetIndexEntry] = []
    inverted: dict[str, list[str]] = defaultdict(list)
    target_fingerprints: list[str] = []
    direct_count = 0
    for target, target_artifact_id in list_targets(workspace):
        target_fingerprints.append(
            f"{target.target_id}:{target_artifact_id}:{target.source_sha256}"
        )
        for voice in extract_monophonic_voices(target):
            notes = voice.notes
            hypothesis = voice.hypothesis
            direct_count += int(hypothesis.hypothesis_kind == "source_monophonic")
            views = target_feature_views(notes, target.ticks_per_beat, target.tempo_map)
            representation = TargetTrackRepresentation(
                target_id=target.target_id,
                track_index=hypothesis.source_track_index,
                event_ids=tuple(note.event_id for note in notes),
                pitch_cents=tuple(int(value) for value in views["pitch_cents"]),
                relative_pitch_cents=tuple(float(value) for value in views["relative_pitch_cents"]),
                interval_cents=tuple(int(value) for value in views["interval_cents"]),
                contour_steps=tuple(int(value) for value in views["contour_steps"]),
                onset_microseconds=tuple(int(value) for value in views["onset_microseconds"]),
                log_ioi_milli=tuple(int(value) for value in views["log_ioi_milli"]),
                start_ticks=tuple(int(value) for value in views["start_ticks"]),
                end_ticks=tuple(int(value) for value in views["end_ticks"]),
                velocities=tuple(int(value) for value in views["velocities"]),
                voice_hypothesis=hypothesis,
            )
            representation_artifact_id = workspace.store_json_artifact(
                representation, artifact_type="index.target_track_representation"
            )
            entries.append(
                TargetIndexEntry(
                    target_id=target.target_id,
                    track_index=hypothesis.source_track_index,
                    event_count=len(notes),
                    representation_artifact_id=representation_artifact_id,
                    voice_id=hypothesis.voice_id,
                    source_channel=hypothesis.source_channel,
                )
            )
            keys = shingle_keys(
                [float(value) for value in representation.interval_cents],
                list(representation.contour_steps),
                [value / 1_000.0 for value in representation.log_ioi_milli],
                shingle_size,
            )
            keys.extend(
                phrase_shingle_keys(
                    [float(value) for value in representation.pitch_cents]
                )
            )
            for position, key in keys:
                inverted[key].append(f"{target.target_id}|{hypothesis.voice_id}|{position}")
    sorted_entries = tuple(
        sorted(entries, key=lambda item: (item.target_id, item.track_index, item.voice_id))
    )
    snapshot = IndexSnapshot(
        index_id=index_id,
        created_at=utc_now(),
        target_entries=sorted_entries,
        shingles={key: tuple(sorted(values)) for key, values in sorted(inverted.items())},
        parameters={
            "shingle_size": shingle_size,
            "views": "interval_semitone,contour_sign,log_ioi_half_nat,interval_contour",
            "phrase_shingles": "run_compressed_multiscale_interval_contour_v1",
            "phrase_interval_widths": "2,3,4",
            "phrase_repeated_pitch_tolerance_cents": 70.0,
            "voice_extraction_version": EXTRACTION_VERSION,
            "indexed_voice_count": len(sorted_entries),
            "source_monophonic_voice_count": direct_count,
            "derived_voice_count": len(sorted_entries) - direct_count,
        },
    )
    artifact_id = workspace.store_json_artifact(snapshot, artifact_type="index.snapshot")
    target_set_hash = hashlib.sha256("\n".join(sorted(target_fingerprints)).encode()).hexdigest()
    workspace.database.execute(
        """
        INSERT INTO indexes(index_id, artifact_id, parameters_json, target_set_hash, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            index_id,
            artifact_id,
            json.dumps(snapshot.parameters, sort_keys=True),
            target_set_hash,
            snapshot.created_at,
        ),
    )
    return index_id


def load_index(workspace: Workspace, index_id: str) -> tuple[IndexSnapshot, str]:
    row = workspace.database.fetchone(
        "SELECT artifact_id FROM indexes WHERE index_id = ?", (index_id,)
    )
    if row is None:
        raise KeyError(f"unknown index: {index_id}")
    artifact_id = str(row["artifact_id"])
    return IndexSnapshot.model_validate(workspace.load_artifact_json(artifact_id)), artifact_id


def latest_index_id(workspace: Workspace) -> str:
    row = workspace.database.fetchone(
        "SELECT index_id FROM indexes ORDER BY created_at DESC LIMIT 1"
    )
    if row is None:
        raise ValueError("no index exists; run index build")
    return str(row["index_id"])


def load_track_representation(workspace: Workspace, artifact_id: str) -> TargetTrackRepresentation:
    return TargetTrackRepresentation.model_validate(workspace.load_artifact_json(artifact_id))


def write_index_manifest(workspace: Workspace, index_id: str, destination: Path) -> None:
    """Optional human-readable inspection export; durable state remains in the workspace."""

    snapshot, artifact_id = load_index(workspace, index_id)
    payload = {"artifact_id": artifact_id, "snapshot": snapshot.model_dump(mode="json")}
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
