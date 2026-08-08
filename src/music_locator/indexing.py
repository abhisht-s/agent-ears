"""Deterministic transparent shingle indexes for small local corpora."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path

from music_locator.models import (
    AudioTargetArtifact,
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


def _index_representation(
    *,
    representation: TargetTrackRepresentation,
    representation_artifact_id: str,
    entries: list[TargetIndexEntry],
    inverted: dict[str, list[str]],
    shingle_size: int,
) -> None:
    """Add one already-persisted matcher-facing view to an index.

    ``view_id`` is the stable selector for real-audio hypotheses.  MIDI keeps
    its existing voice identifier, so old snapshots and posting semantics stay
    valid.
    """

    hypothesis = representation.voice_hypothesis
    selector = representation.view_id or (hypothesis.voice_id if hypothesis is not None else None)
    if selector is None:
        raise ValueError("target representation has neither a view nor voice identifier")
    if "|" in selector:
        raise ValueError("target view/voice identifiers may not contain '|'")
    entries.append(
        TargetIndexEntry(
            target_id=representation.target_id,
            track_index=representation.track_index,
            event_count=len(representation.event_ids),
            representation_artifact_id=representation_artifact_id,
            voice_id=selector,
            view_id=representation.view_id,
            source_channel=(hypothesis.source_channel if hypothesis is not None else None),
        )
    )
    keys = shingle_keys(
        [float(value) for value in representation.interval_cents],
        list(representation.contour_steps),
        [value / 1_000.0 for value in representation.log_ioi_milli],
        shingle_size,
    )
    keys.extend(phrase_shingle_keys([float(value) for value in representation.pitch_cents]))
    for position, key in keys:
        inverted[key].append(f"{representation.target_id}|{selector}|{position}")


def build_index(
    workspace: Workspace,
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    target_ids: Collection[str] | None = None,
) -> str:
    workspace.initialize()
    index_id = new_id("index")
    entries: list[TargetIndexEntry] = []
    inverted: dict[str, list[str]] = defaultdict(list)
    target_fingerprints: list[str] = []
    direct_count = 0
    audio_view_count = 0
    requested_target_ids = set(target_ids) if target_ids is not None else None
    targets = [
        item
        for item in list_targets(workspace)
        if requested_target_ids is None or item[0].target_id in requested_target_ids
    ]
    if requested_target_ids is not None:
        found_target_ids = {target.target_id for target, _ in targets}
        missing_target_ids = sorted(requested_target_ids - found_target_ids)
        if missing_target_ids:
            raise KeyError(f"unknown target(s): {', '.join(missing_target_ids)}")
    for target, target_artifact_id in targets:
        target_fingerprints.append(
            f"{target.target_id}:{target_artifact_id}:{target.source_sha256}"
        )
        if isinstance(target, AudioTargetArtifact):
            # Target analysis is deliberately separate from indexing.  Indexing
            # consumes only a completed, durable evidence bundle and never
            # invokes an extractor implicitly.
            from music_locator.target_audio import load_latest_target_bundle

            bundle, bundle_artifact_id, target_analysis_id = load_latest_target_bundle(
                workspace, target.target_id
            )
            target_fingerprints.append(
                ":".join(
                    (
                        target.target_id,
                        bundle_artifact_id,
                        target_analysis_id,
                        bundle.cache_key,
                    )
                )
            )
            seen_view_ids: set[str] = set()
            for representation_artifact_id in bundle.representation_artifact_ids:
                representation = load_track_representation(workspace, representation_artifact_id)
                if representation.target_id != target.target_id:
                    raise ValueError(
                        "audio representation target does not match its analysis bundle"
                    )
                if representation.view_id is None:
                    raise ValueError("audio target representations require a stable view_id")
                if representation.view_id in seen_view_ids:
                    raise ValueError(f"duplicate audio target view_id: {representation.view_id}")
                seen_view_ids.add(representation.view_id)
                _index_representation(
                    representation=representation,
                    representation_artifact_id=representation_artifact_id,
                    entries=entries,
                    inverted=inverted,
                    shingle_size=shingle_size,
                )
                audio_view_count += 1
            continue
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
            _index_representation(
                representation=representation,
                representation_artifact_id=representation_artifact_id,
                entries=entries,
                inverted=inverted,
                shingle_size=shingle_size,
            )
    sorted_entries = tuple(
        sorted(
            entries,
            key=lambda item: (
                item.target_id,
                -1 if item.track_index is None else item.track_index,
                item.view_id or item.voice_id,
            ),
        )
    )
    snapshot = IndexSnapshot(
        schema_version="0.0.2" if audio_view_count else "0.0.1",
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
            "derived_voice_count": len(sorted_entries) - direct_count - audio_view_count,
            "audio_view_count": audio_view_count,
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
