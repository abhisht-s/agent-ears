"""Authoritative MIDI and timed-note target ingestion."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import mido

from music_locator.models import (
    AssetKind,
    RawMidiEvent,
    TargetArtifact,
    TargetKind,
    TargetNoteEvent,
    TempoChange,
    TimedNoteTargetInput,
)
from music_locator.storage import Workspace, new_id, utc_now
from music_locator.timebase import normalize_tempo_map


def _json_safe_fields(message: Any) -> dict[str, int | str | bool]:
    fields: dict[str, int | str | bool] = {}
    for key, value in message.dict().items():
        if key == "time":
            continue
        if isinstance(value, bool | int | str):
            fields[key] = value
        elif isinstance(value, tuple | list):
            fields[key] = ",".join(str(item) for item in value)
        else:
            fields[key] = str(value)
    return fields


def _message_hex(message: Any) -> str:
    if isinstance(message, mido.MetaMessage):
        return repr(message).encode("utf-8").hex()
    try:
        return bytes(message.bytes()).hex()
    except (TypeError, ValueError):
        return repr(message).encode("utf-8").hex()


def parse_midi(path: Path, target_id: str, asset_id: str, sha256: str) -> TargetArtifact:
    midi = mido.MidiFile(path)
    raw_events: list[RawMidiEvent] = []
    notes: list[TargetNoteEvent] = []
    tempo_changes: list[TempoChange] = []
    active: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message in track:
            delta_tick = int(message.time)
            absolute_tick += delta_tick
            raw_event_index = len(raw_events)
            raw_events.append(
                RawMidiEvent(
                    track_index=track_index,
                    absolute_tick=absolute_tick,
                    delta_tick=delta_tick,
                    event_type=message.type,
                    message_hex=_message_hex(message),
                    fields=_json_safe_fields(message),
                )
            )
            if message.type == "set_tempo":
                tempo_changes.append(
                    TempoChange(tick=absolute_tick, microseconds_per_beat=int(message.tempo))
                )
            if message.type == "note_on" and int(message.velocity) > 0:
                key = (track_index, int(message.channel), int(message.note))
                active[key].append((absolute_tick, int(message.velocity), raw_event_index))
            elif message.type == "note_off" or (
                message.type == "note_on" and int(message.velocity) == 0
            ):
                key = (track_index, int(message.channel), int(message.note))
                if active[key]:
                    start_tick, velocity, note_on_raw_event_index = active[key].pop(0)
                    if absolute_tick > start_tick:
                        notes.append(
                            TargetNoteEvent(
                                event_id=new_id("tevent"),
                                track_index=track_index,
                                channel=int(message.channel),
                                pitch_midi=int(message.note),
                                velocity=velocity,
                                start_tick=start_tick,
                                end_tick=absolute_tick,
                                note_on_raw_event_index=note_on_raw_event_index,
                                note_off_raw_event_index=raw_event_index,
                            )
                        )
    if not notes:
        raise ValueError("MIDI target contains no complete note events")
    notes.sort(key=lambda note: (note.track_index, note.start_tick, note.pitch_midi, note.end_tick))
    return TargetArtifact(
        target_id=target_id,
        name=path.stem,
        kind=TargetKind.MIDI,
        asset_id=asset_id,
        ticks_per_beat=midi.ticks_per_beat,
        tempo_map=normalize_tempo_map(tuple(tempo_changes)),
        raw_events=tuple(raw_events),
        notes=tuple(notes),
        source_sha256=sha256,
    )


def import_target(workspace: Workspace, path: Path, *, name: str | None = None) -> str:
    workspace.initialize()
    suffix = path.suffix.lower()
    target_id = new_id("target")
    if suffix in {".mid", ".midi"}:
        asset = workspace.store_asset(
            path,
            kind=AssetKind.TARGET_MIDI,
            media_type="audio/midi",
            original_name=path.name,
        )
        target = parse_midi(path, target_id, asset.asset_id, asset.sha256)
    elif suffix == ".json":
        asset = workspace.store_asset(
            path,
            kind=AssetKind.TARGET_NOTES,
            media_type="application/json",
            original_name=path.name,
        )
        payload = TimedNoteTargetInput.model_validate(json.loads(path.read_text(encoding="utf-8")))
        target = TargetArtifact(
            target_id=target_id,
            name=payload.name,
            kind=TargetKind.TIMED_NOTES,
            asset_id=asset.asset_id,
            ticks_per_beat=payload.ticks_per_beat,
            tempo_map=normalize_tempo_map(payload.tempo_map),
            raw_events=(),
            notes=tuple(
                sorted(payload.notes, key=lambda note: (note.track_index, note.start_tick))
            ),
            source_sha256=asset.sha256,
        )
    else:
        raise ValueError("supported target formats are .mid, .midi, and timed-note .json")
    if name is not None:
        target = target.model_copy(update={"name": name})
    artifact_id = workspace.store_json_artifact(target, artifact_type="target.timed_notes")
    workspace.database.execute(
        """
        INSERT INTO targets(target_id, name, kind, source_asset_id, artifact_id,
                            ticks_per_beat, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            target.target_id,
            target.name,
            target.kind.value,
            target.asset_id,
            artifact_id,
            target.ticks_per_beat,
            utc_now(),
        ),
    )
    return target.target_id


def load_target(workspace: Workspace, target_id: str) -> tuple[TargetArtifact, str]:
    row = workspace.database.fetchone(
        "SELECT artifact_id FROM targets WHERE target_id = ?", (target_id,)
    )
    if row is None:
        raise KeyError(f"unknown target: {target_id}")
    artifact_id = str(row["artifact_id"])
    return TargetArtifact.model_validate(workspace.load_artifact_json(artifact_id)), artifact_id


def list_targets(workspace: Workspace) -> list[tuple[TargetArtifact, str]]:
    rows = workspace.database.fetchall(
        "SELECT target_id FROM targets ORDER BY created_at, target_id"
    )
    return [load_target(workspace, str(row["target_id"])) for row in rows]
