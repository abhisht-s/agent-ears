"""Audition rendering for authoritative timed-note passages."""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from music_locator.indexing import load_track_representation
from music_locator.models import AssetKind
from music_locator.search import load_search_result
from music_locator.storage import Workspace
from music_locator.targets import load_target
from music_locator.timebase import tick_to_microseconds

RENDER_SAMPLE_RATE = 44_100


def render_match(workspace: Workspace, search_id: str, destination: Path) -> str:
    result, _ = load_search_result(workspace, search_id)
    if (
        result.selected_target_id is None
        or result.selected_track_index is None
        or result.target_range is None
    ):
        raise ValueError("search has no selected target passage to render")
    target, _ = load_target(workspace, result.selected_target_id)
    selected_representation = (
        load_track_representation(
            workspace, result.provenance.target_representation_artifact_id
        )
        if result.provenance is not None
        and result.provenance.target_representation_artifact_id is not None
        else None
    )
    if (
        selected_representation is not None
        and selected_representation.voice_hypothesis is not None
    ):
        source_notes = {note.event_id: note for note in target.notes}
        selected_notes = [
            source_notes[item.source_event_id].model_copy(
                update={
                    "start_tick": item.indexed_start_tick,
                    "end_tick": item.indexed_end_tick,
                }
            )
            for item in selected_representation.voice_hypothesis.note_provenance
            if item.indexed_start_tick < result.target_range.end_tick
            and item.indexed_end_tick > result.target_range.start_tick
        ]
    else:
        selected_notes = [
            note
            for note in target.notes
            if note.track_index == result.selected_track_index
            and note.start_tick < result.target_range.end_tick
            and note.end_tick > result.target_range.start_tick
        ]
    if not selected_notes:
        raise ValueError("selected target passage contains no notes")
    origin_us = tick_to_microseconds(
        result.target_range.start_tick, target.ticks_per_beat, target.tempo_map
    )
    end_us = tick_to_microseconds(
        result.target_range.end_tick, target.ticks_per_beat, target.tempo_map
    )
    frame_count = max(1, math.ceil((end_us - origin_us) * RENDER_SAMPLE_RATE / 1_000_000))
    audio = np.zeros(frame_count, dtype=np.float32)
    for note in selected_notes:
        note_start_us = tick_to_microseconds(
            max(note.start_tick, result.target_range.start_tick),
            target.ticks_per_beat,
            target.tempo_map,
        )
        note_end_us = tick_to_microseconds(
            min(note.end_tick, result.target_range.end_tick),
            target.ticks_per_beat,
            target.tempo_map,
        )
        start = max(0, round((note_start_us - origin_us) * RENDER_SAMPLE_RATE / 1_000_000))
        end = min(frame_count, round((note_end_us - origin_us) * RENDER_SAMPLE_RATE / 1_000_000))
        if end <= start:
            continue
        time = np.arange(end - start, dtype=np.float64) / RENDER_SAMPLE_RATE
        frequency = 440.0 * 2.0 ** ((note.pitch_midi - 69) / 12.0)
        envelope = np.ones(end - start, dtype=np.float64)
        attack = min(len(envelope), round(0.012 * RENDER_SAMPLE_RATE))
        release = min(len(envelope), round(0.035 * RENDER_SAMPLE_RATE))
        if attack:
            envelope[:attack] *= np.linspace(0.0, 1.0, attack)
        if release:
            envelope[-release:] *= np.linspace(1.0, 0.0, release)
        signal = np.sin(2.0 * np.pi * frequency * time) + 0.2 * np.sin(
            4.0 * np.pi * frequency * time
        )
        audio[start:end] += (0.18 * note.velocity / 127.0 * envelope * signal).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.98:
        audio *= 0.98 / peak
    memory = io.BytesIO()
    sf.write(memory, audio, RENDER_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    data = memory.getvalue()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    asset = workspace.store_asset(
        data,
        kind=AssetKind.RENDER,
        media_type="audio/wav",
        original_name=destination.name,
    )
    return asset.asset_id
