from __future__ import annotations

from pathlib import Path

import mido

from music_locator.storage import Workspace
from music_locator.targets import import_target, load_target


def test_midi_import_preserves_raw_events_tempo_and_ticks(tmp_path: Path) -> None:
    midi_path = tmp_path / "melody.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=600_000, time=0))
    track.append(mido.Message("note_on", note=64, velocity=90, time=120))
    track.append(mido.Message("note_off", note=64, velocity=0, time=240))
    midi.save(midi_path)

    workspace = Workspace(tmp_path / "workspace")
    target_id = import_target(workspace, midi_path)
    target, artifact_id = load_target(workspace, target_id)

    assert artifact_id.startswith("artifact_")
    assert target.ticks_per_beat == 480
    assert target.tempo_map[0].microseconds_per_beat == 600_000
    assert target.notes[0].start_tick == 120
    assert target.notes[0].end_tick == 360
    assert any(event.event_type == "set_tempo" for event in target.raw_events)
    assert all(event.message_hex for event in target.raw_events)
