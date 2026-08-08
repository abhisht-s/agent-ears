from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from random import Random

import mido
import pytest
import soundfile as sf

from music_locator.audio import analyze_query, import_query
from music_locator.benchmark import SynthEvent, _synthesize
from music_locator.evidence import verify_search_evidence
from music_locator.indexing import build_index, load_index, load_track_representation
from music_locator.models import ResultStatus
from music_locator.search import search
from music_locator.storage import Workspace
from music_locator.targets import import_target, load_target
from music_locator.voice_extraction import extract_monophonic_voices

TICKS_PER_BEAT = 480
MELODY = (60, 62, 65, 64, 69, 67, 71, 66, 68, 63, 70, 65)
NoteSpec = tuple[int, int, int, int, int]


def _write_track(
    midi: mido.MidiFile,
    name: str,
    notes: Sequence[NoteSpec],
    *,
    program: int = 0,
) -> None:
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    channels = sorted({note[4] for note in notes})
    for channel in channels:
        if channel != 9:
            track.append(mido.Message("program_change", channel=channel, program=program, time=0))
    scheduled: list[tuple[int, int, mido.Message]] = []
    for start, end, pitch, velocity, channel in notes:
        scheduled.append(
            (start, 1, mido.Message("note_on", note=pitch, velocity=velocity, channel=channel))
        )
        scheduled.append(
            (end, 0, mido.Message("note_off", note=pitch, velocity=0, channel=channel))
        )
    previous_tick = 0
    for tick, _, message in sorted(scheduled, key=lambda item: (item[0], item[1])):
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick


def _write_midi(path: Path, tracks: Sequence[tuple[str, Sequence[NoteSpec]]]) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)
    tempo_track = mido.MidiTrack()
    midi.tracks.append(tempo_track)
    tempo_track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    for name, notes in tracks:
        _write_track(midi, name, notes)
    midi.save(path)


def _melody_notes(
    pitches: Sequence[int] = MELODY, *, channel: int = 0, octave_shift: int = 0
) -> list[NoteSpec]:
    return [
        (
            index * TICKS_PER_BEAT,
            index * TICKS_PER_BEAT + 420,
            pitch + octave_shift,
            102,
            channel,
        )
        for index, pitch in enumerate(pitches)
    ]


def _chord_notes(*, channel: int = 0, sustained: bool = False) -> list[NoteSpec]:
    output: list[NoteSpec] = []
    chord_roots = (36, 41, 43, 38)
    if sustained:
        for pitch in (36, 40, 43):
            output.append((0, len(MELODY) * TICKS_PER_BEAT, pitch, 55, channel))
        return output
    for index in range(len(MELODY)):
        root = chord_roots[(index // 2) % len(chord_roots)]
        for offset in (0, 4, 7):
            output.append(
                (
                    index * TICKS_PER_BEAT,
                    index * TICKS_PER_BEAT + 450,
                    root + offset,
                    58,
                    channel,
                )
            )
    return output


@pytest.fixture
def polyphonic_midis(tmp_path: Path) -> dict[str, Path]:
    paths = {name: tmp_path / f"{name}.mid" for name in (
        "melody_chords",
        "melody_bass_chords",
        "repeated_melody",
        "sustained_accompaniment",
        "octave_doubled",
        "drums_and_pitched",
        "ambiguous_lines",
    )}
    _write_midi(
        paths["melody_chords"],
        (("Melody and chords", (*_melody_notes(), *_chord_notes())),),
    )
    bass = [
        (index * 960, index * 960 + 900, 31 + index % 4, 72, 1)
        for index in range(len(MELODY) // 2)
    ]
    _write_midi(
        paths["melody_bass_chords"],
        (
            ("Lead", _melody_notes()),
            ("Bass", bass),
            ("Chords", _chord_notes(channel=2)),
        ),
    )
    repeated = (60, 60, 62, 62, 65, 65, 64, 64, 67, 67, 69, 69)
    _write_midi(
        paths["repeated_melody"],
        (("Repeated lead", (*_melody_notes(repeated), *_chord_notes())),),
    )
    _write_midi(
        paths["sustained_accompaniment"],
        (("Lead over pad", (*_melody_notes(), *_chord_notes(sustained=True))),),
    )
    _write_midi(
        paths["octave_doubled"],
        (("Octave lead", (*_melody_notes(), *_melody_notes(octave_shift=-12))),),
    )
    drums = [
        (index * 240, index * 240 + 80, 36 if index % 2 == 0 else 42, 95, 9)
        for index in range(24)
    ]
    _write_midi(
        paths["drums_and_pitched"],
        (
            ("Drums", drums),
            ("Pitched", (*_melody_notes(), *_chord_notes())),
        ),
    )
    _write_midi(
        paths["ambiguous_lines"],
        (
            ("Line A", _melody_notes()),
            ("Line B", _melody_notes(channel=1, octave_shift=12)),
        ),
    )
    return paths


def _assert_voice_provenance(workspace: Workspace, target_id: str) -> None:
    target, _ = load_target(workspace, target_id)
    source_by_id = {note.event_id: note for note in target.notes}
    voices = extract_monophonic_voices(target)
    assert voices
    for voice in voices:
        notes = voice.notes
        provenance = voice.hypothesis.note_provenance
        assert len(notes) == len(provenance)
        assert all(right.start_tick >= left.end_tick for left, right in pairwise(notes))
        for index, (note, evidence) in enumerate(zip(notes, provenance, strict=True)):
            source = source_by_id[evidence.source_event_id]
            assert evidence.candidate_event_index == index
            assert note.event_id == source.event_id
            assert evidence.indexed_start_tick == note.start_tick == source.start_tick
            assert evidence.indexed_end_tick == note.end_tick <= source.end_tick
            assert evidence.source_note_on_raw_event_index == source.note_on_raw_event_index
            assert evidence.source_note_off_raw_event_index == source.note_off_raw_event_index
            assert source.note_on_raw_event_index is not None
            assert source.note_off_raw_event_index is not None
            assert target.raw_events[source.note_on_raw_event_index].event_type == "note_on"


def test_polyphonic_fixture_strategies_and_complete_provenance(
    tmp_path: Path, polyphonic_midis: dict[str, Path]
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    target_ids = {
        name: import_target(workspace, path) for name, path in polyphonic_midis.items()
    }
    for target_id in target_ids.values():
        _assert_voice_provenance(workspace, target_id)

    melody_target, _ = load_target(workspace, target_ids["melody_chords"])
    melody_voices = extract_monophonic_voices(melody_target)
    supported_methods = {
        method for voice in melody_voices for method in voice.hypothesis.supporting_methods
    }
    assert {
        "highest_at_onset_v1",
        "lowest_at_onset_v1",
        "continuity_path_v1",
        "melodic_salience_path_v1",
    } <= supported_methods

    repeated_target, _ = load_target(workspace, target_ids["repeated_melody"])
    assert any(
        any(left.pitch_midi == right.pitch_midi for left, right in pairwise(voice.notes))
        for voice in extract_monophonic_voices(repeated_target)
    )
    sustained_target, _ = load_target(workspace, target_ids["sustained_accompaniment"])
    assert any(
        item.simultaneous_source_event_ids
        for voice in extract_monophonic_voices(sustained_target)
        for item in voice.hypothesis.note_provenance
    )
    doubled_target, _ = load_target(workspace, target_ids["octave_doubled"])
    doubled_pitch_paths = {
        tuple(note.pitch_midi for note in voice.notes)
        for voice in extract_monophonic_voices(doubled_target)
    }
    assert tuple(MELODY) in doubled_pitch_paths
    assert tuple(pitch - 12 for pitch in MELODY) in doubled_pitch_paths
    drums_target, _ = load_target(workspace, target_ids["drums_and_pitched"])
    assert all(
        voice.hypothesis.source_channel != 9
        for voice in extract_monophonic_voices(drums_target)
    )

    preserved_target, preserved_artifact_id = load_target(
        workspace, target_ids["melody_chords"]
    )
    assert workspace.asset_path(preserved_target.asset_id).read_bytes() == polyphonic_midis[
        "melody_chords"
    ].read_bytes()
    assert any(event.event_type == "program_change" for event in preserved_target.raw_events)
    index_id = build_index(workspace)
    target_after_indexing, artifact_id_after_indexing = load_target(
        workspace, target_ids["melody_chords"]
    )
    assert artifact_id_after_indexing == preserved_artifact_id
    assert target_after_indexing == preserved_target
    snapshot, _ = load_index(workspace, index_id)
    assert snapshot.parameters["derived_voice_count"] > 0
    for entry in snapshot.target_entries:
        representation = load_track_representation(
            workspace, entry.representation_artifact_id
        )
        assert representation.voice_hypothesis is not None
        assert entry.voice_id == representation.voice_hypothesis.voice_id
        assert len(representation.event_ids) == len(
            representation.voice_hypothesis.note_provenance
        )


def _write_hum_like_query(path: Path, pitches: Sequence[int], seed: str) -> None:
    events = [SynthEvent(float(pitch + 5), 0.58) for pitch in pitches]
    audio = _synthesize(events, 44_100, Random(seed))
    sf.write(path, audio, 44_100, subtype="PCM_16")


def test_hum_query_finds_melody_hidden_in_polyphonic_midi(
    tmp_path: Path, polyphonic_midis: dict[str, Path]
) -> None:
    workspace = Workspace(tmp_path / "search-workspace")
    target_id = import_target(workspace, polyphonic_midis["melody_chords"])
    index_id = build_index(workspace)
    query_path = tmp_path / "hum.wav"
    _write_hum_like_query(query_path, MELODY[:10], "polyphonic-hidden-melody")
    query_id = import_query(workspace, query_path)
    analyze_query(workspace, query_id)

    result = search(workspace, query_id, index_id)

    assert result.status == ResultStatus.ACCEPTED
    assert result.selected_target_id == target_id
    assert result.selected_voice_hypothesis is not None
    assert result.selected_voice_hypothesis.hypothesis_kind == "derived_monophonic"
    assert result.target_range is not None
    assert result.target_range.start_tick <= TICKS_PER_BEAT
    assert verify_search_evidence(workspace, result.search_id)["valid"] is True


def test_equally_plausible_melodic_lines_return_ambiguity_and_alternatives(
    tmp_path: Path, polyphonic_midis: dict[str, Path]
) -> None:
    workspace = Workspace(tmp_path / "ambiguous-workspace")
    import_target(workspace, polyphonic_midis["ambiguous_lines"])
    index_id = build_index(workspace)
    query_path = tmp_path / "ambiguous-hum.wav"
    _write_hum_like_query(query_path, MELODY[:10], "ambiguous-lines")
    query_id = import_query(workspace, query_path)
    analyze_query(workspace, query_id)

    result = search(workspace, query_id, index_id)

    assert result.status == ResultStatus.AMBIGUOUS
    assert result.alternatives
    assert result.alternatives[0].score.value == pytest.approx(result.score.value)
    assert any(
        warning.code == "close_source_voice_candidates"
        for warning in result.quality_warnings
    )


def test_percussion_only_midi_retains_typed_unsupported_abstention(tmp_path: Path) -> None:
    midi_path = tmp_path / "percussion-only.mid"
    drums = [
        (index * 240, index * 240 + 80, 36 if index % 2 == 0 else 42, 95, 9)
        for index in range(24)
    ]
    _write_midi(midi_path, (("Drums", drums),))
    workspace = Workspace(tmp_path / "percussion-workspace")
    import_target(workspace, midi_path)
    index_id = build_index(workspace)
    snapshot, _ = load_index(workspace, index_id)
    assert not snapshot.target_entries
    query_path = tmp_path / "pitched-query.wav"
    _write_hum_like_query(query_path, MELODY[:10], "percussion-only")
    query_id = import_query(workspace, query_path)
    analyze_query(workspace, query_id)

    result = search(workspace, query_id, index_id)

    assert result.status == ResultStatus.UNSUPPORTED_TARGET
    assert result.reason_codes == ("index_has_no_usable_pitched_voice_candidates",)
