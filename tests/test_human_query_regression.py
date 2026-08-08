from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from music_locator.alignment import align_candidate
from music_locator.audio import _event_hypothesis, prepare_interpretation_for_search
from music_locator.models import (
    CandidateEvidence,
    OnsetEvidence,
    PitchFrame,
    QueryEvent,
    QueryInterpretation,
    SampleTimeMap,
    TargetTrackRepresentation,
    TimeMapSegment,
)
from music_locator.representation import phrase_shingle_keys, query_representation


def _event(index: int, pitch_midi: float, quality: float = 0.8) -> QueryEvent:
    start = index * 7_200
    return QueryEvent(
        event_id=f"query-event-{index}",
        start_sample=start,
        end_sample=start + 5_400,
        source_start_sample=start * 2,
        source_end_sample=(start + 5_400) * 2,
        pitch_midi_cents=pitch_midi * 100.0,
        pitch_deviation_cents=12.0 if quality >= 0.1 else 8.0,
        voicing=quality,
        onset_strength=0.4 if quality >= 0.1 else 0.0,
        quality_weight=quality,
    )


def _representation(pitches: tuple[int, ...], false_start: int) -> TargetTrackRepresentation:
    count = len(pitches)
    onsets = tuple(index * 300_000 for index in range(count))
    intervals = tuple((right - left) * 100 for left, right in pairwise(pitches))
    return TargetTrackRepresentation(
        target_id="polyphonic-regression-target",
        track_index=1,
        event_ids=tuple(f"target-event-{index}" for index in range(count)),
        pitch_cents=tuple(pitch * 100 for pitch in pitches),
        relative_pitch_cents=tuple(float((pitch - pitches[0]) * 100) for pitch in pitches),
        interval_cents=intervals,
        contour_steps=tuple(1 if value > 35 else -1 if value < -35 else 0 for value in intervals),
        onset_microseconds=onsets,
        log_ioi_milli=tuple(0 for _ in range(count - 1)),
        start_ticks=tuple(index * 288 for index in range(count)),
        end_ticks=tuple(index * 288 + 240 for index in range(count)),
        velocities=tuple(96 if index < false_start else 82 for index in range(count)),
    )


def test_noisy_floor_events_do_not_make_octave_leap_false_phrase_win() -> None:
    true_target = (
        69,
        69,
        70,
        69,
        67,
        67,
        67,
        67,
        69,
        69,
        69,
        69,
        70,
        69,
        67,
        65,
        64,
        62,
        62,
        64,
        65,
    )
    false_target = (60, 62, 74, 76, 77, 76, 77, 79, 77, 79, 81, 79, 77, 74, 74)
    spacer = (45,) * 10
    all_target = true_target + spacer + false_target
    false_start = len(true_target) + len(spacer)
    representation = _representation(all_target, false_start)

    clean_query = tuple(float(pitch - 11) for pitch in true_target[:18])
    raw_pitches = (42.2, 40.4, *clean_query[:7], 39.0, 41.1, *clean_query[7:])
    raw_events = tuple(
        _event(index, pitch, 0.01 if index in {0, 1, 9, 10} else 0.8)
        for index, pitch in enumerate(raw_pitches)
    )
    raw = QueryInterpretation(
        interpretation_id="raw-noisy-interpretation",
        query_id="regression-query",
        rank=1,
        selected=True,
        method="realish_noisy_fixture",
        method_parameters={},
        events=raw_events,
        representation=query_representation(raw_events),
        interpretation_score=0.2,
    )

    prepared = prepare_interpretation_for_search(raw)

    assert len(prepared.events) == 18
    assert {item.source_event_id for item in prepared.event_usability} == {
        event.event_id for event in raw.events
    }
    assert sum(not item.usable for item in prepared.event_usability) == 4
    assert max(abs(value) for value in raw.representation.interval_cents) > 1_000
    assert max(abs(value) for value in prepared.representation.interval_cents) <= 300

    true = align_candidate(
        prepared,
        representation,
        CandidateEvidence(
            target_id=representation.target_id,
            track_index=1,
            start_event_index=0,
            shingle_votes=8,
            matched_shingles=("i:-2,-1", "ic:-2,-1:-1,-1"),
        ),
    )
    false = align_candidate(
        prepared,
        representation,
        CandidateEvidence(
            target_id=representation.target_id,
            track_index=1,
            start_event_index=false_start,
            shingle_votes=12,
            matched_shingles=("c:-1,1", "r:1,-1"),
        ),
    )

    assert true.end_index <= false_start
    assert false.start_index >= false_start - 3
    assert true.score.value > false.score.value
    assert true.score.components["interval_transition_cost"] < false.score.components[
        "interval_transition_cost"
    ]
    assert false.transformations.query_insertions + false.transformations.target_insertions >= 1


def test_repeated_attack_boundary_is_not_merged_and_phrase_view_keeps_mapping() -> None:
    length = 12_288
    frames = tuple(
        PitchFrame(
            frame_index=index,
            center_sample=index * 256,
            f0_hz=261.63,
            midi_cents=6_000.0,
            voiced_probability=0.9,
            voiced=True,
            quality_weight=0.9,
        )
        for index in range(length // 256)
    )
    audio = np.asarray(
        [0.1 * math.sin(2.0 * math.pi * 261.63 * sample / 22_050) for sample in range(length)],
        dtype=np.float32,
    )
    mapping = SampleTimeMap(
        source_stream_id="source",
        derived_stream_id="derived",
        source_sample_rate=22_050,
        derived_sample_rate=22_050,
        segments=(
            TimeMapSegment(
                source_start_sample=0,
                source_end_sample=length,
                derived_start_sample=0,
                derived_end_sample=length,
            ),
        ),
    )
    interpretation = _event_hypothesis(
        "repeat-query",
        frames,
        (OnsetEvidence(sample=6_144, strength=0.20),),
        mapping,
        length,
        audio,
        float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        rank=1,
        pitch_jump_cents=75.0,
        minimum_event_samples=1_000,
    )

    assert len(interpretation.events) == 2
    assert interpretation.events[1].start_sample == 6_144
    keys = phrase_shingle_keys([6_000.0, 6_000.0, 6_200.0, 6_000.0, 5_800.0])
    assert keys
    assert all(position in {0, 2} for position, _ in keys)
    assert any(key.startswith("msi2:") for _, key in keys)
