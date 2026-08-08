from __future__ import annotations

from music_locator.alignment import align_candidate
from music_locator.models import (
    CandidateEvidence,
    QueryEvent,
    QueryInterpretation,
    TargetTrackRepresentation,
)
from music_locator.representation import query_representation


def _query_event(index: int, pitch_cents: float, onset: int) -> QueryEvent:
    return QueryEvent(
        event_id=f"q-{index}",
        start_sample=onset,
        end_sample=onset + 8_000,
        source_start_sample=onset * 2,
        source_end_sample=(onset + 8_000) * 2,
        pitch_midi_cents=pitch_cents,
        pitch_deviation_cents=12.0,
        voicing=0.9,
        onset_strength=1.0,
        quality_weight=0.9,
    )


def test_open_target_dp_handles_transposition_tempo_edits_split_and_octave() -> None:
    target_pitches = (6_000, 6_200, 6_500, 6_400, 6_900, 6_700, 7_100, 6_600, 6_800, 6_300)
    events = [
        _query_event(index, pitch + 505.0, round(index * 11_025 * 1.18))
        for index, pitch in enumerate(target_pitches)
    ]
    events[3] = events[3].model_copy(
        update={"pitch_midi_cents": events[3].pitch_midi_cents + 1_200}
    )
    events.insert(
        6,
        _query_event(
            100, events[5].pitch_midi_cents, (events[5].start_sample + events[6].start_sample) // 2
        ),
    )
    event_tuple = tuple(events)
    interpretation = QueryInterpretation(
        interpretation_id="interpretation-test",
        query_id="query-test",
        rank=1,
        selected=True,
        method="fixture",
        method_parameters={},
        events=event_tuple,
        representation=query_representation(event_tuple),
        interpretation_score=0.9,
    )
    prefix = (5_500, 5_700, 5_600, 5_900)
    all_pitches = prefix + target_pitches + (7_300, 6_100)
    representation = TargetTrackRepresentation(
        target_id="target-test",
        track_index=0,
        event_ids=tuple(f"t-{index}" for index in range(len(all_pitches))),
        pitch_cents=all_pitches,
        relative_pitch_cents=tuple(float(pitch - 6_400) for pitch in all_pitches),
        interval_cents=tuple(
            all_pitches[index + 1] - all_pitches[index] for index in range(len(all_pitches) - 1)
        ),
        contour_steps=tuple(
            1 if all_pitches[index + 1] > all_pitches[index] else -1
            for index in range(len(all_pitches) - 1)
        ),
        onset_microseconds=tuple(index * 500_000 for index in range(len(all_pitches))),
        log_ioi_milli=tuple(0 for _ in range(len(all_pitches) - 1)),
        start_ticks=tuple(index * 480 for index in range(len(all_pitches))),
        end_ticks=tuple(index * 480 + 430 for index in range(len(all_pitches))),
        velocities=tuple(90 for _ in all_pitches),
    )
    candidate = CandidateEvidence(
        target_id="target-test",
        track_index=0,
        start_event_index=4,
        shingle_votes=8,
        matched_shingles=("fixture",),
    )
    result = align_candidate(interpretation, representation, candidate)
    assert abs(result.start_index - 4) <= 1
    assert result.score.value > 60
    assert abs(result.transformations.transposition_cents - 505) < 80
    assert result.transformations.octave_repairs >= 1
    assert (
        result.transformations.query_insertions
        + result.transformations.splits
        + result.transformations.merges
        >= 1
    )
