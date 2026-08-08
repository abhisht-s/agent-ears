from __future__ import annotations

from music_locator.models import SampleTimeMap, TempoChange, TimeMapSegment
from music_locator.timebase import microseconds_to_tick, tick_to_microseconds


def make_map(source_rate: int, derived_rate: int, duration_seconds: int = 12) -> SampleTimeMap:
    return SampleTimeMap(
        source_stream_id="source",
        derived_stream_id="derived",
        source_sample_rate=source_rate,
        derived_sample_rate=derived_rate,
        segments=(
            TimeMapSegment(
                source_start_sample=0,
                source_end_sample=source_rate * duration_seconds,
                derived_start_sample=0,
                derived_end_sample=derived_rate * duration_seconds,
            ),
        ),
    )


def test_44100_to_22050_round_trip_within_one_source_sample() -> None:
    mapping = make_map(44_100, 22_050)
    for source_sample in range(0, 44_100 * 12 + 1, 37):
        recovered = mapping.derived_to_source(mapping.source_to_derived(source_sample))
        assert abs(recovered - source_sample) <= 1


def test_48000_to_22050_round_trip_within_one_source_sample() -> None:
    mapping = make_map(48_000, 22_050)
    for source_sample in range(0, 48_000 * 12 + 1, 43):
        recovered = mapping.derived_to_source(mapping.source_to_derived(source_sample))
        assert abs(recovered - source_sample) <= 1


def test_piecewise_time_map_round_trip_within_one_sample() -> None:
    mapping = SampleTimeMap(
        source_stream_id="source",
        derived_stream_id="derived",
        source_sample_rate=48_000,
        derived_sample_rate=22_050,
        segments=(
            TimeMapSegment(
                source_start_sample=0,
                source_end_sample=240_000,
                derived_start_sample=0,
                derived_end_sample=110_250,
            ),
            TimeMapSegment(
                source_start_sample=240_000,
                source_end_sample=480_000,
                derived_start_sample=110_250,
                derived_end_sample=220_500,
            ),
        ),
    )
    for source_sample in range(0, 480_001, 47):
        recovered = mapping.derived_to_source(mapping.source_to_derived(source_sample))
        assert abs(recovered - source_sample) <= 1


def test_midi_tempo_map_round_trip() -> None:
    changes = (
        TempoChange(tick=0, microseconds_per_beat=500_000),
        TempoChange(tick=1_920, microseconds_per_beat=600_000),
    )
    for tick in range(0, 4_800, 17):
        elapsed_us = tick_to_microseconds(tick, 480, changes)
        assert abs(microseconds_to_tick(elapsed_us, 480, changes) - tick) <= 1
