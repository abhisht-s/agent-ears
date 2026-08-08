"""Synchronized transparent melodic representations."""

from __future__ import annotations

import math
from itertools import pairwise
from statistics import median
from typing import TypedDict

from music_locator.models import QueryEvent, QueryRepresentation, TargetNoteEvent, TempoChange
from music_locator.timebase import tick_to_microseconds


def contour_step(interval_cents: float, deadband_cents: float = 35.0) -> int:
    if interval_cents > deadband_cents:
        return 1
    if interval_cents < -deadband_cents:
        return -1
    return 0


def query_representation(events: tuple[QueryEvent, ...]) -> QueryRepresentation:
    if not events:
        return QueryRepresentation(
            relative_pitch_cents=(),
            interval_cents=(),
            contour_steps=(),
            onset_samples=(),
            log_ioi_ratios=(),
            quality_weights=(),
        )
    pitches = [event.pitch_midi_cents for event in events]
    pitch_center = median(pitches)
    intervals = [right - left for left, right in pairwise(pitches)]
    onsets = [event.start_sample for event in events]
    iois = [right - left for left, right in pairwise(onsets)]
    positive_iois = [ioi for ioi in iois if ioi > 0]
    median_ioi = median(positive_iois) if positive_iois else 1
    return QueryRepresentation(
        relative_pitch_cents=tuple(pitch - pitch_center for pitch in pitches),
        interval_cents=tuple(intervals),
        contour_steps=tuple(contour_step(value) for value in intervals),
        onset_samples=tuple(onsets),
        log_ioi_ratios=tuple(math.log(max(ioi, 1) / median_ioi) for ioi in iois),
        quality_weights=tuple(event.quality_weight for event in events),
    )


class TargetFeatureViews(TypedDict):
    pitch_cents: list[int]
    relative_pitch_cents: list[float]
    interval_cents: list[int]
    contour_steps: list[int]
    onset_microseconds: list[int]
    log_ioi_milli: list[int]
    start_ticks: list[int]
    end_ticks: list[int]
    velocities: list[int]


def target_feature_views(
    notes: tuple[TargetNoteEvent, ...],
    ticks_per_beat: int,
    tempo_map: tuple[TempoChange, ...],
) -> TargetFeatureViews:
    """Create key-invariant views while retaining original tick coordinates."""

    pitches = [note.pitch_midi * 100 for note in notes]
    intervals = [right - left for left, right in pairwise(pitches)]
    contour = [contour_step(float(value)) for value in intervals]
    onset_us = [tick_to_microseconds(note.start_tick, ticks_per_beat, tempo_map) for note in notes]
    iois = [right - left for left, right in pairwise(onset_us)]
    positive_iois = [ioi for ioi in iois if ioi > 0]
    median_ioi = median(positive_iois) if positive_iois else 1
    log_ioi_milli = [round(1_000 * math.log(max(ioi, 1) / median_ioi)) for ioi in iois]
    center = median(pitches) if pitches else 0
    return {
        "pitch_cents": pitches,
        "relative_pitch_cents": [pitch - center for pitch in pitches],
        "interval_cents": intervals,
        "contour_steps": contour,
        "onset_microseconds": onset_us,
        "log_ioi_milli": log_ioi_milli,
        "start_ticks": [note.start_tick for note in notes],
        "end_ticks": [note.end_tick for note in notes],
        "velocities": [note.velocity for note in notes],
    }


def quantized_interval(value_cents: float) -> int:
    return round(value_cents / 100.0)


def quantized_rhythm(value: float) -> int:
    return round(value * 2.0)


def shingle_keys(
    intervals_cents: list[float], contour: list[int], log_ioi: list[float], size: int = 3
) -> list[tuple[int, str]]:
    """Return position-key pairs combining interval, contour, and rhythm evidence."""

    if size < 2:
        raise ValueError("shingle size must be at least 2")
    count = min(len(intervals_cents), len(contour))
    output: list[tuple[int, str]] = []
    width = size - 1
    for start in range(max(0, count - width + 1)):
        interval_values = tuple(
            quantized_interval(value) for value in intervals_cents[start : start + width]
        )
        contour_values = tuple(contour[start : start + width])
        rhythm_slice = log_ioi[start : start + width]
        rhythm_values = tuple(quantized_rhythm(value) for value in rhythm_slice)
        output.extend(
            (
                (start, "i:" + ",".join(map(str, interval_values))),
                (start, "c:" + ",".join(map(str, contour_values))),
                (start, "r:" + ",".join(map(str, rhythm_values))),
                (
                    start,
                    "ic:"
                    + ",".join(map(str, interval_values))
                    + ":"
                    + ",".join(map(str, contour_values)),
                ),
            )
        )
    return output


def phrase_shingle_keys(
    pitches_cents: list[float],
    *,
    interval_widths: tuple[int, ...] = (2, 3, 4),
    repeated_pitch_tolerance_cents: float = 70.0,
) -> list[tuple[int, str]]:
    """Return multi-scale interval keys from a retrieval-only repeated-note view.

    Adjacent near-equal pitches are collapsed only for candidate retrieval.  Each
    posting retains the first original event position in its run; alignment and
    provenance continue to use every original event.
    """

    if not pitches_cents:
        return []
    compressed_pitches = [pitches_cents[0]]
    original_positions = [0]
    for position, pitch in enumerate(pitches_cents[1:], start=1):
        if abs(pitch - compressed_pitches[-1]) <= repeated_pitch_tolerance_cents:
            continue
        compressed_pitches.append(pitch)
        original_positions.append(position)
    compressed_intervals = [
        right - left for left, right in pairwise(compressed_pitches)
    ]
    compressed_contour = [contour_step(value) for value in compressed_intervals]
    output: list[tuple[int, str]] = []
    for width in interval_widths:
        if width < 2:
            raise ValueError("phrase interval widths must be at least 2")
        for start in range(max(0, len(compressed_intervals) - width + 1)):
            intervals = compressed_intervals[start : start + width]
            interval_bins = tuple(quantized_interval(value) for value in intervals)
            coarse_bins = tuple(round(value / 200.0) for value in intervals)
            contour = tuple(compressed_contour[start : start + width])
            original_start = original_positions[start]
            output.extend(
                (
                    (
                        original_start,
                        f"msi{width}:" + ",".join(map(str, interval_bins)),
                    ),
                    (
                        original_start,
                        f"msic{width}:"
                        + ",".join(map(str, coarse_bins))
                        + ":"
                        + ",".join(map(str, contour)),
                    ),
                    (
                        original_start,
                        f"msc{width}:" + ",".join(map(str, contour)),
                    ),
                )
            )
    return output
