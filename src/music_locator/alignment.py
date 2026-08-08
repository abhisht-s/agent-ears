"""Custom open-target melodic sequence alignment and contour refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise, product
from statistics import median

import librosa
import numpy as np

from music_locator.models import (
    AlignmentPair,
    CandidateEvidence,
    QueryEvent,
    QueryInterpretation,
    ScoreEvidence,
    TargetTrackRepresentation,
    TransformationEvidence,
)


@dataclass(frozen=True)
class AlignmentComputation:
    start_index: int
    end_index: int
    pairs: tuple[AlignmentPair, ...]
    transformations: TransformationEvidence
    score: ScoreEvidence
    dtw_cost: float
    raw_cost: float


@dataclass(frozen=True)
class _BackPointer:
    previous_i: int
    previous_j: int
    transition: str
    query_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    pitch_residual: float | None
    timing_residual_ppm: int | None
    local_cost: float
    octave_repairs: int


def _nearest_octave_residual(value_cents: float) -> tuple[float, int]:
    octave_count = round(value_cents / 1_200.0)
    residual = value_cents - octave_count * 1_200.0
    return residual, abs(octave_count)


def _pitch_cost(residual_cents: float) -> float:
    """Penalize large residuals super-linearly without imposing a hard cap."""

    scaled = abs(residual_cents) / 100.0
    if scaled <= 1.0:
        return scaled
    return min(6.0, 1.0 + 1.5 * (scaled - 1.0))


def _timing_cost(
    query_onset: int,
    query_first: int,
    query_span: int,
    target_onset_us: int,
    target_first_us: int,
    target_span_us: int,
) -> tuple[float, int]:
    query_position = (query_onset - query_first) / max(query_span, 1)
    target_position = (target_onset_us - target_first_us) / max(target_span_us, 1)
    residual = query_position - target_position
    residual_ppm = round(residual * 1_000_000)
    absolute = abs(residual)
    if absolute <= 0.12:
        return absolute / 0.12, residual_ppm
    return 1.0 + (absolute - 0.12) * 8.0, residual_ppm


def _match_cost(
    query: QueryEvent,
    target_pitch_cents: float,
    target_onset_us: int,
    transposition_cents: float,
    query_first: int,
    query_span: int,
    target_first_us: int,
    target_span_us: int,
) -> tuple[float, float, int, int]:
    raw_pitch_residual = query.pitch_midi_cents - target_pitch_cents - transposition_cents
    pitch_residual, repairs = _nearest_octave_residual(raw_pitch_residual)
    pitch_cost = _pitch_cost(pitch_residual)
    timing_cost, timing_residual = _timing_cost(
        query.start_sample,
        query_first,
        query_span,
        target_onset_us,
        target_first_us,
        target_span_us,
    )
    repair_cost = repairs * 0.8
    quality = max(0.2, query.quality_weight)
    local = quality * (0.64 * pitch_cost + 0.36 * timing_cost) + repair_cost
    return local, pitch_residual, timing_residual, repairs


def _estimate_transposition(
    query: tuple[QueryEvent, ...], target_pitches: tuple[int, ...]
) -> float:
    count = min(len(query), len(target_pitches))
    if count == 0:
        return 0.0
    differences = [
        query[round(index * (len(query) - 1) / max(count - 1, 1))].pitch_midi_cents
        - target_pitches[round(index * (len(target_pitches) - 1) / max(count - 1, 1))]
        for index in range(count)
    ]
    return float(median(differences))


def _contour_dtw(query_pitches: list[float], target_pitches: list[float]) -> float:
    if len(query_pitches) < 2 or len(target_pitches) < 2:
        return 2.0
    query_array = np.asarray(query_pitches, dtype=np.float64)
    target_array = np.asarray(target_pitches, dtype=np.float64)
    query_array -= np.median(query_array)
    target_array -= np.median(target_array)
    try:
        accumulated, path = librosa.sequence.dtw(
            X=query_array[np.newaxis, :] / 100.0,
            Y=target_array[np.newaxis, :] / 100.0,
            metric="euclidean",
            global_constraints=True,
            band_rad=0.35,
        )
    except (ParameterError, TypeError, ValueError):
        return 4.0
    if len(path) == 0:
        return 4.0
    return float(accumulated[-1, -1] / len(path))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 600.0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def _alignment_structure(
    query: tuple[QueryEvent, ...],
    representation: TargetTrackRepresentation,
    pointers: list[_BackPointer],
    transposition_cents: float,
) -> dict[str, float]:
    """Measure transposition-invariant transitions on the chosen DP path."""

    anchors: list[tuple[int, int, float, float, float]] = []
    pitch_residuals: list[float] = []
    for pointer in pointers:
        if pointer.pitch_residual is not None:
            pitch_residuals.append(abs(pointer.pitch_residual))
        if not pointer.query_indices or not pointer.target_indices:
            continue
        if pointer.transition == "split":
            target_index = pointer.target_indices[0]
            for query_index in pointer.query_indices:
                event = query[query_index]
                anchors.append(
                    (
                        query_index,
                        target_index,
                        event.pitch_midi_cents,
                        float(representation.pitch_cents[target_index]),
                        event.quality_weight,
                    )
                )
            continue
        query_index = pointer.query_indices[-1]
        event = query[query_index]
        target_index = min(
            pointer.target_indices,
            key=lambda index: abs(
                _nearest_octave_residual(
                    event.pitch_midi_cents
                    - representation.pitch_cents[index]
                    - transposition_cents
                )[0]
            ),
        )
        anchors.append(
            (
                query_index,
                target_index,
                event.pitch_midi_cents,
                float(representation.pitch_cents[target_index]),
                event.quality_weight,
            )
        )
    interval_cost_total = 0.0
    interval_weight_total = 0.0
    interval_octave_repairs = 0
    contour_agreement_total = 0.0
    rhythm_ratios: list[float] = []
    for left, right in pairwise(anchors):
        if right[0] <= left[0] or right[1] <= left[1]:
            continue
        query_interval = right[2] - left[2]
        target_interval = right[3] - left[3]
        raw_interval_residual = query_interval - target_interval
        interval_octaves = round(raw_interval_residual / 1_200.0)
        interval_residual = raw_interval_residual - interval_octaves * 1_200.0
        weight = max(0.25, min(left[4], right[4]))
        interval_cost_total += weight * min(
            6.0, abs(interval_residual) / 100.0 + 1.5 * abs(interval_octaves)
        )
        interval_weight_total += weight
        interval_octave_repairs += abs(interval_octaves)
        query_contour = 1 if query_interval > 35.0 else -1 if query_interval < -35.0 else 0
        target_contour = 1 if target_interval > 35.0 else -1 if target_interval < -35.0 else 0
        contour_agreement_total += weight * float(query_contour == target_contour)
        query_ioi = query[right[0]].start_sample - query[left[0]].start_sample
        target_ioi = (
            representation.onset_microseconds[right[1]]
            - representation.onset_microseconds[left[1]]
        )
        if query_ioi > 0 and target_ioi > 0:
            rhythm_ratios.append((query_ioi / 22_050.0) / (target_ioi / 1_000_000.0))
    interval_transition_cost = interval_cost_total / max(interval_weight_total, 1e-9)
    contour_agreement = contour_agreement_total / max(interval_weight_total, 1e-9)
    if rhythm_ratios:
        rhythm_center = float(median(rhythm_ratios))
        rhythm_transition_cost = sum(
            abs(math.log(ratio / rhythm_center)) for ratio in rhythm_ratios
        ) / len(rhythm_ratios)
    else:
        rhythm_transition_cost = 2.0
    return {
        "aligned_anchor_count": float(len(anchors)),
        "interval_transition_cost": interval_transition_cost,
        "interval_octave_repair_count": float(interval_octave_repairs),
        "contour_agreement": contour_agreement,
        "rhythm_transition_cost": min(4.0, rhythm_transition_cost),
        "pitch_residual_median_cents": float(median(pitch_residuals))
        if pitch_residuals
        else 600.0,
        "pitch_residual_p90_cents": _percentile(pitch_residuals, 0.9),
    }


try:
    from librosa.util.exceptions import ParameterError
except ImportError:  # pragma: no cover - version-pinned defensive fallback
    ParameterError = ValueError  # type: ignore[misc, assignment]


def _align_fixed_window(
    query: tuple[QueryEvent, ...],
    representation: TargetTrackRepresentation,
    start_index: int,
    end_index: int,
    candidate: CandidateEvidence,
) -> AlignmentComputation:
    target_pitches = representation.pitch_cents[start_index:end_index]
    target_onsets = representation.onset_microseconds[start_index:end_index]
    if not query or not target_pitches:
        raise ValueError("alignment requires non-empty query and target sequences")
    transposition = _estimate_transposition(query, target_pitches)
    query_first = query[0].start_sample
    query_span = max(query[-1].start_sample - query_first, query[-1].end_sample - query_first, 1)
    target_first = target_onsets[0]
    target_span = max(target_onsets[-1] - target_first, 1)
    rows = len(query) + 1
    columns = len(target_pitches) + 1
    costs = np.full((rows, columns), np.inf, dtype=np.float64)
    costs[0, 0] = 0.0
    back: dict[tuple[int, int], _BackPointer] = {}

    def update(i: int, j: int, value: float, pointer: _BackPointer) -> None:
        if value < costs[i, j]:
            costs[i, j] = value
            back[(i, j)] = pointer

    for i, j in product(range(rows), range(columns)):
        current = float(costs[i, j])
        if not math.isfinite(current):
            continue
        if i < len(query) and j < len(target_pitches):
            local, pitch_residual, timing_residual, repairs = _match_cost(
                query[i],
                float(target_pitches[j]),
                target_onsets[j],
                transposition,
                query_first,
                query_span,
                target_first,
                target_span,
            )
            update(
                i + 1,
                j + 1,
                current + local,
                _BackPointer(
                    i,
                    j,
                    "match",
                    (i,),
                    (start_index + j,),
                    pitch_residual,
                    timing_residual,
                    local,
                    repairs,
                ),
            )
        if i < len(query):
            local = 0.95 * max(0.3, query[i].quality_weight)
            update(
                i + 1,
                j,
                current + local,
                _BackPointer(i, j, "query_insertion", (i,), (), None, None, local, 0),
            )
        if j < len(target_pitches):
            local = 0.95
            update(
                i,
                j + 1,
                current + local,
                _BackPointer(
                    i,
                    j,
                    "target_insertion",
                    (),
                    (start_index + j,),
                    None,
                    None,
                    local,
                    0,
                ),
            )
        if i + 1 < len(query) and j < len(target_pitches):
            first = _match_cost(
                query[i],
                float(target_pitches[j]),
                target_onsets[j],
                transposition,
                query_first,
                query_span,
                target_first,
                target_span,
            )
            second = _match_cost(
                query[i + 1],
                float(target_pitches[j]),
                target_onsets[j],
                transposition,
                query_first,
                query_span,
                target_first,
                target_span,
            )
            query_span_cents = abs(
                query[i + 1].pitch_midi_cents - query[i].pitch_midi_cents
            )
            divergence_penalty = min(1.2, max(0.0, query_span_cents - 100.0) / 250.0)
            base_penalty = 0.20 if query_span_cents <= 70.0 else 0.55
            local = base_penalty + 0.5 * (first[0] + second[0]) + divergence_penalty
            update(
                i + 2,
                j + 1,
                current + local,
                _BackPointer(
                    i,
                    j,
                    "split",
                    (i, i + 1),
                    (start_index + j,),
                    (first[1] + second[1]) / 2.0,
                    round((first[2] + second[2]) / 2),
                    local,
                    first[3] + second[3],
                ),
            )
        if i < len(query) and j + 1 < len(target_pitches):
            left_match = _match_cost(
                query[i],
                float(target_pitches[j]),
                target_onsets[j],
                transposition,
                query_first,
                query_span,
                target_first,
                target_span,
            )
            right_match = _match_cost(
                query[i],
                float(target_pitches[j + 1]),
                target_onsets[j + 1],
                transposition,
                query_first,
                query_span,
                target_first,
                target_span,
            )
            local_match = min(left_match, right_match, key=lambda value: value[0])
            target_span_cents = abs(target_pitches[j + 1] - target_pitches[j])
            divergence_penalty = min(1.2, max(0.0, target_span_cents - 100.0) / 250.0)
            base_penalty = 0.20 if target_span_cents <= 70 else 0.55
            local = base_penalty + local_match[0] + divergence_penalty
            update(
                i + 1,
                j + 2,
                current + local,
                _BackPointer(
                    i,
                    j,
                    "merge",
                    (i,),
                    (start_index + j, start_index + j + 1),
                    local_match[1],
                    local_match[2],
                    local,
                    local_match[3],
                ),
            )

    i, j = len(query), len(target_pitches)
    pointers: list[_BackPointer] = []
    while (i, j) != (0, 0):
        pointer = back.get((i, j))
        if pointer is None:
            raise RuntimeError("alignment backtrace is incomplete")
        pointers.append(pointer)
        i, j = pointer.previous_i, pointer.previous_j
    pointers.reverse()
    pairs = tuple(
        AlignmentPair(
            query_event_indices=pointer.query_indices,
            target_event_indices=pointer.target_indices,
            transition=pointer.transition,  # type: ignore[arg-type]
            pitch_residual_cents=pointer.pitch_residual,
            timing_residual_ppm=pointer.timing_residual_ppm,
            local_cost=pointer.local_cost,
        )
        for pointer in pointers
    )
    query_insertions = sum(pointer.transition == "query_insertion" for pointer in pointers)
    target_insertions = sum(pointer.transition == "target_insertion" for pointer in pointers)
    splits = sum(pointer.transition == "split" for pointer in pointers)
    merges = sum(pointer.transition == "merge" for pointer in pointers)
    octave_repairs = sum(pointer.octave_repairs for pointer in pointers)
    query_span_us = round(query_span * 1_000_000 / 22_050)
    tempo_ratio_ppm = max(1, round(query_span_us * 1_000_000 / target_span))
    dtw_cost = _contour_dtw(
        [event.pitch_midi_cents for event in query], [float(value) for value in target_pitches]
    )
    raw_cost = float(costs[-1, -1])
    normalizer = max(len(query), len(target_pitches), 1)
    normalized_cost = raw_cost / normalizer
    edit_fraction = (query_insertions + target_insertions + splits + merges) / normalizer
    weighted_edit_count = 0.0
    for pointer in pointers:
        if pointer.transition == "merge":
            left, right = pointer.target_indices
            weighted_edit_count += (
                0.25
                if abs(representation.pitch_cents[right] - representation.pitch_cents[left])
                <= 70
                else 1.0
            )
        elif pointer.transition == "split":
            left, right = pointer.query_indices
            weighted_edit_count += (
                0.25
                if abs(query[right].pitch_midi_cents - query[left].pitch_midi_cents) <= 70.0
                else 1.0
            )
        elif pointer.transition == "target_insertion":
            target_index = pointer.target_indices[0]
            neighboring_pitches = [
                representation.pitch_cents[index]
                for index in (target_index - 1, target_index + 1)
                if start_index <= index < end_index
            ]
            weighted_edit_count += (
                0.25
                if any(
                    abs(representation.pitch_cents[target_index] - pitch) <= 70
                    for pitch in neighboring_pitches
                )
                else 1.0
            )
        elif pointer.transition == "query_insertion":
            weighted_edit_count += 1.0
    weighted_edit_fraction = weighted_edit_count / normalizer
    shingle_support = candidate.shingle_votes / max(1, len(query) - 2)
    weighted_shingle_support = candidate.weighted_shingle_score / max(1, len(query) - 2)
    structure = _alignment_structure(query, representation, pointers, transposition)
    tempo_log_deviation = abs(math.log(tempo_ratio_ppm / 1_000_000.0))
    pitch_tail_cost = min(3.0, structure["pitch_residual_p90_cents"] / 200.0)
    structural_penalty = (
        0.62 * normalized_cost
        + 0.055 * dtw_cost
        + 0.16 * structure["interval_transition_cost"]
        + 0.08 * structure["rhythm_transition_cost"]
        + 0.14 * tempo_log_deviation
        + 0.28 * weighted_edit_fraction
        + 0.07 * pitch_tail_cost
    )
    similarity = 100.0 * math.exp(-structural_penalty)
    retrieval_bonus = min(4.0, 0.35 * weighted_shingle_support)
    similarity = min(100.0, similarity + retrieval_bonus)
    score = ScoreEvidence(
        value=similarity,
        components={
            "dynamic_program_cost": raw_cost,
            "normalized_dynamic_program_cost": normalized_cost,
            "contour_dtw_cost": dtw_cost,
            "edit_fraction": edit_fraction,
            "weighted_edit_fraction": weighted_edit_fraction,
            "shingle_support": shingle_support,
            "weighted_shingle_support": weighted_shingle_support,
            "retrieval_bonus": retrieval_bonus,
            **structure,
            "tempo_log_deviation": tempo_log_deviation,
            "pitch_tail_cost": pitch_tail_cost,
            "structural_penalty": structural_penalty,
        },
    )
    transformations = TransformationEvidence(
        transposition_cents=round(transposition),
        tempo_ratio_ppm=tempo_ratio_ppm,
        octave_repairs=octave_repairs,
        query_insertions=query_insertions,
        target_insertions=target_insertions,
        splits=splits,
        merges=merges,
    )
    return AlignmentComputation(
        start_index=start_index,
        end_index=end_index,
        pairs=pairs,
        transformations=transformations,
        score=score,
        dtw_cost=dtw_cost,
        raw_cost=raw_cost,
    )


def align_candidate(
    interpretation: QueryInterpretation,
    representation: TargetTrackRepresentation,
    candidate: CandidateEvidence,
) -> AlignmentComputation:
    """Evaluate constrained local windows around a shingle-derived open-target start."""

    effective_query_count = 0
    previous_pitch: float | None = None
    for event in interpretation.events:
        if previous_pitch is None or abs(event.pitch_midi_cents - previous_pitch) > 70.0:
            effective_query_count += 1
            previous_pitch = event.pitch_midi_cents
    target_count = len(representation.pitch_cents)
    starts = range(
        max(0, candidate.start_event_index - 3),
        min(target_count, candidate.start_event_index + 4),
    )
    computations: list[AlignmentComputation] = []
    for start in starts:
        minimum_length = max(2, effective_query_count - 3)
        lengths = set(range(minimum_length, effective_query_count + 4))
        lengths.update(
            round(effective_query_count * ratio)
            for ratio in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75)
        )
        for length in sorted(lengths):
            if length > target_count - start:
                continue
            end = min(target_count, start + length)
            if end - start < 2:
                continue
            computations.append(
                _align_fixed_window(interpretation.events, representation, start, end, candidate)
            )
    if not computations:
        raise ValueError("candidate has no alignable target window")
    return max(computations, key=lambda value: value.score.value)
