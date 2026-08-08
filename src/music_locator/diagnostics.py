"""Read-only replay diagnostics for query perception, retrieval, and alignment.

The report intentionally lives outside the production search path.  It reads the
immutable artifacts used by a completed search and replays existing V0.0.1 helpers;
it never writes workspace state or changes candidate ranking and scoring.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from statistics import median
from typing import Any

from music_locator.alignment import AlignmentComputation, _align_fixed_window, _pitch_cost
from music_locator.audio import load_interpretation, load_query_analysis
from music_locator.indexing import load_index, load_track_representation
from music_locator.models import (
    AlignmentArtifact,
    AlignmentPair,
    CandidateEvidence,
    IndexSnapshot,
    QueryEvent,
    QueryInterpretation,
    TargetArtifact,
    TargetTrackRepresentation,
    TransformationEvidence,
    VoiceHypothesis,
)
from music_locator.representation import phrase_shingle_keys, shingle_keys
from music_locator.search import (
    _candidate_retrieval,
    _weighted_shingle_vote,
    load_search_result,
)
from music_locator.storage import Workspace
from music_locator.targets import load_midi_target
from music_locator.timebase import tick_to_microseconds

JsonObject = dict[str, Any]
RetrievalBucket = tuple[str, str, int]


@lru_cache(maxsize=256)
def _diagnostic_target(workspace: Workspace, target_id: str) -> TargetArtifact:
    """Avoid repeatedly decoding a target's potentially large raw-event artifact."""

    return load_midi_target(workspace, target_id)[0]


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _seconds(value: int, rate: int) -> float:
    return _rounded(value / rate)


def _microseconds_to_seconds(value: int) -> float:
    return _rounded(value / 1_000_000.0)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return _rounded(ordered[lower])
    weight = position - lower
    return _rounded(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _midi_note_name(midi_value: float) -> str:
    note = round(midi_value)
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[note % 12]}{note // 12 - 1}"


def _event_report(
    event: QueryEvent,
    relative_cents: float,
    *,
    analysis_rate: int,
    source_rate: int,
) -> JsonObject:
    midi_pitch = event.pitch_midi_cents / 100.0
    return {
        "event_id": event.event_id,
        "analysis_sample_range": {
            "start": event.start_sample,
            "end": event.end_sample,
            "sample_rate": analysis_rate,
        },
        "analysis_time_range_seconds": {
            "start": _seconds(event.start_sample, analysis_rate),
            "end": _seconds(event.end_sample, analysis_rate),
        },
        "source_sample_range": {
            "start": event.source_start_sample,
            "end": event.source_end_sample,
            "sample_rate": source_rate,
        },
        "source_time_range_seconds": {
            "start": _seconds(event.source_start_sample, source_rate),
            "end": _seconds(event.source_end_sample, source_rate),
        },
        "duration_seconds": _seconds(event.end_sample - event.start_sample, analysis_rate),
        "median_f0_hz": _rounded(440.0 * 2.0 ** ((midi_pitch - 69.0) / 12.0)),
        "midi_equivalent_pitch": _rounded(midi_pitch),
        "nearest_midi_note": _midi_note_name(midi_pitch),
        "relative_cents": _rounded(relative_cents),
        "pitch_deviation_cents": _rounded(event.pitch_deviation_cents),
        "onset_confidence": _rounded(event.onset_strength),
        "voicing_probability": _rounded(event.voicing),
        "quality_weight": _rounded(event.quality_weight),
    }


def _interpretation_report(
    interpretation: QueryInterpretation, *, analysis_rate: int, source_rate: int
) -> JsonObject:
    representation = interpretation.representation
    iois = [
        right - left
        for left, right in zip(
            representation.onset_samples,
            representation.onset_samples[1:],
            strict=False,
        )
    ]
    events = [
        _event_report(
            event,
            representation.relative_pitch_cents[index],
            analysis_rate=analysis_rate,
            source_rate=source_rate,
        )
        for index, event in enumerate(interpretation.events)
    ]
    return {
        "rank": interpretation.rank,
        "selected": interpretation.selected,
        "interpretation_id": interpretation.interpretation_id,
        "method": interpretation.method,
        "method_parameters": interpretation.method_parameters,
        "interpretation_score": _rounded(interpretation.interpretation_score),
        "event_count": len(events),
        "events": events,
        "feature_sequences": {
            "relative_pitch_cents": [
                _rounded(value) for value in representation.relative_pitch_cents
            ],
            "interval_cents": [_rounded(value) for value in representation.interval_cents],
            "contour_steps": list(representation.contour_steps),
            "onset_samples": list(representation.onset_samples),
            "ioi_samples": iois,
            "ioi_seconds": [_seconds(value, analysis_rate) for value in iois],
            "log_ioi_ratios": [_rounded(value) for value in representation.log_ioi_ratios],
            "quality_weights": [_rounded(value) for value in representation.quality_weights],
        },
    }


def _side_by_side(
    interpretations: list[QueryInterpretation], analysis_rate: int
) -> list[JsonObject]:
    rows: list[JsonObject] = []
    maximum = max((len(item.events) for item in interpretations), default=0)
    for event_position in range(maximum):
        cells: JsonObject = {}
        for interpretation in interpretations:
            key = f"rank_{interpretation.rank}"
            if event_position >= len(interpretation.events):
                cells[key] = None
                continue
            event = interpretation.events[event_position]
            cells[key] = {
                "event_id": event.event_id,
                "start_seconds": _seconds(event.start_sample, analysis_rate),
                "end_seconds": _seconds(event.end_sample, analysis_rate),
                "midi_pitch": _rounded(event.pitch_midi_cents / 100.0),
                "relative_cents": _rounded(
                    interpretation.representation.relative_pitch_cents[event_position]
                ),
                "pitch_deviation_cents": _rounded(event.pitch_deviation_cents),
                "onset_confidence": _rounded(event.onset_strength),
                "voicing_probability": _rounded(event.voicing),
                "quality_weight": _rounded(event.quality_weight),
            }
        rows.append({"event_position": event_position, "interpretations": cells})
    return rows


def _f0_summary(frames: list[JsonObject], analysis_rate: int) -> JsonObject:
    voiced = [
        frame for frame in frames if bool(frame.get("voiced")) and frame.get("f0_hz") is not None
    ]
    frequencies = [float(frame["f0_hz"]) for frame in voiced]
    midi_cents = [float(frame["midi_cents"]) for frame in voiced]
    probabilities = [float(frame["voiced_probability"]) for frame in frames]
    voiced_probabilities = [float(frame["voiced_probability"]) for frame in voiced]
    octave_jumps = 0
    large_pitch_jumps = 0
    voiced_runs = 0
    previous_index: int | None = None
    previous_pitch: float | None = None
    for frame in voiced:
        frame_index = int(frame["frame_index"])
        pitch = float(frame["midi_cents"])
        if previous_index is None or frame_index != previous_index + 1:
            voiced_runs += 1
        elif previous_pitch is not None:
            difference = abs(pitch - previous_pitch)
            octave_jumps += int(difference >= 1_000.0)
            large_pitch_jumps += int(difference >= 300.0)
        previous_index = frame_index
        previous_pitch = pitch
    return {
        "frame_count": len(frames),
        "voiced_frame_count": len(voiced),
        "voiced_fraction_recomputed": _rounded(len(voiced) / max(len(frames), 1)),
        "voiced_run_count": voiced_runs,
        "first_voiced_center_seconds": (
            _seconds(int(voiced[0]["center_sample"]), analysis_rate) if voiced else None
        ),
        "last_voiced_center_seconds": (
            _seconds(int(voiced[-1]["center_sample"]), analysis_rate) if voiced else None
        ),
        "f0_hz": {
            "minimum": _percentile(frequencies, 0.0),
            "p05": _percentile(frequencies, 0.05),
            "median": _percentile(frequencies, 0.5),
            "p95": _percentile(frequencies, 0.95),
            "maximum": _percentile(frequencies, 1.0),
        },
        "midi_cents": {
            "minimum": _percentile(midi_cents, 0.0),
            "p05": _percentile(midi_cents, 0.05),
            "median": _percentile(midi_cents, 0.5),
            "p95": _percentile(midi_cents, 0.95),
            "maximum": _percentile(midi_cents, 1.0),
        },
        "voiced_probability": {
            "all_frame_median": _percentile(probabilities, 0.5),
            "voiced_frame_median": _percentile(voiced_probabilities, 0.5),
            "voiced_frame_p05": _percentile(voiced_probabilities, 0.05),
        },
        "consecutive_voiced_pitch_jumps_at_least_300_cents": large_pitch_jumps,
        "consecutive_voiced_pitch_jumps_at_least_1000_cents": octave_jumps,
    }


def _voice_report(hypothesis: VoiceHypothesis | None, fallback_voice_id: str) -> JsonObject:
    if hypothesis is None:
        return {
            "voice_id": fallback_voice_id,
            "hypothesis_kind": "legacy",
            "source_track_index": None,
            "source_channel": None,
            "source_track_name": None,
            "extraction_method": None,
        }
    return {
        "voice_id": hypothesis.voice_id,
        "hypothesis_kind": hypothesis.hypothesis_kind,
        "authoritative": hypothesis.authoritative,
        "source_track_index": hypothesis.source_track_index,
        "source_channel": hypothesis.source_channel,
        "source_track_name": hypothesis.source_track_name,
        "extraction_method": hypothesis.extraction_method,
        "supporting_methods": list(hypothesis.supporting_methods),
        "extraction_version": hypothesis.extraction_version,
        "quality": hypothesis.quality.model_dump(mode="json"),
    }


def _target_window(
    workspace: Workspace,
    representation: TargetTrackRepresentation,
    start_index: int,
    end_index: int,
) -> JsonObject:
    target = _diagnostic_target(workspace, representation.target_id)
    count = len(representation.pitch_cents)
    if not 0 <= start_index < end_index <= count:
        raise ValueError("target window indices are outside the representation")
    start_tick = representation.start_ticks[start_index]
    end_tick = representation.end_ticks[end_index - 1]
    start_us = tick_to_microseconds(start_tick, target.ticks_per_beat, target.tempo_map)
    end_us = tick_to_microseconds(end_tick, target.ticks_per_beat, target.tempo_map)
    return {
        "event_start_index": start_index,
        "event_end_index_exclusive": end_index,
        "event_count": end_index - start_index,
        "tick_range": {
            "start": start_tick,
            "end": end_tick,
            "ticks_per_beat": target.ticks_per_beat,
        },
        "time_range_seconds": {
            "start": _microseconds_to_seconds(start_us),
            "end": _microseconds_to_seconds(end_us),
        },
    }


def _estimated_candidate_window(
    workspace: Workspace,
    representation: TargetTrackRepresentation,
    start_index: int,
    query_event_count: int,
) -> JsonObject:
    end_index = min(
        len(representation.pitch_cents),
        start_index + max(2, query_event_count),
    )
    if end_index <= start_index:
        end_index = min(len(representation.pitch_cents), start_index + 1)
    return _target_window(workspace, representation, start_index, end_index)


def _retrieval_replay(
    interpretation: QueryInterpretation,
    index: IndexSnapshot,
) -> tuple[
    list[tuple[int, str]],
    dict[RetrievalBucket, int],
    dict[RetrievalBucket, float],
    dict[RetrievalBucket, set[str]],
    dict[RetrievalBucket, list[JsonObject]],
    dict[RetrievalBucket, int],
]:
    keys = shingle_keys(
        list(interpretation.representation.interval_cents),
        list(interpretation.representation.contour_steps),
        list(interpretation.representation.log_ioi_ratios),
        int(index.parameters["shingle_size"]),
    )
    if "phrase_shingles" in index.parameters:
        keys.extend(
            phrase_shingle_keys([event.pitch_midi_cents for event in interpretation.events])
        )
    votes: dict[RetrievalBucket, int] = defaultdict(int)
    weighted_votes: dict[RetrievalBucket, float] = defaultdict(float)
    matched: dict[RetrievalBucket, set[str]] = defaultdict(set)
    occurrences: dict[RetrievalBucket, list[JsonObject]] = defaultdict(list)
    indexed_positions = max(1, sum(entry.event_count for entry in index.target_entries))
    for query_position, key in keys:
        postings = index.shingles.get(key, ())
        if not postings:
            continue
        query_quality = interpretation.events[
            min(query_position, len(interpretation.events) - 1)
        ].quality_weight
        weighted_vote = _weighted_shingle_vote(
            key,
            indexed_positions=indexed_positions,
            posting_count=len(postings),
            query_quality=query_quality,
        )
        for posting in postings:
            target_id, voice_id, target_position_text = posting.split("|")
            target_position = int(target_position_text)
            estimated_start = max(0, target_position - query_position)
            bucket = (target_id, voice_id, estimated_start)
            votes[bucket] += 1
            weighted_votes[bucket] += weighted_vote
            matched[bucket].add(key)
            occurrences[bucket].append(
                {
                    "query_position": query_position,
                    "key": key,
                    "target_position": target_position,
                    "estimated_start_event_index": estimated_start,
                    "weighted_vote": _rounded(weighted_vote),
                }
            )
    ranked = sorted(
        votes,
        key=lambda item: (
            -weighted_votes[item],
            -votes[item],
            item[1],
            item[2],
            item[0],
        ),
    )
    global_ranks = {bucket: rank for rank, bucket in enumerate(ranked, start=1)}
    return keys, votes, weighted_votes, matched, occurrences, global_ranks


def _candidate_report(
    workspace: Workspace,
    candidate: CandidateEvidence,
    rank: int,
    representation: TargetTrackRepresentation,
    query_event_count: int,
    global_vote_rank: int | None,
    matched_occurrences: list[JsonObject],
) -> JsonObject:
    target = _diagnostic_target(workspace, candidate.target_id)
    return {
        "pre_alignment_rank": rank,
        "retrieval_mode": "shingle_vote" if candidate.shingle_votes else "fallback_scan",
        "global_shingle_vote_bucket_rank": global_vote_rank,
        "target_id": candidate.target_id,
        "target_name": target.name,
        "track_index": candidate.track_index,
        "source_channel": candidate.source_channel,
        "voice": _voice_report(representation.voice_hypothesis, candidate.voice_id),
        "estimated_start_event_index": candidate.start_event_index,
        "estimated_query_length_window": _estimated_candidate_window(
            workspace,
            representation,
            candidate.start_event_index,
            query_event_count,
        ),
        "shingle_votes": candidate.shingle_votes,
        "weighted_shingle_score": _rounded(candidate.weighted_shingle_score),
        "matched_shingles": list(candidate.matched_shingles),
        "matched_query_to_target_shingle_occurrences": matched_occurrences,
    }


def _pitch_timing_components(
    query: QueryEvent,
    target_pitch_cents: float,
    target_onset_us: int,
    transposition_cents: float,
    query_first: int,
    query_span: int,
    target_first_us: int,
    target_span_us: int,
) -> tuple[float, float, float]:
    raw_pitch_residual = query.pitch_midi_cents - target_pitch_cents - transposition_cents
    octave_count = round(raw_pitch_residual / 1_200.0)
    pitch_residual = raw_pitch_residual - octave_count * 1_200.0
    pitch_cost = _pitch_cost(pitch_residual)
    query_position = (query.start_sample - query_first) / max(query_span, 1)
    target_position = (target_onset_us - target_first_us) / max(target_span_us, 1)
    timing_absolute = abs(query_position - target_position)
    timing_cost = (
        timing_absolute / 0.12 if timing_absolute <= 0.12 else 1.0 + (timing_absolute - 0.12) * 8.0
    )
    quality = max(0.2, query.quality_weight)
    return (
        quality * 0.64 * pitch_cost,
        quality * 0.36 * timing_cost,
        abs(octave_count) * 0.8,
    )


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


def _cost_contributions(
    interpretation: QueryInterpretation,
    representation: TargetTrackRepresentation,
    start_index: int,
    end_index: int,
    pairs: tuple[AlignmentPair, ...],
) -> JsonObject:
    query = interpretation.events
    target_pitches = representation.pitch_cents[start_index:end_index]
    transposition = _estimate_transposition(query, target_pitches)
    query_first = query[0].start_sample
    query_span = max(
        query[-1].start_sample - query_first,
        query[-1].end_sample - query_first,
        1,
    )
    target_first = representation.onset_microseconds[start_index]
    target_span = max(
        representation.onset_microseconds[end_index - 1] - target_first,
        1,
    )
    buckets = {
        "pitch": 0.0,
        "timing": 0.0,
        "query_insertion": 0.0,
        "target_insertion_deletion": 0.0,
        "split_penalty": 0.0,
        "merge_penalty": 0.0,
        "octave_repair": 0.0,
        "tempo_scaling": 0.0,
    }
    transition_totals: dict[str, float] = defaultdict(float)
    for pair in pairs:
        transition_totals[pair.transition] += pair.local_cost
        if pair.transition == "query_insertion":
            buckets["query_insertion"] += pair.local_cost
            continue
        if pair.transition == "target_insertion":
            buckets["target_insertion_deletion"] += pair.local_cost
            continue
        if pair.transition == "match":
            query_event = query[pair.query_event_indices[0]]
            target_index = pair.target_event_indices[0]
            pitch, timing, octave = _pitch_timing_components(
                query_event,
                float(representation.pitch_cents[target_index]),
                representation.onset_microseconds[target_index],
                transposition,
                query_first,
                query_span,
                target_first,
                target_span,
            )
            buckets["pitch"] += pitch
            buckets["timing"] += timing
            buckets["octave_repair"] += octave
            continue
        if pair.transition == "split":
            target_index = pair.target_event_indices[0]
            for query_index in pair.query_event_indices:
                pitch, timing, octave = _pitch_timing_components(
                    query[query_index],
                    float(representation.pitch_cents[target_index]),
                    representation.onset_microseconds[target_index],
                    transposition,
                    query_first,
                    query_span,
                    target_first,
                    target_span,
                )
                buckets["pitch"] += 0.5 * pitch
                buckets["timing"] += 0.5 * timing
                buckets["octave_repair"] += 0.5 * octave
            query_span_cents = abs(
                query[pair.query_event_indices[-1]].pitch_midi_cents
                - query[pair.query_event_indices[0]].pitch_midi_cents
            )
            split_base = 0.20 if query_span_cents <= 70.0 else 0.55
            buckets["split_penalty"] += split_base + min(
                1.2, max(0.0, query_span_cents - 100.0) / 250.0
            )
            continue
        if pair.transition == "merge":
            query_event = query[pair.query_event_indices[0]]
            left, right = pair.target_event_indices
            components = [
                _pitch_timing_components(
                    query_event,
                    float(representation.pitch_cents[target_index]),
                    representation.onset_microseconds[target_index],
                    transposition,
                    query_first,
                    query_span,
                    target_first,
                    target_span,
                )
                for target_index in (left, right)
            ]
            pitch, timing, octave = min(components, key=sum)
            buckets["pitch"] += pitch
            buckets["timing"] += timing
            buckets["octave_repair"] += octave
            target_span_cents = abs(
                representation.pitch_cents[right] - representation.pitch_cents[left]
            )
            merge_base = 0.20 if target_span_cents <= 70 else 0.55
            buckets["merge_penalty"] += merge_base + min(
                1.2, max(0.0, target_span_cents - 100.0) / 250.0
            )
    persisted_total = sum(pair.local_cost for pair in pairs)
    replay_total = sum(buckets.values())
    return {
        "exclusive_raw_dp_cost_buckets": {key: _rounded(value) for key, value in buckets.items()},
        "local_cost_by_transition": {
            key: _rounded(value) for key, value in sorted(transition_totals.items())
        },
        "persisted_path_local_cost_total": _rounded(persisted_total),
        "reconstructed_raw_dp_cost_total": _rounded(replay_total),
        "reconstruction_delta": _rounded(persisted_total - replay_total),
        "tempo_scaling_note": (
            "The raw DP derives tempo_ratio_ppm after path selection; the V0.0.1 "
            "structural final score explicitly penalizes its log deviation."
        ),
        "split_merge_note": (
            "Pitch, timing, and octave terms inside split/merge matches are assigned to "
            "their respective buckets; split_penalty and merge_penalty contain only the "
            "0.20 same-pitch-repeat or 0.55 distinct-pitch structural base plus any "
            "pitch-divergence penalty."
        ),
    }


def _path_report(
    pairs: tuple[AlignmentPair, ...],
    interpretation: QueryInterpretation,
    representation: TargetTrackRepresentation,
) -> list[JsonObject]:
    output: list[JsonObject] = []
    for step, pair in enumerate(pairs):
        query_events = [
            {
                "index": index,
                "event_id": interpretation.events[index].event_id,
                "pitch_midi": _rounded(interpretation.events[index].pitch_midi_cents / 100.0),
                "start_seconds": _seconds(interpretation.events[index].start_sample, 22_050),
                "quality_weight": _rounded(interpretation.events[index].quality_weight),
            }
            for index in pair.query_event_indices
        ]
        target_events = [
            {
                "index": index,
                "event_id": representation.event_ids[index],
                "pitch_midi": _rounded(representation.pitch_cents[index] / 100.0),
                "onset_seconds": _microseconds_to_seconds(representation.onset_microseconds[index]),
                "start_tick": representation.start_ticks[index],
                "end_tick": representation.end_ticks[index],
            }
            for index in pair.target_event_indices
        ]
        output.append(
            {
                "path_step": step,
                "transition": pair.transition,
                "query_events": query_events,
                "target_events": target_events,
                "pitch_residual_cents": (
                    _rounded(pair.pitch_residual_cents)
                    if pair.pitch_residual_cents is not None
                    else None
                ),
                "timing_residual_ppm": pair.timing_residual_ppm,
                "local_cost": _rounded(pair.local_cost),
            }
        )
    return output


def _alignment_body(
    workspace: Workspace,
    interpretation: QueryInterpretation,
    representation: TargetTrackRepresentation,
    *,
    start_index: int,
    end_index: int,
    pairs: tuple[AlignmentPair, ...],
    transformations: TransformationEvidence,
    score: JsonObject,
    contour_dtw_cost: float,
) -> JsonObject:
    transition_counts = Counter(pair.transition for pair in pairs)
    return {
        "target_window": _target_window(workspace, representation, start_index, end_index),
        "matrix_path_summary": {
            "matrix_shape": {
                "rows": len(interpretation.events) + 1,
                "columns": end_index - start_index + 1,
            },
            "path_step_count": len(pairs),
            "transition_counts": dict(sorted(transition_counts.items())),
            "matrix_cells_persisted": False,
            "matrix_note": (
                "V0.0.1 persists the winning backtrace and local path costs, not every DP "
                "cell. This diagnostic reports the complete persisted/replayed winning path."
            ),
        },
        "transformations": transformations.model_dump(mode="json"),
        "score": score,
        "contour_dtw_cost": _rounded(contour_dtw_cost),
        "cost_contributions": _cost_contributions(
            interpretation,
            representation,
            start_index,
            end_index,
            pairs,
        ),
        "full_alignment_path": _path_report(pairs, interpretation, representation),
    }


def _persisted_alignment_report(
    workspace: Workspace,
    artifact: AlignmentArtifact,
    persisted_rank: int,
    interpretation: QueryInterpretation,
    representation: TargetTrackRepresentation,
    candidate_rank: int | None,
) -> JsonObject:
    target = _diagnostic_target(workspace, artifact.target_id)
    output = {
        "persisted_alignment_rank": persisted_rank,
        "alignment_id": artifact.alignment_id,
        "target_id": artifact.target_id,
        "target_name": target.name,
        "track_index": artifact.track_index,
        "source_channel": artifact.candidate.source_channel,
        "voice": _voice_report(representation.voice_hypothesis, artifact.candidate.voice_id),
        "candidate_pre_alignment_rank": candidate_rank,
        "candidate": artifact.candidate.model_dump(mode="json"),
    }
    output.update(
        _alignment_body(
            workspace,
            interpretation,
            representation,
            start_index=artifact.target_event_start_index,
            end_index=artifact.target_event_end_index,
            pairs=artifact.aligned_events,
            transformations=artifact.transformations,
            score=artifact.score.model_dump(mode="json"),
            contour_dtw_cost=artifact.contour_dtw_cost,
        )
    )
    return output


def _computation_report(
    workspace: Workspace,
    computation: AlignmentComputation,
    interpretation: QueryInterpretation,
    representation: TargetTrackRepresentation,
) -> JsonObject:
    return _alignment_body(
        workspace,
        interpretation,
        representation,
        start_index=computation.start_index,
        end_index=computation.end_index,
        pairs=computation.pairs,
        transformations=computation.transformations,
        score=computation.score.model_dump(mode="json"),
        contour_dtw_cost=computation.dtw_cost,
    )


def _ranges_overlap(
    left_start: float, left_end: float, right_start: float, right_end: float
) -> bool:
    return left_start < right_end and right_start < left_end


def _event_overlap_indices(
    workspace: Workspace,
    representation: TargetTrackRepresentation,
    expected_start: float,
    expected_end: float,
) -> tuple[int, int] | None:
    target = _diagnostic_target(workspace, representation.target_id)
    indices: list[int] = []
    for index, (start_tick, end_tick) in enumerate(
        zip(representation.start_ticks, representation.end_ticks, strict=True)
    ):
        start = tick_to_microseconds(start_tick, target.ticks_per_beat, target.tempo_map) / 1e6
        end = tick_to_microseconds(end_tick, target.ticks_per_beat, target.tempo_map) / 1e6
        if _ranges_overlap(start, end, expected_start, expected_end):
            indices.append(index)
    if not indices:
        return None
    return min(indices), max(indices) + 1


def _expected_passage_report(
    workspace: Workspace,
    expected_start: float,
    expected_end: float,
    selected_target_id: str,
    selected_score: float | None,
    interpretation: QueryInterpretation,
    representations: dict[tuple[str, str], TargetTrackRepresentation],
    candidates: list[CandidateEvidence],
    candidate_rank_by_bucket: dict[RetrievalBucket, int],
    global_vote_ranks: dict[RetrievalBucket, int],
    votes: dict[RetrievalBucket, int],
    weighted_votes: dict[RetrievalBucket, float],
    matched: dict[RetrievalBucket, set[str]],
) -> JsonObject:
    target = _diagnostic_target(workspace, selected_target_id)
    voices: list[JsonObject] = []
    for (target_id, voice_id), representation in sorted(representations.items()):
        if target_id != selected_target_id:
            continue
        overlap = _event_overlap_indices(
            workspace,
            representation,
            expected_start,
            expected_end,
        )
        if overlap is None:
            continue
        start_index, end_index = overlap
        shortlist_seeds = [
            candidate
            for candidate in candidates
            if candidate.target_id == target_id
            and candidate.voice_id == voice_id
            and abs(candidate.start_event_index - start_index) <= 3
        ]
        shortlist_seeds.sort(
            key=lambda item: candidate_rank_by_bucket[
                (item.target_id, item.voice_id, item.start_event_index)
            ]
        )
        voted_seeds = [
            bucket
            for bucket in votes
            if bucket[0] == target_id
            and bucket[1] == voice_id
            and abs(bucket[2] - start_index) <= 3
        ]
        voted_seeds.sort(key=lambda bucket: global_vote_ranks[bucket])
        if shortlist_seeds:
            seed = shortlist_seeds[0]
            seed_bucket = (seed.target_id, seed.voice_id, seed.start_event_index)
            seed_rank = candidate_rank_by_bucket[seed_bucket]
            synthetic_seed = False
        else:
            seed_bucket = (target_id, voice_id, start_index)
            seed = CandidateEvidence(
                target_id=target_id,
                track_index=representation.track_index,
                start_event_index=start_index,
                shingle_votes=votes.get(seed_bucket, 0),
                matched_shingles=tuple(sorted(matched.get(seed_bucket, set()))),
                weighted_shingle_score=weighted_votes.get(seed_bucket, 0.0),
                voice_id=voice_id,
                source_channel=(
                    representation.voice_hypothesis.source_channel
                    if representation.voice_hypothesis is not None
                    else 0
                ),
            )
            seed_rank = None
            synthetic_seed = True
        item: JsonObject = {
            "target_id": target_id,
            "target_name": target.name,
            "track_index": representation.track_index,
            "voice": _voice_report(representation.voice_hypothesis, voice_id),
            "overlapping_voice_window": _target_window(
                workspace,
                representation,
                start_index,
                end_index,
            ),
            "candidate_generation": {
                "reached_pre_dp_shortlist": bool(shortlist_seeds),
                "best_pre_alignment_rank_capable_of_seeding_window": seed_rank,
                "best_global_shingle_vote_bucket_rank_capable_of_seeding_window": (
                    global_vote_ranks[voted_seeds[0]] if voted_seeds else None
                ),
                "best_global_shingle_votes_capable_of_seeding_window": (
                    votes[voted_seeds[0]] if voted_seeds else 0
                ),
                "shortlist_seed_candidates": [
                    {
                        "pre_alignment_rank": candidate_rank_by_bucket[
                            (candidate.target_id, candidate.voice_id, candidate.start_event_index)
                        ],
                        "start_event_index": candidate.start_event_index,
                        "shingle_votes": candidate.shingle_votes,
                        "matched_shingles": list(candidate.matched_shingles),
                    }
                    for candidate in shortlist_seeds
                ],
            },
            "forced_alignment_seed": {
                "synthetic_for_diagnostic_only": synthetic_seed,
                "pre_alignment_rank": seed_rank,
                "candidate": seed.model_dump(mode="json"),
            },
        }
        try:
            forced = _align_fixed_window(
                interpretation.events,
                representation,
                start_index,
                end_index,
                seed,
            )
        except (RuntimeError, ValueError) as error:
            item["forced_alignment_error"] = str(error)
        else:
            forced_report = _computation_report(
                workspace,
                forced,
                interpretation,
                representation,
            )
            forced_score = forced.score.value
            forced_report["comparison_to_selected_result"] = {
                "selected_result_score": (
                    _rounded(selected_score) if selected_score is not None else None
                ),
                "forced_expected_score": _rounded(forced_score),
                "forced_minus_selected_score": (
                    _rounded(forced_score - selected_score) if selected_score is not None else None
                ),
            }
            item["forced_exact_window_alignment"] = forced_report
        voices.append(item)
    scored = [
        item for item in voices if isinstance(item.get("forced_exact_window_alignment"), dict)
    ]
    scored.sort(
        key=lambda item: float(item["forced_exact_window_alignment"]["score"]["value"]),
        reverse=True,
    )
    return {
        "expected_target_id": selected_target_id,
        "expected_target_name": target.name,
        "supplied_time_range_seconds": {
            "start": _rounded(expected_start),
            "end": _rounded(expected_end),
        },
        "overlapping_voice_count": len(voices),
        "best_forced_voice_id": (scored[0]["voice"]["voice_id"] if scored else None),
        "best_forced_score": (
            _rounded(float(scored[0]["forced_exact_window_alignment"]["score"]["value"]))
            if scored
            else None
        ),
        "voices": voices,
    }


def diagnose_search(
    workspace: Workspace,
    search_id: str,
    *,
    expected_target_start: float | None = None,
    expected_target_end: float | None = None,
) -> JsonObject:
    """Build a deterministic JSON-ready report for a completed search."""

    workspace.initialize()
    if (expected_target_start is None) != (expected_target_end is None):
        raise ValueError("expected target start and end must be supplied together")
    if expected_target_start is not None and expected_target_end is not None:
        if expected_target_start < 0.0:
            raise ValueError("expected target start must be non-negative")
        if expected_target_end <= expected_target_start:
            raise ValueError("expected target end must be greater than start")
    search_row = workspace.database.fetchone(
        """
        SELECT query_id, interpretation_id, index_id, status
        FROM searches WHERE search_id = ?
        """,
        (search_id,),
    )
    if search_row is None:
        raise KeyError(f"unknown search: {search_id}")
    if search_row["index_id"] is None:
        raise ValueError(f"search has no index: {search_id}")
    query_id = str(search_row["query_id"])
    index_id = str(search_row["index_id"])
    result, result_artifact_id = load_search_result(workspace, search_id)
    analysis, analysis_artifact_id = load_query_analysis(workspace, query_id)
    source_row = workspace.database.fetchone(
        """
        SELECT audio_streams.frame_count, audio_streams.sample_rate,
               audio_streams.channels, audio_streams.sample_format,
               assets.asset_id, assets.sha256, assets.original_name
        FROM queries
        JOIN audio_streams ON queries.original_stream_id = audio_streams.stream_id
        JOIN assets ON queries.original_asset_id = assets.asset_id
        WHERE queries.query_id = ?
        """,
        (query_id,),
    )
    analysis_stream_row = workspace.database.fetchone(
        "SELECT frame_count, sample_rate FROM audio_streams WHERE stream_id = ?",
        (analysis.analysis_stream_id,),
    )
    if source_row is None or analysis_stream_row is None:
        raise RuntimeError("query audio stream evidence is missing")
    source_rate = int(source_row["sample_rate"])
    analysis_rate = int(analysis_stream_row["sample_rate"])
    interpretations = [
        load_interpretation(workspace, interpretation_id)[0]
        for interpretation_id in analysis.interpretation_ids
    ]
    interpretations.sort(key=lambda item: item.rank)
    selected_id = (
        str(search_row["interpretation_id"])
        if search_row["interpretation_id"] is not None
        else analysis.selected_interpretation_id
    )
    source_selected = next(
        (item for item in interpretations if item.interpretation_id == selected_id),
        None,
    )
    selected = source_selected
    if (
        result.provenance is not None
        and result.provenance.search_interpretation_artifact_id is not None
    ):
        selected = QueryInterpretation.model_validate(
            workspace.load_artifact_json(result.provenance.search_interpretation_artifact_id)
        )
    pitch_payload = workspace.load_artifact_json(analysis.pitch_frames_artifact_id)
    onset_payload = workspace.load_artifact_json(analysis.onsets_artifact_id)
    pitch_frames = list(pitch_payload.get("frames", []))
    onsets = list(onset_payload.get("onsets", []))
    report: JsonObject = {
        "diagnostic_schema_version": "0.0.1",
        "scope": "read_only_replay_of_persisted_v0.0.1_evidence",
        "search": {
            "search_id": search_id,
            "status": result.status.value,
            "reason_codes": list(result.reason_codes),
            "result_artifact_id": result_artifact_id,
            "query_id": query_id,
            "selected_interpretation_id": selected_id,
            "search_interpretation_id": (
                selected.interpretation_id if selected is not None else None
            ),
            "index_id": index_id,
            "selected_target_id": result.selected_target_id,
            "selected_target_name": result.selected_target_name,
            "selected_score": result.score.model_dump(mode="json") if result.score else None,
        },
        "query_audio": {
            "asset_id": str(source_row["asset_id"]),
            "source_sha256": str(source_row["sha256"]),
            "original_name": str(source_row["original_name"]),
            "source_stream": {
                "duration_samples": int(source_row["frame_count"]),
                "sample_rate": source_rate,
                "duration_seconds": _seconds(int(source_row["frame_count"]), source_rate),
                "channels": int(source_row["channels"]),
                "sample_format": str(source_row["sample_format"]),
            },
            "analysis_stream": {
                "duration_samples": int(analysis_stream_row["frame_count"]),
                "sample_rate": analysis_rate,
                "duration_seconds": _seconds(
                    int(analysis_stream_row["frame_count"]), analysis_rate
                ),
            },
            "quality": analysis.quality.model_dump(mode="json"),
            "analysis_artifact_id": analysis_artifact_id,
        },
        "query_perception": {
            "f0_contour_summary": _f0_summary(pitch_frames, analysis_rate),
            "onset_evidence": [
                {
                    "sample": int(onset["sample"]),
                    "seconds": _seconds(int(onset["sample"]), analysis_rate),
                    "strength": _rounded(float(onset["strength"])),
                }
                for onset in onsets
            ],
            "interpretations": [
                _interpretation_report(
                    interpretation,
                    analysis_rate=analysis_rate,
                    source_rate=source_rate,
                )
                for interpretation in interpretations
            ],
            "interpretation_side_by_side": _side_by_side(interpretations, analysis_rate),
            "selected_search_interpretation": (
                _interpretation_report(
                    selected,
                    analysis_rate=analysis_rate,
                    source_rate=source_rate,
                )
                if selected is not None
                else None
            ),
        },
    }
    if selected is None:
        report["candidate_generation"] = {
            "available": False,
            "reason": "search has no selected query interpretation",
        }
        report["persisted_top_alignments"] = []
        return report
    index, index_artifact_id = load_index(workspace, index_id)
    representations = {
        (entry.target_id, entry.voice_id): load_track_representation(
            workspace,
            entry.representation_artifact_id,
        )
        for entry in index.target_entries
    }
    candidates = _candidate_retrieval(selected, index, representations)
    keys, votes, weighted_votes, matched, occurrences, global_vote_ranks = _retrieval_replay(
        selected,
        index,
    )
    candidate_rank_by_bucket = {
        (candidate.target_id, candidate.voice_id, candidate.start_event_index): rank
        for rank, candidate in enumerate(candidates, start=1)
    }
    candidate_reports = [
        _candidate_report(
            workspace,
            candidate,
            rank,
            representations[(candidate.target_id, candidate.voice_id)],
            len(selected.events),
            global_vote_ranks.get(
                (candidate.target_id, candidate.voice_id, candidate.start_event_index)
            ),
            occurrences.get(
                (candidate.target_id, candidate.voice_id, candidate.start_event_index),
                [],
            ),
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]
    posting_evidence: list[JsonObject] = []
    for query_position, key in keys:
        postings: list[JsonObject] = []
        for posting in index.shingles.get(key, ()):
            target_id, voice_id, target_position_text = posting.split("|")
            target_position = int(target_position_text)
            estimated_start = max(0, target_position - query_position)
            bucket = (target_id, voice_id, estimated_start)
            representation = representations[(target_id, voice_id)]
            target = _diagnostic_target(workspace, target_id)
            postings.append(
                {
                    "target_id": target_id,
                    "target_name": target.name,
                    "voice_id": voice_id,
                    "voice_method": (
                        representation.voice_hypothesis.extraction_method
                        if representation.voice_hypothesis is not None
                        else None
                    ),
                    "target_position": target_position,
                    "estimated_start_event_index": estimated_start,
                    "global_shingle_vote_bucket_rank": global_vote_ranks.get(bucket),
                    "pre_alignment_shortlist_rank": candidate_rank_by_bucket.get(bucket),
                }
            )
        posting_evidence.append(
            {
                "query_position": query_position,
                "shingle_key": key,
                "view": key.split(":", maxsplit=1)[0],
                "posting_count": len(postings),
                "query_occurrence_weight": _rounded(
                    _weighted_shingle_vote(
                        key,
                        indexed_positions=max(
                            1, sum(entry.event_count for entry in index.target_entries)
                        ),
                        posting_count=max(1, len(postings)),
                        query_quality=selected.events[
                            min(query_position, len(selected.events) - 1)
                        ].quality_weight,
                    )
                )
                if postings
                else 0.0,
                "postings": postings,
            }
        )
    report["candidate_generation"] = {
        "available": True,
        "index_artifact_id": index_artifact_id,
        "shingle_size": int(index.parameters["shingle_size"]),
        "query_shingle_occurrence_count": len(keys),
        "unique_query_shingle_count": len({key for _, key in keys}),
        "global_shingle_vote_bucket_count_before_limit": len(votes),
        "global_weighted_shingle_bucket_count_before_limit": len(weighted_votes),
        "shortlist_limit": 160,
        "shortlist_count": len(candidates),
        "query_shingle_posting_evidence": posting_evidence,
        "pre_dp_shortlist": candidate_reports,
    }
    alignment_rows = workspace.database.fetchall(
        """
        SELECT rank, artifact_id FROM alignments
        WHERE search_id = ? ORDER BY rank
        """,
        (search_id,),
    )
    persisted_alignments: list[JsonObject] = []
    for row in alignment_rows:
        artifact = AlignmentArtifact.model_validate(
            workspace.load_artifact_json(str(row["artifact_id"]))
        )
        representation = representations[(artifact.target_id, artifact.candidate.voice_id)]
        alignment_interpretation = selected
        if artifact.query_interpretation_artifact_id is not None:
            alignment_interpretation = QueryInterpretation.model_validate(
                workspace.load_artifact_json(artifact.query_interpretation_artifact_id)
            )
        if alignment_interpretation is None:
            raise RuntimeError("alignment query interpretation evidence is missing")
        bucket = (
            artifact.candidate.target_id,
            artifact.candidate.voice_id,
            artifact.candidate.start_event_index,
        )
        persisted_alignments.append(
            _persisted_alignment_report(
                workspace,
                artifact,
                int(row["rank"]),
                alignment_interpretation,
                representation,
                candidate_rank_by_bucket.get(bucket),
            )
        )
    report["persisted_top_alignments"] = persisted_alignments
    if expected_target_start is not None and expected_target_end is not None:
        if result.selected_target_id is None:
            report["expected_passage"] = {
                "available": False,
                "reason": "search selected no target; expected seconds cannot identify a target",
            }
        else:
            report["expected_passage"] = _expected_passage_report(
                workspace,
                expected_target_start,
                expected_target_end,
                result.selected_target_id,
                result.score.value if result.score is not None else None,
                selected,
                representations,
                candidates,
                candidate_rank_by_bucket,
                global_vote_ranks,
                votes,
                weighted_votes,
                matched,
            )
    return report
