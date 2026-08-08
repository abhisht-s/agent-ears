"""Candidate retrieval, ranking, abstention, and durable search results."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from music_locator.alignment import AlignmentComputation, align_candidate
from music_locator.audio import (
    load_interpretation,
    load_query_analysis,
    prepare_interpretation_for_search,
)
from music_locator.indexing import load_index, load_track_representation
from music_locator.models import (
    AlignmentArtifact,
    AudioTargetArtifact,
    CandidateEvidence,
    IndexSnapshot,
    MatchAlternative,
    MediaMicrosecondRange,
    MicrosecondRange,
    ProvenanceEvidence,
    QualityWarning,
    QueryInterpretation,
    ResultStatus,
    SampleRange,
    SearchResult,
    TargetArtifact,
    TargetCoordinateKind,
    TargetRepresentationBundle,
    TargetTimeMap,
    TargetTrackRepresentation,
    TickRange,
    VoiceHypothesis,
    VoiceHypothesisSummary,
)
from music_locator.representation import phrase_shingle_keys, shingle_keys
from music_locator.storage import Workspace, new_id, utc_now
from music_locator.targets import load_target
from music_locator.timebase import tick_to_microseconds


@dataclass(frozen=True)
class _TargetLocation:
    target: TargetArtifact | AudioTargetArtifact
    target_artifact_id: str
    coordinate_kind: TargetCoordinateKind
    tick_range: TickRange | None
    sample_range: SampleRange | None
    time_range: MicrosecondRange | MediaMicrosecondRange
    target_analysis_artifact_id: str | None = None
    decode_manifest_artifact_id: str | None = None
    time_map_artifact_id: str | None = None
    evidence_artifact_id: str | None = None


def _load_audio_bundle_for_representation(
    workspace: Workspace,
    *,
    target_id: str,
    representation_artifact_id: str,
) -> tuple[TargetRepresentationBundle, str]:
    """Resolve the evidence bundle that was actually indexed, not merely latest.

    The exact representation-to-analysis relationship is durable in
    ``target_views``.  The fallback supports early/migrated workspaces while
    still refusing to attach a representation to an unrelated newer bundle.
    """

    row = workspace.database.fetchone(
        """
        SELECT target_analyses.bundle_artifact_id
        FROM target_views
        JOIN target_analyses
          ON target_analyses.target_analysis_id = target_views.target_analysis_id
        WHERE target_views.target_id = ?
          AND target_views.representation_artifact_id = ?
        ORDER BY target_analyses.created_at DESC
        LIMIT 1
        """,
        (target_id, representation_artifact_id),
    )
    if row is not None:
        bundle_artifact_id = str(row["bundle_artifact_id"])
        bundle = TargetRepresentationBundle.model_validate(
            workspace.load_artifact_json(bundle_artifact_id)
        )
    else:
        from music_locator.target_audio import load_latest_target_bundle

        bundle, bundle_artifact_id, _ = load_latest_target_bundle(workspace, target_id)
    if bundle.target_id != target_id:
        raise ValueError("audio representation bundle belongs to a different target")
    if representation_artifact_id not in bundle.representation_artifact_ids:
        raise ValueError("indexed audio representation is absent from its evidence bundle")
    return bundle, bundle_artifact_id


def _target_location(
    workspace: Workspace,
    *,
    representation: TargetTrackRepresentation,
    representation_artifact_id: str,
    start_index: int,
    end_index: int,
) -> _TargetLocation:
    """Map one aligned event window into its native target coordinates."""

    target, target_artifact_id = load_target(workspace, representation.target_id)
    if end_index <= start_index:
        raise ValueError("target alignment window must contain at least one event")
    if representation.coordinate_kind == TargetCoordinateKind.MIDI_TICKS:
        if isinstance(target, AudioTargetArtifact):
            raise ValueError("audio target cannot use MIDI tick coordinates")
        tick_range = TickRange(
            start_tick=representation.start_ticks[start_index],
            end_tick=representation.end_ticks[end_index - 1],
            ticks_per_beat=target.ticks_per_beat,
        )
        return _TargetLocation(
            target=target,
            target_artifact_id=target_artifact_id,
            coordinate_kind=TargetCoordinateKind.MIDI_TICKS,
            tick_range=tick_range,
            sample_range=None,
            time_range=MicrosecondRange(
                start_microsecond=tick_to_microseconds(
                    tick_range.start_tick, target.ticks_per_beat, target.tempo_map
                ),
                end_microsecond=tick_to_microseconds(
                    tick_range.end_tick, target.ticks_per_beat, target.tempo_map
                ),
            ),
        )
    if not isinstance(target, AudioTargetArtifact):
        raise ValueError("MIDI/timed-note target cannot use decoded sample coordinates")
    bundle, bundle_artifact_id = _load_audio_bundle_for_representation(
        workspace,
        target_id=target.target_id,
        representation_artifact_id=representation_artifact_id,
    )
    target_time_map = TargetTimeMap.model_validate(
        workspace.load_artifact_json(bundle.target_time_map_artifact_id)
    )
    if representation.reference_sample_rate != target_time_map.decoded_sample_rate:
        raise ValueError("audio representation sample rate disagrees with its target time map")
    if (
        representation.reference_zero_media_microseconds is not None
        and representation.reference_zero_media_microseconds
        != target_time_map.reference_zero_media_microseconds
    ):
        raise ValueError("audio representation media origin disagrees with its target time map")
    sample_range = SampleRange(
        start_sample=representation.start_samples[start_index],
        end_sample=representation.end_samples[end_index - 1],
        sample_rate=target_time_map.decoded_sample_rate,
    )
    audio_hypothesis = representation.audio_voice_hypothesis
    return _TargetLocation(
        target=target,
        target_artifact_id=target_artifact_id,
        coordinate_kind=TargetCoordinateKind.DECODED_SAMPLES,
        tick_range=None,
        sample_range=sample_range,
        time_range=MediaMicrosecondRange(
            start_microsecond=target_time_map.reference_sample_to_media_microseconds(
                sample_range.start_sample
            ),
            end_microsecond=target_time_map.reference_sample_to_media_microseconds(
                sample_range.end_sample
            ),
        ),
        target_analysis_artifact_id=bundle_artifact_id,
        decode_manifest_artifact_id=bundle.decode_manifest_artifact_id,
        time_map_artifact_id=bundle.target_time_map_artifact_id,
        evidence_artifact_id=(
            audio_hypothesis.evidence_artifact_id if audio_hypothesis is not None else None
        ),
    )


def _sample_overlap(left: SampleRange, right: SampleRange) -> float:
    if left.sample_rate != right.sample_rate:
        return 0.0
    overlap = max(
        0,
        min(left.end_sample, right.end_sample) - max(left.start_sample, right.start_sample),
    )
    shorter = min(
        left.end_sample - left.start_sample,
        right.end_sample - right.start_sample,
    )
    return overlap / max(1, shorter)


def _shingle_view_weight(key: str) -> float:
    if key.startswith("ic:"):
        return 2.0
    if key.startswith("i:"):
        return 1.5
    if key.startswith("msic"):
        return 1.6
    if key.startswith("msi"):
        return 1.8
    if key.startswith("msc"):
        return 0.45
    if key.startswith(("c:", "r:")):
        return 0.25
    return 1.0


def _weighted_shingle_vote(
    key: str,
    *,
    indexed_positions: int,
    posting_count: int,
    query_quality: float,
) -> float:
    inverse_frequency = 1.0 + math.log((1.0 + indexed_positions) / (1.0 + posting_count))
    return _shingle_view_weight(key) * inverse_frequency * (0.5 + 0.5 * query_quality)


def _voice_summary(hypothesis: VoiceHypothesis | None) -> VoiceHypothesisSummary | None:
    if hypothesis is None:
        return None
    return VoiceHypothesisSummary(
        voice_id=hypothesis.voice_id,
        hypothesis_kind=hypothesis.hypothesis_kind,
        authoritative=hypothesis.authoritative,
        source_track_index=hypothesis.source_track_index,
        source_channel=hypothesis.source_channel,
        source_track_name=hypothesis.source_track_name,
        source_programs=hypothesis.source_programs,
        extraction_method=hypothesis.extraction_method,
        supporting_methods=hypothesis.supporting_methods,
        extraction_version=hypothesis.extraction_version,
        parameters=hypothesis.parameters,
        quality=hypothesis.quality,
        note_count=len(hypothesis.note_provenance),
    )


def _source_range(workspace: Workspace, query_id: str) -> SampleRange:
    row = workspace.database.fetchone(
        """
        SELECT audio_streams.frame_count, audio_streams.sample_rate
        FROM queries JOIN audio_streams ON queries.original_stream_id = audio_streams.stream_id
        WHERE queries.query_id = ?
        """,
        (query_id,),
    )
    if row is None:
        raise KeyError(f"unknown query: {query_id}")
    return SampleRange(
        start_sample=0, end_sample=int(row["frame_count"]), sample_rate=int(row["sample_rate"])
    )


def _candidate_retrieval(
    interpretation: QueryInterpretation,
    index: IndexSnapshot,
    representations: dict[tuple[str, str], TargetTrackRepresentation],
    *,
    limit: int = 160,
) -> list[CandidateEvidence]:
    query = interpretation
    snapshot = index
    query_keys = shingle_keys(
        list(query.representation.interval_cents),
        list(query.representation.contour_steps),
        list(query.representation.log_ioi_ratios),
        int(snapshot.parameters["shingle_size"]),
    )
    if "phrase_shingles" in snapshot.parameters:
        query_keys.extend(phrase_shingle_keys([event.pitch_midi_cents for event in query.events]))
    votes: dict[tuple[str, str, int], int] = defaultdict(int)
    weighted_votes: dict[tuple[str, str, int], float] = defaultdict(float)
    evidence: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    indexed_positions = max(1, sum(entry.event_count for entry in snapshot.target_entries))

    for query_position, key in query_keys:
        postings = tuple(
            posting
            for posting in snapshot.shingles.get(key, ())
            if tuple(posting.split("|")[:2]) in representations
        )
        if not postings:
            continue
        query_quality = query.events[min(query_position, len(query.events) - 1)].quality_weight
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
            evidence[bucket].add(key)
    ranked = sorted(
        votes,
        key=lambda item: (-weighted_votes[item], -votes[item], item[1], item[2], item[0]),
    )
    candidates: list[CandidateEvidence] = []
    for target_id, voice_id, start in ranked[:limit]:
        representation = representations.get((target_id, voice_id))
        if representation is None:
            continue
        hypothesis = representation.voice_hypothesis
        candidates.append(
            CandidateEvidence(
                target_id=target_id,
                track_index=representation.track_index,
                start_event_index=start,
                shingle_votes=votes[(target_id, voice_id, start)],
                matched_shingles=tuple(sorted(evidence[(target_id, voice_id, start)])),
                weighted_shingle_score=weighted_votes[(target_id, voice_id, start)],
                voice_id=voice_id,
                view_id=representation.view_id,
                source_channel=hypothesis.source_channel if hypothesis is not None else None,
            )
        )
    represented = {(item.target_id, item.voice_id, item.start_event_index) for item in candidates}
    fallback_starts: dict[tuple[str, str], list[int]] = {}
    for representation_key, representation in sorted(representations.items()):
        fallback_starts[representation_key] = list(
            range(0, len(representation.pitch_cents), max(1, len(query.events) // 2))
        )
    fallback_depth = max((len(starts) for starts in fallback_starts.values()), default=0)
    for depth in range(fallback_depth):
        for (target_id, voice_id), representation in sorted(representations.items()):
            starts = fallback_starts[(target_id, voice_id)]
            if depth >= len(starts):
                continue
            start = starts[depth]
            range_key = (target_id, voice_id, start)
            if range_key not in represented:
                hypothesis = representation.voice_hypothesis
                candidates.append(
                    CandidateEvidence(
                        target_id=target_id,
                        track_index=representation.track_index,
                        start_event_index=start,
                        shingle_votes=0,
                        matched_shingles=(),
                        voice_id=voice_id,
                        view_id=representation.view_id,
                        source_channel=(
                            hypothesis.source_channel if hypothesis is not None else None
                        ),
                    )
                )
            if len(candidates) >= limit:
                return candidates
    return candidates


def _typed_early_result(
    workspace: Workspace,
    query_id: str,
    search_id: str,
    status: ResultStatus,
    reasons: tuple[str, ...],
    explanation: str,
    quality_warnings: tuple[object, ...],
) -> SearchResult:
    return SearchResult(
        search_id=search_id,
        status=status,
        reason_codes=reasons,
        status_explanation=explanation,
        selected_target_id=None,
        selected_target_name=None,
        selected_track_index=None,
        target_range=None,
        target_time_range=None,
        query_source_range=_source_range(workspace, query_id),
        score=None,
        transformations=None,
        aligned_events=(),
        strong_rhythmic_target_ticks=(),
        alternatives=(),
        quality_warnings=quality_warnings,
        provenance=None,
    )


def _persist_result(
    workspace: Workspace, search_id: str, result: SearchResult, run_id: str
) -> SearchResult:
    artifact_id = workspace.store_json_artifact(
        result, artifact_type="search.result", analysis_run_id=run_id
    )
    workspace.database.execute(
        "UPDATE searches SET result_artifact_id = ?, status = ? WHERE search_id = ?",
        (artifact_id, result.status.value, search_id),
    )
    return result


def search(
    workspace: Workspace,
    query_id: str,
    index_id: str,
    *,
    target_id: str | None = None,
) -> SearchResult:
    workspace.initialize()
    analysis, query_analysis_artifact_id = load_query_analysis(workspace, query_id)
    search_id = new_id("search")
    run_parameters = {
        "query_id": query_id,
        "index_id": index_id,
        "matcher": "open_target_dp_v1_structural_score_v1",
        "interpretation_policy": "all_usable_interpretations_v1",
    }
    if target_id is not None:
        run_parameters["target_id"] = target_id
    run_id = workspace.create_analysis_run(
        "search",
        run_parameters,
    )
    interpretation_id = analysis.selected_interpretation_id
    workspace.database.execute(
        """
        INSERT INTO searches(search_id, query_id, interpretation_id, index_id,
                             analysis_run_id, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (search_id, query_id, interpretation_id, index_id, run_id, "running", utc_now()),
    )
    hard_quality_codes = {
        warning.code for warning in analysis.quality.warnings if warning.severity == "error"
    }
    if interpretation_id is None or hard_quality_codes:
        result = _typed_early_result(
            workspace,
            query_id,
            search_id,
            ResultStatus.INSUFFICIENT_QUERY,
            tuple(sorted(hard_quality_codes or {"no_selected_interpretation"})),
            "The query does not contain enough valid pitched evidence for reliable retrieval.",
            analysis.quality.warnings,
        )
        workspace.finish_analysis_run(run_id)
        return _persist_result(workspace, search_id, result, run_id)
    source_interpretations = [
        load_interpretation(workspace, item_id)[0] for item_id in analysis.interpretation_ids
    ]
    prepared_interpretations: list[tuple[QueryInterpretation, str, str]] = []
    evaluated_source_interpretation_ids: list[str] = []
    for source_interpretation in source_interpretations:
        prepared = prepare_interpretation_for_search(source_interpretation)
        artifact_id = workspace.store_json_artifact(
            prepared,
            artifact_type="search.query_interpretation",
            analysis_run_id=run_id,
        )
        if len(prepared.events) >= 3:
            evaluated_source_interpretation_ids.append(source_interpretation.interpretation_id)
            prepared_interpretations.append(
                (prepared, artifact_id, source_interpretation.interpretation_id)
            )
    deduplicated: dict[tuple[tuple[int, int, int], ...], tuple[QueryInterpretation, str, str]] = {}
    for prepared_item in prepared_interpretations:
        prepared = prepared_item[0]
        signature = tuple(
            (event.start_sample, event.end_sample, round(event.pitch_midi_cents))
            for event in prepared.events
        )
        previous_prepared = deduplicated.get(signature)
        if (
            previous_prepared is None
            or prepared.interpretation_score > previous_prepared[0].interpretation_score
        ):
            deduplicated[signature] = prepared_item
    prepared_interpretations = list(deduplicated.values())
    if not prepared_interpretations:
        result = _typed_early_result(
            workspace,
            query_id,
            search_id,
            ResultStatus.INSUFFICIENT_QUERY,
            ("fewer_than_three_usable_events_across_interpretations",),
            "At least three stable, usable pitched events are required.",
            analysis.quality.warnings,
        )
        workspace.finish_analysis_run(run_id)
        return _persist_result(workspace, search_id, result, run_id)
    index, index_artifact_id = load_index(workspace, index_id)
    if target_id is not None:
        scoped_entries = tuple(
            entry for entry in index.target_entries if entry.target_id == target_id
        )
        index = index.model_copy(update={"target_entries": scoped_entries})
    if not index.target_entries:
        reason = (
            "requested_target_not_present_in_index"
            if target_id is not None
            else "index_has_no_usable_pitched_voice_candidates"
        )
        explanation = (
            "The requested target has no indexed pitched audio or MIDI views."
            if target_id is not None
            else "The index contains no source or derived monophonic pitched voice candidates."
        )
        result = _typed_early_result(
            workspace,
            query_id,
            search_id,
            ResultStatus.UNSUPPORTED_TARGET,
            (reason,),
            explanation,
            analysis.quality.warnings,
        )
        workspace.finish_analysis_run(run_id)
        return _persist_result(workspace, search_id, result, run_id)
    map_row = workspace.database.fetchone(
        "SELECT artifact_id FROM time_maps WHERE time_map_id = ?", (analysis.time_map_id,)
    )
    if map_row is None or map_row["artifact_id"] is None:
        result = _typed_early_result(
            workspace,
            query_id,
            search_id,
            ResultStatus.SOURCE_TIME_UNMAPPABLE,
            ("missing_query_time_map",),
            "The analysis derivative cannot be mapped to original query samples.",
            analysis.quality.warnings,
        )
        workspace.finish_analysis_run(run_id)
        return _persist_result(workspace, search_id, result, run_id)
    representations = {
        (entry.target_id, entry.view_id or entry.voice_id): load_track_representation(
            workspace, entry.representation_artifact_id
        )
        for entry in index.target_entries
    }
    representation_artifact_ids = {
        (entry.target_id, entry.view_id or entry.voice_id): entry.representation_artifact_id
        for entry in index.target_entries
    }
    computations: list[
        tuple[
            AlignmentComputation,
            CandidateEvidence,
            TargetTrackRepresentation,
            QueryInterpretation,
            str,
            str,
        ]
    ] = []
    for prepared, prepared_artifact_id, source_interpretation_id in prepared_interpretations:
        candidates = _candidate_retrieval(
            prepared,
            index,
            representations,
            limit=24,
        )
        for candidate in candidates:
            selector = candidate.view_id or candidate.voice_id
            representation = representations[(candidate.target_id, selector)]
            try:
                computation = align_candidate(prepared, representation, candidate)
            except ValueError:
                continue
            computations.append(
                (
                    computation,
                    candidate,
                    representation,
                    prepared,
                    prepared_artifact_id,
                    source_interpretation_id,
                )
            )
    best_by_window: dict[
        tuple[str, str, int, int],
        tuple[
            AlignmentComputation,
            CandidateEvidence,
            TargetTrackRepresentation,
            QueryInterpretation,
            str,
            str,
        ],
    ] = {}
    for alignment_item in computations:
        computation, candidate, _, _, _, _ = alignment_item
        key = (
            candidate.target_id,
            candidate.voice_id,
            computation.start_index,
            computation.end_index,
        )
        previous_alignment = best_by_window.get(key)
        if (
            previous_alignment is None
            or computation.score.value > previous_alignment[0].score.value
        ):
            best_by_window[key] = alignment_item
    computations = sorted(
        best_by_window.values(), key=lambda value: value[0].score.value, reverse=True
    )
    if not computations:
        result = _typed_early_result(
            workspace,
            query_id,
            search_id,
            ResultStatus.NO_RELIABLE_MATCH,
            ("no_alignable_candidates",),
            "No target window could be aligned to the query.",
            analysis.quality.warnings,
        )
        workspace.finish_analysis_run(run_id)
        return _persist_result(workspace, search_id, result, run_id)

    alignment_records: list[
        tuple[
            AlignmentComputation,
            CandidateEvidence,
            TargetTrackRepresentation,
            QueryInterpretation,
            str,
            str,
            str,
        ]
    ] = []
    for rank, (
        computation,
        candidate,
        representation,
        prepared,
        prepared_artifact_id,
        source_interpretation_id,
    ) in enumerate(computations[:24], start=1):
        selector = candidate.view_id or candidate.voice_id
        location = _target_location(
            workspace,
            representation=representation,
            representation_artifact_id=representation_artifact_ids[(candidate.target_id, selector)],
            start_index=computation.start_index,
            end_index=computation.end_index,
        )
        alignment_id = new_id("alignment")
        artifact = AlignmentArtifact(
            schema_version=(
                "0.0.2"
                if location.coordinate_kind == TargetCoordinateKind.DECODED_SAMPLES
                else "0.0.1"
            ),
            alignment_id=alignment_id,
            search_id=search_id,
            target_id=location.target.target_id,
            track_index=candidate.track_index,
            target_event_start_index=computation.start_index,
            target_event_end_index=computation.end_index,
            coordinate_kind=location.coordinate_kind,
            target_range=location.tick_range,
            target_sample_range=location.sample_range,
            target_time_range=location.time_range,
            candidate=candidate,
            aligned_events=computation.pairs,
            transformations=computation.transformations,
            score=computation.score,
            contour_dtw_cost=computation.dtw_cost,
            voice_hypothesis=_voice_summary(representation.voice_hypothesis),
            audio_voice_hypothesis=representation.audio_voice_hypothesis,
            query_interpretation_id=prepared.interpretation_id,
            query_interpretation_artifact_id=prepared_artifact_id,
        )
        artifact_id = workspace.store_json_artifact(
            artifact, artifact_type="search.alignment", analysis_run_id=run_id
        )
        workspace.database.execute(
            """
            INSERT INTO alignments(alignment_id, search_id, target_id, rank, artifact_id,
                                   uncalibrated_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alignment_id,
                search_id,
                location.target.target_id,
                rank,
                artifact_id,
                computation.score.value,
                utc_now(),
            ),
        )
        alignment_records.append(
            (
                computation,
                candidate,
                representation,
                prepared,
                prepared_artifact_id,
                source_interpretation_id,
                alignment_id,
            )
        )
    (
        best,
        best_candidate,
        best_representation,
        best_interpretation,
        best_interpretation_artifact_id,
        best_source_interpretation_id,
        best_alignment_id,
    ) = alignment_records[0]
    workspace.database.execute(
        "UPDATE searches SET interpretation_id = ? WHERE search_id = ?",
        (best_source_interpretation_id, search_id),
    )
    best_selector = best_candidate.view_id or best_candidate.voice_id
    best_representation_artifact_id = representation_artifact_ids[
        (best_candidate.target_id, best_selector)
    ]
    best_location = _target_location(
        workspace,
        representation=best_representation,
        representation_artifact_id=best_representation_artifact_id,
        start_index=best.start_index,
        end_index=best.end_index,
    )
    best_target = best_location.target
    best_target_artifact_id = best_location.target_artifact_id
    best_target_row = workspace.database.fetchone(
        "SELECT source_asset_id FROM targets WHERE target_id = ?", (best_target.target_id,)
    )
    query_row = workspace.database.fetchone(
        "SELECT original_asset_id FROM queries WHERE query_id = ?", (query_id,)
    )
    if best_target_row is None or query_row is None:
        raise RuntimeError("selected evidence rows disappeared during search")
    target_asset = workspace.asset_row(str(best_target_row["source_asset_id"]))
    query_asset = workspace.asset_row(str(query_row["original_asset_id"]))
    best_range = best_location.tick_range
    best_sample_range = best_location.sample_range
    best_time_range = best_location.time_range
    distinct_alternatives: list[MatchAlternative] = []
    best_signature = tuple(best_representation.event_ids[best.start_index : best.end_index])
    best_signature_set = set(best_signature)
    seen_ranges: set[tuple[str, tuple[str, ...]]] = {(best_candidate.target_id, best_signature)}
    audio_occurrences: list[tuple[str, SampleRange]] = (
        [(best_candidate.target_id, best_sample_range)] if best_sample_range is not None else []
    )
    competing_occurrence_score = 0.0
    close_distinct_voice = False
    for (
        computation,
        candidate,
        representation,
        _alternative_interpretation,
        _,
        alternative_source_interpretation_id,
        _,
    ) in alignment_records[1:]:
        alternative_selector = candidate.view_id or candidate.voice_id
        alternative_location = _target_location(
            workspace,
            representation=representation,
            representation_artifact_id=representation_artifact_ids[
                (candidate.target_id, alternative_selector)
            ],
            start_index=computation.start_index,
            end_index=computation.end_index,
        )
        range_key = (
            candidate.target_id,
            tuple(representation.event_ids[computation.start_index : computation.end_index]),
        )
        if range_key in seen_ranges:
            continue
        seen_ranges.add(range_key)
        alternative_signature_set = set(range_key[1])
        event_overlap = len(best_signature_set & alternative_signature_set) / max(
            1, min(len(best_signature_set), len(alternative_signature_set))
        )
        same_audio_occurrence = alternative_location.sample_range is not None and any(
            candidate.target_id == occurrence_target_id
            and _sample_overlap(occurrence_range, alternative_location.sample_range) >= 0.5
            for occurrence_target_id, occurrence_range in audio_occurrences
        )
        if same_audio_occurrence:
            # Different extraction paths often describe the same real-world
            # passage.  Treat them as corroborating views rather than distinct
            # occurrences that would manufacture an ambiguous result.
            continue
        same_source_voice_path = same_audio_occurrence or (
            best_range is not None
            and alternative_location.tick_range is not None
            and candidate.target_id == best_candidate.target_id
            and candidate.track_index == best_candidate.track_index
            and candidate.source_channel == best_candidate.source_channel
            and event_overlap >= 0.5
        )
        if not same_source_voice_path and competing_occurrence_score == 0.0:
            competing_occurrence_score = computation.score.value
        if (
            not same_source_voice_path
            and best.score.value - computation.score.value < 4.0
            and candidate.voice_id != best_candidate.voice_id
        ):
            close_distinct_voice = True
        alternative = MatchAlternative(
            rank=len(distinct_alternatives) + 2,
            target_id=alternative_location.target.target_id,
            target_name=alternative_location.target.name,
            track_index=candidate.track_index,
            coordinate_kind=alternative_location.coordinate_kind,
            target_range=alternative_location.tick_range,
            target_sample_range=alternative_location.sample_range,
            target_time_range=alternative_location.time_range,
            score=computation.score,
            transformations=computation.transformations,
            candidate=candidate,
            source_channel=candidate.source_channel,
            voice_hypothesis=_voice_summary(representation.voice_hypothesis),
            audio_voice_hypothesis=representation.audio_voice_hypothesis,
            query_interpretation_id=alternative_source_interpretation_id,
        )
        if alternative_location.sample_range is not None:
            audio_occurrences.append((candidate.target_id, alternative_location.sample_range))
        if len(distinct_alternatives) < 5:
            distinct_alternatives.append(alternative)
    score_margin = best.score.value - competing_occurrence_score
    if best.score.value < 52.0:
        status = ResultStatus.NO_RELIABLE_MATCH
        reason_codes = ("top_score_below_acceptance_floor",)
        explanation = "The best alignment is too weak to ground an edit."
    elif score_margin < 4.0:
        status = ResultStatus.AMBIGUOUS
        reason_codes = ("top_two_score_margin_too_small",)
        explanation = (
            "Multiple distinct real-audio passages explain the query nearly equally well."
            if best_location.coordinate_kind == TargetCoordinateKind.DECODED_SAMPLES
            else (
                "Multiple distinct passages or source voices explain the query nearly equally well."
            )
        )
    else:
        status = ResultStatus.ACCEPTED
        reason_codes = ("score_and_margin_passed",)
        explanation = "The selected passage has sufficient alignment evidence and separation."
    matched_target_indices = sorted(
        {
            index
            for pair in best.pairs
            for index in pair.target_event_indices
            if best.start_index <= index < best.end_index
        }
    )
    if best_location.coordinate_kind == TargetCoordinateKind.MIDI_TICKS:
        velocities = [best_representation.velocities[index] for index in matched_target_indices]
        velocity_floor = median(velocities) if velocities else 0
        strong_ticks = tuple(
            best_representation.start_ticks[index]
            for index in matched_target_indices
            if best_representation.velocities[index] >= velocity_floor
        )
        strong_samples: tuple[int, ...] = ()
    else:
        confidences = (
            [best_representation.event_confidences[index] for index in matched_target_indices]
            if best_representation.event_confidences
            else [1.0 for _ in matched_target_indices]
        )
        confidence_floor = median(confidences) if confidences else 0.0
        strong_ticks = ()
        strong_samples = tuple(
            best_representation.start_samples[index]
            for index, confidence in zip(matched_target_indices, confidences, strict=True)
            if confidence >= confidence_floor
        )
    selected_entry = next(
        entry
        for entry in index.target_entries
        if entry.target_id == best_candidate.target_id and entry.voice_id == best_candidate.voice_id
    )
    provenance = ProvenanceEvidence(
        query_asset_id=str(query_asset["asset_id"]),
        query_asset_sha256=str(query_asset["sha256"]),
        query_analysis_artifact_id=query_analysis_artifact_id,
        query_interpretation_id=best_source_interpretation_id,
        target_asset_id=str(target_asset["asset_id"]),
        target_asset_sha256=str(target_asset["sha256"]),
        target_artifact_id=best_target_artifact_id,
        index_id=index_id,
        index_artifact_id=index_artifact_id,
        search_id=search_id,
        alignment_id=best_alignment_id,
        target_representation_artifact_id=selected_entry.representation_artifact_id,
        target_voice_id=(
            best_candidate.voice_id
            if best_location.coordinate_kind == TargetCoordinateKind.MIDI_TICKS
            else None
        ),
        search_interpretation_id=best_interpretation.interpretation_id,
        search_interpretation_artifact_id=best_interpretation_artifact_id,
        target_decode_manifest_artifact_id=best_location.decode_manifest_artifact_id,
        target_time_map_artifact_id=best_location.time_map_artifact_id,
        target_analysis_artifact_id=best_location.target_analysis_artifact_id,
        target_evidence_artifact_id=best_location.evidence_artifact_id,
        target_view_id=best_representation.view_id,
    )
    result_warnings = list(analysis.quality.warnings)
    excluded_event_ids = tuple(
        item.source_event_id for item in best_interpretation.event_usability if not item.usable
    )
    if excluded_event_ids:
        result_warnings.append(
            QualityWarning(
                code="unusable_query_events_excluded",
                severity="info",
                message=(
                    f"Excluded {len(excluded_event_ids)} low-confidence or unstable query "
                    "events from retrieval while retaining their source evidence."
                ),
            )
        )
    if (
        best_representation.voice_hypothesis is not None
        and best_representation.voice_hypothesis.hypothesis_kind == "derived_monophonic"
    ):
        result_warnings.append(
            QualityWarning(
                code="derived_voice_hypothesis_selected",
                severity="info",
                message=(
                    "The selected match uses a transparent monophonic hypothesis derived "
                    "from a polyphonic source track; it is not an authoritative melody label."
                ),
            )
        )
    if close_distinct_voice:
        alternatives_are_derived = (
            best_representation.voice_hypothesis is not None
            and best_representation.voice_hypothesis.hypothesis_kind == "derived_monophonic"
        ) or (
            any(
                alternative.voice_hypothesis is not None
                and alternative.voice_hypothesis.hypothesis_kind == "derived_monophonic"
                for alternative in distinct_alternatives
            )
        )
        result_warnings.append(
            QualityWarning(
                code=(
                    "close_audio_view_hypotheses"
                    if best_location.coordinate_kind == TargetCoordinateKind.DECODED_SAMPLES
                    else (
                        "close_derived_voice_hypotheses"
                        if alternatives_are_derived
                        else "close_source_voice_candidates"
                    )
                ),
                severity="warning",
                message=(
                    "Several target evidence views explain distinct passages with similar scores."
                    if best_location.coordinate_kind == TargetCoordinateKind.DECODED_SAMPLES
                    else "Several source voice candidates explain the query with similar scores."
                ),
            )
        )
    result = SearchResult(
        schema_version=(
            "0.0.2"
            if best_location.coordinate_kind == TargetCoordinateKind.DECODED_SAMPLES
            else "0.0.1"
        ),
        search_id=search_id,
        status=status,
        reason_codes=reason_codes,
        status_explanation=explanation,
        selected_target_id=best_target.target_id,
        selected_target_name=best_target.name,
        selected_track_index=best_candidate.track_index,
        target_range=best_range,
        target_sample_range=best_sample_range,
        target_time_range=best_time_range,
        query_source_range=_source_range(workspace, query_id),
        score=best.score,
        transformations=best.transformations,
        aligned_events=best.pairs,
        strong_rhythmic_target_ticks=strong_ticks,
        strong_rhythmic_target_samples=strong_samples,
        alternatives=tuple(distinct_alternatives),
        quality_warnings=tuple(result_warnings),
        provenance=provenance,
        selected_source_channel=best_candidate.source_channel,
        selected_voice_hypothesis=_voice_summary(best_representation.voice_hypothesis),
        selected_audio_voice_hypothesis=best_representation.audio_voice_hypothesis,
        selected_target_coordinate_kind=best_location.coordinate_kind,
        selected_target_original_name=(
            best_target.original_name
            if isinstance(best_target, AudioTargetArtifact)
            else str(target_asset["original_name"])
        ),
        selected_target_view_id=best_representation.view_id,
        selected_target_source_role=(
            best_representation.audio_voice_hypothesis.source_role
            if best_representation.audio_voice_hypothesis is not None
            else None
        ),
        selected_target_backend=(
            best_representation.audio_voice_hypothesis.extraction_backend
            if best_representation.audio_voice_hypothesis is not None
            else None
        ),
        selected_query_interpretation_id=best_source_interpretation_id,
        evaluated_query_interpretation_ids=tuple(evaluated_source_interpretation_ids),
        usable_query_event_ids=tuple(event.event_id for event in best_interpretation.events),
        excluded_query_event_ids=excluded_event_ids,
    )
    workspace.finish_analysis_run(run_id)
    return _persist_result(workspace, search_id, result, run_id)


def load_search_result(workspace: Workspace, search_id: str) -> tuple[SearchResult, str]:
    row = workspace.database.fetchone(
        "SELECT result_artifact_id FROM searches WHERE search_id = ?", (search_id,)
    )
    if row is None:
        raise KeyError(f"unknown search: {search_id}")
    if row["result_artifact_id"] is None:
        raise ValueError(f"search is not complete: {search_id}")
    artifact_id = str(row["result_artifact_id"])
    return SearchResult.model_validate(workspace.load_artifact_json(artifact_id)), artifact_id


def export_search_result(workspace: Workspace, search_id: str, destination: Path) -> None:
    result, artifact_id = load_search_result(workspace, search_id)
    destination.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
