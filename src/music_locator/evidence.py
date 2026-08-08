"""End-to-end evidence graph integrity checks."""

from __future__ import annotations

import hashlib

from music_locator.models import (
    AlignmentArtifact,
    AnalysisDerivativeEvidence,
    AudioTargetArtifact,
    DecodedReferenceManifest,
    IndexSnapshot,
    QueryAnalysisArtifact,
    QueryInterpretation,
    SampleTimeMap,
    SearchResult,
    TargetArtifact,
    TargetCoordinateKind,
    TargetEventEvidenceArtifact,
    TargetRepresentationBundle,
    TargetTimeMap,
    TargetTrackRepresentation,
)
from music_locator.search import load_search_result
from music_locator.storage import Workspace


def _verify_audio_target_evidence(
    workspace: Workspace,
    *,
    target: AudioTargetArtifact,
    representation: TargetTrackRepresentation,
    result: SearchResult,
) -> tuple[int, int]:
    """Verify the selected audio view back through observations to decoded PCM."""

    # SearchResult is intentionally not imported only to avoid widening the public helper.
    provenance = result.provenance
    if provenance is None:
        raise ValueError("audio result is missing provenance")
    if representation.coordinate_kind != TargetCoordinateKind.DECODED_SAMPLES:
        raise ValueError("audio target selected a non-audio representation")
    hypothesis = representation.audio_voice_hypothesis
    if hypothesis is None:
        raise ValueError("audio target representation has no source hypothesis")
    if result.selected_audio_voice_hypothesis != hypothesis:
        raise ValueError("selected audio view summary differs from its representation")
    if result.selected_target_view_id != hypothesis.view_id:
        raise ValueError("selected audio view identity differs from provenance")
    bundle_artifact_id = provenance.target_analysis_artifact_id
    if bundle_artifact_id is None:
        raise ValueError("audio result is missing its representation bundle")
    bundle_row = workspace.artifact_row(bundle_artifact_id)
    _verify_blob(workspace, str(bundle_row["sha256"]))
    bundle = TargetRepresentationBundle.model_validate(
        workspace.load_artifact_json(bundle_artifact_id)
    )
    if bundle.target_id != target.target_id:
        raise ValueError("audio bundle points at a different target")
    if provenance.target_decode_manifest_artifact_id not in {
        None,
        bundle.decode_manifest_artifact_id,
    }:
        raise ValueError("audio decode manifest differs between result and bundle")
    if provenance.target_time_map_artifact_id not in {
        None,
        bundle.target_time_map_artifact_id,
    }:
        raise ValueError("audio target time map differs between result and bundle")
    if provenance.target_representation_artifact_id not in bundle.representation_artifact_ids:
        raise ValueError("selected audio view is not present in its bundle")
    checked_artifacts = 1
    for artifact_id in (
        bundle.decode_manifest_artifact_id,
        bundle.target_time_map_artifact_id,
        *bundle.derivative_artifact_ids,
        *bundle.evidence_artifact_ids,
        *bundle.backend_invocation_artifact_ids,
        *bundle.representation_artifact_ids,
    ):
        row = workspace.artifact_row(artifact_id)
        _verify_blob(workspace, str(row["sha256"]))
        checked_artifacts += 1
    manifest = DecodedReferenceManifest.model_validate(
        workspace.load_artifact_json(bundle.decode_manifest_artifact_id)
    )
    target_time_map = TargetTimeMap.model_validate(
        workspace.load_artifact_json(bundle.target_time_map_artifact_id)
    )
    if (
        manifest.target_id != target.target_id
        or manifest.source_asset_id != target.asset_id
        or manifest.source_sha256 != target.source_sha256
    ):
        raise ValueError("decoded reference does not descend from the target source asset")
    if (
        target_time_map.decoded_stream_id != bundle.decoded_stream_id
        or target_time_map.decoded_sample_rate != manifest.decoded_sample_rate
        or target_time_map.decoded_frame_count != manifest.decoded_frame_count
    ):
        raise ValueError("audio target time map differs from the decoded reference")
    ffprobe_row = workspace.artifact_row(manifest.ffprobe_artifact_id)
    _verify_blob(workspace, str(ffprobe_row["sha256"]))
    checked_artifacts += 1
    decoded_asset = workspace.asset_row(manifest.decoded_asset_id)
    _verify_blob(workspace, str(decoded_asset["sha256"]))
    if str(decoded_asset["sha256"]) != manifest.decoded_sha256:
        raise ValueError("decoded reference digest differs from its manifest")
    checked_assets = 1
    selected_evidence_id = hypothesis.evidence_artifact_id
    if selected_evidence_id != provenance.target_evidence_artifact_id:
        raise ValueError("selected evidence artifact differs from result provenance")
    evidence = TargetEventEvidenceArtifact.model_validate(
        workspace.load_artifact_json(selected_evidence_id)
    )
    if (
        evidence.target_id != target.target_id
        or evidence.source_role != hypothesis.source_role
        or evidence.backend != hypothesis.extraction_backend
    ):
        raise ValueError("audio event evidence differs from its view hypothesis")
    if evidence.derivative_artifact_id not in bundle.derivative_artifact_ids:
        raise ValueError("audio evidence points outside the representation derivative bundle")
    if hypothesis.source_derivative_artifact_id != evidence.derivative_artifact_id:
        raise ValueError("audio view and observation point at different derivatives")
    selected_derivative = AnalysisDerivativeEvidence.model_validate(
        workspace.load_artifact_json(evidence.derivative_artifact_id)
    )
    if selected_derivative.derivative_id != evidence.derivative_id:
        raise ValueError("audio evidence derivative identity differs from its artifact")
    valid_refs = {
        *(frame.frame_id for frame in evidence.pitch_frames),
        *(onset.onset_id for onset in evidence.onsets),
        *(candidate.candidate_id for candidate in evidence.note_candidates),
    }
    if any(reference not in valid_refs for reference in representation.source_evidence_refs):
        raise ValueError("audio view event does not map to retained observation evidence")
    if len(representation.event_ids) != len(representation.start_samples):
        raise ValueError("audio view sample evidence does not cover every event")
    for start, end in zip(representation.start_samples, representation.end_samples, strict=True):
        if end <= start or end > manifest.decoded_frame_count:
            raise ValueError("audio view event support is outside the decoded reference")
    checked_asset_ids = {manifest.decoded_asset_id}
    derivative_maps: dict[str, SampleTimeMap] = {}
    derivative_by_artifact: dict[str, AnalysisDerivativeEvidence] = {}
    for derivative_artifact_id in bundle.derivative_artifact_ids:
        derivative = AnalysisDerivativeEvidence.model_validate(
            workspace.load_artifact_json(derivative_artifact_id)
        )
        derivative_by_artifact[derivative_artifact_id] = derivative
        if derivative.target_id != target.target_id:
            raise ValueError("analysis derivative points at a different target")
        map_row = workspace.artifact_row(derivative.sample_time_map_artifact_id)
        _verify_blob(workspace, str(map_row["sha256"]))
        mapping = SampleTimeMap.model_validate(
            workspace.load_artifact_json(derivative.sample_time_map_artifact_id)
        )
        derivative_maps[derivative_artifact_id] = mapping
        if (
            mapping.source_stream_id != derivative.source_stream_id
            or mapping.derived_stream_id != derivative.derived_stream_id
            or mapping.source_sample_rate != derivative.source_sample_rate
            or mapping.derived_sample_rate != derivative.derived_sample_rate
            or mapping.segments[0].source_start_sample != 0
            or mapping.segments[0].derived_start_sample != 0
            or mapping.segments[-1].source_end_sample != derivative.source_frame_count
            or mapping.segments[-1].derived_end_sample != derivative.derived_frame_count
        ):
            raise ValueError("analysis derivative time map differs from its stream evidence")
        source_stream = workspace.database.fetchone(
            "SELECT * FROM audio_streams WHERE stream_id = ?", (derivative.source_stream_id,)
        )
        derived_stream = workspace.database.fetchone(
            "SELECT * FROM audio_streams WHERE stream_id = ?", (derivative.derived_stream_id,)
        )
        if source_stream is None or derived_stream is None:
            raise ValueError("analysis derivative references a missing audio stream")
        if (
            str(source_stream["asset_id"]) != derivative.source_asset_id
            or int(source_stream["sample_rate"]) != derivative.source_sample_rate
            or int(source_stream["frame_count"]) != derivative.source_frame_count
            or str(derived_stream["asset_id"]) != derivative.derived_asset_id
            or int(derived_stream["sample_rate"]) != derivative.derived_sample_rate
            or int(derived_stream["frame_count"]) != derivative.derived_frame_count
        ):
            raise ValueError("analysis derivative stream rows differ from its evidence")
        for asset_id in (derivative.source_asset_id, derivative.derived_asset_id):
            asset_row = workspace.asset_row(asset_id)
            _verify_blob(workspace, str(asset_row["sha256"]))
            if asset_id not in checked_asset_ids:
                checked_asset_ids.add(asset_id)
                checked_assets += 1
        derived_asset = workspace.asset_row(derivative.derived_asset_id)
        if str(derived_asset["sha256"]) != derivative.artifact_sha256:
            raise ValueError("analysis derivative audio digest differs from its evidence")
        invocation_id = derivative.parameters.get("invocation_artifact_id")
        if isinstance(invocation_id, str):
            invocation_row = workspace.artifact_row(invocation_id)
            _verify_blob(workspace, str(invocation_row["sha256"]))
            checked_artifacts += 1
        checked_artifacts += 1
    for evidence_artifact_id in bundle.evidence_artifact_ids:
        child = TargetEventEvidenceArtifact.model_validate(
            workspace.load_artifact_json(evidence_artifact_id)
        )
        if (
            child.target_id != target.target_id
            or child.derivative_artifact_id not in derivative_by_artifact
            or derivative_by_artifact[child.derivative_artifact_id].derivative_id
            != child.derivative_id
        ):
            raise ValueError("audio observation does not descend from a bundled derivative")
        invocation_id = child.backend_config.get("invocation_artifact_id")
        if isinstance(invocation_id, str):
            invocation_row = workspace.artifact_row(invocation_id)
            _verify_blob(workspace, str(invocation_row["sha256"]))
            checked_artifacts += 1
        if child.activation_grid is not None:
            raw = workspace.artifact_row(child.activation_grid.raw_artifact_id)
            _verify_blob(workspace, str(raw["sha256"]))
            if str(raw["sha256"]) != child.activation_grid.raw_artifact_sha256:
                raise ValueError("raw activation grid digest differs from its evidence")
            checked_artifacts += 1
    selected_mapping: SampleTimeMap | None = None
    if selected_derivative.source_stream_id != bundle.decoded_stream_id:
        for derivative_artifact_id, derivative in derivative_by_artifact.items():
            if (
                derivative.source_stream_id == bundle.decoded_stream_id
                and derivative.derived_stream_id == selected_derivative.source_stream_id
            ):
                selected_mapping = derivative_maps[derivative_artifact_id]
                break
        if selected_mapping is None:
            raise ValueError("audio stem observation has no map back to decoded reference PCM")
    observation_support: dict[str, tuple[int, int]] = {
        **{
            frame.frame_id: (frame.reference_start_sample, frame.reference_end_sample)
            for frame in evidence.pitch_frames
        },
        **{
            onset.onset_id: (onset.reference_sample, onset.reference_sample)
            for onset in evidence.onsets
        },
        **{
            candidate.candidate_id: (
                candidate.reference_start_sample,
                candidate.reference_end_sample,
            )
            for candidate in evidence.note_candidates
        },
    }
    for view_start, view_end, evidence_ref in zip(
        representation.start_samples,
        representation.end_samples,
        representation.source_evidence_refs,
        strict=True,
    ):
        support_start, support_end = observation_support[evidence_ref]
        if selected_mapping is not None:
            support_start = selected_mapping.derived_to_source(support_start)
            support_end = selected_mapping.derived_to_source(support_end)
        if support_end < view_start or support_start > view_end:
            raise ValueError("audio view event does not overlap its mapped observation support")
    target_sample_range = result.target_sample_range
    target_time_range = result.target_time_range
    if target_sample_range is None or target_time_range is None:
        raise ValueError("audio result is missing decoded-sample or media-time coordinates")
    if target_sample_range.sample_rate != manifest.decoded_sample_rate:
        raise ValueError("audio result sample rate differs from the decoded reference")
    reconstructed_start = target_time_map.reference_sample_to_media_microseconds(
        target_sample_range.start_sample
    )
    reconstructed_end = target_time_map.reference_sample_to_media_microseconds(
        target_sample_range.end_sample
    )
    if (
        reconstructed_start != target_time_range.start_microsecond
        or reconstructed_end != target_time_range.end_microsecond
    ):
        raise ValueError("audio result media range does not reconstruct from source samples")
    if result.target_range is not None:
        raise ValueError("audio result contains a fabricated MIDI tick range")
    if result.strong_rhythmic_target_ticks:
        raise ValueError("audio result contains fabricated rhythmic MIDI ticks")
    return checked_artifacts, checked_assets


def _verify_blob(workspace: Workspace, digest: str) -> None:
    data = workspace.blobs.get_bytes(digest)
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise ValueError(f"blob digest mismatch: expected {digest}, got {actual}")


def verify_search_evidence(workspace: Workspace, search_id: str) -> dict[str, int | str | bool]:
    """Traverse and hash-check every durable reference exposed by a search result."""

    result, result_artifact_id = load_search_result(workspace, search_id)
    result_artifact = workspace.artifact_row(result_artifact_id)
    _verify_blob(workspace, str(result_artifact["sha256"]))
    checked_artifacts = 1
    checked_assets = 0
    if result.provenance is not None:
        artifact_ids = (
            result.provenance.query_analysis_artifact_id,
            result.provenance.target_artifact_id,
            result.provenance.index_artifact_id,
        )
        for artifact_id in artifact_ids:
            row = workspace.artifact_row(artifact_id)
            _verify_blob(workspace, str(row["sha256"]))
            checked_artifacts += 1
        query_analysis = QueryAnalysisArtifact.model_validate(
            workspace.load_artifact_json(result.provenance.query_analysis_artifact_id)
        )
        if query_analysis.original_asset_id != result.provenance.query_asset_id:
            raise ValueError("query analysis does not reference the original query asset")
        analysis_asset_row = workspace.asset_row(query_analysis.analysis_asset_id)
        _verify_blob(workspace, str(analysis_asset_row["sha256"]))
        checked_assets += 1
        interpretation_row = workspace.database.fetchone(
            "SELECT artifact_id FROM interpretations WHERE interpretation_id = ?",
            (result.provenance.query_interpretation_id,),
        )
        if interpretation_row is None:
            raise ValueError("selected query interpretation row is missing")
        interpretation_artifact_id = str(interpretation_row["artifact_id"])
        interpretation_artifact = workspace.artifact_row(interpretation_artifact_id)
        _verify_blob(workspace, str(interpretation_artifact["sha256"]))
        source_interpretation = QueryInterpretation.model_validate(
            workspace.load_artifact_json(interpretation_artifact_id)
        )
        interpretation = source_interpretation
        checked_artifacts += 1
        if result.provenance.search_interpretation_artifact_id is not None:
            search_interpretation_artifact = workspace.artifact_row(
                result.provenance.search_interpretation_artifact_id
            )
            _verify_blob(workspace, str(search_interpretation_artifact["sha256"]))
            interpretation = QueryInterpretation.model_validate(
                workspace.load_artifact_json(result.provenance.search_interpretation_artifact_id)
            )
            if interpretation.source_interpretation_id != source_interpretation.interpretation_id:
                raise ValueError("search interpretation does not reference its source evidence")
            if interpretation.interpretation_id != result.provenance.search_interpretation_id:
                raise ValueError("search interpretation identity differs from provenance")
            source_event_ids = {event.event_id for event in source_interpretation.events}
            if any(event.event_id not in source_event_ids for event in interpretation.events):
                raise ValueError("search interpretation invented a query event")
            usability_ids = {item.source_event_id for item in interpretation.event_usability}
            if usability_ids != source_event_ids:
                raise ValueError("search interpretation usability does not cover source events")
            usable_event_ids = tuple(
                item.source_event_id for item in interpretation.event_usability if item.usable
            )
            if tuple(event.event_id for event in interpretation.events) != usable_event_ids:
                raise ValueError(
                    "search interpretation events differ from the events marked usable"
                )
            excluded_event_ids = tuple(
                item.source_event_id for item in interpretation.event_usability if not item.usable
            )
            if result.selected_query_interpretation_id != (source_interpretation.interpretation_id):
                raise ValueError("result selected interpretation differs from provenance")
            if result.usable_query_event_ids != usable_event_ids:
                raise ValueError("result usable query events differ from search evidence")
            if result.excluded_query_event_ids != excluded_event_ids:
                raise ValueError("result excluded query events differ from search evidence")
            analysis_interpretation_ids = set(query_analysis.interpretation_ids)
            if (
                not set(result.evaluated_query_interpretation_ids) <= analysis_interpretation_ids
                or source_interpretation.interpretation_id
                not in result.evaluated_query_interpretation_ids
            ):
                raise ValueError("result evaluated interpretations differ from analysis evidence")
            checked_artifacts += 1
        for child_artifact_id in (
            query_analysis.pitch_frames_artifact_id,
            query_analysis.onsets_artifact_id,
        ):
            child_row = workspace.artifact_row(child_artifact_id)
            _verify_blob(workspace, str(child_row["sha256"]))
            checked_artifacts += 1
        time_map_row = workspace.database.fetchone(
            "SELECT artifact_id FROM time_maps WHERE time_map_id = ?",
            (query_analysis.time_map_id,),
        )
        if time_map_row is None or time_map_row["artifact_id"] is None:
            raise ValueError("query time map evidence is missing")
        time_map_artifact = workspace.artifact_row(str(time_map_row["artifact_id"]))
        _verify_blob(workspace, str(time_map_artifact["sha256"]))
        sample_time_map = SampleTimeMap.model_validate(
            workspace.load_artifact_json(str(time_map_row["artifact_id"]))
        )
        if (
            sample_time_map.source_stream_id != query_analysis.original_stream_id
            or sample_time_map.derived_stream_id != query_analysis.analysis_stream_id
        ):
            raise ValueError("query time map stream endpoints do not match the analysis")
        derivative_row = workspace.database.fetchone(
            "SELECT * FROM derivatives WHERE derivative_id = ?",
            (query_analysis.derivative_id,),
        )
        if derivative_row is None or str(derivative_row["time_map_id"]) != (
            query_analysis.time_map_id
        ):
            raise ValueError("query derivative does not reference its time map")
        checked_artifacts += 1
        target_payload = workspace.load_artifact_json(result.provenance.target_artifact_id)
        if target_payload.get("kind") == "audio":
            target: TargetArtifact | AudioTargetArtifact = AudioTargetArtifact.model_validate(
                target_payload
            )
        else:
            target = TargetArtifact.model_validate(target_payload)
        if target.asset_id != result.provenance.target_asset_id:
            raise ValueError("target artifact does not reference the selected source asset")
        snapshot = IndexSnapshot.model_validate(
            workspace.load_artifact_json(result.provenance.index_artifact_id)
        )
        selected_entries = [
            entry
            for entry in snapshot.target_entries
            if entry.target_id == result.selected_target_id
            and (
                result.provenance.target_view_id is None
                or entry.view_id == result.provenance.target_view_id
            )
            and (
                result.provenance.target_view_id is not None
                or result.provenance.target_voice_id is None
                or entry.voice_id == result.provenance.target_voice_id
            )
            and (
                result.provenance.target_view_id is not None
                or result.provenance.target_voice_id is not None
                or entry.track_index == result.selected_track_index
            )
        ]
        if len(selected_entries) != 1:
            raise ValueError("index does not have exactly one selected target-track representation")
        representation_artifact_id = selected_entries[0].representation_artifact_id
        if (
            result.provenance.target_representation_artifact_id is not None
            and representation_artifact_id != result.provenance.target_representation_artifact_id
        ):
            raise ValueError("result provenance points at a different voice representation")
        representation_row = workspace.artifact_row(representation_artifact_id)
        _verify_blob(workspace, str(representation_row["sha256"]))
        representation = TargetTrackRepresentation.model_validate(
            workspace.load_artifact_json(representation_artifact_id)
        )
        hypothesis = representation.voice_hypothesis
        if isinstance(target, AudioTargetArtifact):
            audio_artifacts, audio_assets = _verify_audio_target_evidence(
                workspace,
                target=target,
                representation=representation,
                result=result,
            )
            checked_artifacts += audio_artifacts
            checked_assets += audio_assets
        if hypothesis is not None:
            if isinstance(target, AudioTargetArtifact):
                raise ValueError("audio target representation contains MIDI voice provenance")
            if hypothesis.voice_id != selected_entries[0].voice_id:
                raise ValueError("voice identity differs between index entry and representation")
            selected_summary = result.selected_voice_hypothesis
            if (
                selected_summary is None
                or selected_summary.voice_id != hypothesis.voice_id
                or selected_summary.extraction_method != hypothesis.extraction_method
                or selected_summary.extraction_version != hypothesis.extraction_version
                or selected_summary.quality != hypothesis.quality
                or selected_summary.note_count != len(hypothesis.note_provenance)
            ):
                raise ValueError("selected voice evidence differs from its representation")
            if result.selected_source_channel != hypothesis.source_channel:
                raise ValueError("selected source channel differs from voice provenance")
            if len(hypothesis.note_provenance) != len(representation.event_ids):
                raise ValueError("voice provenance does not cover every indexed note")
            target_notes = {note.event_id: note for note in target.notes}
            for index, note_provenance in enumerate(hypothesis.note_provenance):
                source_note = target_notes.get(note_provenance.source_event_id)
                if source_note is None:
                    raise ValueError("voice provenance refers to a missing source note")
                if note_provenance.candidate_event_index != index:
                    raise ValueError("voice provenance event indices are not contiguous")
                if representation.event_ids[index] != source_note.event_id:
                    raise ValueError("indexed voice event does not map to its source note")
                if (
                    representation.start_ticks[index] != note_provenance.indexed_start_tick
                    or representation.end_ticks[index] != note_provenance.indexed_end_tick
                ):
                    raise ValueError("indexed voice timing differs from note provenance")
                if (
                    source_note.track_index != hypothesis.source_track_index
                    or source_note.channel != hypothesis.source_channel
                    or source_note.start_tick != note_provenance.source_start_tick
                    or source_note.end_tick != note_provenance.source_end_tick
                ):
                    raise ValueError("source note fields differ from voice provenance")
                raw_on = note_provenance.source_note_on_raw_event_index
                raw_off = note_provenance.source_note_off_raw_event_index
                if raw_on is not None:
                    if raw_on >= len(target.raw_events):
                        raise ValueError("voice provenance note-on raw event is missing")
                    raw_event = target.raw_events[raw_on]
                    if raw_event.event_type != "note_on":
                        raise ValueError("voice provenance note-on index is not a note-on")
                if raw_off is not None:
                    if raw_off >= len(target.raw_events):
                        raise ValueError("voice provenance note-off raw event is missing")
                    raw_event = target.raw_events[raw_off]
                    if raw_event.event_type not in {"note_off", "note_on"}:
                        raise ValueError("voice provenance note-off index is not a note release")
        checked_artifacts += 1
        for asset_id in (
            result.provenance.query_asset_id,
            result.provenance.target_asset_id,
        ):
            row = workspace.asset_row(asset_id)
            _verify_blob(workspace, str(row["sha256"]))
            checked_assets += 1
        if result.provenance.alignment_id is not None:
            alignment_row = workspace.database.fetchone(
                "SELECT artifact_id FROM alignments WHERE alignment_id = ?",
                (result.provenance.alignment_id,),
            )
            if alignment_row is None:
                raise ValueError("selected alignment row is missing")
            artifact_id = str(alignment_row["artifact_id"])
            artifact_row = workspace.artifact_row(artifact_id)
            _verify_blob(workspace, str(artifact_row["sha256"]))
            alignment = AlignmentArtifact.model_validate(workspace.load_artifact_json(artifact_id))
            if alignment.target_range != result.target_range:
                raise ValueError("result target range diverges from alignment evidence")
            if alignment.target_sample_range != result.target_sample_range:
                raise ValueError("result target sample range diverges from alignment evidence")
            if alignment.target_time_range != result.target_time_range:
                raise ValueError("result target time range diverges from alignment evidence")
            if alignment.score != result.score:
                raise ValueError("result score diverges from alignment evidence")
            if (
                alignment.query_interpretation_id is not None
                and alignment.query_interpretation_id != interpretation.interpretation_id
            ):
                raise ValueError("alignment uses a different search interpretation")
            if alignment.query_interpretation_artifact_id != (
                result.provenance.search_interpretation_artifact_id
            ):
                raise ValueError("alignment points at a different search interpretation artifact")
            if (
                alignment.audio_voice_hypothesis != result.selected_audio_voice_hypothesis
                and alignment.alignment_id == result.provenance.alignment_id
            ):
                raise ValueError("alignment audio view differs from the selected result")
            query_indices = {
                index for pair in alignment.aligned_events for index in pair.query_event_indices
            }
            target_indices = {
                index for pair in alignment.aligned_events for index in pair.target_event_indices
            }
            if query_indices and max(query_indices) >= len(interpretation.events):
                raise ValueError("alignment refers to a missing query event")
            if target_indices and max(target_indices) >= len(representation.event_ids):
                raise ValueError("alignment refers to a missing target event")
            checked_artifacts += 1
        if str(workspace.asset_row(result.provenance.query_asset_id)["sha256"]) != (
            result.provenance.query_asset_sha256
        ):
            raise ValueError("query provenance digest differs from the asset row")
        if str(workspace.asset_row(result.provenance.target_asset_id)["sha256"]) != (
            result.provenance.target_asset_sha256
        ):
            raise ValueError("target provenance digest differs from the asset row")
    return {
        "search_id": search_id,
        "valid": True,
        "checked_artifacts": checked_artifacts,
        "checked_assets": checked_assets,
    }
