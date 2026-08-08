"""End-to-end evidence graph integrity checks."""

from __future__ import annotations

import hashlib

from music_locator.models import (
    AlignmentArtifact,
    IndexSnapshot,
    QueryAnalysisArtifact,
    QueryInterpretation,
    SampleTimeMap,
    TargetArtifact,
    TargetTrackRepresentation,
)
from music_locator.search import load_search_result
from music_locator.storage import Workspace


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
                workspace.load_artifact_json(
                    result.provenance.search_interpretation_artifact_id
                )
            )
            if interpretation.source_interpretation_id != source_interpretation.interpretation_id:
                raise ValueError("search interpretation does not reference its source evidence")
            if interpretation.interpretation_id != result.provenance.search_interpretation_id:
                raise ValueError("search interpretation identity differs from provenance")
            source_event_ids = {event.event_id for event in source_interpretation.events}
            if any(event.event_id not in source_event_ids for event in interpretation.events):
                raise ValueError("search interpretation invented a query event")
            usability_ids = {
                item.source_event_id for item in interpretation.event_usability
            }
            if usability_ids != source_event_ids:
                raise ValueError("search interpretation usability does not cover source events")
            usable_event_ids = tuple(
                item.source_event_id
                for item in interpretation.event_usability
                if item.usable
            )
            if tuple(event.event_id for event in interpretation.events) != usable_event_ids:
                raise ValueError(
                    "search interpretation events differ from the events marked usable"
                )
            excluded_event_ids = tuple(
                item.source_event_id
                for item in interpretation.event_usability
                if not item.usable
            )
            if result.selected_query_interpretation_id != (
                source_interpretation.interpretation_id
            ):
                raise ValueError("result selected interpretation differs from provenance")
            if result.usable_query_event_ids != usable_event_ids:
                raise ValueError("result usable query events differ from search evidence")
            if result.excluded_query_event_ids != excluded_event_ids:
                raise ValueError("result excluded query events differ from search evidence")
            analysis_interpretation_ids = set(query_analysis.interpretation_ids)
            if (
                not set(result.evaluated_query_interpretation_ids)
                <= analysis_interpretation_ids
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
        target = TargetArtifact.model_validate(
            workspace.load_artifact_json(result.provenance.target_artifact_id)
        )
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
                result.provenance.target_voice_id is None
                or entry.voice_id == result.provenance.target_voice_id
            )
            and (
                result.provenance.target_voice_id is not None
                or entry.track_index == result.selected_track_index
            )
        ]
        if len(selected_entries) != 1:
            raise ValueError("index does not have exactly one selected target-track representation")
        representation_artifact_id = selected_entries[0].representation_artifact_id
        if (
            result.provenance.target_representation_artifact_id is not None
            and representation_artifact_id
            != result.provenance.target_representation_artifact_id
        ):
            raise ValueError("result provenance points at a different voice representation")
        representation_row = workspace.artifact_row(representation_artifact_id)
        _verify_blob(workspace, str(representation_row["sha256"]))
        representation = TargetTrackRepresentation.model_validate(
            workspace.load_artifact_json(representation_artifact_id)
        )
        hypothesis = representation.voice_hypothesis
        if hypothesis is not None:
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
