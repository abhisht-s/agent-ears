"""Persistent real-audio target analysis and matcher-facing view construction."""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any, cast

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from music_locator.decoding import (
    DecodedTargetReference,
    FFmpegTargetAudioDecoder,
    insert_audio_stream,
)
from music_locator.models import (
    ANALYSIS_SAMPLE_RATE,
    AnalysisDerivativeEvidence,
    AssetKind,
    AudioTargetArtifact,
    AudioVoiceHypothesisSummary,
    DerivativeRecord,
    SampleTimeMap,
    TargetCoordinateKind,
    TargetEventEvidenceArtifact,
    TargetNoteCandidate,
    TargetOnsetEvidence,
    TargetPitchFrame,
    TargetRepresentationBundle,
    TargetTrackRepresentation,
    TimeMapSegment,
)
from music_locator.representation import contour_step
from music_locator.storage import Workspace, new_id, utc_now
from music_locator.targets import load_target

TARGET_ANALYSIS_VERSION = "audio_target_parallel_evidence_v3"
TARGET_VIEW_VERSION = "audio_event_compatibility_view_v2"
TARGET_ANALYSIS_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _analysis_code_sha256() -> str:
    """Return the code identity frozen when this analysis process started."""

    return TARGET_ANALYSIS_CODE_SHA256


@dataclass(frozen=True)
class TargetAnalysisOptions:
    enable_pyin: bool = True
    enable_basic_pitch: bool = True
    basic_pitch_python: str | None = None
    basic_pitch_model_path: str | None = None
    separator: str = "none"
    separator_python: str | None = None
    selected_stream_index: int | None = None
    max_paths_per_source: int = 4
    max_total_paths: int = 16

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "enable_pyin": self.enable_pyin,
            "enable_basic_pitch": self.enable_basic_pitch,
            "basic_pitch_python": self.basic_pitch_python or "auto",
            "basic_pitch_model_path": self.basic_pitch_model_path or "package_default",
            "separator": self.separator,
            "separator_python": self.separator_python or "auto",
            "selected_stream_index": (
                self.selected_stream_index if self.selected_stream_index is not None else "default"
            ),
            "max_paths_per_source": self.max_paths_per_source,
            "max_total_paths": self.max_total_paths,
            "analysis_version": TARGET_ANALYSIS_VERSION,
            "analysis_code_sha256": _analysis_code_sha256(),
            "view_version": TARGET_VIEW_VERSION,
        }


@dataclass(frozen=True)
class TargetAnalysisResult:
    target_analysis_id: str
    bundle: TargetRepresentationBundle
    bundle_artifact_id: str
    cache_hit: bool
    decode_cache_hit: bool


@dataclass(frozen=True)
class _AnalysisSource:
    role: str
    asset_id: str
    stream_id: str
    sample_rate: int
    channels: int
    frame_count: int
    lineage: str
    decoded_reference_map: SampleTimeMap | None = None


@dataclass(frozen=True)
class _DerivedEvent:
    evidence_ref: str
    pitch_cents: float
    start_sample: int
    end_sample: int
    confidence: float


def _stable_id(prefix: str, *parts: object) -> str:
    content = "\x1f".join(str(part) for part in parts).encode()
    return f"{prefix}_{hashlib.sha256(content).hexdigest()[:24]}"


def _cache_key(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _artifact_sha256(workspace: Workspace, artifact_id: str) -> str:
    return str(workspace.artifact_row(artifact_id)["sha256"])


def _load_sample_map(workspace: Workspace, artifact_id: str) -> SampleTimeMap:
    return SampleTimeMap.model_validate(workspace.load_artifact_json(artifact_id))


def _analysis_derivative(
    workspace: Workspace,
    target_id: str,
    source: _AnalysisSource,
    *,
    run_id: str,
) -> tuple[AnalysisDerivativeEvidence, str, SampleTimeMap]:
    """Create explicit mean-mono/22.05-kHz evidence without touching the reference."""

    audio, source_rate = sf.read(
        workspace.asset_path(source.asset_id), dtype="float32", always_2d=True
    )
    if source_rate != source.sample_rate or audio.shape != (source.frame_count, source.channels):
        raise RuntimeError("analysis source does not match its persisted stream contract")
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("analysis source contains non-finite samples")
    weights = np.full(source.channels, 1.0 / source.channels, dtype=np.float64)
    # A direct reduction avoids platform BLAS behavior for the unusually tall
    # [frames, channels] matrix while retaining explicit float64 downmix math.
    mono = np.sum(np.asarray(audio, dtype=np.float64), axis=1, dtype=np.float64) / source.channels
    divisor = math.gcd(source_rate, ANALYSIS_SAMPLE_RATE)
    derived = resample_poly(
        mono,
        up=ANALYSIS_SAMPLE_RATE // divisor,
        down=source_rate // divisor,
    ).astype(np.float32)
    memory = io.BytesIO()
    sf.write(memory, derived, ANALYSIS_SAMPLE_RATE, format="WAV", subtype="FLOAT")
    asset = workspace.store_asset(
        memory.getvalue(),
        kind=AssetKind.TARGET_AUDIO_DERIVATIVE,
        media_type="audio/wav",
        original_name=(
            f"{target_id}-{source.role}-mono-{ANALYSIS_SAMPLE_RATE}-"
            f"{hashlib.sha256(source.lineage.encode()).hexdigest()[:10]}.wav"
        ),
    )
    stream = insert_audio_stream(
        workspace,
        asset_id=asset.asset_id,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        channels=1,
        frame_count=len(derived),
        sample_format="FLOAT",
    )
    sample_map = SampleTimeMap(
        source_stream_id=source.stream_id,
        derived_stream_id=stream.stream_id,
        source_sample_rate=source.sample_rate,
        derived_sample_rate=ANALYSIS_SAMPLE_RATE,
        segments=(
            TimeMapSegment(
                source_start_sample=0,
                source_end_sample=source.frame_count,
                derived_start_sample=0,
                derived_end_sample=stream.frame_count,
            ),
        ),
    )
    map_artifact_id = workspace.store_json_artifact(
        sample_map,
        artifact_type="time_map.target_analysis_derivative",
        analysis_run_id=run_id,
    )
    time_map_id = new_id("timemap")
    workspace.database.execute(
        """
        INSERT INTO time_maps(time_map_id, source_stream_id, derived_stream_id,
                              artifact_id, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (time_map_id, source.stream_id, stream.stream_id, map_artifact_id, utc_now()),
    )
    parameters: dict[str, int | float | str | bool] = {
        "source_role": source.role,
        "source_sample_rate": source.sample_rate,
        "derived_sample_rate": ANALYSIS_SAMPLE_RATE,
        "source_channels": source.channels,
        "channel_matrix": ",".join(f"{value:.17g}" for value in weights),
        "resampler": "scipy.signal.resample_poly",
        "resampler_up": ANALYSIS_SAMPLE_RATE // divisor,
        "resampler_down": source_rate // divisor,
        "normalization": "none",
        "limiting": "none",
    }
    derivative_record = DerivativeRecord(
        derivative_id=new_id("derivative"),
        source_stream_id=source.stream_id,
        derived_stream_id=stream.stream_id,
        operation="explicit_mean_downmix_then_polyphase_resample",
        parameters=parameters,
        time_map_id=time_map_id,
        created_at=utc_now(),
    )
    workspace.database.execute(
        """
        INSERT INTO derivatives(derivative_id, source_stream_id, derived_stream_id,
                                operation, parameters_json, time_map_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            derivative_record.derivative_id,
            derivative_record.source_stream_id,
            derivative_record.derived_stream_id,
            derivative_record.operation,
            json.dumps(parameters, sort_keys=True),
            derivative_record.time_map_id,
            derivative_record.created_at,
        ),
    )
    evidence = AnalysisDerivativeEvidence(
        derivative_id=derivative_record.derivative_id,
        target_id=target_id,
        source_role=source.role,
        source_asset_id=source.asset_id,
        source_stream_id=source.stream_id,
        derived_asset_id=asset.asset_id,
        derived_stream_id=stream.stream_id,
        operation=derivative_record.operation,
        parameters=parameters,
        channel_matrix=(tuple(float(value) for value in weights),),
        sample_time_map_artifact_id=map_artifact_id,
        source_sample_rate=source.sample_rate,
        derived_sample_rate=stream.sample_rate,
        source_frame_count=source.frame_count,
        derived_frame_count=stream.frame_count,
        artifact_sha256=asset.sha256,
    )
    evidence_artifact_id = workspace.store_json_artifact(
        evidence,
        artifact_type="target.analysis_derivative",
        analysis_run_id=run_id,
    )
    return evidence, evidence_artifact_id, sample_map


def _f0_events(
    frames: tuple[TargetPitchFrame, ...],
    onsets: tuple[TargetOnsetEvidence, ...],
    *,
    pitch_jump_cents: float,
) -> tuple[_DerivedEvent, ...]:
    usable = [
        frame
        for frame in frames
        if frame.voiced and frame.pitch_midi_cents is not None and frame.voiced_probability >= 0.10
    ]
    if not usable:
        return ()
    events: list[list[TargetPitchFrame]] = []
    current: list[TargetPitchFrame] = []
    onset_samples = tuple(sorted(onset.reference_sample for onset in onsets))
    for frame in usable:
        if current:
            center = float(
                median(
                    float(item.pitch_midi_cents)
                    for item in current
                    if item.pitch_midi_cents is not None
                )
            )
            previous = current[-1]
            hop = (
                max(
                    1,
                    previous.derivative_center_sample - current[-2].derivative_center_sample,
                )
                if len(current) > 1
                else 256
            )
            separated = frame.derivative_center_sample - previous.derivative_center_sample > 3 * hop
            frame_pitch = frame.pitch_midi_cents
            if frame_pitch is None:
                raise RuntimeError("usable pitch frame unexpectedly lost its pitch")
            changed = abs(frame_pitch - center) >= pitch_jump_cents
            repeated_attack = len(current) >= 3 and any(
                previous.reference_center_sample < onset_sample <= frame.reference_center_sample
                for onset_sample in onset_samples
            )
            if separated or changed or repeated_attack:
                events.append(current)
                current = []
        current.append(frame)
    if current:
        events.append(current)
    output: list[_DerivedEvent] = []
    for group in events:
        if len(group) < 3:
            continue
        pitches = [
            float(frame.pitch_midi_cents) for frame in group if frame.pitch_midi_cents is not None
        ]
        confidences = [frame.voiced_probability for frame in group]
        start = group[0].reference_start_sample
        end = group[-1].reference_end_sample
        if end <= start:
            continue
        strongest = max(group, key=lambda frame: frame.voiced_probability)
        output.append(
            _DerivedEvent(
                evidence_ref=strongest.frame_id,
                pitch_cents=float(median(pitches)),
                start_sample=start,
                end_sample=end,
                confidence=float(sum(confidences) / len(confidences)),
            )
        )
    return tuple(output)


def _events_in_decoded_reference(
    events: tuple[_DerivedEvent, ...], source: _AnalysisSource
) -> tuple[_DerivedEvent, ...]:
    """Compose immediate-source support (for example a stem) into reference PCM."""

    mapping = source.decoded_reference_map
    if mapping is None:
        return events
    if mapping.derived_stream_id != source.stream_id:
        raise ValueError("analysis source map does not terminate at the source stream")
    output: list[_DerivedEvent] = []
    for event in events:
        start = mapping.derived_to_source(event.start_sample)
        end = mapping.derived_to_source(event.end_sample)
        if end <= start:
            raise ValueError("analysis source map collapsed positive event support")
        output.append(
            _DerivedEvent(
                evidence_ref=event.evidence_ref,
                pitch_cents=event.pitch_cents,
                start_sample=start,
                end_sample=end,
                confidence=event.confidence,
            )
        )
    return tuple(output)


def _select_diverse_representations(
    representations: list[TargetTrackRepresentation], limit: int
) -> list[TargetTrackRepresentation]:
    """Keep at least one searchable path per observation backend when possible."""

    if limit <= 0:
        return []

    def score(item: TargetTrackRepresentation) -> float:
        hypothesis = item.audio_voice_hypothesis
        return hypothesis.path_score if hypothesis is not None else 0.0

    ordered = sorted(representations, key=score, reverse=True)
    selected: list[TargetTrackRepresentation] = []
    seen_backends: set[str] = set()
    for representation in ordered:
        hypothesis = representation.audio_voice_hypothesis
        backend = hypothesis.extraction_backend if hypothesis is not None else "unknown"
        if backend not in seen_backends:
            selected.append(representation)
            seen_backends.add(backend)
            if len(selected) == limit:
                return selected
    selected_ids = {item.view_id for item in selected}
    selected.extend(item for item in ordered if item.view_id not in selected_ids)
    return selected[:limit]


def _cluster_note_candidates(
    candidates: tuple[TargetNoteCandidate, ...], sample_rate: int
) -> tuple[tuple[TargetNoteCandidate, ...], ...]:
    tolerance = max(1, round(sample_rate * 0.045))
    groups: list[list[TargetNoteCandidate]] = []
    for candidate in sorted(
        candidates, key=lambda item: (item.reference_start_sample, item.pitch_midi_cents)
    ):
        if (
            not groups
            or candidate.reference_start_sample - groups[-1][0].reference_start_sample > tolerance
        ):
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    return tuple(tuple(group) for group in groups)


def _candidate_path(
    groups: tuple[tuple[TargetNoteCandidate, ...], ...], method: str
) -> tuple[TargetNoteCandidate, ...]:
    if not groups:
        return ()
    if method == "highest_at_onset":
        return tuple(max(group, key=lambda item: item.pitch_midi_cents) for group in groups)
    if method == "lowest_at_onset":
        return tuple(min(group, key=lambda item: item.pitch_midi_cents) for group in groups)
    if method == "salience_at_onset":
        return tuple(
            max(group, key=lambda item: (item.note_confidence, item.amplitude)) for group in groups
        )
    if method != "continuity_path":
        raise ValueError(f"unknown audio path method: {method}")
    costs: list[dict[str, tuple[float, str | None]]] = []
    for group_index, group in enumerate(groups):
        state: dict[str, tuple[float, str | None]] = {}
        for candidate in group:
            node = 0.8 * (1.0 - candidate.note_confidence) + 0.2 * (1.0 - candidate.amplitude)
            if group_index == 0:
                state[candidate.candidate_id] = (node, None)
                continue
            choices: list[tuple[float, str]] = []
            previous_by_id = {item.candidate_id: item for item in groups[group_index - 1]}
            for previous_id, (previous_cost, _) in costs[-1].items():
                previous = previous_by_id[previous_id]
                jump = abs(candidate.pitch_midi_cents - previous.pitch_midi_cents)
                choices.append((previous_cost + min(4.0, jump / 450.0), previous_id))
            best_cost, best_id = min(choices)
            state[candidate.candidate_id] = (best_cost + node, best_id)
        costs.append(state)
    final_id: str = min(costs[-1], key=lambda item: costs[-1][item][0])
    selected: list[TargetNoteCandidate] = []
    for group_index in range(len(groups) - 1, -1, -1):
        candidate = next(item for item in groups[group_index] if item.candidate_id == final_id)
        selected.append(candidate)
        parent_id = costs[group_index][final_id][1]
        if parent_id is None:
            break
        final_id = parent_id
    return tuple(reversed(selected))


def _note_events_for_methods(
    evidence: TargetEventEvidenceArtifact,
    reference_sample_rate: int,
) -> tuple[tuple[str, tuple[_DerivedEvent, ...], float], ...]:
    groups = _cluster_note_candidates(evidence.note_candidates, reference_sample_rate)
    output: list[tuple[str, tuple[_DerivedEvent, ...], float]] = []
    signatures: set[tuple[str, ...]] = set()
    for method in (
        "salience_at_onset",
        "continuity_path",
        "highest_at_onset",
        "lowest_at_onset",
    ):
        path = _candidate_path(groups, method)
        signature = tuple(candidate.candidate_id for candidate in path)
        if len(path) < 3 or signature in signatures:
            continue
        signatures.add(signature)
        events: list[_DerivedEvent] = []
        for index, candidate in enumerate(path):
            end = candidate.reference_end_sample
            if index + 1 < len(path):
                end = min(end, path[index + 1].reference_start_sample)
            if end <= candidate.reference_start_sample:
                end = candidate.reference_end_sample
            events.append(
                _DerivedEvent(
                    evidence_ref=candidate.candidate_id,
                    pitch_cents=candidate.pitch_midi_cents,
                    start_sample=candidate.reference_start_sample,
                    end_sample=end,
                    confidence=min(1.0, max(0.0, candidate.note_confidence)),
                )
            )
        path_score = sum(event.confidence for event in events) / len(events)
        output.append((method, tuple(events), path_score))
    return tuple(output)


def _representation(
    *,
    target_id: str,
    source_role: str,
    derivative_artifact_id: str,
    evidence_artifact_id: str,
    backend: str,
    backend_version: str,
    method: str,
    rank: int,
    events: tuple[_DerivedEvent, ...],
    path_score: float,
    reference: DecodedTargetReference,
) -> TargetTrackRepresentation:
    view_id = _stable_id(
        "view",
        target_id,
        source_role,
        evidence_artifact_id,
        backend,
        method,
        rank,
    )
    pitches = [round(event.pitch_cents) for event in events]
    intervals = [right - left for left, right in pairwise(pitches)]
    onsets = [
        round(event.start_sample * 1_000_000 / reference.decoded_stream.sample_rate)
        for event in events
    ]
    iois = [right - left for left, right in pairwise(onsets)]
    positive_iois = [value for value in iois if value > 0]
    median_ioi = median(positive_iois) if positive_iois else 1
    center = median(pitches) if pitches else 0
    event_ids = tuple(
        _stable_id("aevent", view_id, index, event.evidence_ref)
        for index, event in enumerate(events)
    )
    hypothesis = AudioVoiceHypothesisSummary(
        view_id=view_id,
        source_role=source_role,
        source_derivative_artifact_id=derivative_artifact_id,
        evidence_artifact_id=evidence_artifact_id,
        extraction_backend=backend,
        extraction_backend_version=backend_version,
        path_method=method,
        path_rank=rank,
        path_score=path_score,
        event_count=len(events),
    )
    return TargetTrackRepresentation(
        schema_version="0.0.2",
        target_id=target_id,
        track_index=None,
        view_id=view_id,
        source_group_id=f"{target_id}:{source_role}",
        event_ids=event_ids,
        pitch_cents=tuple(pitches),
        relative_pitch_cents=tuple(float(pitch - center) for pitch in pitches),
        interval_cents=tuple(intervals),
        contour_steps=tuple(contour_step(float(value)) for value in intervals),
        onset_microseconds=tuple(onsets),
        log_ioi_milli=tuple(round(1_000 * math.log(max(value, 1) / median_ioi)) for value in iois),
        coordinate_kind=TargetCoordinateKind.DECODED_SAMPLES,
        start_samples=tuple(event.start_sample for event in events),
        end_samples=tuple(event.end_sample for event in events),
        reference_sample_rate=reference.decoded_stream.sample_rate,
        reference_zero_media_microseconds=(reference.time_map.reference_zero_media_microseconds),
        event_confidences=tuple(event.confidence for event in events),
        source_evidence_refs=tuple(event.evidence_ref for event in events),
        audio_voice_hypothesis=hypothesis,
    )


def _cached_analysis(
    workspace: Workspace, cache_key: str, *, target_id: str
) -> TargetAnalysisResult | None:
    row = workspace.database.fetchone(
        """
        SELECT target_analyses.*
        FROM target_analyses
        JOIN analysis_runs USING (analysis_run_id)
        WHERE target_analyses.cache_key = ?
          AND target_analyses.target_id = ?
          AND analysis_runs.status = 'succeeded'
        """,
        (cache_key, target_id),
    )
    if row is None:
        return None
    bundle = TargetRepresentationBundle.model_validate(
        workspace.load_artifact_json(str(row["bundle_artifact_id"]))
    )
    if bundle.target_id != target_id or bundle.cache_key != cache_key:
        raise RuntimeError("cached target representation bundle identity is inconsistent")
    decode_row = workspace.database.fetchone(
        """
        SELECT target_decodes.cache_key
        FROM target_decodes
        JOIN analysis_runs USING (analysis_run_id)
        WHERE target_decodes.decode_id = ? AND analysis_runs.status = 'succeeded'
        """,
        (str(row["decode_id"]),),
    )
    if decode_row is None:
        raise RuntimeError("cached target analysis lost its decoded reference")
    view_rows = workspace.database.fetchall(
        """
        SELECT representation_artifact_id
        FROM target_views
        WHERE target_analysis_id = ? AND target_id = ?
        """,
        (str(row["target_analysis_id"]), target_id),
    )
    persisted_views = {str(item["representation_artifact_id"]) for item in view_rows}
    expected_views = set(bundle.representation_artifact_ids)
    if (
        len(view_rows) != len(bundle.representation_artifact_ids)
        or persisted_views != expected_views
    ):
        raise RuntimeError("cached target analysis has an incomplete target-view set")
    return TargetAnalysisResult(
        target_analysis_id=str(row["target_analysis_id"]),
        bundle=bundle,
        bundle_artifact_id=str(row["bundle_artifact_id"]),
        cache_hit=True,
        decode_cache_hit=True,
    )


def analyze_audio_target(
    workspace: Workspace,
    target_id: str,
    *,
    options: TargetAnalysisOptions | None = None,
    decoder: FFmpegTargetAudioDecoder | None = None,
) -> TargetAnalysisResult:
    """Build cacheable observation artifacts and a small K of matcher views."""

    workspace.initialize()
    configured = options or TargetAnalysisOptions()
    if configured.separator not in {"none", "htdemucs"}:
        raise ValueError("separator must be 'none' or 'htdemucs'")
    loaded, target_artifact_id = load_target(workspace, target_id)
    if not isinstance(loaded, AudioTargetArtifact):
        raise ValueError(f"target is not encoded real audio: {target_id}")
    reference = (decoder or FFmpegTargetAudioDecoder()).decode(
        workspace,
        loaded,
        selected_stream_index=configured.selected_stream_index,
    )
    # Backend identity is intentionally part of the cache. An unavailable optional
    # backend and a later installed backend cannot silently share analysis results.
    from music_locator.target_backends import (
        BasicPitchSubprocessTranscriber,
        HTDemucsSubprocessSeparator,
        PyinPitchTracker,
    )

    pyin = PyinPitchTracker() if configured.enable_pyin else None
    basic_pitch = (
        BasicPitchSubprocessTranscriber(
            configured.basic_pitch_python,
            model_path=configured.basic_pitch_model_path,
        )
        if configured.enable_basic_pitch
        else None
    )
    separator = (
        HTDemucsSubprocessSeparator(configured.separator_python)
        if configured.separator == "htdemucs"
        else None
    )
    backend_identity = {
        "pyin": pyin.cache_identity() if pyin is not None else "disabled",
        "basic_pitch": (basic_pitch.cache_identity() if basic_pitch is not None else "disabled"),
        "separator": separator.cache_identity() if separator is not None else "disabled",
    }
    basic_pitch_identity = backend_identity["basic_pitch"]
    basic_pitch_model_identity = (
        basic_pitch_identity.get("model_identity")
        if isinstance(basic_pitch_identity, dict)
        else None
    )
    basic_pitch_probe_identity = (
        basic_pitch_identity.get("probe") if isinstance(basic_pitch_identity, dict) else None
    )
    basic_pitch_available = bool(
        basic_pitch is not None
        and isinstance(basic_pitch_probe_identity, dict)
        and basic_pitch_probe_identity.get("available") in {True, "true"}
        and isinstance(basic_pitch_model_identity, dict)
        and basic_pitch_model_identity.get("available") in {True, "true"}
        and isinstance(basic_pitch_model_identity.get("sha256"), str)
    )
    separator_identity = backend_identity["separator"]
    separator_model_identity = (
        separator_identity.get("model_identity") if isinstance(separator_identity, dict) else None
    )
    separator_probe_identity = (
        separator_identity.get("probe") if isinstance(separator_identity, dict) else None
    )
    separator_importable = bool(
        separator is not None
        and isinstance(separator_probe_identity, dict)
        and separator_probe_identity.get("available") in {True, "true"}
    )
    separator_model_identified = bool(
        isinstance(separator_model_identity, dict)
        and separator_model_identity.get("available") in {True, "true"}
        and isinstance(separator_model_identity.get("model_state_sha256"), str)
    )
    if separator_importable and not separator_model_identified:
        raise RuntimeError("HTDemucs is importable but its exact model identity is unavailable")
    separator_available = separator_importable and separator_model_identified
    analysis_cache_key = _cache_key(
        {
            "target_id": target_id,
            "target_sha256": loaded.source_sha256,
            "decode_cache_key": reference.cache_key,
            "options": configured.as_dict(),
            "backends": backend_identity,
        }
    )
    cached = _cached_analysis(workspace, analysis_cache_key, target_id=target_id)
    if cached is not None:
        return cached
    run_id = workspace.create_analysis_run(
        "analyze_audio_target",
        {
            "target_id": target_id,
            "cache_key": analysis_cache_key,
            **configured.as_dict(),
        },
    )
    try:
        warnings: list[str] = list(reference.manifest.warnings)
        sources: list[_AnalysisSource] = [
            _AnalysisSource(
                role="full_mix",
                asset_id=reference.manifest.decoded_asset_id,
                stream_id=reference.decoded_stream.stream_id,
                sample_rate=reference.decoded_stream.sample_rate,
                channels=reference.decoded_stream.channels,
                frame_count=reference.decoded_stream.frame_count,
                lineage=reference.manifest.decoded_sha256,
                decoded_reference_map=None,
            )
        ]
        separator_derivative_artifact_ids: list[str] = []
        if separator is not None:
            if not separator_available:
                warnings.append("htdemucs_requested_but_worker_unavailable")
            else:
                separated = separator.separate(workspace, reference, run_id=run_id)
                expected_model_hash = str(
                    cast(dict[str, Any], separator_model_identity)["model_state_sha256"]
                )
                if any(
                    item.derivative.parameters.get("model_state_sha256") != expected_model_hash
                    for item in separated
                ):
                    raise RuntimeError(
                        "HTDemucs model state changed between cache identity and separation"
                    )
                separator_derivative_artifact_ids.extend(
                    item.derivative_artifact_id for item in separated
                )
                sources.extend(
                    _AnalysisSource(
                        role=item.role,
                        asset_id=item.asset.asset_id,
                        stream_id=item.stream.stream_id,
                        sample_rate=item.stream.sample_rate,
                        channels=item.stream.channels,
                        frame_count=item.stream.frame_count,
                        lineage=item.asset.sha256,
                        decoded_reference_map=item.sample_map,
                    )
                    for item in separated
                    if item.role in {"vocals", "bass", "other"}
                )
        derivative_artifact_ids: list[str] = list(separator_derivative_artifact_ids)
        evidence_artifact_ids: list[str] = []
        representations: list[TargetTrackRepresentation] = []
        for source in sources:
            if len(representations) >= configured.max_total_paths:
                break
            derivative, derivative_artifact_id, sample_map = _analysis_derivative(
                workspace, target_id, source, run_id=run_id
            )
            derivative_artifact_ids.append(derivative_artifact_id)
            source_representations: list[TargetTrackRepresentation] = []
            if pyin is not None:
                pyin_evidence, pyin_artifact_id = pyin.analyze(
                    workspace,
                    derivative,
                    sample_map,
                    run_id=run_id,
                    derivative_artifact_id=derivative_artifact_id,
                )
                evidence_artifact_ids.append(pyin_artifact_id)
                thresholds = (75.0, 110.0, 150.0)
                for rank, threshold in enumerate(thresholds, start=1):
                    events = _f0_events(
                        pyin_evidence.pitch_frames,
                        pyin_evidence.onsets,
                        pitch_jump_cents=threshold,
                    )
                    events = _events_in_decoded_reference(events, source)
                    if len(events) < 3:
                        continue
                    source_representations.append(
                        _representation(
                            target_id=target_id,
                            source_role=source.role,
                            derivative_artifact_id=derivative_artifact_id,
                            evidence_artifact_id=pyin_artifact_id,
                            backend=pyin_evidence.backend,
                            backend_version=pyin_evidence.backend_version,
                            method=f"predominant_f0_events_{round(threshold)}c",
                            rank=rank,
                            events=events,
                            path_score=sum(event.confidence for event in events) / len(events),
                            reference=reference,
                        )
                    )
            if basic_pitch is not None:
                if not basic_pitch_available:
                    warnings.append("basic_pitch_unavailable_pyin_views_only")
                else:
                    try:
                        bp_evidence, bp_artifact_id = basic_pitch.transcribe(
                            workspace,
                            derivative,
                            sample_map,
                            run_id=run_id,
                            derivative_artifact_id=derivative_artifact_id,
                        )
                    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                        warnings.append(
                            "basic_pitch_failed_pyin_views_only:"
                            f"{type(error).__name__}:{str(error)[:500]}"
                        )
                    else:
                        expected_model_hash = str(
                            cast(dict[str, Any], basic_pitch_model_identity)["sha256"]
                        )
                        if bp_evidence.backend_config.get("model_sha256") != expected_model_hash:
                            warnings.append(
                                "basic_pitch_failed_pyin_views_only:"
                                "RuntimeError:model changed between cache identity and inference"
                            )
                        else:
                            evidence_artifact_ids.append(bp_artifact_id)
                            for rank, (method, events, score) in enumerate(
                                _note_events_for_methods(bp_evidence, source.sample_rate),
                                start=1,
                            ):
                                events = _events_in_decoded_reference(events, source)
                                source_representations.append(
                                    _representation(
                                        target_id=target_id,
                                        source_role=source.role,
                                        derivative_artifact_id=derivative_artifact_id,
                                        evidence_artifact_id=bp_artifact_id,
                                        backend=bp_evidence.backend,
                                        backend_version=bp_evidence.backend_version,
                                        method=f"basic_pitch_{method}",
                                        rank=rank,
                                        events=events,
                                        path_score=score,
                                        reference=reference,
                                    )
                                )
            deduplicated: dict[tuple[str, ...], TargetTrackRepresentation] = {}
            for representation in source_representations:
                signature = representation.source_evidence_refs
                previous = deduplicated.get(signature)
                if previous is None or (
                    representation.audio_voice_hypothesis is not None
                    and previous.audio_voice_hypothesis is not None
                    and representation.audio_voice_hypothesis.path_score
                    > previous.audio_voice_hypothesis.path_score
                ):
                    deduplicated[signature] = representation
            selected = _select_diverse_representations(
                list(deduplicated.values()), configured.max_paths_per_source
            )
            representations.extend(
                selected[: max(0, configured.max_total_paths - len(representations))]
            )
        if not representations:
            raise ValueError("audio analysis produced no searchable pitched event views")
        representation_artifact_ids = tuple(
            workspace.store_json_artifact(
                representation,
                artifact_type="index.audio_target_view",
                analysis_run_id=run_id,
            )
            for representation in representations
        )
        degraded_backend_failure = any(
            warning.startswith("basic_pitch_failed_") for warning in warnings
        )
        published_cache_key = (
            _cache_key(
                {
                    "requested_cache_key": analysis_cache_key,
                    "degraded_backend_failure_run_id": run_id,
                }
            )
            if degraded_backend_failure
            else analysis_cache_key
        )
        invocation_rows = workspace.database.fetchall(
            """
            SELECT artifact_id
            FROM artifacts
            WHERE analysis_run_id = ? AND artifact_type LIKE 'target.backend_invocation.%'
            ORDER BY created_at, artifact_id
            """,
            (run_id,),
        )
        bundle = TargetRepresentationBundle(
            bundle_id=new_id("bundle"),
            target_id=target_id,
            target_artifact_id=target_artifact_id,
            decode_manifest_artifact_id=reference.manifest_artifact_id,
            target_time_map_artifact_id=reference.time_map_artifact_id,
            decoded_asset_id=reference.manifest.decoded_asset_id,
            decoded_stream_id=reference.decoded_stream.stream_id,
            analysis_config={
                **configured.as_dict(),
                "pyin_identity": backend_identity["pyin"],
                "basic_pitch_identity": backend_identity["basic_pitch"],
                "separator_identity": backend_identity["separator"],
                "requested_cache_key": analysis_cache_key,
                "cacheable_requested_outcome": not degraded_backend_failure,
            },
            cache_key=published_cache_key,
            derivative_artifact_ids=tuple(derivative_artifact_ids),
            evidence_artifact_ids=tuple(evidence_artifact_ids),
            backend_invocation_artifact_ids=tuple(
                str(row["artifact_id"]) for row in invocation_rows
            ),
            representation_artifact_ids=representation_artifact_ids,
            warnings=tuple(dict.fromkeys(warnings)),
        )
        bundle_artifact_id = workspace.store_json_artifact(
            bundle,
            artifact_type="target.representation_bundle",
            analysis_run_id=run_id,
        )
        target_analysis_id = new_id("targetanalysis")
        view_rows: list[tuple[str, str, str, str, str, str, str]] = []
        for representation, artifact_id in zip(
            representations, representation_artifact_ids, strict=True
        ):
            hypothesis = representation.audio_voice_hypothesis
            if hypothesis is None or representation.view_id is None:
                raise RuntimeError("audio representation lost its hypothesis identity")
            view_rows.append(
                (
                    representation.view_id,
                    target_analysis_id,
                    target_id,
                    artifact_id,
                    hypothesis.source_role,
                    hypothesis.extraction_backend,
                    utc_now(),
                )
            )
        # The cache row and every searchable view become visible in one
        # transaction, and only after the evidence-producing run succeeded.
        workspace.finish_analysis_run(run_id)
        with workspace.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO target_analyses(target_analysis_id, target_id, decode_id, cache_key,
                                            analysis_run_id, bundle_artifact_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    target_analysis_id,
                    target_id,
                    reference.decode_id,
                    published_cache_key,
                    run_id,
                    bundle_artifact_id,
                    utc_now(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO target_views(view_id, target_analysis_id, target_id,
                                         representation_artifact_id, source_role,
                                         backend, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                view_rows,
            )
        return TargetAnalysisResult(
            target_analysis_id=target_analysis_id,
            bundle=bundle,
            bundle_artifact_id=bundle_artifact_id,
            cache_hit=False,
            decode_cache_hit=reference.cache_hit,
        )
    except Exception as error:
        workspace.finish_analysis_run(run_id, error=str(error))
        raise


def load_latest_target_bundle(
    workspace: Workspace, target_id: str
) -> tuple[TargetRepresentationBundle, str, str]:
    row = workspace.database.fetchone(
        """
        SELECT target_analyses.target_analysis_id, target_analyses.bundle_artifact_id
        FROM target_analyses
        JOIN analysis_runs USING (analysis_run_id)
        WHERE target_analyses.target_id = ? AND analysis_runs.status = 'succeeded'
        ORDER BY created_at DESC, target_analysis_id DESC
        LIMIT 1
        """,
        (target_id,),
    )
    if row is None:
        raise ValueError(f"audio target has not been analyzed: {target_id}")
    artifact_id = str(row["bundle_artifact_id"])
    return (
        TargetRepresentationBundle.model_validate(workspace.load_artifact_json(artifact_id)),
        artifact_id,
        str(row["target_analysis_id"]),
    )


def load_audio_view(workspace: Workspace, artifact_id: str) -> TargetTrackRepresentation:
    representation = TargetTrackRepresentation.model_validate(
        workspace.load_artifact_json(artifact_id)
    )
    if representation.coordinate_kind != TargetCoordinateKind.DECODED_SAMPLES:
        raise ValueError("artifact is not an audio target view")
    return representation
