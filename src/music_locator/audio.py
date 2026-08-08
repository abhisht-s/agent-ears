"""Immutable query ingestion, analysis derivatives, and transparent perception."""

from __future__ import annotations

import io
import math
from itertools import pairwise
from pathlib import Path
from statistics import median

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample_poly

from music_locator.models import (
    ANALYSIS_SAMPLE_RATE,
    AssetKind,
    AudioStreamRecord,
    DerivativeRecord,
    OnsetEvidence,
    PitchFrame,
    QualityReport,
    QualityWarning,
    QueryAnalysisArtifact,
    QueryEvent,
    QueryEventUsabilityEvidence,
    QueryInterpretation,
    SampleTimeMap,
    TimeMapSegment,
)
from music_locator.representation import query_representation
from music_locator.storage import Workspace, new_id, utc_now

HOP_LENGTH = 256
FRAME_LENGTH = 2_048
MIN_USABLE_EVENT_QUALITY = 0.10
MIN_USABLE_EVENT_VOICING = 0.10
MAX_USABLE_PITCH_DEVIATION_CENTS = 180.0
MIN_CONFIDENT_FRAME_FRACTION = 0.20
MIN_EVENT_RELATIVE_RMS = 0.04
REPEATED_ATTACK_ONSET_FLOOR = 0.12


def _event_usability(event: QueryEvent) -> QueryEventUsabilityEvidence:
    reasons: list[str] = []
    if event.quality_weight < MIN_USABLE_EVENT_QUALITY:
        reasons.append("quality_below_usable_floor")
    if event.voicing < MIN_USABLE_EVENT_VOICING:
        reasons.append("voicing_probability_below_usable_floor")
    if event.pitch_deviation_cents > MAX_USABLE_PITCH_DEVIATION_CENTS:
        reasons.append("pitch_instability_above_usable_ceiling")
    if (
        event.confident_frame_fraction is not None
        and event.confident_frame_fraction < MIN_CONFIDENT_FRAME_FRACTION
    ):
        reasons.append("confident_frame_coverage_below_usable_floor")
    if event.relative_rms is not None and event.relative_rms < MIN_EVENT_RELATIVE_RMS:
        reasons.append("event_energy_below_adaptive_usable_floor")
    return QueryEventUsabilityEvidence(
        source_event_id=event.event_id,
        usable=not reasons,
        reason_codes=tuple(reasons),
        quality_weight=event.quality_weight,
        voicing_probability=event.voicing,
        pitch_deviation_cents=event.pitch_deviation_cents,
        onset_strength=event.onset_strength,
        confident_frame_fraction=event.confident_frame_fraction,
        rms_amplitude=event.rms_amplitude,
        relative_rms=event.relative_rms,
    )


def _interpretation_evidence_score(
    events: tuple[QueryEvent, ...], usability: tuple[QueryEventUsabilityEvidence, ...]
) -> float:
    usable = [event for event, evidence in zip(events, usability, strict=True) if evidence.usable]
    if not usable:
        return 0.0
    mean_quality = sum(event.quality_weight for event in usable) / len(usable)
    usable_fraction = len(usable) / max(len(events), 1)
    onset_fraction = sum(event.onset_strength >= 0.10 for event in usable) / len(usable)
    count_adequacy = min(1.0, len(usable) / 8.0)
    return (
        0.55 * mean_quality
        + 0.20 * usable_fraction
        + 0.15 * onset_fraction
        + 0.10 * count_adequacy
    )


def prepare_interpretation_for_search(
    interpretation: QueryInterpretation,
) -> QueryInterpretation:
    """Create a provenance-linked usable-event view without mutating raw evidence."""

    usability = tuple(_event_usability(event) for event in interpretation.events)
    usable_events = tuple(
        event
        for event, evidence in zip(interpretation.events, usability, strict=True)
        if evidence.usable
    )
    parameters = dict(interpretation.method_parameters)
    parameters.update(
        {
            "usable_event_policy": "quality_voicing_stability_v1",
            "minimum_usable_event_quality": MIN_USABLE_EVENT_QUALITY,
            "minimum_usable_event_voicing": MIN_USABLE_EVENT_VOICING,
            "maximum_usable_pitch_deviation_cents": MAX_USABLE_PITCH_DEVIATION_CENTS,
            "minimum_confident_frame_fraction": MIN_CONFIDENT_FRAME_FRACTION,
            "minimum_event_relative_rms": MIN_EVENT_RELATIVE_RMS,
            "source_event_count": len(interpretation.events),
            "usable_event_count": len(usable_events),
        }
    )
    return QueryInterpretation(
        interpretation_id=new_id("searchinterp"),
        query_id=interpretation.query_id,
        rank=interpretation.rank,
        selected=False,
        method=f"{interpretation.method}+usable_event_filter",
        method_parameters=parameters,
        events=usable_events,
        representation=query_representation(usable_events),
        interpretation_score=_interpretation_evidence_score(
            interpretation.events, usability
        ),
        source_interpretation_id=interpretation.interpretation_id,
        event_usability=usability,
    )


def _audio_stream(workspace: Workspace, asset_id: str, path: Path) -> AudioStreamRecord:
    info = sf.info(path)
    record = AudioStreamRecord(
        stream_id=new_id("stream"),
        asset_id=asset_id,
        sample_rate=info.samplerate,
        channels=info.channels,
        frame_count=info.frames,
        sample_format=info.subtype,
        created_at=utc_now(),
    )
    workspace.database.execute(
        """
        INSERT INTO audio_streams(stream_id, asset_id, sample_rate, channels,
                                  frame_count, sample_format, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.stream_id,
            record.asset_id,
            record.sample_rate,
            record.channels,
            record.frame_count,
            record.sample_format,
            record.created_at,
        ),
    )
    return record


def import_query(workspace: Workspace, path: Path) -> str:
    workspace.initialize()
    if path.suffix.lower() != ".wav":
        raise ValueError("V0.0.1 accepts WAV query recordings only")
    info = sf.info(path)
    if info.format != "WAV":
        raise ValueError("query must be a valid WAV container")
    asset = workspace.store_asset(
        path,
        kind=AssetKind.QUERY_AUDIO,
        media_type="audio/wav",
        original_name=path.name,
    )
    stream = _audio_stream(workspace, asset.asset_id, workspace.asset_path(asset.asset_id))
    query_id = new_id("query")
    workspace.database.execute(
        """
        INSERT INTO queries(query_id, original_asset_id, original_stream_id, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (query_id, asset.asset_id, stream.stream_id, utc_now()),
    )
    return query_id


def _analysis_derivative(
    workspace: Workspace, source_stream: AudioStreamRecord
) -> tuple[AudioStreamRecord, DerivativeRecord, SampleTimeMap]:
    source_path = workspace.asset_path(source_stream.asset_id)
    audio, source_rate = sf.read(source_path, dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
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
        kind=AssetKind.ANALYSIS_AUDIO,
        media_type="audio/wav",
        original_name=f"{source_stream.asset_id}-mono-22050.wav",
    )
    stream = _audio_stream(workspace, asset.asset_id, workspace.asset_path(asset.asset_id))
    mapping = SampleTimeMap(
        source_stream_id=source_stream.stream_id,
        derived_stream_id=stream.stream_id,
        source_sample_rate=source_rate,
        derived_sample_rate=ANALYSIS_SAMPLE_RATE,
        segments=(
            TimeMapSegment(
                source_start_sample=0,
                source_end_sample=source_stream.frame_count,
                derived_start_sample=0,
                derived_end_sample=stream.frame_count,
            ),
        ),
    )
    map_artifact_id = workspace.store_json_artifact(mapping, artifact_type="time_map.sample")
    time_map_id = new_id("timemap")
    workspace.database.execute(
        """
        INSERT INTO time_maps(time_map_id, source_stream_id, derived_stream_id,
                              artifact_id, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (time_map_id, source_stream.stream_id, stream.stream_id, map_artifact_id, utc_now()),
    )
    derivative = DerivativeRecord(
        derivative_id=new_id("derivative"),
        source_stream_id=source_stream.stream_id,
        derived_stream_id=stream.stream_id,
        operation="downmix_mean_and_resample_polyphase",
        parameters={
            "source_sample_rate": source_rate,
            "derived_sample_rate": ANALYSIS_SAMPLE_RATE,
            "channels_to_mono": source_stream.channels,
        },
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
            derivative.derivative_id,
            derivative.source_stream_id,
            derivative.derived_stream_id,
            derivative.operation,
            __import__("json").dumps(derivative.parameters, sort_keys=True),
            derivative.time_map_id,
            derivative.created_at,
        ),
    )
    return stream, derivative, mapping


def _pitch_frames(audio: npt.NDArray[np.float32]) -> tuple[PitchFrame, ...]:
    f0, voiced, probability = librosa.pyin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=ANALYSIS_SAMPLE_RATE,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        fill_na=np.nan,
    )
    output: list[PitchFrame] = []
    for index, (frequency, is_voiced, voice_probability) in enumerate(
        zip(f0, voiced, probability, strict=True)
    ):
        frequency_value = None if np.isnan(frequency) else float(frequency)
        cents = None
        if frequency_value is not None and frequency_value > 0:
            cents = float(librosa.hz_to_midi(frequency_value) * 100.0)
        probability_value = 0.0 if np.isnan(voice_probability) else float(voice_probability)
        output.append(
            PitchFrame(
                frame_index=index,
                center_sample=index * HOP_LENGTH,
                f0_hz=frequency_value,
                midi_cents=cents,
                voiced_probability=min(1.0, max(0.0, probability_value)),
                voiced=bool(is_voiced),
                quality_weight=min(1.0, max(0.0, probability_value)),
            )
        )
    return tuple(output)


def _onsets(audio: npt.NDArray[np.float32]) -> tuple[OnsetEvidence, ...]:
    envelope = librosa.onset.onset_strength(y=audio, sr=ANALYSIS_SAMPLE_RATE, hop_length=HOP_LENGTH)
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope,
        sr=ANALYSIS_SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        backtrack=True,
        units="frames",
    )
    return tuple(
        OnsetEvidence(
            sample=int(frame * HOP_LENGTH), strength=float(envelope[min(frame, len(envelope) - 1)])
        )
        for frame in frames
    )


def _quality(audio: npt.NDArray[np.float32], pitch_frames: tuple[PitchFrame, ...]) -> QualityReport:
    duration_samples = len(audio)
    peak = float(np.max(np.abs(audio))) if duration_samples else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if duration_samples else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.999)) if duration_samples else 0.0
    voiced_fraction = (
        sum(frame.voiced for frame in pitch_frames) / len(pitch_frames) if pitch_frames else 0.0
    )
    warnings: list[QualityWarning] = []
    minimum = 4 * ANALYSIS_SAMPLE_RATE
    maximum = 12 * ANALYSIS_SAMPLE_RATE
    if duration_samples < minimum:
        warnings.append(
            QualityWarning(
                code="duration_too_short",
                severity="error",
                message="Query is shorter than 4 seconds.",
            )
        )
    if duration_samples > maximum:
        warnings.append(
            QualityWarning(
                code="duration_too_long",
                severity="error",
                message="Query is longer than 12 seconds.",
            )
        )
    if rms < 0.005:
        warnings.append(
            QualityWarning(
                code="signal_too_quiet", severity="error", message="Signal RMS is too low."
            )
        )
    if clipping > 0.01:
        warnings.append(
            QualityWarning(
                code="excessive_clipping",
                severity="warning",
                message="More than 1% of samples clip.",
            )
        )
    if voiced_fraction < 0.2:
        warnings.append(
            QualityWarning(
                code="low_voiced_fraction",
                severity="error",
                message="Too little reliable pitched content.",
            )
        )
    return QualityReport(
        duration_samples=duration_samples,
        sample_rate=ANALYSIS_SAMPLE_RATE,
        peak_amplitude=peak,
        rms_amplitude=rms,
        clipping_fraction=min(1.0, max(0.0, clipping)),
        voiced_fraction=min(1.0, max(0.0, voiced_fraction)),
        warnings=tuple(warnings),
    )


def _pitch_change_boundaries(frames: tuple[PitchFrame, ...], threshold_cents: float) -> list[int]:
    values = [frame.midi_cents for frame in frames]
    boundaries: list[int] = []
    for index in range(2, len(values) - 2):
        left = [value for value in values[index - 2 : index] if value is not None]
        right = [value for value in values[index : index + 2] if value is not None]
        if left and right and abs(median(right) - median(left)) >= threshold_cents:
            candidate = index * HOP_LENGTH
            if not boundaries or candidate - boundaries[-1] >= 3 * HOP_LENGTH:
                boundaries.append(candidate)
    return boundaries


def _event_hypothesis(
    query_id: str,
    frames: tuple[PitchFrame, ...],
    onsets: tuple[OnsetEvidence, ...],
    mapping: SampleTimeMap,
    derived_length: int,
    audio: npt.NDArray[np.float32],
    query_rms: float,
    *,
    rank: int,
    pitch_jump_cents: float,
    minimum_event_samples: int,
) -> QueryInterpretation:
    confident_samples = [
        frame.center_sample
        for frame in frames
        if frame.voiced
        and frame.midi_cents is not None
        and frame.voiced_probability >= MIN_USABLE_EVENT_VOICING
    ]
    voiced_samples = confident_samples or [
        frame.center_sample for frame in frames if frame.voiced and frame.midi_cents is not None
    ]
    if not voiced_samples:
        boundaries = [0, derived_length]
    else:
        first = max(0, voiced_samples[0] - HOP_LENGTH)
        last = min(derived_length, voiced_samples[-1] + HOP_LENGTH)
        boundaries = [first]
        boundaries.extend(onset.sample for onset in onsets if first < onset.sample < last)
        boundaries.extend(
            value
            for value in _pitch_change_boundaries(frames, pitch_jump_cents)
            if first < value < last
        )
        boundaries.append(last)
    boundaries = sorted(set(boundaries))
    events: list[QueryEvent] = []
    for start, end in pairwise(boundaries):
        if end - start < minimum_event_samples:
            continue
        time_frames = [
            frame for frame in frames if start <= frame.center_sample < end
        ]
        segment_frames = [
            frame
            for frame in time_frames
            if start <= frame.center_sample < end and frame.voiced and frame.midi_cents is not None
        ]
        if len(segment_frames) < 2:
            continue
        confident_frames = [
            frame
            for frame in segment_frames
            if frame.voiced_probability >= MIN_USABLE_EVENT_VOICING
        ]
        pitch_frames = confident_frames if len(confident_frames) >= 2 else segment_frames
        pitches = [
            float(frame.midi_cents) for frame in pitch_frames if frame.midi_cents is not None
        ]
        center = float(median(pitches))
        deviation = float(median(abs(pitch - center) for pitch in pitches))
        voice = float(np.mean([frame.voiced_probability for frame in segment_frames]))
        nearest_onset = max(
            (onset.strength for onset in onsets if abs(onset.sample - start) <= HOP_LENGTH * 2),
            default=0.0,
        )
        stability = max(0.0, 1.0 - deviation / 150.0)
        weight = min(1.0, max(0.0, voice * stability))
        confident_fraction = len(confident_frames) / max(len(time_frames), 1)
        segment_audio = audio[start:end]
        event_rms = (
            float(np.sqrt(np.mean(np.square(segment_audio, dtype=np.float64))))
            if len(segment_audio)
            else 0.0
        )
        relative_rms = event_rms / max(query_rms, 1e-9)
        source_start = mapping.derived_to_source(start)
        source_end = mapping.derived_to_source(end)
        if source_end <= source_start:
            source_end = source_start + 1
        events.append(
            QueryEvent(
                event_id=new_id("qevent"),
                start_sample=start,
                end_sample=end,
                source_start_sample=source_start,
                source_end_sample=source_end,
                pitch_midi_cents=center,
                pitch_deviation_cents=deviation,
                voicing=min(1.0, max(0.0, voice)),
                onset_strength=nearest_onset,
                quality_weight=weight,
                confident_frame_fraction=min(1.0, max(0.0, confident_fraction)),
                rms_amplitude=event_rms,
                relative_rms=max(0.0, relative_rms),
            )
        )
    merged_events: list[QueryEvent] = []
    confident_onset_boundaries = {
        onset.sample for onset in onsets if onset.strength >= REPEATED_ATTACK_ONSET_FLOOR
    }
    for event in events:
        if (
            merged_events
            and abs(event.pitch_midi_cents - merged_events[-1].pitch_midi_cents) <= 70.0
            and event.start_sample - merged_events[-1].end_sample <= 3 * HOP_LENGTH
            and event.start_sample not in confident_onset_boundaries
        ):
            previous = merged_events[-1]
            previous_weight = max(previous.end_sample - previous.start_sample, 1)
            event_weight = max(event.end_sample - event.start_sample, 1)
            total_weight = previous_weight + event_weight
            merged_events[-1] = previous.model_copy(
                update={
                    "end_sample": event.end_sample,
                    "source_end_sample": event.source_end_sample,
                    "pitch_midi_cents": (
                        previous.pitch_midi_cents * previous_weight
                        + event.pitch_midi_cents * event_weight
                    )
                    / total_weight,
                    "pitch_deviation_cents": max(
                        previous.pitch_deviation_cents, event.pitch_deviation_cents
                    ),
                    "voicing": (previous.voicing * previous_weight + event.voicing * event_weight)
                    / total_weight,
                    "onset_strength": max(previous.onset_strength, event.onset_strength),
                    "quality_weight": (
                        previous.quality_weight * previous_weight
                        + event.quality_weight * event_weight
                    )
                    / total_weight,
                    "confident_frame_fraction": (
                        (previous.confident_frame_fraction or 0.0) * previous_weight
                        + (event.confident_frame_fraction or 0.0) * event_weight
                    )
                    / total_weight,
                    "rms_amplitude": math.sqrt(
                        ((previous.rms_amplitude or 0.0) ** 2 * previous_weight)
                        + ((event.rms_amplitude or 0.0) ** 2 * event_weight)
                    )
                    / math.sqrt(total_weight),
                    "relative_rms": math.sqrt(
                        ((previous.relative_rms or 0.0) ** 2 * previous_weight)
                        + ((event.relative_rms or 0.0) ** 2 * event_weight)
                    )
                    / math.sqrt(total_weight),
                }
            )
        else:
            merged_events.append(event)
    interpretation_id = new_id("interpretation")
    event_tuple = tuple(merged_events)
    usability = tuple(_event_usability(event) for event in event_tuple)
    score = _interpretation_evidence_score(event_tuple, usability)
    return QueryInterpretation(
        interpretation_id=interpretation_id,
        query_id=query_id,
        rank=rank,
        selected=False,
        method="pyin_onset_pitch_change_segmentation",
        method_parameters={
            "hop_length": HOP_LENGTH,
            "pitch_jump_cents": pitch_jump_cents,
            "minimum_event_samples": minimum_event_samples,
            "active_region_voicing_probability_floor": MIN_USABLE_EVENT_VOICING,
            "repeated_attack_policy": "preserve_confident_onset_boundaries_v1",
            "repeated_attack_onset_strength_floor": REPEATED_ATTACK_ONSET_FLOOR,
        },
        events=event_tuple,
        representation=query_representation(event_tuple),
        interpretation_score=score,
        event_usability=usability,
    )


def analyze_query(workspace: Workspace, query_id: str) -> QueryAnalysisArtifact:
    workspace.initialize()
    row = workspace.database.fetchone("SELECT * FROM queries WHERE query_id = ?", (query_id,))
    if row is None:
        raise KeyError(f"unknown query: {query_id}")
    if row["analysis_artifact_id"] is not None:
        return QueryAnalysisArtifact.model_validate(
            workspace.load_artifact_json(str(row["analysis_artifact_id"]))
        )
    source_row = workspace.database.fetchone(
        "SELECT * FROM audio_streams WHERE stream_id = ?", (str(row["original_stream_id"]),)
    )
    if source_row is None:
        raise RuntimeError("query source stream is missing")
    source_stream = AudioStreamRecord.model_validate(dict(source_row))
    run_id = workspace.create_analysis_run(
        "analyze_query",
        {"pitch": "librosa.pyin", "sample_rate": ANALYSIS_SAMPLE_RATE, "hop_length": HOP_LENGTH},
    )
    try:
        derived_stream, derivative, mapping = _analysis_derivative(workspace, source_stream)
        audio, _ = sf.read(workspace.asset_path(derived_stream.asset_id), dtype="float32")
        audio_array = np.asarray(audio, dtype=np.float32)
        pitch_frames = _pitch_frames(audio_array)
        onsets = _onsets(audio_array)
        quality = _quality(audio_array, pitch_frames)
        pitch_artifact_id = workspace.store_json_artifact(
            {
                "schema_version": "0.0.1",
                "frames": [frame.model_dump(mode="json") for frame in pitch_frames],
            },
            artifact_type="query.pitch_frames",
            analysis_run_id=run_id,
        )
        onset_artifact_id = workspace.store_json_artifact(
            {
                "schema_version": "0.0.1",
                "onsets": [onset.model_dump(mode="json") for onset in onsets],
            },
            artifact_type="query.onsets",
            analysis_run_id=run_id,
        )
        configurations = ((75.0, 1_323), (110.0, 1_764), (55.0, 882), (150.0, 2_205))
        interpretations: list[QueryInterpretation] = []
        for rank, (pitch_jump, minimum_length) in enumerate(configurations, start=1):
            hypothesis = _event_hypothesis(
                query_id,
                pitch_frames,
                onsets,
                mapping,
                derived_stream.frame_count,
                audio_array,
                quality.rms_amplitude,
                rank=rank,
                pitch_jump_cents=pitch_jump,
                minimum_event_samples=minimum_length,
            )
            interpretations.append(hypothesis)
        if interpretations:
            selected_id = max(
                interpretations,
                key=lambda item: (item.interpretation_score, -item.rank),
            ).interpretation_id
            interpretations = [
                item.model_copy(
                    update={"selected": item.interpretation_id == selected_id}
                )
                for item in interpretations
            ]
        for hypothesis in interpretations:
            artifact_id = workspace.store_json_artifact(
                hypothesis, artifact_type="query.interpretation", analysis_run_id=run_id
            )
            workspace.database.execute(
                """
                INSERT INTO interpretations(interpretation_id, query_id, artifact_id, rank,
                                            selected, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis.interpretation_id,
                    query_id,
                    artifact_id,
                    hypothesis.rank,
                    int(hypothesis.selected),
                    hypothesis.interpretation_score,
                    utc_now(),
                ),
            )
        analysis = QueryAnalysisArtifact(
            query_id=query_id,
            original_asset_id=str(row["original_asset_id"]),
            original_stream_id=source_stream.stream_id,
            analysis_asset_id=derived_stream.asset_id,
            analysis_stream_id=derived_stream.stream_id,
            derivative_id=derivative.derivative_id,
            time_map_id=derivative.time_map_id,
            analysis_run_id=run_id,
            quality=quality,
            pitch_frames_artifact_id=pitch_artifact_id,
            onsets_artifact_id=onset_artifact_id,
            interpretation_ids=tuple(item.interpretation_id for item in interpretations),
            selected_interpretation_id=next(
                (item.interpretation_id for item in interpretations if item.selected),
                None,
            ),
        )
        artifact_id = workspace.store_json_artifact(
            analysis, artifact_type="query.analysis", analysis_run_id=run_id
        )
        workspace.database.execute(
            "UPDATE queries SET analysis_artifact_id = ? WHERE query_id = ?",
            (artifact_id, query_id),
        )
        workspace.finish_analysis_run(run_id)
        return analysis
    except Exception as error:
        workspace.finish_analysis_run(run_id, error=str(error))
        raise


def load_query_analysis(workspace: Workspace, query_id: str) -> tuple[QueryAnalysisArtifact, str]:
    row = workspace.database.fetchone(
        "SELECT analysis_artifact_id FROM queries WHERE query_id = ?", (query_id,)
    )
    if row is None:
        raise KeyError(f"unknown query: {query_id}")
    if row["analysis_artifact_id"] is None:
        raise ValueError(f"query has not been analyzed: {query_id}")
    artifact_id = str(row["analysis_artifact_id"])
    return QueryAnalysisArtifact.model_validate(
        workspace.load_artifact_json(artifact_id)
    ), artifact_id


def load_interpretation(
    workspace: Workspace, interpretation_id: str
) -> tuple[QueryInterpretation, str]:
    row = workspace.database.fetchone(
        "SELECT artifact_id FROM interpretations WHERE interpretation_id = ?", (interpretation_id,)
    )
    if row is None:
        raise KeyError(f"unknown interpretation: {interpretation_id}")
    artifact_id = str(row["artifact_id"])
    return QueryInterpretation.model_validate(
        workspace.load_artifact_json(artifact_id)
    ), artifact_id
