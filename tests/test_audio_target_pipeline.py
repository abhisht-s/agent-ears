from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_locator.audio import analyze_query, import_query
from music_locator.evidence import verify_search_evidence
from music_locator.indexing import build_index, load_index, load_track_representation
from music_locator.models import (
    AnalysisDerivativeEvidence,
    AudioTargetArtifact,
    ResultStatus,
    SampleTimeMap,
    TargetCoordinateKind,
    TargetEventEvidenceArtifact,
    TargetKind,
    TargetNoteCandidate,
    TargetTimeMap,
)
from music_locator.search import search
from music_locator.storage import Workspace, new_id
from music_locator.target_audio import TargetAnalysisOptions, analyze_audio_target
from music_locator.targets import import_target, load_target

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _write_note_sequence(
    path: Path,
    pitches: tuple[int, ...],
    *,
    note_seconds: float = 0.52,
    gap_seconds: float = 0.08,
    leading_seconds: float = 0.30,
    trailing_seconds: float = 0.30,
    sample_rate: int = 44_100,
) -> None:
    """Write original synthetic pitched evidence with discrete, stable attacks."""

    note_frames = round(note_seconds * sample_rate)
    gap_frames = round(gap_seconds * sample_rate)
    fade_frames = round(0.025 * sample_rate)
    pieces = [np.zeros(round(leading_seconds * sample_rate), dtype=np.float64)]
    for pitch in pitches:
        frequency = 440.0 * 2.0 ** ((pitch - 69) / 12.0)
        time = np.arange(note_frames, dtype=np.float64) / sample_rate
        tone = 0.36 * np.sin(2.0 * np.pi * frequency * time)
        tone += 0.045 * np.sin(4.0 * np.pi * frequency * time)
        envelope = np.ones(note_frames, dtype=np.float64)
        ramp = np.linspace(0.0, 1.0, fade_frames, endpoint=False)
        envelope[:fade_frames] = ramp
        envelope[-fade_frames:] = ramp[::-1]
        pieces.extend((tone * envelope, np.zeros(gap_frames, dtype=np.float64)))
    pieces.append(np.zeros(round(trailing_seconds * sample_rate), dtype=np.float64))
    mono = np.concatenate(pieces).astype(np.float32)
    sf.write(path, mono, sample_rate, subtype="PCM_16")


def test_audio_import_descriptor_roundtrip_has_no_ppq_and_preserves_source_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generated-target.wav"
    _write_note_sequence(source, (60, 64, 67), note_seconds=0.12, gap_seconds=0.03)
    source_bytes = source.read_bytes()
    expected_sha256 = hashlib.sha256(source_bytes).hexdigest()
    workspace = Workspace(tmp_path / "workspace")

    target_id = import_target(workspace, source, name="Generated target")
    loaded, artifact_id = load_target(workspace, target_id)

    assert isinstance(loaded, AudioTargetArtifact)
    assert loaded.schema_version == "0.0.2"
    assert loaded.kind is TargetKind.AUDIO
    assert loaded.name == "Generated target"
    assert loaded.original_name == source.name
    assert loaded.source_sha256 == expected_sha256
    assert loaded.byte_length == len(source_bytes)
    assert workspace.asset_path(loaded.asset_id).read_bytes() == source_bytes
    assert AudioTargetArtifact.model_validate(workspace.load_artifact_json(artifact_id)) == loaded

    target_row = workspace.database.fetchone(
        "SELECT * FROM targets WHERE target_id = ?", (target_id,)
    )
    assert target_row is not None
    assert target_row["ticks_per_beat"] is None
    assert target_row["kind"] == TargetKind.AUDIO.value
    assert target_row["source_asset_id"] == loaded.asset_id
    assert workspace.artifact_row(artifact_id)["schema_version"] == "0.0.2"


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg and ffprobe are required")
def test_pyin_audio_target_analysis_is_cached_and_its_views_are_indexed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cache-target.wav"
    _write_note_sequence(source, (60, 62, 65, 64, 67, 65, 69, 67, 71))
    workspace = Workspace(tmp_path / "workspace")
    target_id = import_target(workspace, source)
    options = TargetAnalysisOptions(
        enable_pyin=True,
        enable_basic_pitch=False,
        separator="none",
    )

    first = analyze_audio_target(workspace, target_id, options=options)
    second = analyze_audio_target(workspace, target_id, options=options)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.decode_cache_hit is True
    assert second.target_analysis_id == first.target_analysis_id
    assert second.bundle_artifact_id == first.bundle_artifact_id
    assert second.bundle == first.bundle
    assert first.bundle.analysis_config["enable_basic_pitch"] is False
    assert first.bundle.analysis_config["basic_pitch_identity"] == "disabled"
    assert first.bundle.representation_artifact_ids

    index_id = build_index(workspace, target_ids=(target_id,))
    index, _ = load_index(workspace, index_id)

    assert index.parameters["audio_view_count"] == len(first.bundle.representation_artifact_ids)
    assert len(index.target_entries) == len(first.bundle.representation_artifact_ids)
    assert {entry.representation_artifact_id for entry in index.target_entries} == set(
        first.bundle.representation_artifact_ids
    )
    for entry in index.target_entries:
        representation = load_track_representation(workspace, entry.representation_artifact_id)
        assert entry.target_id == target_id
        assert entry.track_index is None
        assert entry.view_id == representation.view_id
        assert representation.coordinate_kind is TargetCoordinateKind.DECODED_SAMPLES
        assert representation.audio_voice_hypothesis is not None
        assert representation.audio_voice_hypothesis.extraction_backend == "librosa.pyin"


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg and ffprobe are required")
@pytest.mark.parametrize("failure_kind", ("runtime", "timeout"))
def test_optional_basic_pitch_failure_keeps_pyin_views_but_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    source = tmp_path / f"basic-pitch-retry-{failure_kind}.wav"
    _write_note_sequence(source, (60, 62, 65, 64, 67, 65, 69, 67, 71))
    workspace = Workspace(tmp_path / "workspace")
    target_id = import_target(workspace, source)
    model_sha256 = "b" * 64

    class FakeBasicPitchTranscriber:
        attempts = 0

        def __init__(self, *_: object, **__: object) -> None:
            pass

        def available(self) -> bool:
            return True

        def cache_identity(self) -> dict[str, object]:
            return {
                "backend": "spotify.basic_pitch",
                "contract": "fake_basic_pitch_retry_v1",
                "adapter_sha256": "a" * 64,
                "worker_sha256": "c" * 64,
                "probe": {"available": "true"},
                "model_identity": {
                    "available": "true",
                    "sha256": model_sha256,
                    "path": "generated-test-model.onnx",
                },
            }

        def transcribe(
            self,
            current: Workspace,
            derivative: AnalysisDerivativeEvidence,
            sample_map: SampleTimeMap,
            *,
            run_id: str,
            derivative_artifact_id: str | None = None,
        ) -> tuple[TargetEventEvidenceArtifact, str]:
            FakeBasicPitchTranscriber.attempts += 1
            attempt = FakeBasicPitchTranscriber.attempts
            invocation_artifact_id = current.store_json_artifact(
                {
                    "schema_version": "0.0.2",
                    "backend": "spotify.basic_pitch",
                    "attempt": attempt,
                    "outcome": "failure" if attempt == 1 else "success",
                    "failure_kind": failure_kind if attempt == 1 else None,
                },
                artifact_type="target.backend_invocation.basic_pitch",
                analysis_run_id=run_id,
            )
            if attempt == 1:
                if failure_kind == "timeout":
                    raise subprocess.TimeoutExpired("fake-basic-pitch", 0.01)
                raise RuntimeError("generated Basic Pitch failure")

            derivative_id = derivative.derivative_id
            derivative_target_id = derivative.target_id
            source_role = derivative.source_role
            derived_frame_count = derivative.derived_frame_count
            step = max(2, derived_frame_count // 10)
            candidates: list[TargetNoteCandidate] = []
            for index, pitch in enumerate((60, 62, 65, 64, 67, 65), start=1):
                derivative_start = index * step
                derivative_end = min(derived_frame_count, derivative_start + step // 2)
                reference_start = sample_map.derived_to_source(derivative_start)
                reference_end = sample_map.derived_to_source(derivative_end)
                candidates.append(
                    TargetNoteCandidate(
                        candidate_id=f"fake_bp_note_{index}",
                        pitch_midi_cents=float(pitch * 100),
                        derivative_start_sample=derivative_start,
                        derivative_end_sample=derivative_end,
                        reference_start_sample=reference_start,
                        reference_end_sample=reference_end,
                        onset_confidence=0.9,
                        note_confidence=0.9,
                        amplitude=0.8,
                        source_role=source_role,
                        backend="spotify.basic_pitch",
                        supporting_evidence_ids=(invocation_artifact_id,),
                    )
                )
            evidence = TargetEventEvidenceArtifact(
                evidence_id=new_id("target_evidence"),
                target_id=derivative_target_id,
                source_role=source_role,
                derivative_id=derivative_id,
                derivative_artifact_id=derivative_artifact_id or derivative_id,
                backend="spotify.basic_pitch",
                backend_version="fake-basic-pitch-1",
                backend_config={
                    "model_sha256": model_sha256,
                    "invocation_artifact_id": invocation_artifact_id,
                },
                note_candidates=tuple(candidates),
            )
            artifact_id = current.store_json_artifact(
                evidence,
                artifact_type="target.evidence.basic_pitch",
                analysis_run_id=run_id,
            )
            return evidence, artifact_id

    monkeypatch.setattr(
        "music_locator.target_backends.BasicPitchSubprocessTranscriber",
        FakeBasicPitchTranscriber,
    )
    options = TargetAnalysisOptions(
        enable_pyin=True,
        enable_basic_pitch=True,
        separator="none",
    )

    degraded = analyze_audio_target(workspace, target_id, options=options)
    degraded_backends = {
        load_track_representation(workspace, artifact_id).audio_voice_hypothesis.extraction_backend
        for artifact_id in degraded.bundle.representation_artifact_ids
    }
    assert degraded.cache_hit is False
    assert degraded_backends == {"librosa.pyin"}
    assert degraded.bundle.analysis_config["cacheable_requested_outcome"] is False
    requested_cache_key = str(degraded.bundle.analysis_config["requested_cache_key"])
    assert degraded.bundle.cache_key != requested_cache_key
    assert any(
        warning.startswith(
            "basic_pitch_failed_pyin_views_only:"
            + ("TimeoutExpired" if failure_kind == "timeout" else "RuntimeError")
        )
        for warning in degraded.bundle.warnings
    )
    assert len(degraded.bundle.backend_invocation_artifact_ids) == 1
    failed_invocation = workspace.load_artifact_json(
        degraded.bundle.backend_invocation_artifact_ids[0]
    )
    assert failed_invocation["outcome"] == "failure"
    assert failed_invocation["failure_kind"] == failure_kind

    recovered = analyze_audio_target(workspace, target_id, options=options)
    recovered_backends = {
        load_track_representation(workspace, artifact_id).audio_voice_hypothesis.extraction_backend
        for artifact_id in recovered.bundle.representation_artifact_ids
    }
    assert FakeBasicPitchTranscriber.attempts == 2
    assert recovered.cache_hit is False
    assert recovered.target_analysis_id != degraded.target_analysis_id
    assert recovered.bundle.analysis_config["requested_cache_key"] == requested_cache_key
    assert recovered.bundle.analysis_config["cacheable_requested_outcome"] is True
    assert recovered.bundle.cache_key == requested_cache_key
    assert "librosa.pyin" in recovered_backends
    assert "spotify.basic_pitch" in recovered_backends
    assert not any(
        warning.startswith("basic_pitch_failed_") for warning in recovered.bundle.warnings
    )
    assert len(recovered.bundle.backend_invocation_artifact_ids) == 1
    successful_invocation = workspace.load_artifact_json(
        recovered.bundle.backend_invocation_artifact_ids[0]
    )
    assert successful_invocation["outcome"] == "success"

    cached = analyze_audio_target(workspace, target_id, options=options)
    assert cached.cache_hit is True
    assert cached.target_analysis_id == recovered.target_analysis_id
    assert cached.bundle_artifact_id == recovered.bundle_artifact_id
    assert FakeBasicPitchTranscriber.attempts == 2


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg and ffprobe are required")
def test_transposed_synthetic_hum_search_returns_source_grounded_audio_range(
    tmp_path: Path,
) -> None:
    phrase = (64, 67, 65, 69, 66, 71, 68, 73)
    target_path = tmp_path / "pitched-target.wav"
    query_path = tmp_path / "transposed-hum.wav"
    _write_note_sequence(target_path, (62, 59, *phrase, 61, 63))
    _write_note_sequence(
        query_path,
        tuple(pitch + 3 for pitch in phrase),
        note_seconds=0.58,
    )
    workspace = Workspace(tmp_path / "workspace")

    target_id = import_target(workspace, target_path)
    analysis = analyze_audio_target(
        workspace,
        target_id,
        options=TargetAnalysisOptions(enable_basic_pitch=False, separator="none"),
    )
    index_id = build_index(workspace, target_ids=(target_id,))
    query_id = import_query(workspace, query_path)
    analyze_query(workspace, query_id)

    result = search(workspace, query_id, index_id, target_id=target_id)

    assert result.status is ResultStatus.ACCEPTED
    assert result.selected_target_id == target_id
    assert result.selected_target_coordinate_kind is TargetCoordinateKind.DECODED_SAMPLES
    assert result.target_sample_range is not None
    assert result.target_time_range is not None
    assert result.target_range is None
    assert result.strong_rhythmic_target_ticks == ()
    assert result.strong_rhythmic_target_samples
    assert result.selected_track_index is None
    assert result.selected_voice_hypothesis is None
    assert result.selected_audio_voice_hypothesis is not None
    assert result.selected_target_view_id == result.selected_audio_voice_hypothesis.view_id
    assert result.selected_target_source_role == "full_mix"
    assert result.selected_target_backend == "librosa.pyin"
    assert result.transformations is not None
    assert abs(result.transformations.transposition_cents - 300) <= 35
    assert result.provenance is not None
    assert result.provenance.target_view_id == result.selected_target_view_id
    assert result.provenance.target_analysis_artifact_id == analysis.bundle_artifact_id
    assert result.provenance.target_decode_manifest_artifact_id == (
        analysis.bundle.decode_manifest_artifact_id
    )
    assert result.provenance.target_time_map_artifact_id == (
        analysis.bundle.target_time_map_artifact_id
    )
    assert result.provenance.target_evidence_artifact_id == (
        result.selected_audio_voice_hypothesis.evidence_artifact_id
    )

    time_map = TargetTimeMap.model_validate(
        workspace.load_artifact_json(analysis.bundle.target_time_map_artifact_id)
    )
    assert result.target_time_range.start_microsecond == (
        time_map.reference_sample_to_media_microseconds(result.target_sample_range.start_sample)
    )
    assert result.target_time_range.end_microsecond == (
        time_map.reference_sample_to_media_microseconds(result.target_sample_range.end_sample)
    )
    verification = verify_search_evidence(workspace, result.search_id)
    assert verification["valid"] is True
    assert int(verification["checked_artifacts"]) >= 1
    assert int(verification["checked_assets"]) >= 1
