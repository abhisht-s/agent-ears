from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from music_locator.decoding import DecodedTargetReference, insert_audio_stream
from music_locator.models import (
    AnalysisDerivativeEvidence,
    AssetKind,
    DecodedReferenceManifest,
    DecoderInvocation,
    SampleTimeMap,
    TargetEventEvidenceArtifact,
    TargetNoteCandidate,
    TargetPitchFrame,
    TargetTimeMap,
    TargetTrackRepresentation,
    TimeMapSegment,
)
from music_locator.storage import Workspace, new_id
from music_locator.target_audio import TargetAnalysisOptions, analyze_audio_target
from music_locator.target_backends import (
    BASIC_PITCH_BACKEND,
    BasicPitchSubprocessTranscriber,
    HTDemucsSubprocessSeparator,
)
from music_locator.targets import import_target

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _float_wav_bytes(
    samples: np.ndarray[Any, Any], sample_rate: int, *, always_2d: bool = True
) -> bytes:
    memory = io.BytesIO()
    values = samples if always_2d else np.asarray(samples).reshape(-1)
    sf.write(memory, values, sample_rate, format="WAV", subtype="FLOAT")
    return memory.getvalue()


def _analysis_derivative_fixture(
    workspace: Workspace,
) -> tuple[AnalysisDerivativeEvidence, SampleTimeMap, str]:
    workspace.initialize()
    reference_rate = 44_100
    derivative_rate = 22_050
    reference_frames = 44_100
    derivative_frames = 22_050
    time = np.arange(reference_frames, dtype=np.float64) / reference_rate
    reference_audio = np.column_stack(
        (
            0.1 * np.sin(2.0 * np.pi * 220.0 * time),
            0.1 * np.sin(2.0 * np.pi * 330.0 * time),
        )
    ).astype(np.float32)
    derivative_audio = np.asarray(reference_audio[::2].mean(axis=1), dtype=np.float32)
    reference_asset = workspace.store_asset(
        _float_wav_bytes(reference_audio, reference_rate),
        kind=AssetKind.TARGET_AUDIO_REFERENCE,
        media_type="audio/wav",
        original_name="generated-reference.wav",
    )
    derivative_asset = workspace.store_asset(
        _float_wav_bytes(derivative_audio, derivative_rate, always_2d=False),
        kind=AssetKind.TARGET_AUDIO_DERIVATIVE,
        media_type="audio/wav",
        original_name="generated-analysis.wav",
    )
    reference_stream = insert_audio_stream(
        workspace,
        asset_id=reference_asset.asset_id,
        sample_rate=reference_rate,
        channels=2,
        frame_count=reference_frames,
        sample_format="FLOAT",
    )
    derivative_stream = insert_audio_stream(
        workspace,
        asset_id=derivative_asset.asset_id,
        sample_rate=derivative_rate,
        channels=1,
        frame_count=derivative_frames,
        sample_format="FLOAT",
    )
    sample_map = SampleTimeMap(
        source_stream_id=reference_stream.stream_id,
        derived_stream_id=derivative_stream.stream_id,
        source_sample_rate=reference_rate,
        derived_sample_rate=derivative_rate,
        segments=(
            TimeMapSegment(
                source_start_sample=0,
                source_end_sample=reference_frames,
                derived_start_sample=0,
                derived_end_sample=derivative_frames,
            ),
        ),
    )
    map_artifact_id = workspace.store_json_artifact(
        sample_map, artifact_type="time_map.test_target_analysis"
    )
    derivative = AnalysisDerivativeEvidence(
        derivative_id="derivative_test_basic_pitch",
        target_id="target_test_basic_pitch",
        source_role="full_mix",
        source_asset_id=reference_asset.asset_id,
        source_stream_id=reference_stream.stream_id,
        derived_asset_id=derivative_asset.asset_id,
        derived_stream_id=derivative_stream.stream_id,
        operation="explicit_test_downmix_resample",
        parameters={"fixture": True},
        channel_matrix=((0.5, 0.5),),
        sample_time_map_artifact_id=map_artifact_id,
        source_sample_rate=reference_rate,
        derived_sample_rate=derivative_rate,
        source_frame_count=reference_frames,
        derived_frame_count=derivative_frames,
        artifact_sha256=derivative_asset.sha256,
    )
    derivative_artifact_id = workspace.store_json_artifact(
        derivative, artifact_type="target.analysis_derivative.test"
    )
    return derivative, sample_map, derivative_artifact_id


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_basic_pitch_adapter_normalizes_notes_and_preserves_raw_activation_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    derivative, sample_map, derivative_artifact_id = _analysis_derivative_fixture(workspace)
    raw_arrays: dict[str, np.ndarray[Any, Any]] = {
        "onset": np.zeros((5, 88), dtype=np.float32),
        "note": np.zeros((5, 88), dtype=np.float32),
        "contour": np.zeros((5, 264), dtype=np.float32),
        "frame_times_seconds": np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "event_start_seconds": np.asarray([0.1], dtype=np.float64),
        "event_end_seconds": np.asarray([0.3], dtype=np.float64),
        "event_pitch_midi": np.asarray([60], dtype=np.int16),
        "event_amplitude": np.asarray([0.8], dtype=np.float32),
        "pitch_bend_offsets": np.asarray([0, 2], dtype=np.int64),
        "pitch_bend_values": np.asarray([-1, 2], dtype=np.int16),
    }
    raw_arrays["onset"][1, 39] = 0.75
    raw_arrays["note"][1:4, 39] = np.asarray([0.6, 0.8, 1.0], dtype=np.float32)
    raw_arrays["contour"][1:4, 117:120] = 0.55

    def fake_run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_npz = Path(_option(argv, "--output-npz"))
        output_manifest = Path(_option(argv, "--output-manifest"))
        np.savez_compressed(
            output_npz,
            onset=raw_arrays["onset"],
            note=raw_arrays["note"],
            contour=raw_arrays["contour"],
            frame_times_seconds=raw_arrays["frame_times_seconds"],
            event_start_seconds=raw_arrays["event_start_seconds"],
            event_end_seconds=raw_arrays["event_end_seconds"],
            event_pitch_midi=raw_arrays["event_pitch_midi"],
            event_amplitude=raw_arrays["event_amplitude"],
            pitch_bend_offsets=raw_arrays["pitch_bend_offsets"],
            pitch_bend_values=raw_arrays["pitch_bend_values"],
        )
        raw_digest = hashlib.sha256(output_npz.read_bytes()).hexdigest()
        output_manifest.write_text(
            json.dumps(
                {
                    "contract": "basic_pitch_raw_evidence_v1",
                    "basic_pitch_version": "test-basic-pitch-1",
                    "model_path": "test-model.onnx",
                    "model_sha256": "a" * 64,
                    "npz_sha256": raw_digest,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="test worker", stderr="")

    monkeypatch.setattr("music_locator.target_backends.subprocess.run", fake_run)
    run_id = workspace.create_analysis_run("test_basic_pitch", {})
    evidence, evidence_artifact_id = BasicPitchSubprocessTranscriber(
        "/fake/basic-pitch-python"
    ).transcribe(
        workspace,
        derivative,
        sample_map,
        run_id=run_id,
        derivative_artifact_id=derivative_artifact_id,
    )

    assert evidence.backend == BASIC_PITCH_BACKEND
    assert evidence.backend_version == "test-basic-pitch-1"
    assert evidence.derivative_artifact_id == derivative_artifact_id
    assert len(evidence.note_candidates) == 1
    candidate = evidence.note_candidates[0]
    assert candidate.pitch_midi_cents == 6_000.0
    assert candidate.derivative_start_sample == 2_205
    assert candidate.derivative_end_sample == 6_615
    assert candidate.reference_start_sample == 4_410
    assert candidate.reference_end_sample == 13_230
    assert candidate.onset_confidence == pytest.approx(0.75)
    assert candidate.note_confidence == pytest.approx(0.8)
    assert candidate.amplitude == pytest.approx(0.8)

    assert evidence.activation_grid is not None
    grid = evidence.activation_grid
    assert candidate.supporting_evidence_ids == (grid.raw_artifact_id,)
    assert grid.arrays == {name: value.shape for name, value in raw_arrays.items()}
    raw_row = workspace.artifact_row(grid.raw_artifact_id)
    assert raw_row["artifact_type"] == "target.evidence.basic_pitch.raw_npz"
    assert raw_row["media_type"] == "application/x-npz"
    assert raw_row["sha256"] == grid.raw_artifact_sha256
    raw_path = workspace.root / str(raw_row["blob_path"])
    with np.load(raw_path, allow_pickle=False) as persisted:
        assert set(persisted.files) == set(raw_arrays)
        np.testing.assert_array_equal(persisted["contour"], raw_arrays["contour"])
        np.testing.assert_array_equal(persisted["pitch_bend_values"], np.asarray([-1, 2]))

    assert len(evidence.onsets) == 1
    assert evidence.onsets[0].derivative_sample == 2_205
    assert evidence.onsets[0].reference_sample == 4_410
    assert evidence.onsets[0].strength == pytest.approx(0.75)
    assert (
        workspace.load_artifact_json(evidence_artifact_id)["activation_grid"]["raw_artifact_id"]
        == grid.raw_artifact_id
    )
    invocation_id = str(evidence.backend_config["invocation_artifact_id"])
    invocation = workspace.load_artifact_json(invocation_id)
    assert invocation["return_code"] == 0
    assert invocation["worker_manifest"]["npz_sha256"] == grid.raw_artifact_sha256


def _decoded_reference_fixture(workspace: Workspace) -> DecodedTargetReference:
    workspace.initialize()
    sample_rate = 44_100
    frame_count = 44_100
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    audio = np.column_stack(
        (
            0.1 * np.sin(2.0 * np.pi * 220.0 * time),
            0.1 * np.sin(2.0 * np.pi * 330.0 * time),
        )
    ).astype(np.float32)
    source_asset = workspace.store_asset(
        b"generated encoded source fixture",
        kind=AssetKind.TARGET_AUDIO,
        media_type="audio/test",
        original_name="generated-source.audio",
    )
    decoded_asset = workspace.store_asset(
        _float_wav_bytes(audio, sample_rate),
        kind=AssetKind.TARGET_AUDIO_REFERENCE,
        media_type="audio/wav",
        original_name="generated-decoded-reference.wav",
    )
    decoded_stream = insert_audio_stream(
        workspace,
        asset_id=decoded_asset.asset_id,
        sample_rate=sample_rate,
        channels=2,
        frame_count=frame_count,
        sample_format="FLOAT",
    )
    invocation = DecoderInvocation(
        executable="test-ffmpeg",
        arguments=(),
        version_line="test",
        build_configuration="test",
        return_code=0,
        stderr="",
    )
    ffprobe_artifact_id = workspace.store_json_artifact(
        {"schema_version": "0.0.2", "probe": "generated"}, artifact_type="target.ffprobe.test"
    )
    manifest = DecodedReferenceManifest(
        target_id="target_test_separator",
        source_asset_id=source_asset.asset_id,
        source_sha256=source_asset.sha256,
        source_byte_length=source_asset.byte_length,
        source_media_type=source_asset.media_type,
        source_original_name=source_asset.original_name,
        container_format_names=("test",),
        selected_stream_index=0,
        codec_name="test",
        source_sample_rate=sample_rate,
        source_channels=2,
        media_start_microseconds=0,
        presentation_frame_count=frame_count,
        canonicalization="generated_exact_native_samples",
        decoded_asset_id=decoded_asset.asset_id,
        decoded_sha256=decoded_asset.sha256,
        decoded_sample_rate=sample_rate,
        decoded_channels=2,
        decoded_frame_count=frame_count,
        decoded_sample_format="FLOAT",
        ffprobe_artifact_id=ffprobe_artifact_id,
        probe=invocation,
        decoder=invocation,
    )
    manifest_artifact_id = workspace.store_json_artifact(
        manifest, artifact_type="target.decoded_reference_manifest.test"
    )
    time_map = TargetTimeMap(
        target_id=manifest.target_id,
        decoded_stream_id=decoded_stream.stream_id,
        decoded_sample_rate=sample_rate,
        decoded_frame_count=frame_count,
        interpretation="generated reference sample zero is media time zero",
    )
    time_map_artifact_id = workspace.store_json_artifact(
        time_map, artifact_type="time_map.target_reference.test"
    )
    return DecodedTargetReference(
        decode_id="decode_test_separator",
        manifest=manifest,
        manifest_artifact_id=manifest_artifact_id,
        time_map=time_map,
        time_map_artifact_id=time_map_artifact_id,
        decoded_stream=decoded_stream,
        cache_key="generated-decoded-reference",
        cache_hit=False,
    )


def test_htdemucs_adapter_persists_stem_artifacts_and_exact_reference_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "workspace")
    reference = _decoded_reference_fixture(workspace)
    output_rate = 48_000
    output_frames = 48_000

    def fake_run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        output_dir = Path(_option(argv, "--output-dir"))
        output_manifest = Path(_option(argv, "--output-manifest"))
        output_dir.mkdir()
        items: list[dict[str, Any]] = []
        time = np.arange(output_frames, dtype=np.float64) / output_rate
        for role, frequency in (("vocals", 440.0), ("bass", 110.0)):
            mono = (0.05 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
            samples = np.column_stack((mono, mono))
            path = output_dir / f"{role}.wav"
            sf.write(path, samples, output_rate, format="WAV", subtype="FLOAT")
            data = path.read_bytes()
            items.append(
                {
                    "role": role,
                    "filename": path.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "frames": output_frames,
                    "channels": 2,
                    "sample_rate": output_rate,
                    "sample_format": "FLOAT",
                }
            )
        output_manifest.write_text(
            json.dumps(
                {
                    "contract": "htdemucs_float_stems_v1",
                    "worker_sha256": "b" * 64,
                    "demucs_version": "test-demucs-1",
                    "model_state_sha256": "c" * 64,
                    "output_sample_rate": output_rate,
                    "output_channels": 2,
                    "output_frames": output_frames,
                    "stems": items,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="test worker", stderr="")

    monkeypatch.setattr("music_locator.target_backends.subprocess.run", fake_run)
    run_id = workspace.create_analysis_run("test_htdemucs", {})
    sources = HTDemucsSubprocessSeparator("/fake/demucs-python").separate(
        workspace, reference, run_id=run_id
    )

    assert {source.role for source in sources} == {"vocals", "bass"}
    assert len({source.invocation_artifact_id for source in sources}) == 1
    for source in sources:
        assert source.asset.kind is AssetKind.TARGET_STEM
        assert source.stream.asset_id == source.asset.asset_id
        assert source.stream.sample_rate == output_rate
        assert source.stream.channels == 2
        assert source.stream.frame_count == output_frames
        assert source.sample_map.source_stream_id == reference.decoded_stream.stream_id
        assert source.sample_map.derived_stream_id == source.stream.stream_id
        assert source.sample_map.source_to_derived(0) == 0
        assert source.sample_map.source_to_derived(22_050) == 24_000
        assert source.sample_map.source_to_derived(44_100) == 48_000
        assert source.sample_map.derived_to_source(24_000) == 22_050
        assert source.sample_map.derived_to_source(48_000) == 44_100
        assert (
            SampleTimeMap.model_validate(
                workspace.load_artifact_json(source.sample_map_artifact_id)
            )
            == source.sample_map
        )

        assert source.derivative.source_asset_id == reference.decoded_stream.asset_id
        assert source.derivative.source_stream_id == reference.decoded_stream.stream_id
        assert source.derivative.derived_asset_id == source.asset.asset_id
        assert source.derivative.derived_stream_id == source.stream.stream_id
        assert source.derivative.source_role == source.role
        assert source.derivative.normalization == "none"
        assert source.derivative.limiting == "none"
        assert source.derivative.operation == "htdemucs_nonlinear_source_hypothesis"
        assert (
            AnalysisDerivativeEvidence.model_validate(
                workspace.load_artifact_json(source.derivative_artifact_id)
            )
            == source.derivative
        )
        derivative_row = workspace.database.fetchone(
            "SELECT * FROM derivatives WHERE derivative_id = ?", (source.derivative.derivative_id,)
        )
        assert derivative_row is not None
        assert derivative_row["source_stream_id"] == reference.decoded_stream.stream_id
        assert derivative_row["derived_stream_id"] == source.stream.stream_id
        assert derivative_row["operation"] == "htdemucs_nonlinear_source_hypothesis"


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg and ffprobe are required")
def test_public_target_analysis_retains_pyin_and_basic_pitch_view_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "generated-target.wav"
    sample_rate = 44_100
    frame_count = sample_rate
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    sf.write(
        source_path,
        (0.08 * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32),
        sample_rate,
        subtype="PCM_16",
    )
    workspace = Workspace(tmp_path / "workspace")
    target_id = import_target(workspace, source_path)

    class FakePyinPitchTracker:
        def available(self) -> bool:
            return True

        def cache_identity(self) -> dict[str, str]:
            return {"backend": "test.pyin", "version": "1"}

        def analyze(
            self,
            workspace: Workspace,
            derivative: AnalysisDerivativeEvidence,
            sample_map: SampleTimeMap,
            *,
            run_id: str,
            derivative_artifact_id: str | None = None,
        ) -> tuple[TargetEventEvidenceArtifact, str]:
            frames: list[TargetPitchFrame] = []
            frame_index = 0
            for group_index, pitch in enumerate((6_000.0, 6_300.0, 6_700.0)):
                group_start = 1_000 + group_index * 3_000
                for within_group in range(3):
                    center = group_start + within_group * 256
                    frames.append(
                        TargetPitchFrame(
                            frame_id=f"pyin-frame-{frame_index}",
                            frame_index=frame_index,
                            derivative_start_sample=center - 128,
                            derivative_center_sample=center,
                            derivative_end_sample=center + 128,
                            reference_start_sample=sample_map.derived_to_source(center - 128),
                            reference_center_sample=sample_map.derived_to_source(center),
                            reference_end_sample=sample_map.derived_to_source(center + 128),
                            f0_hz=220.0,
                            pitch_midi_cents=pitch,
                            voiced_probability=0.2,
                            voiced=True,
                            backend="librosa.pyin",
                        )
                    )
                    frame_index += 1
            evidence = TargetEventEvidenceArtifact(
                evidence_id=new_id("target_evidence"),
                target_id=derivative.target_id,
                source_role=derivative.source_role,
                derivative_id=derivative.derivative_id,
                derivative_artifact_id=derivative_artifact_id or derivative.derivative_id,
                backend="librosa.pyin",
                backend_version="test-pyin-1",
                backend_config={"fixture": True},
                pitch_frames=tuple(frames),
            )
            artifact_id = workspace.store_json_artifact(
                evidence, artifact_type="target.evidence.test.pyin", analysis_run_id=run_id
            )
            return evidence, artifact_id

    class FakeBasicPitchTranscriber:
        def __init__(
            self,
            _: str | None = None,
            *,
            model_path: str | Path | None = None,
        ) -> None:
            pass

        def available(self) -> bool:
            return True

        def cache_identity(self) -> dict[str, object]:
            return {
                "backend": "test.basic_pitch",
                "version": "1",
                "probe": {"available": "true"},
                "model_identity": {
                    "available": True,
                    "sha256": "fixture-basic-pitch-model-sha256",
                },
            }

        def transcribe(
            self,
            workspace: Workspace,
            derivative: AnalysisDerivativeEvidence,
            sample_map: SampleTimeMap,
            *,
            run_id: str,
            derivative_artifact_id: str | None = None,
        ) -> tuple[TargetEventEvidenceArtifact, str]:
            candidates = tuple(
                TargetNoteCandidate(
                    candidate_id=f"basic-pitch-note-{index}",
                    pitch_midi_cents=pitch,
                    derivative_start_sample=start,
                    derivative_end_sample=start + 800,
                    reference_start_sample=sample_map.derived_to_source(start),
                    reference_end_sample=sample_map.derived_to_source(start + 800),
                    onset_confidence=0.95,
                    note_confidence=0.95,
                    amplitude=0.95,
                    source_role=derivative.source_role,
                    backend=BASIC_PITCH_BACKEND,
                )
                for index, (start, pitch) in enumerate(
                    ((1_000, 6_500.0), (4_000, 6_900.0), (7_000, 7_200.0))
                )
            )
            evidence = TargetEventEvidenceArtifact(
                evidence_id=new_id("target_evidence"),
                target_id=derivative.target_id,
                source_role=derivative.source_role,
                derivative_id=derivative.derivative_id,
                derivative_artifact_id=derivative_artifact_id or derivative.derivative_id,
                backend=BASIC_PITCH_BACKEND,
                backend_version="test-basic-pitch-1",
                backend_config={
                    "fixture": True,
                    "model_sha256": "fixture-basic-pitch-model-sha256",
                },
                note_candidates=candidates,
            )
            artifact_id = workspace.store_json_artifact(
                evidence,
                artifact_type="target.evidence.test.basic_pitch",
                analysis_run_id=run_id,
            )
            return evidence, artifact_id

    monkeypatch.setattr("music_locator.target_backends.PyinPitchTracker", FakePyinPitchTracker)
    monkeypatch.setattr(
        "music_locator.target_backends.BasicPitchSubprocessTranscriber",
        FakeBasicPitchTranscriber,
    )
    analysis = analyze_audio_target(
        workspace,
        target_id,
        options=TargetAnalysisOptions(
            enable_pyin=True,
            enable_basic_pitch=True,
            separator="none",
            max_paths_per_source=2,
            max_total_paths=2,
        ),
    )

    representations = tuple(
        TargetTrackRepresentation.model_validate(workspace.load_artifact_json(artifact_id))
        for artifact_id in analysis.bundle.representation_artifact_ids
    )
    assert len(representations) == 2
    assert {
        representation.audio_voice_hypothesis.extraction_backend
        for representation in representations
        if representation.audio_voice_hypothesis is not None
    } == {"librosa.pyin", BASIC_PITCH_BACKEND}
    assert {
        representation.audio_voice_hypothesis.path_method
        for representation in representations
        if representation.audio_voice_hypothesis is not None
    } == {"predominant_f0_events_75c", "basic_pitch_salience_at_onset"}
