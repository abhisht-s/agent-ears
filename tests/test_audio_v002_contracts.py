from __future__ import annotations

import shutil
import subprocess
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from music_locator.decoding import (
    FFmpegTargetAudioDecoder,
    packet_timing_summary,
    select_audio_stream,
)
from music_locator.models import (
    AudioTargetArtifact,
    AudioVoiceHypothesisSummary,
    Rational,
    TargetCoordinateKind,
    TargetTimeMap,
    TargetTrackRepresentation,
)
from music_locator.storage import Workspace
from music_locator.targets import import_target, load_target


def test_select_audio_stream_prefers_default_and_honors_explicit_index() -> None:
    probe = {
        "streams": [
            {"index": 0, "codec_type": "video", "disposition": {"default": 1}},
            {"index": 3, "codec_type": "audio", "disposition": {"default": 0}},
            {"index": 2, "codec_type": "audio", "disposition": {"default": 1}},
            {"index": 5, "codec_type": "audio", "disposition": {"default": 1}},
        ]
    }

    assert select_audio_stream(probe)["index"] == 2
    assert select_audio_stream(probe, selected_stream_index=3)["index"] == 3

    with pytest.raises(ValueError, match="audio stream index 4 does not exist"):
        select_audio_stream(probe, selected_stream_index=4)
    with pytest.raises(ValueError, match="no decodable audio stream"):
        select_audio_stream({"streams": probe["streams"][:1]})


def test_packet_timing_summary_preserves_duro_like_gapless_metadata() -> None:
    # Duro's observed MP3 packet geometry: 7,111 MPEG frames of 1,152 samples,
    # with FFmpeg reporting 1,105 leading skip samples and 815 trailing samples.
    getcontext().prec = 40
    sample_rate = 44_100
    packet_count = 7_111
    samples_per_packet = 1_152
    duration_time = str(Decimal(samples_per_packet) / Decimal(sample_rate))
    packets = [
        {
            "stream_index": 1,
            "duration_time": duration_time,
            "side_data_list": [],
        }
        for _ in range(packet_count)
    ]
    packets[0]["side_data_list"] = [
        {
            "side_data_type": "Skip Samples",
            "skip_samples": 1_105,
            "discard_padding": 0,
        }
    ]
    packets[-1]["side_data_list"] = [
        {
            "side_data_type": "Skip Samples",
            "skip_samples": 0,
            "discard_padding": 815,
        }
    ]
    # A packet for another stream must not affect the selected stream summary.
    packets.append(
        {
            "stream_index": 2,
            "duration_time": "10.0",
            "side_data_list": [
                {
                    "side_data_type": "Skip Samples",
                    "skip_samples": 99_999,
                    "discard_padding": 99_999,
                }
            ],
        }
    )

    skip, discard, coded = packet_timing_summary(
        {"packets": packets},
        {"index": 1, "sample_rate": str(sample_rate)},
    )

    assert skip == 1_105
    assert discard == 815
    assert coded == packet_count * samples_per_packet == 8_191_872
    assert coded - skip - discard == 8_189_952


def test_target_time_map_round_trips_from_nonzero_media_start() -> None:
    mapping = TargetTimeMap(
        target_id="target-audio",
        decoded_stream_id="stream-reference",
        decoded_sample_rate=48_000,
        decoded_frame_count=96_000,
        media_time_base=Rational(numerator=1, denominator=90_000),
        reference_zero_media_pts=112_500,
        reference_zero_media_microseconds=1_250_000,
        interpretation="synthetic nonzero media start",
    )

    assert mapping.reference_sample_to_media_microseconds(0) == 1_250_000
    assert mapping.reference_sample_to_media_microseconds(96_000) == 3_250_000
    for sample in range(0, 96_001, 137):
        media_time = mapping.reference_sample_to_media_microseconds(sample)
        recovered = mapping.media_microseconds_to_reference_sample(media_time)
        assert abs(recovered - sample) <= 1

    with pytest.raises(ValueError, match="precedes"):
        mapping.media_microseconds_to_reference_sample(1_249_999)


def _audio_representation(**updates: object) -> TargetTrackRepresentation:
    hypothesis = AudioVoiceHypothesisSummary(
        view_id="view-full-mix-0",
        source_role="full_mix",
        source_derivative_artifact_id="artifact-derivative",
        evidence_artifact_id="artifact-evidence",
        extraction_backend="generated-test",
        extraction_backend_version="1",
        path_method="generated",
        path_rank=1,
        path_score=0.82,
        event_count=1,
    )
    values: dict[str, object] = {
        "schema_version": "0.0.2",
        "target_id": "target-audio",
        "track_index": None,
        "view_id": "view-full-mix-0",
        "source_group_id": "full_mix",
        "event_ids": ("event-1",),
        "pitch_cents": (6_900,),
        "relative_pitch_cents": (0.0,),
        "interval_cents": (),
        "contour_steps": (),
        "onset_microseconds": (1_252_083,),
        "log_ioi_milli": (),
        "coordinate_kind": TargetCoordinateKind.DECODED_SAMPLES,
        "start_samples": (100,),
        "end_samples": (340,),
        "reference_sample_rate": 48_000,
        "reference_zero_media_microseconds": 1_250_000,
        "event_confidences": (0.82,),
        "source_evidence_refs": ("evidence-1",),
        "audio_voice_hypothesis": hypothesis,
    }
    values.update(updates)
    return TargetTrackRepresentation.model_validate(values)


def test_audio_target_representation_requires_decoded_sample_ranges() -> None:
    representation = _audio_representation()

    assert representation.coordinate_kind is TargetCoordinateKind.DECODED_SAMPLES
    assert representation.start_samples == (100,)
    assert representation.end_samples == (340,)
    assert representation.start_ticks == ()
    assert representation.end_ticks == ()

    with pytest.raises(ValidationError, match="decoded sample ranges"):
        _audio_representation(start_samples=(), end_samples=())


def test_audio_target_representation_rejects_fake_midi_tick_ranges() -> None:
    with pytest.raises(ValidationError, match="cannot carry MIDI tick ranges"):
        _audio_representation(start_ticks=(1,), end_ticks=(2,))


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for the decoder integration test",
)
def test_ffmpeg_decoder_preserves_native_stereo_clock_frames_and_peak(
    tmp_path: Path,
) -> None:
    sample_rate = 48_000
    frame_count = 4_800
    signal = np.zeros((frame_count, 2), dtype=np.float32)
    signal[137, 0] = 0.75
    signal[911, 1] = -0.375
    source = tmp_path / "synthetic-stereo.wav"
    sf.write(source, signal, sample_rate, subtype="FLOAT")

    workspace = Workspace(tmp_path / "workspace")
    target_id = import_target(workspace, source)
    target, _ = load_target(workspace, target_id)
    assert isinstance(target, AudioTargetArtifact)

    decoded = FFmpegTargetAudioDecoder().decode(workspace, target)
    reference_path = workspace.asset_path(decoded.decoded_stream.asset_id)
    samples, decoded_rate = sf.read(reference_path, always_2d=True, dtype="float32")

    assert decoded_rate == sample_rate
    assert decoded.manifest.source_sample_rate == sample_rate
    assert decoded.manifest.decoded_sample_rate == sample_rate
    assert decoded.manifest.source_channels == 2
    assert decoded.manifest.decoded_channels == 2
    assert decoded.manifest.decoded_frame_count == frame_count
    assert decoded.decoded_stream.frame_count == frame_count
    assert samples.shape == signal.shape
    np.testing.assert_allclose(
        np.max(np.abs(samples), axis=0),
        np.max(np.abs(signal), axis=0),
        rtol=0.0,
        atol=1e-7,
    )
    assert decoded.time_map.reference_sample_to_media_microseconds(frame_count) == 100_000


def _encode_synthetic_aac_m4a(path: Path, *, output_timestamp_offset: int | None = None) -> None:
    """Create a one-second AAC fixture without checking any private media into Git."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:a",
        "aac",
    ]
    if output_timestamp_offset is not None:
        command.extend(("-output_ts_offset", str(output_timestamp_offset)))
    command.extend(("-y", str(path)))
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for the AAC priming regression test",
)
def test_ffmpeg_decoder_trims_plain_aac_to_exact_presentation_frames(
    tmp_path: Path,
) -> None:
    source = tmp_path / "plain.m4a"
    _encode_synthetic_aac_m4a(source)

    workspace = Workspace(tmp_path / "workspace")
    target_id = import_target(workspace, source)
    target, _ = load_target(workspace, target_id)
    assert isinstance(target, AudioTargetArtifact)

    decoded = FFmpegTargetAudioDecoder().decode(workspace, target)

    assert decoded.manifest.codec_name == "aac"
    assert decoded.manifest.source_sample_rate == 48_000
    assert decoded.manifest.presentation_frame_count == 48_000
    assert decoded.manifest.decoded_frame_count == 48_000
    assert decoded.decoded_stream.frame_count == 48_000
    assert decoded.manifest.canonicalization == "stream_duration_ts_exact_native_samples"
    assert "-af" in decoded.manifest.decoder.arguments
    assert "atrim=end_sample=48000" in decoded.manifest.decoder.arguments
    assert decoded.time_map.reference_sample_to_media_microseconds(0) == 0
    assert decoded.time_map.reference_sample_to_media_microseconds(48_000) == 1_000_000


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for the AAC nonzero-start regression test",
)
def test_ffmpeg_decoder_preserves_nonzero_m4a_start_and_exact_end_mapping(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonzero-start.m4a"
    _encode_synthetic_aac_m4a(source, output_timestamp_offset=2)

    workspace = Workspace(tmp_path / "workspace")
    target_id = import_target(workspace, source)
    target, _ = load_target(workspace, target_id)
    assert isinstance(target, AudioTargetArtifact)

    decoded = FFmpegTargetAudioDecoder().decode(workspace, target)
    manifest = decoded.manifest
    time_map = decoded.time_map

    # FFmpeg's native AAC encoder contributes 1,024 samples of encoder delay.
    # With a two-second output offset, the MP4 edit-list geometry is therefore
    # start_pts=94,944 and duration_ts=49,024 on the 48 kHz media clock.
    assert manifest.media_time_base == Rational(numerator=1, denominator=48_000)
    assert manifest.media_start_pts == 94_944
    assert manifest.media_start_microseconds == 1_978_000
    assert manifest.presentation_frame_count == 49_024
    assert manifest.decoded_frame_count == 49_024
    assert decoded.decoded_stream.frame_count == 49_024
    assert manifest.canonicalization == "stream_duration_ts_exact_native_samples"
    assert "atrim=end_sample=49024" in manifest.decoder.arguments

    start_us = time_map.reference_sample_to_media_microseconds(0)
    end_us = time_map.reference_sample_to_media_microseconds(49_024)
    assert start_us == manifest.media_start_microseconds
    assert end_us == 2_999_333
    assert end_us - start_us == 1_021_333
