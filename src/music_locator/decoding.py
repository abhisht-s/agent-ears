"""Timestamp-safe encoded target decoding through ffprobe and FFmpeg."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import soundfile as sf

from music_locator.models import (
    AssetKind,
    AudioStreamRecord,
    AudioTargetArtifact,
    DecodedReferenceManifest,
    DecoderInvocation,
    Rational,
    TargetTimeMap,
)
from music_locator.storage import Workspace, new_id, utc_now

DECODE_CONTRACT_VERSION = "ffmpeg_native_float_reference_v2"
DECODER_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _decoder_code_sha256() -> str:
    """Return the decoder code identity frozen when this process started."""

    return DECODER_CODE_SHA256


class TargetAudioDecoder(Protocol):
    def decode(
        self,
        workspace: Workspace,
        target: AudioTargetArtifact,
        *,
        selected_stream_index: int | None = None,
    ) -> DecodedTargetReference: ...


@dataclass(frozen=True)
class DecodedTargetReference:
    decode_id: str
    manifest: DecodedReferenceManifest
    manifest_artifact_id: str
    time_map: TargetTimeMap
    time_map_artifact_id: str
    decoded_stream: AudioStreamRecord
    cache_key: str
    cache_hit: bool


def _bounded(value: str, limit: int = 65_536) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} characters]"


def _microseconds(value: object | None) -> int | None:
    if value is None or str(value) in {"", "N/A"}:
        return None
    return int((Decimal(str(value)) * Decimal(1_000_000)).quantize(Decimal(1), ROUND_HALF_UP))


def _rational(value: object | None) -> Rational | None:
    if value is None or str(value) in {"", "N/A", "0/0"}:
        return None
    fraction = Fraction(str(value))
    return Rational.from_fraction(fraction)


def select_audio_stream(
    probe: dict[str, Any], selected_stream_index: int | None = None
) -> dict[str, Any]:
    streams = [item for item in probe.get("streams", []) if item.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("target contains no decodable audio stream")
    if selected_stream_index is None:
        defaults = [
            item for item in streams if int(item.get("disposition", {}).get("default", 0)) == 1
        ]
        return dict(min(defaults or streams, key=lambda item: int(item["index"])))
    selected = [item for item in streams if int(item["index"]) == selected_stream_index]
    if len(selected) != 1:
        raise ValueError(f"audio stream index {selected_stream_index} does not exist")
    return dict(selected[0])


def packet_timing_summary(
    probe: dict[str, Any], stream: dict[str, Any]
) -> tuple[int, int, int | None]:
    """Return skip, discard, and coded samples without applying either adjustment."""

    stream_index = int(stream["index"])
    packets = [
        packet
        for packet in probe.get("packets", [])
        if int(packet.get("stream_index", -1)) == stream_index
    ]
    skip_samples = 0
    discard_padding = 0
    for packet in packets:
        for side_data in packet.get("side_data_list", []):
            if "skip" in str(side_data.get("side_data_type", "")).lower():
                skip_samples = max(skip_samples, int(side_data.get("skip_samples", 0)))
                discard_padding = max(discard_padding, int(side_data.get("discard_padding", 0)))
    sample_rate = int(stream["sample_rate"])
    coded_samples: int | None = None
    time_base_value = stream.get("time_base")
    packet_durations = [packet.get("duration") for packet in packets]
    if (
        time_base_value not in {None, "N/A", "0/0"}
        and packet_durations
        and all(value not in {None, "N/A"} for value in packet_durations)
    ):
        total_ticks = sum(int(value) for value in packet_durations)
        exact_samples = total_ticks * Fraction(str(time_base_value)) * sample_rate
        quotient, remainder = divmod(exact_samples.numerator, exact_samples.denominator)
        coded_samples = quotient + int(2 * remainder >= exact_samples.denominator)
    else:
        duration_total = Decimal(0)
        duration_count = 0
        for packet in packets:
            duration_time = packet.get("duration_time")
            if duration_time not in {None, "N/A"}:
                duration_total += Decimal(str(duration_time))
                duration_count += 1
        if duration_count:
            coded_samples = int((duration_total * sample_rate).quantize(Decimal(1), ROUND_HALF_UP))
    return skip_samples, discard_padding, coded_samples


def presentation_sample_count(
    stream: dict[str, Any],
    *,
    sample_rate: int,
    coded_samples: int | None,
    skip_samples: int,
    discard_padding: int,
) -> tuple[int | None, str]:
    """Resolve the selected stream's presentation duration on its native sample grid."""

    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts not in {None, "N/A"} and time_base not in {None, "N/A", "0/0"}:
        exact = int(str(duration_ts)) * Fraction(str(time_base)) * sample_rate
        if exact.denominator == 1 and exact.numerator >= 0:
            return exact.numerator, "stream_duration_ts_exact_native_samples"
    if coded_samples is not None:
        adjusted = coded_samples - skip_samples - discard_padding
        if adjusted >= 0:
            return adjusted, "packet_duration_minus_skip_and_discard"
    return None, "unbounded_decoder_output_no_exact_presentation_count"


def _version(executable: str) -> tuple[str, str, str]:
    completed = subprocess.run(
        [executable, "-version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    lines = completed.stdout.splitlines()
    version_line = lines[0] if lines else executable
    configuration = next(
        (
            line.removeprefix("configuration: ")
            for line in lines
            if line.startswith("configuration:")
        ),
        "",
    )
    return version_line, configuration, completed.stdout


def _cache_key(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def insert_audio_stream(
    workspace: Workspace,
    *,
    asset_id: str,
    sample_rate: int,
    channels: int,
    frame_count: int,
    sample_format: str,
) -> AudioStreamRecord:
    record = AudioStreamRecord(
        stream_id=new_id("stream"),
        asset_id=asset_id,
        sample_rate=sample_rate,
        channels=channels,
        frame_count=frame_count,
        sample_format=sample_format,
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


class FFmpegTargetAudioDecoder:
    """Decode one selected stream to immutable native-rate/channel float WAV."""

    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        timeout_seconds: int = 3_600,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("decoder timeout must be positive")
        self.ffmpeg = shutil.which(ffmpeg) or ffmpeg
        self.ffprobe = shutil.which(ffprobe) or ffprobe
        self.timeout_seconds = timeout_seconds

    def _probe(self, source: Path) -> tuple[dict[str, Any], DecoderInvocation]:
        version_line, configuration, _ = _version(self.ffprobe)
        arguments = (
            "-v",
            "error",
            "-print_format",
            "json",
            "-select_streams",
            "a",
            "-show_format",
            "-show_streams",
            "-show_packets",
            str(source),
        )
        completed = subprocess.run(
            [self.ffprobe, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        invocation = DecoderInvocation(
            executable=self.ffprobe,
            arguments=arguments,
            version_line=version_line,
            build_configuration=configuration,
            return_code=completed.returncode,
            stderr=_bounded(completed.stderr),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {_bounded(completed.stderr)}")
        return json.loads(completed.stdout), invocation

    def decode(
        self,
        workspace: Workspace,
        target: AudioTargetArtifact,
        *,
        selected_stream_index: int | None = None,
    ) -> DecodedTargetReference:
        workspace.initialize()
        source_path = workspace.asset_path(target.asset_id)
        probe, probe_invocation = self._probe(source_path)
        stream = select_audio_stream(probe, selected_stream_index)
        stream_index = int(stream["index"])
        ffmpeg_version, ffmpeg_configuration, _ = _version(self.ffmpeg)
        skip_samples, discard_padding, coded_samples = packet_timing_summary(probe, stream)
        expected_frames, canonicalization = presentation_sample_count(
            stream,
            sample_rate=int(stream["sample_rate"]),
            coded_samples=coded_samples,
            skip_samples=skip_samples,
            discard_padding=discard_padding,
        )
        key = _cache_key(
            {
                "contract": DECODE_CONTRACT_VERSION,
                "decoder_code_sha256": _decoder_code_sha256(),
                "target_id": target.target_id,
                "source_sha256": target.source_sha256,
                "stream_index": stream_index,
                "ffprobe_version": probe_invocation.version_line,
                "ffprobe_configuration": probe_invocation.build_configuration,
                "ffmpeg_version": ffmpeg_version,
                "ffmpeg_configuration": ffmpeg_configuration,
                "codec": "pcm_f32le",
                "presentation_frame_count": expected_frames,
                "canonicalization": canonicalization,
            }
        )
        cached = workspace.database.fetchone(
            """
            SELECT target_decodes.*
            FROM target_decodes
            JOIN analysis_runs USING (analysis_run_id)
            WHERE target_decodes.cache_key = ? AND analysis_runs.status = 'succeeded'
            """,
            (key,),
        )
        if cached is not None:
            manifest = DecodedReferenceManifest.model_validate(
                workspace.load_artifact_json(str(cached["manifest_artifact_id"]))
            )
            time_map = TargetTimeMap.model_validate(
                workspace.load_artifact_json(str(cached["time_map_artifact_id"]))
            )
            stream_row = workspace.database.fetchone(
                "SELECT * FROM audio_streams WHERE stream_id = ?",
                (str(cached["decoded_stream_id"]),),
            )
            if stream_row is None:
                raise RuntimeError("cached decoded stream row is missing")
            if (
                manifest.target_id != target.target_id
                or manifest.source_asset_id != target.asset_id
                or manifest.source_sha256 != target.source_sha256
                or time_map.target_id != target.target_id
                or time_map.decoded_stream_id != str(cached["decoded_stream_id"])
            ):
                raise RuntimeError("cached decoded reference does not belong to this target")
            decoded_asset_row = workspace.asset_row(str(cached["decoded_asset_id"]))
            if str(decoded_asset_row["sha256"]) != manifest.decoded_sha256:
                raise RuntimeError("cached decoded reference digest differs from its manifest")
            return DecodedTargetReference(
                decode_id=str(cached["decode_id"]),
                manifest=manifest,
                manifest_artifact_id=str(cached["manifest_artifact_id"]),
                time_map=time_map,
                time_map_artifact_id=str(cached["time_map_artifact_id"]),
                decoded_stream=AudioStreamRecord.model_validate(dict(stream_row)),
                cache_key=key,
                cache_hit=True,
            )

        run_id = workspace.create_analysis_run(
            "decode_audio_target",
            {
                "target_id": target.target_id,
                "stream_index": stream_index,
                "decoder_contract": DECODE_CONTRACT_VERSION,
                "decoder_code_sha256": _decoder_code_sha256(),
                "cache_key": key,
            },
        )
        try:
            probe_artifact_id = workspace.store_json_artifact(
                {
                    "schema_version": "0.0.2",
                    "invocation": probe_invocation.model_dump(mode="json"),
                    "probe": probe,
                },
                artifact_type="target.ffprobe",
                analysis_run_id=run_id,
            )
            with tempfile.TemporaryDirectory(prefix="music-locator-decode-") as temporary:
                output = Path(temporary) / "reference.wav"
                arguments = (
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-i",
                    str(source_path),
                    "-map",
                    f"0:{stream_index}",
                    "-vn",
                    "-sn",
                    "-dn",
                    "-map_metadata",
                    "-1",
                    *(
                        ("-af", f"atrim=end_sample={expected_frames}")
                        if expected_frames is not None
                        else ()
                    ),
                    "-c:a",
                    "pcm_f32le",
                    "-y",
                    str(output),
                )
                completed = subprocess.run(
                    [self.ffmpeg, *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                decode_invocation = DecoderInvocation(
                    executable=self.ffmpeg,
                    arguments=arguments,
                    version_line=ffmpeg_version,
                    build_configuration=ffmpeg_configuration,
                    return_code=completed.returncode,
                    stderr=_bounded(completed.stderr),
                )
                if completed.returncode != 0 or not output.is_file():
                    raise RuntimeError(f"FFmpeg decode failed: {_bounded(completed.stderr)}")
                info = sf.info(output)
                if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
                    raise RuntimeError("FFmpeg produced an empty or invalid reference stream")
                if expected_frames is not None and info.frames != expected_frames:
                    raise RuntimeError(
                        "FFmpeg canonical decode length differs from the exact selected-stream "
                        f"presentation count: emitted {info.frames}, expected {expected_frames}"
                    )
                decoded_asset = workspace.store_asset(
                    output,
                    kind=AssetKind.TARGET_AUDIO_REFERENCE,
                    media_type="audio/wav",
                    original_name=f"{target.target_id}-stream-{stream_index}-reference.wav",
                )

            decoded_stream = insert_audio_stream(
                workspace,
                asset_id=decoded_asset.asset_id,
                sample_rate=info.samplerate,
                channels=info.channels,
                frame_count=info.frames,
                sample_format=info.subtype,
            )
            time_base = _rational(stream.get("time_base"))
            media_start_us = _microseconds(stream.get("start_time")) or 0
            warnings: list[str] = []
            if decoded_stream.sample_rate != int(stream["sample_rate"]):
                warnings.append("decoded_sample_rate_differs_from_probed_stream")
            if decoded_stream.channels != int(stream["channels"]):
                warnings.append("decoded_channel_count_differs_from_probed_stream")
            if decode_invocation.stderr:
                warnings.append("ffmpeg_emitted_decode_warnings")
            if expected_frames is None:
                warnings.append("presentation_sample_count_unavailable_unbounded_decode")
            if ffmpeg_configuration and "--enable-gpl" in ffmpeg_configuration:
                warnings.append("local_ffmpeg_build_is_gpl_enabled_do_not_redistribute")
            manifest = DecodedReferenceManifest(
                target_id=target.target_id,
                source_asset_id=target.asset_id,
                source_sha256=target.source_sha256,
                source_byte_length=target.byte_length,
                source_media_type=target.media_type,
                source_original_name=target.original_name,
                container_format_names=tuple(
                    name
                    for name in str(probe.get("format", {}).get("format_name", "")).split(",")
                    if name
                ),
                selected_stream_index=stream_index,
                codec_name=str(stream.get("codec_name", "unknown")),
                codec_long_name=(
                    str(stream["codec_long_name"]) if stream.get("codec_long_name") else None
                ),
                source_sample_format=(
                    str(stream["sample_fmt"]) if stream.get("sample_fmt") else None
                ),
                source_sample_rate=int(stream["sample_rate"]),
                source_channels=int(stream["channels"]),
                source_channel_layout=(
                    str(stream["channel_layout"]) if stream.get("channel_layout") else None
                ),
                media_time_base=time_base,
                media_start_pts=(
                    int(stream["start_pts"])
                    if stream.get("start_pts") not in {None, "N/A"}
                    else None
                ),
                media_start_microseconds=media_start_us,
                reported_duration_microseconds=(
                    _microseconds(stream.get("duration"))
                    or _microseconds(probe.get("format", {}).get("duration"))
                ),
                presentation_frame_count=expected_frames,
                canonicalization=canonicalization,
                coded_sample_count=coded_samples,
                skip_samples=skip_samples,
                discard_padding=discard_padding,
                decoded_asset_id=decoded_asset.asset_id,
                decoded_sha256=decoded_asset.sha256,
                decoded_sample_rate=decoded_stream.sample_rate,
                decoded_channels=decoded_stream.channels,
                decoded_channel_layout=(
                    str(stream["channel_layout"]) if stream.get("channel_layout") else None
                ),
                decoded_frame_count=decoded_stream.frame_count,
                decoded_sample_format=decoded_stream.sample_format,
                ffprobe_artifact_id=probe_artifact_id,
                probe=probe_invocation,
                decoder=decode_invocation,
                warnings=tuple(warnings),
            )
            manifest_artifact_id = workspace.store_json_artifact(
                manifest,
                artifact_type="target.decoded_reference_manifest",
                analysis_run_id=run_id,
            )
            time_map = TargetTimeMap(
                target_id=target.target_id,
                decoded_stream_id=decoded_stream.stream_id,
                decoded_sample_rate=decoded_stream.sample_rate,
                decoded_frame_count=decoded_stream.frame_count,
                media_time_base=time_base,
                reference_zero_media_pts=manifest.media_start_pts,
                reference_zero_media_microseconds=media_start_us,
                interpretation=(
                    "Decoded sample zero is the first sample emitted after FFmpeg applies codec "
                    "priming/skip and discard metadata; stream start_time is reported separately "
                    "as the container presentation coordinate and is not added twice."
                ),
            )
            time_map_artifact_id = workspace.store_json_artifact(
                time_map,
                artifact_type="time_map.target_reference_to_media",
                analysis_run_id=run_id,
            )
            decode_id = new_id("decode")
            # Publish a cache row only after the producing run is complete. A
            # crash before this point leaves immutable orphan evidence, never a
            # partially reusable decode.
            workspace.finish_analysis_run(run_id)
            workspace.database.execute(
                """
                INSERT INTO target_decodes(
                    decode_id, target_id, cache_key, analysis_run_id,
                    manifest_artifact_id, time_map_artifact_id,
                    decoded_asset_id, decoded_stream_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decode_id,
                    target.target_id,
                    key,
                    run_id,
                    manifest_artifact_id,
                    time_map_artifact_id,
                    decoded_asset.asset_id,
                    decoded_stream.stream_id,
                    utc_now(),
                ),
            )
            return DecodedTargetReference(
                decode_id=decode_id,
                manifest=manifest,
                manifest_artifact_id=manifest_artifact_id,
                time_map=time_map,
                time_map_artifact_id=time_map_artifact_id,
                decoded_stream=decoded_stream,
                cache_key=key,
                cache_hit=False,
            )
        except Exception as error:
            workspace.finish_analysis_run(run_id, error=str(error))
            raise
