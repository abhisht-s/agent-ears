"""Standalone demucs.api worker with a dependency-free FLOAT WAV writer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import struct
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

CONTRACT = "htdemucs_float_stems_v1"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--expected-sample-rate", type=int)
    parser.add_argument("--expected-channels", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shifts", type=int, required=True)
    parser.add_argument("--overlap", type=float, required=True)
    parser.add_argument("--split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--segment", type=int)
    parser.add_argument("--jobs", type=int, required=True)
    return parser.parse_args()


def _model_state_sha256(model: Any) -> str:
    import torch  # type: ignore[import-not-found]

    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        shape_bytes = json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        for part in (name_bytes, dtype_bytes, shape_bytes):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        try:
            raw = tensor.numpy().tobytes(order="C")
        except TypeError:
            raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _float32_wave_bytes(samples: np.ndarray[Any, Any], sample_rate: int) -> bytes:
    """Serialize [frames, channels] IEEE float32 without TorchCodec/torchaudio."""

    if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("FLOAT WAV samples must have non-empty [frames, channels] shape")
    if sample_rate <= 0:
        raise ValueError("FLOAT WAV sample rate must be positive")
    if not np.all(np.isfinite(samples)):
        raise ValueError("FLOAT WAV samples contain non-finite values")
    frames = np.ascontiguousarray(samples, dtype="<f4")
    channels = int(frames.shape[1])
    bits_per_sample = 32
    block_align = channels * bits_per_sample // 8
    byte_rate = sample_rate * block_align
    payload = frames.tobytes(order="C")
    # Non-PCM WAVE convention includes a fact chunk.  The sample count is exact.
    fact = struct.pack("<4sII", b"fact", 4, int(frames.shape[0]))
    fmt = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        3,  # WAVE_FORMAT_IEEE_FLOAT
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    data_header = struct.pack("<4sI", b"data", len(payload))
    riff_size = 4 + len(fmt) + len(fact) + len(data_header) + len(payload)
    if riff_size > 0xFFFF_FFFF:
        raise ValueError("stem is too large for a classic RIFF/WAVE artifact")
    return struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE") + fmt + fact + data_header + payload


def main() -> None:
    args = _arguments()
    from demucs.api import Separator  # type: ignore[import-not-found]

    separator = Separator(
        model=args.model,
        repo=args.repo,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
        split=args.split,
        segment=args.segment,
        jobs=args.jobs,
        progress=False,
    )
    worker_path = Path(__file__)
    identity = {
        "contract": CONTRACT,
        "worker_sha256": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "demucs_version": _distribution_version("demucs"),
        "torch_version": _distribution_version("torch"),
        "torchaudio_version": _distribution_version("torchaudio"),
        "numpy_version": _distribution_version("numpy"),
        "soundfile_version": _distribution_version("soundfile"),
        "model": args.model,
        "model_class": f"{type(separator.model).__module__}.{type(separator.model).__qualname__}",
        "model_state_sha256": _model_state_sha256(separator.model),
        "repo": str(args.repo.resolve()) if args.repo is not None else "demucs-default-remote",
    }
    if args.identity_only:
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(
            json.dumps(
                {**identity, "mode": "identity"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        return
    if (
        args.input is None
        or args.output_dir is None
        or args.expected_sample_rate is None
        or args.expected_channels is None
        or args.expected_frames is None
    ):
        raise ValueError("Demucs separation mode requires input/output and stream dimensions")
    info = sf.info(args.input)
    if (
        info.format != "WAV"
        or info.subtype != "FLOAT"
        or info.samplerate != args.expected_sample_rate
        or info.channels != args.expected_channels
        or info.frames != args.expected_frames
    ):
        raise ValueError("Demucs input differs from the canonical FLOAT reference contract")
    input_audio, input_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if input_rate != args.expected_sample_rate or input_audio.shape != (
        args.expected_frames,
        args.expected_channels,
    ):
        raise ValueError("Demucs decoded input shape/rate differs from its manifest")
    if not np.all(np.isfinite(input_audio)):
        raise ValueError("Demucs input contains non-finite samples")

    separated_input, stems = separator.separate_audio_file(args.input)
    mix = separated_input.detach().cpu().numpy()
    if mix.ndim != 2:
        raise RuntimeError("demucs.api returned an input tensor with an invalid shape")
    output_channels = int(separator.audio_channels)
    output_rate = int(separator.samplerate)
    # demucs.api.AudioFile delegates sample-rate conversion to FFmpeg, whose emitted
    # finite-duration stream covers the final fractional output sample (ceiling).
    expected_output_frames = math.ceil(args.expected_frames * output_rate / input_rate)
    if mix.shape != (output_channels, expected_output_frames):
        raise RuntimeError(
            "demucs.api returned an unexpected resampled reference shape: "
            f"{mix.shape}, expected {(output_channels, expected_output_frames)}"
        )
    if not np.all(np.isfinite(mix)):
        raise RuntimeError("demucs.api returned a non-finite resampled reference")
    if set(stems) != set(separator.model.sources):
        raise RuntimeError("demucs.api stem roles differ from the loaded model sources")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem_manifest: list[dict[str, Any]] = []
    for role in separator.model.sources:
        if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in role):
            raise RuntimeError(f"demucs.api returned an unsafe source role: {role!r}")
        tensor = stems[role].detach().cpu().numpy()
        if tensor.shape != mix.shape:
            raise RuntimeError(f"demucs.api stem shape differs from its input: {role}")
        frames_channels = np.asarray(tensor.T, dtype=np.float32)
        data = _float32_wave_bytes(frames_channels, output_rate)
        filename = f"{role}.wav"
        path = args.output_dir / filename
        path.write_bytes(data)
        # Re-open every artifact so the worker verifies its own serializer contract.
        stem_info = sf.info(path)
        if (
            stem_info.format != "WAV"
            or stem_info.subtype != "FLOAT"
            or stem_info.samplerate != output_rate
            or stem_info.channels != output_channels
            or stem_info.frames != expected_output_frames
        ):
            raise RuntimeError(f"self-written FLOAT WAV failed conformance: {role}")
        stem_manifest.append(
            {
                "role": role,
                "filename": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "frames": expected_output_frames,
                "channels": output_channels,
                "sample_rate": output_rate,
                "sample_format": "FLOAT",
                "peak": float(np.max(np.abs(frames_channels))) if frames_channels.size else 0.0,
            }
        )

    manifest = {
        **identity,
        "mode": "separate",
        "config": {
            "device": args.device,
            "shifts": args.shifts,
            "overlap": args.overlap,
            "split": args.split,
            "segment": args.segment,
            "jobs": args.jobs,
            "progress": False,
            "output_serialization": "own_ieee_float32_wave_v1",
            "output_normalization": "none",
            "output_limiting": "none",
        },
        "input": {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        },
        "output_sample_rate": output_rate,
        "output_channels": output_channels,
        "output_frames": expected_output_frames,
        "stems": stem_manifest,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
