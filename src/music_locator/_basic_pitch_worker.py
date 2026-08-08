"""Standalone Basic Pitch worker for an isolated Python environment.

This file deliberately imports no music_locator modules so it can be executed by
an environment containing Basic Pitch without installing the application package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

CONTRACT = "basic_pitch_raw_evidence_v1"


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {path}")
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = item.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--expected-sample-rate", type=int, required=True)
    parser.add_argument("--expected-channels", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--onset-threshold", type=float, required=True)
    parser.add_argument("--frame-threshold", type=float, required=True)
    parser.add_argument("--minimum-note-length-ms", type=float, required=True)
    parser.add_argument("--minimum-frequency-hz", type=float)
    parser.add_argument("--maximum-frequency-hz", type=float)
    parser.add_argument("--melodia-trick", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.expected_sample_rate != 22_050 or args.expected_channels != 1:
        raise ValueError("Basic Pitch worker accepts only an explicit mono 22050-Hz derivative")
    info = sf.info(args.input)
    if (
        info.samplerate != args.expected_sample_rate
        or info.channels != args.expected_channels
        or info.frames != args.expected_frames
    ):
        raise ValueError("Basic Pitch input header differs from the adapter contract")
    audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if sample_rate != args.expected_sample_rate or audio.shape != (args.expected_frames, 1):
        raise ValueError("Basic Pitch decoded input differs from the adapter contract")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Basic Pitch input contains non-finite samples")

    from basic_pitch.inference import (  # type: ignore[import-not-found]
        ICASSP_2022_MODEL_PATH,
        predict,
    )
    from basic_pitch.note_creation import (  # type: ignore[import-not-found]
        model_frames_to_time,
    )

    model_path = (args.model_path or Path(ICASSP_2022_MODEL_PATH)).resolve()
    model_output, _, note_events = predict(
        args.input,
        model_or_model_path=model_path,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_length=args.minimum_note_length_ms,
        minimum_frequency=args.minimum_frequency_hz,
        maximum_frequency=args.maximum_frequency_hz,
        multiple_pitch_bends=False,
        melodia_trick=args.melodia_trick,
    )
    expected_keys = {"onset", "note", "contour"}
    if set(model_output) != expected_keys:
        raise RuntimeError(f"unexpected Basic Pitch model output keys: {sorted(model_output)}")
    onset = np.asarray(model_output["onset"], dtype=np.float32)
    note = np.asarray(model_output["note"], dtype=np.float32)
    contour = np.asarray(model_output["contour"], dtype=np.float32)
    if onset.ndim != 2 or note.shape != onset.shape or onset.shape[1] != 88:
        raise RuntimeError("Basic Pitch note/onset grids do not have [frames, 88] shape")
    if contour.shape != (onset.shape[0], 264):
        raise RuntimeError("Basic Pitch contour grid does not have [frames, 264] shape")
    for name, values in (("onset", onset), ("note", note), ("contour", contour)):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Basic Pitch {name} grid contains non-finite values")
        if values.size and (float(np.min(values)) < -1e-6 or float(np.max(values)) > 1.000001):
            raise RuntimeError(f"Basic Pitch {name} grid lies outside [0, 1]")

    frame_times = np.asarray(model_frames_to_time(onset.shape[0]), dtype=np.float64)
    if not np.all(np.isfinite(frame_times)) or np.any(np.diff(frame_times) < 0):
        raise RuntimeError("Basic Pitch frame times are not finite and monotonic")
    event_starts: list[float] = []
    event_ends: list[float] = []
    event_pitches: list[int] = []
    event_amplitudes: list[float] = []
    pitch_bend_offsets = [0]
    pitch_bend_values: list[int] = []
    for raw_event in note_events:
        if len(raw_event) != 5:
            raise RuntimeError("Basic Pitch returned an unexpected note-event tuple")
        start, end, pitch, amplitude, bends = raw_event
        if not float(end) > float(start):
            raise RuntimeError("Basic Pitch returned a non-positive note event")
        if not 0 <= int(pitch) <= 127:
            raise RuntimeError("Basic Pitch returned a pitch outside the MIDI note domain")
        if not 0.0 <= float(amplitude) <= 1.0:
            raise RuntimeError("Basic Pitch returned a note amplitude outside [0, 1]")
        event_starts.append(float(start))
        event_ends.append(float(end))
        event_pitches.append(int(pitch))
        event_amplitudes.append(float(amplitude))
        if bends is not None:
            pitch_bend_values.extend(int(value) for value in bends)
        pitch_bend_offsets.append(len(pitch_bend_values))

    arrays: dict[str, np.ndarray[Any, Any]] = {
        "onset": onset,
        "note": note,
        "contour": contour,
        "frame_times_seconds": frame_times,
        "event_start_seconds": np.asarray(event_starts, dtype=np.float64),
        "event_end_seconds": np.asarray(event_ends, dtype=np.float64),
        "event_pitch_midi": np.asarray(event_pitches, dtype=np.int16),
        "event_amplitude": np.asarray(event_amplitudes, dtype=np.float32),
        "pitch_bend_offsets": np.asarray(pitch_bend_offsets, dtype=np.int64),
        "pitch_bend_values": np.asarray(pitch_bend_values, dtype=np.int16),
    }
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        onset=arrays["onset"],
        note=arrays["note"],
        contour=arrays["contour"],
        frame_times_seconds=arrays["frame_times_seconds"],
        event_start_seconds=arrays["event_start_seconds"],
        event_end_seconds=arrays["event_end_seconds"],
        event_pitch_midi=arrays["event_pitch_midi"],
        event_amplitude=arrays["event_amplitude"],
        pitch_bend_offsets=arrays["pitch_bend_offsets"],
        pitch_bend_values=arrays["pitch_bend_values"],
    )
    npz_digest = hashlib.sha256(args.output_npz.read_bytes()).hexdigest()
    manifest = {
        "contract": CONTRACT,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "basic_pitch_version": _distribution_version("basic-pitch"),
        "numpy_version": _distribution_version("numpy"),
        "soundfile_version": _distribution_version("soundfile"),
        "onnxruntime_version": _distribution_version("onnxruntime"),
        "tensorflow_version": _distribution_version("tensorflow"),
        "model_path": str(model_path),
        "model_sha256": _path_sha256(model_path),
        "input": {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
        },
        "config": {
            "onset_threshold": args.onset_threshold,
            "frame_threshold": args.frame_threshold,
            "minimum_note_length_ms": args.minimum_note_length_ms,
            "minimum_frequency_hz": args.minimum_frequency_hz,
            "maximum_frequency_hz": args.maximum_frequency_hz,
            "melodia_trick": args.melodia_trick,
            "multiple_pitch_bends": False,
        },
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "note_event_count": len(note_events),
        "timing": {
            "sample_rate": 22_050,
            "fft_hop": 256,
            "frame_times": "basic_pitch.note_creation.model_frames_to_time",
        },
        "npz_sha256": npz_digest,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
