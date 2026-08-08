"""Local, evidence-preserving target extraction backend adapters.

Heavy model stacks intentionally run in caller-selected Python environments.  The
main music-locator environment only owns validation, provenance, immutable storage,
and conversion from backend coordinates to decoded-reference sample coordinates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal, Protocol

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from music_locator.decoding import DecodedTargetReference
from music_locator.models import (
    ActivationGridEvidence,
    AnalysisDerivativeEvidence,
    AssetKind,
    AssetRecord,
    AudioStreamRecord,
    DerivativeRecord,
    SampleTimeMap,
    TargetEventEvidenceArtifact,
    TargetNoteCandidate,
    TargetOnsetEvidence,
    TargetPitchFrame,
    TimeMapSegment,
)
from music_locator.storage import Workspace, new_id, utc_now

PYIN_BACKEND = "librosa.pyin"
BASIC_PITCH_BACKEND = "spotify.basic_pitch"
HTDEMUCS_BACKEND = "demucs.api.Separator"
_BASIC_PITCH_WORKER_CONTRACT = "basic_pitch_raw_evidence_v1"
_HTDEMUCS_WORKER_CONTRACT = "htdemucs_float_stems_v1"
TARGET_BACKENDS_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class PitchTracker(Protocol):
    def available(self) -> bool: ...

    def cache_identity(self) -> dict[str, Any]: ...

    def analyze(
        self,
        workspace: Workspace,
        derivative: AnalysisDerivativeEvidence,
        sample_map: SampleTimeMap,
        *,
        run_id: str,
        derivative_artifact_id: str | None = None,
    ) -> tuple[TargetEventEvidenceArtifact, str]: ...


class PolyphonicTranscriber(Protocol):
    def available(self) -> bool: ...

    def cache_identity(self) -> dict[str, Any]: ...

    def transcribe(
        self,
        workspace: Workspace,
        derivative: AnalysisDerivativeEvidence,
        sample_map: SampleTimeMap,
        *,
        run_id: str,
        derivative_artifact_id: str | None = None,
    ) -> tuple[TargetEventEvidenceArtifact, str]: ...


@dataclass(frozen=True)
class SeparatedSource:
    """One non-authoritative separator output and its decoded-reference lineage."""

    role: str
    asset: AssetRecord
    stream: AudioStreamRecord
    sample_map: SampleTimeMap
    sample_map_artifact_id: str
    derivative: AnalysisDerivativeEvidence
    derivative_artifact_id: str
    invocation_artifact_id: str


class StemSeparator(Protocol):
    def available(self) -> bool: ...

    def cache_identity(self) -> dict[str, Any]: ...

    def separate(
        self,
        workspace: Workspace,
        decoded_ref: DecodedTargetReference,
        *,
        run_id: str,
    ) -> tuple[SeparatedSource, ...]: ...


@dataclass(frozen=True)
class PyinConfig:
    fmin_hz: float = float(librosa.note_to_hz("C2"))
    fmax_hz: float = float(librosa.note_to_hz("C7"))
    frame_length: int = 2_048
    hop_length: int = 256
    center: bool = True
    pad_mode: Literal["constant"] = "constant"
    onset_backtrack: bool = True

    def validate(self) -> None:
        if not (0.0 < self.fmin_hz < self.fmax_hz):
            raise ValueError("pYIN frequency bounds must be positive and ordered")
        if self.frame_length <= 0 or self.hop_length <= 0:
            raise ValueError("pYIN frame and hop lengths must be positive")
        if self.frame_length < self.hop_length:
            raise ValueError("pYIN frame length must be at least the hop length")
        if self.pad_mode != "constant":
            raise ValueError("only deterministic constant pYIN padding is supported")


def _artifact_sha256(workspace: Workspace, artifact_id: str) -> str:
    return str(workspace.artifact_row(artifact_id)["sha256"])


def _resolve_executable(executable: str | Path | None) -> str:
    value = sys.executable if executable is None else str(executable)
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.parent != Path("."):
        # Preserve virtual-environment launcher symlinks: resolving them to the
        # base interpreter discards the venv's site-packages context.
        return str(path.absolute())
    resolved = shutil.which(value)
    return resolved or value


def _bounded_log(value: str, limit: int = 1_000_000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} characters]"


def _worker_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        return "missing"
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = item.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _external_python_probe(executable: str, module: str, distribution: str) -> dict[str, str]:
    """Return stable import/version evidence without importing the backend in-process."""

    code = (
        "import importlib,importlib.metadata,json,platform;"
        f"importlib.import_module({module!r});"
        f"print(json.dumps({{'version':importlib.metadata.version({distribution!r}),"
        "'python_version':platform.python_version()},sort_keys=True))"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": "false", "error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "available": "false",
            "error": _bounded_log(completed.stderr.strip() or completed.stdout.strip(), 2_000),
        }
    try:
        parsed = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        return {"available": "false", "error": f"invalid probe output: {error}"}
    if not isinstance(parsed, dict):
        return {"available": "false", "error": "invalid probe result type"}
    return {
        "available": "true",
        "version": str(parsed.get("version", "unknown")),
        "python_version": str(parsed.get("python_version", "unknown")),
    }


def _basic_pitch_runtime_identity(executable: str) -> dict[str, str]:
    """Record every runtime Basic Pitch may select for a supplied model format."""

    code = """
import importlib.metadata, json, platform
def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'
print(json.dumps({
    'python_version': platform.python_version(),
    'platform': platform.platform(),
    'numpy_version': version('numpy'),
    'onnxruntime_version': version('onnxruntime'),
    'tensorflow_version': version('tensorflow'),
    'tflite_runtime_version': version('tflite-runtime'),
    'coremltools_version': version('coremltools'),
}, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": "false", "error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "available": "false",
            "error": _bounded_log(completed.stderr.strip() or completed.stdout.strip(), 2_000),
        }
    try:
        parsed = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        return {"available": "false", "error": f"invalid runtime probe output: {error}"}
    if not isinstance(parsed, dict):
        return {"available": "false", "error": "invalid runtime probe result type"}
    return {"available": "true", **{str(key): str(value) for key, value in parsed.items()}}


def _basic_pitch_model_format(path: Path) -> str:
    suffix_format = {
        ".onnx": "onnx",
        ".tflite": "tflite",
        ".mlmodel": "coreml",
        ".mlpackage": "coreml",
    }.get(path.suffix.lower())
    if suffix_format is not None:
        return suffix_format
    if path.is_dir():
        return "tensorflow_saved_model"
    return "unknown"


def _basic_pitch_default_model_identity(executable: str) -> dict[str, str]:
    """Resolve and hash the exact package-default model before cache lookup."""

    code = """
import hashlib, importlib.metadata, json
from pathlib import Path
from basic_pitch.inference import ICASSP_2022_MODEL_PATH
path = Path(ICASSP_2022_MODEL_PATH).resolve()
digest = hashlib.sha256()
if path.is_file():
    digest.update(path.read_bytes())
else:
    for item in sorted(p for p in path.rglob('*') if p.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        data = item.read_bytes()
        digest.update(len(relative).to_bytes(8, 'big')); digest.update(relative)
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
payload = {'path': str(path), 'sha256': digest.hexdigest(),
           'basic_pitch_version': importlib.metadata.version('basic-pitch'),
           'onnxruntime_version': importlib.metadata.version('onnxruntime')}
print(json.dumps(payload, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": "false", "error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {
            "available": "false",
            "error": _bounded_log(completed.stderr.strip() or completed.stdout.strip(), 2_000),
        }
    try:
        parsed = json.loads(completed.stdout.splitlines()[-1])
        return {
            "available": "true",
            "path": str(parsed["path"]),
            "sha256": str(parsed["sha256"]),
            "basic_pitch_version": str(parsed["basic_pitch_version"]),
            "onnxruntime_version": str(parsed["onnxruntime_version"]),
        }
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        return {"available": "false", "error": f"invalid model probe output: {error}"}


def _validate_mono_derivative(
    workspace: Workspace,
    derivative: AnalysisDerivativeEvidence,
    sample_map: SampleTimeMap,
) -> tuple[Path, npt.NDArray[np.float32]]:
    """Validate the derivative and its complete reference map before inference."""

    if sample_map.source_stream_id != derivative.source_stream_id:
        raise ValueError("sample map source stream differs from derivative evidence")
    if sample_map.derived_stream_id != derivative.derived_stream_id:
        raise ValueError("sample map derived stream differs from derivative evidence")
    if sample_map.source_sample_rate != derivative.source_sample_rate:
        raise ValueError("sample map source rate differs from derivative evidence")
    if sample_map.derived_sample_rate != derivative.derived_sample_rate:
        raise ValueError("sample map derived rate differs from derivative evidence")
    if sample_map.segments[0].source_start_sample != 0:
        raise ValueError("analysis time map must begin at decoded-reference sample zero")
    if sample_map.segments[0].derived_start_sample != 0:
        raise ValueError("analysis time map must begin at derivative sample zero")
    if sample_map.segments[-1].source_end_sample != derivative.source_frame_count:
        raise ValueError("analysis time map does not cover the complete reference")
    if sample_map.segments[-1].derived_end_sample != derivative.derived_frame_count:
        raise ValueError("analysis time map does not cover the complete derivative")

    persisted_map = SampleTimeMap.model_validate(
        workspace.load_artifact_json(derivative.sample_time_map_artifact_id)
    )
    if persisted_map != sample_map:
        raise ValueError("provided analysis time map differs from persisted evidence")

    source_row = workspace.database.fetchone(
        "SELECT * FROM audio_streams WHERE stream_id = ?", (derivative.source_stream_id,)
    )
    derived_row = workspace.database.fetchone(
        "SELECT * FROM audio_streams WHERE stream_id = ?", (derivative.derived_stream_id,)
    )
    if source_row is None or derived_row is None:
        raise ValueError("analysis derivative references an unknown audio stream")
    if (
        str(source_row["asset_id"]) != derivative.source_asset_id
        or int(source_row["sample_rate"]) != derivative.source_sample_rate
        or int(source_row["frame_count"]) != derivative.source_frame_count
    ):
        raise ValueError("source audio stream row differs from derivative evidence")
    if (
        str(derived_row["asset_id"]) != derivative.derived_asset_id
        or int(derived_row["sample_rate"]) != derivative.derived_sample_rate
        or int(derived_row["frame_count"]) != derivative.derived_frame_count
        or int(derived_row["channels"]) != 1
    ):
        raise ValueError("derived audio stream is not the declared explicit mono derivative")
    if len(derivative.channel_matrix) != 1:
        raise ValueError("pitch/transcription backends require an explicit mono derivative")
    if len(derivative.channel_matrix[0]) != int(source_row["channels"]):
        raise ValueError("analysis derivative channel matrix does not cover every source channel")
    if not all(math.isfinite(weight) for weight in derivative.channel_matrix[0]):
        raise ValueError("analysis derivative channel matrix contains non-finite weights")

    asset_row = workspace.asset_row(derivative.derived_asset_id)
    if str(asset_row["sha256"]) != derivative.artifact_sha256:
        raise ValueError("analysis derivative asset hash differs from evidence")
    path = workspace.asset_path(derivative.derived_asset_id)
    info = sf.info(path)
    if (
        info.channels != 1
        or info.samplerate != derivative.derived_sample_rate
        or info.frames != derivative.derived_frame_count
    ):
        raise ValueError("analysis derivative WAV header differs from evidence")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != derivative.derived_sample_rate or audio.shape != (derivative.derived_frame_count, 1):
        raise ValueError("decoded analysis derivative shape/rate differs from evidence")
    mono = np.asarray(audio[:, 0], dtype=np.float32)
    if not np.all(np.isfinite(mono)):
        raise ValueError("analysis derivative contains non-finite samples")
    return path, mono


class LibrosaPyinPitchTracker:
    """Monophonic pYIN/onset baseline with explicit centered-frame support."""

    def __init__(self, config: PyinConfig | None = None) -> None:
        self.config = config or PyinConfig()
        self.config.validate()

    def available(self) -> bool:
        return True

    def cache_identity(self) -> dict[str, int | float | str | bool]:
        config = self.config
        return {
            "backend": PYIN_BACKEND,
            "backend_version": importlib.metadata.version("librosa"),
            "adapter_sha256": TARGET_BACKENDS_CODE_SHA256,
            "fmin_hz": config.fmin_hz,
            "fmax_hz": config.fmax_hz,
            "frame_length": config.frame_length,
            "hop_length": config.hop_length,
            "center": config.center,
            "pad_mode": config.pad_mode,
            "onset_backtrack": config.onset_backtrack,
        }

    def analyze(
        self,
        workspace: Workspace,
        derivative: AnalysisDerivativeEvidence,
        sample_map: SampleTimeMap,
        *,
        run_id: str,
        derivative_artifact_id: str | None = None,
    ) -> tuple[TargetEventEvidenceArtifact, str]:
        _, audio = _validate_mono_derivative(workspace, derivative, sample_map)
        if audio.size == 0:
            raise ValueError("pYIN cannot analyze an empty derivative")

        config = self.config
        f0, voiced, probabilities = librosa.pyin(
            audio,
            fmin=config.fmin_hz,
            fmax=config.fmax_hz,
            sr=derivative.derived_sample_rate,
            frame_length=config.frame_length,
            hop_length=config.hop_length,
            center=config.center,
            pad_mode=config.pad_mode,
            fill_na=np.nan,
        )
        if not (len(f0) == len(voiced) == len(probabilities)):
            raise RuntimeError("librosa.pyin returned unsynchronized frame arrays")

        pitch_frames: list[TargetPitchFrame] = []
        half_window = config.frame_length // 2
        for index, (frequency, voiced_flag, probability) in enumerate(
            zip(f0, voiced, probabilities, strict=True)
        ):
            nominal_center = index * config.hop_length
            if config.center:
                start = max(0, nominal_center - half_window)
                center = min(nominal_center, derivative.derived_frame_count)
                end = min(derivative.derived_frame_count, nominal_center + half_window)
            else:
                start = min(nominal_center, derivative.derived_frame_count)
                end = min(derivative.derived_frame_count, start + config.frame_length)
                center = min(start + half_window, end)
            frequency_value = float(frequency) if np.isfinite(frequency) and frequency > 0 else None
            probability_value = float(probability) if np.isfinite(probability) else 0.0
            if not 0.0 <= probability_value <= 1.0:
                raise RuntimeError("librosa.pyin returned a probability outside [0, 1]")
            pitch_frames.append(
                TargetPitchFrame(
                    frame_id=new_id("target_pitch_frame"),
                    frame_index=index,
                    derivative_start_sample=start,
                    derivative_center_sample=center,
                    derivative_end_sample=end,
                    reference_start_sample=sample_map.derived_to_source(start),
                    reference_center_sample=sample_map.derived_to_source(center),
                    reference_end_sample=sample_map.derived_to_source(end),
                    f0_hz=frequency_value,
                    pitch_midi_cents=(
                        float(librosa.hz_to_midi(frequency_value) * 100.0)
                        if frequency_value is not None
                        else None
                    ),
                    voiced_probability=probability_value,
                    voiced=bool(voiced_flag) and frequency_value is not None,
                    backend=PYIN_BACKEND,
                )
            )

        envelope = librosa.onset.onset_strength(
            y=audio,
            sr=derivative.derived_sample_rate,
            hop_length=config.hop_length,
            center=config.center,
        )
        if not np.all(np.isfinite(envelope)):
            raise RuntimeError("librosa onset strength returned non-finite evidence")
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=envelope,
            sr=derivative.derived_sample_rate,
            hop_length=config.hop_length,
            backtrack=config.onset_backtrack,
            units="frames",
        )
        onsets: list[TargetOnsetEvidence] = []
        for frame_value in onset_frames:
            frame = int(frame_value)
            derivative_sample = min(frame * config.hop_length, derivative.derived_frame_count)
            envelope_index = min(max(frame, 0), max(0, len(envelope) - 1))
            onsets.append(
                TargetOnsetEvidence(
                    onset_id=new_id("target_onset"),
                    derivative_sample=derivative_sample,
                    reference_sample=sample_map.derived_to_source(derivative_sample),
                    strength=float(envelope[envelope_index]),
                    backend=PYIN_BACKEND,
                )
            )

        warnings: list[str] = []
        if not any(frame.voiced for frame in pitch_frames):
            warnings.append("pyin_produced_no_voiced_frames")
        if not onsets:
            warnings.append("librosa_onset_detector_produced_no_peaks")
        evidence = TargetEventEvidenceArtifact(
            evidence_id=new_id("target_evidence"),
            target_id=derivative.target_id,
            source_role=derivative.source_role,
            derivative_id=derivative.derivative_id,
            derivative_artifact_id=derivative_artifact_id or derivative.derivative_id,
            backend=PYIN_BACKEND,
            backend_version=importlib.metadata.version("librosa"),
            backend_config={
                "fmin_hz": config.fmin_hz,
                "fmax_hz": config.fmax_hz,
                "frame_length": config.frame_length,
                "hop_length": config.hop_length,
                "center": config.center,
                "pad_mode": config.pad_mode,
                "onset_backtrack": config.onset_backtrack,
                "numpy_version": importlib.metadata.version("numpy"),
                "soundfile_version": importlib.metadata.version("soundfile"),
                "frame_support": (
                    "clipped_centered_window" if config.center else "left_aligned_window"
                ),
                "input_asset_sha256": derivative.artifact_sha256,
            },
            pitch_frames=tuple(pitch_frames),
            onsets=tuple(onsets),
            warnings=tuple(warnings),
        )
        artifact_id = workspace.store_json_artifact(
            evidence,
            artifact_type="target.evidence.pyin",
            analysis_run_id=run_id,
        )
        return evidence, artifact_id


# The shorter name is kept for the orchestration API requested by the V0.0.2 plan.
PyinPitchTracker = LibrosaPyinPitchTracker


@dataclass(frozen=True)
class BasicPitchConfig:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 127.7
    minimum_frequency_hz: float | None = None
    maximum_frequency_hz: float | None = None
    melodia_trick: bool = True

    def validate(self) -> None:
        for name, value in (
            ("onset_threshold", self.onset_threshold),
            ("frame_threshold", self.frame_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Basic Pitch {name} must be in [0, 1]")
        if self.minimum_note_length_ms <= 0:
            raise ValueError("Basic Pitch minimum note length must be positive")
        if (
            self.minimum_frequency_hz is not None
            and self.maximum_frequency_hz is not None
            and self.minimum_frequency_hz >= self.maximum_frequency_hz
        ):
            raise ValueError("Basic Pitch frequency bounds must be ordered")


class BasicPitchSubprocessTranscriber:
    """Run Basic Pitch outside the base environment and retain its raw grids."""

    def __init__(
        self,
        external_python: str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        config: BasicPitchConfig | None = None,
        timeout_seconds: int = 3_600,
    ) -> None:
        self.external_python = _resolve_executable(external_python)
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.config = config or BasicPitchConfig()
        self.config.validate()
        if timeout_seconds <= 0:
            raise ValueError("backend timeout must be positive")
        self.timeout_seconds = timeout_seconds

    def available(self) -> bool:
        if self.model_path is not None and not self.model_path.exists():
            return False
        return (
            _external_python_probe(
                self.external_python, "basic_pitch.inference", "basic-pitch"
            ).get("available")
            == "true"
        )

    def cache_identity(self) -> dict[str, Any]:
        config = self.config
        model_identity: dict[str, Any]
        if self.model_path is None:
            model_identity = _basic_pitch_default_model_identity(self.external_python)
        else:
            model_identity = {
                "available": str(self.model_path.exists()).lower(),
                "path": str(self.model_path),
                "sha256": _path_sha256(self.model_path),
            }
        model_path_value = model_identity.get("path")
        if isinstance(model_path_value, str):
            model_identity["format"] = _basic_pitch_model_format(Path(model_path_value))
        return {
            "backend": BASIC_PITCH_BACKEND,
            "contract": _BASIC_PITCH_WORKER_CONTRACT,
            "adapter_sha256": TARGET_BACKENDS_CODE_SHA256,
            "external_python": self.external_python,
            "probe": _external_python_probe(
                self.external_python, "basic_pitch.inference", "basic-pitch"
            ),
            "runtime_identity": _basic_pitch_runtime_identity(self.external_python),
            "worker_sha256": _worker_sha256(Path(__file__).with_name("_basic_pitch_worker.py")),
            "model_identity": model_identity,
            "onset_threshold": config.onset_threshold,
            "frame_threshold": config.frame_threshold,
            "minimum_note_length_ms": config.minimum_note_length_ms,
            "minimum_frequency_hz": config.minimum_frequency_hz,
            "maximum_frequency_hz": config.maximum_frequency_hz,
            "melodia_trick": config.melodia_trick,
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
        source_path, _ = _validate_mono_derivative(workspace, derivative, sample_map)
        if derivative.derived_sample_rate != 22_050:
            raise ValueError(
                "Basic Pitch requires an explicit 22050-Hz derivative; "
                "hidden resampling is forbidden"
            )
        worker = Path(__file__).with_name("_basic_pitch_worker.py")
        worker_digest = _worker_sha256(worker)
        config = self.config

        with tempfile.TemporaryDirectory(prefix="music-locator-basic-pitch-") as temporary_name:
            temporary = Path(temporary_name)
            input_path = temporary / "analysis.wav"
            output_npz = temporary / "raw-evidence.npz"
            output_manifest = temporary / "worker-manifest.json"
            shutil.copyfile(source_path, input_path)
            arguments = [
                str(worker),
                "--input",
                str(input_path),
                "--output-npz",
                str(output_npz),
                "--output-manifest",
                str(output_manifest),
                "--expected-sample-rate",
                str(derivative.derived_sample_rate),
                "--expected-channels",
                "1",
                "--expected-frames",
                str(derivative.derived_frame_count),
                "--onset-threshold",
                repr(config.onset_threshold),
                "--frame-threshold",
                repr(config.frame_threshold),
                "--minimum-note-length-ms",
                repr(config.minimum_note_length_ms),
            ]
            if config.minimum_frequency_hz is not None:
                arguments.extend(["--minimum-frequency-hz", repr(config.minimum_frequency_hz)])
            if config.maximum_frequency_hz is not None:
                arguments.extend(["--maximum-frequency-hz", repr(config.maximum_frequency_hz)])
            if not config.melodia_trick:
                arguments.append("--no-melodia-trick")
            if self.model_path is not None:
                arguments.extend(["--model-path", str(self.model_path)])
            argv = [self.external_python, *arguments]
            try:
                completed = subprocess.run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.SubprocessError) as error:
                invocation_artifact_id = workspace.store_json_artifact(
                    {
                        "schema_version": "0.0.2",
                        "contract": _BASIC_PITCH_WORKER_CONTRACT,
                        "worker_sha256": worker_digest,
                        "executable": self.external_python,
                        "arguments": arguments,
                        "return_code": None,
                        "stdout": "",
                        "stderr": _bounded_log(f"{type(error).__name__}: {error}"),
                        "worker_manifest": None,
                    },
                    artifact_type="target.backend_invocation.basic_pitch",
                    analysis_run_id=run_id,
                )
                raise RuntimeError(
                    "Basic Pitch worker could not complete; invocation evidence "
                    f"{invocation_artifact_id}: {type(error).__name__}: {error}"
                ) from error
            manifest: dict[str, Any] | None = None
            manifest_error: str | None = None
            if output_manifest.is_file():
                try:
                    parsed_manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
                    if isinstance(parsed_manifest, dict):
                        manifest = parsed_manifest
                    else:
                        manifest_error = "worker manifest root is not an object"
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    manifest_error = f"{type(error).__name__}: {error}"
            invocation_artifact_id = workspace.store_json_artifact(
                {
                    "schema_version": "0.0.2",
                    "contract": _BASIC_PITCH_WORKER_CONTRACT,
                    "worker_sha256": worker_digest,
                    "executable": self.external_python,
                    "arguments": arguments,
                    "return_code": completed.returncode,
                    "stdout": _bounded_log(completed.stdout),
                    "stderr": _bounded_log(completed.stderr),
                    "worker_manifest": manifest,
                    "worker_manifest_error": manifest_error,
                },
                artifact_type="target.backend_invocation.basic_pitch",
                analysis_run_id=run_id,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Basic Pitch worker failed; invocation evidence "
                    f"{invocation_artifact_id}: {_bounded_log(completed.stderr, 8_000)}"
                )
            if manifest_error is not None:
                raise RuntimeError(
                    "Basic Pitch worker manifest is invalid; invocation evidence "
                    f"{invocation_artifact_id}: {manifest_error}"
                )
            if manifest is None or not output_npz.is_file():
                raise RuntimeError("Basic Pitch worker did not produce its declared outputs")
            if manifest.get("contract") != _BASIC_PITCH_WORKER_CONTRACT:
                raise RuntimeError("Basic Pitch worker contract version differs from adapter")
            raw_bytes = output_npz.read_bytes()
            expected_digest = str(manifest.get("npz_sha256", ""))
            actual_digest = hashlib.sha256(raw_bytes).hexdigest()
            if actual_digest != expected_digest:
                raise RuntimeError("Basic Pitch raw evidence hash differs from worker manifest")
            arrays = _validated_basic_pitch_npz(output_npz)
            raw_artifact_id = workspace.store_artifact_bytes(
                raw_bytes,
                artifact_type="target.evidence.basic_pitch.raw_npz",
                media_type="application/x-npz",
                analysis_run_id=run_id,
            )

        backend_version = str(manifest.get("basic_pitch_version", "unknown"))
        frame_times = arrays["frame_times_seconds"]
        onset_grid = arrays["onset"]
        note_grid = arrays["note"]
        event_starts = arrays["event_start_seconds"]
        event_ends = arrays["event_end_seconds"]
        event_pitches = arrays["event_pitch_midi"]
        event_amplitudes = arrays["event_amplitude"]
        warnings: list[str] = []
        candidates: list[TargetNoteCandidate] = []
        for index in range(len(event_starts)):
            start_seconds = float(event_starts[index])
            end_seconds = float(event_ends[index])
            pitch_midi = int(event_pitches[index])
            amplitude = float(event_amplitudes[index])
            if not (math.isfinite(start_seconds) and math.isfinite(end_seconds)):
                raise RuntimeError("Basic Pitch returned a non-finite note time")
            start_sample = _seconds_to_sample(
                start_seconds, derivative.derived_sample_rate, derivative.derived_frame_count
            )
            end_sample = _seconds_to_sample(
                end_seconds, derivative.derived_sample_rate, derivative.derived_frame_count
            )
            if end_sample <= start_sample:
                warnings.append(f"discarded_nonpositive_basic_pitch_candidate_{index}")
                continue
            onset_confidence, note_confidence = _basic_pitch_candidate_confidences(
                frame_times,
                onset_grid,
                note_grid,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                pitch_midi=pitch_midi,
                fallback=amplitude,
            )
            candidates.append(
                TargetNoteCandidate(
                    candidate_id=new_id("target_note_candidate"),
                    pitch_midi_cents=float(pitch_midi * 100),
                    derivative_start_sample=start_sample,
                    derivative_end_sample=end_sample,
                    reference_start_sample=sample_map.derived_to_source(start_sample),
                    reference_end_sample=sample_map.derived_to_source(end_sample),
                    onset_confidence=onset_confidence,
                    note_confidence=note_confidence,
                    amplitude=_unit_float(amplitude, "Basic Pitch note amplitude"),
                    source_role=derivative.source_role,
                    backend=BASIC_PITCH_BACKEND,
                    supporting_evidence_ids=(raw_artifact_id,),
                )
            )

        onsets: list[TargetOnsetEvidence] = []
        maximum_onset = np.max(onset_grid, axis=1)
        for frame_index in np.flatnonzero(maximum_onset >= config.onset_threshold):
            seconds = float(frame_times[int(frame_index)])
            derivative_sample = _seconds_to_sample(
                seconds, derivative.derived_sample_rate, derivative.derived_frame_count
            )
            onsets.append(
                TargetOnsetEvidence(
                    onset_id=new_id("target_onset"),
                    derivative_sample=derivative_sample,
                    reference_sample=sample_map.derived_to_source(derivative_sample),
                    strength=float(maximum_onset[int(frame_index)]),
                    backend=BASIC_PITCH_BACKEND,
                )
            )
        if not candidates:
            warnings.append("basic_pitch_produced_no_note_candidates")

        array_shapes = {
            name: tuple(int(value) for value in array.shape) for name, array in arrays.items()
        }
        activation_grid = ActivationGridEvidence(
            backend=BASIC_PITCH_BACKEND,
            backend_version=backend_version,
            raw_artifact_id=raw_artifact_id,
            raw_artifact_sha256=_artifact_sha256(workspace, raw_artifact_id),
            arrays=array_shapes,
            dtype="float32 activations; per-array dtypes recorded in worker manifest",
            nominal_frames_per_second=22_050 / 256,
            timing_interpretation=(
                "frame_times_seconds is Basic Pitch model_frames_to_time output, including its "
                "overlap-window correction; note bounds use the same clock and are rounded to "
                "explicit derivative samples with nearest-half-up"
            ),
        )
        evidence = TargetEventEvidenceArtifact(
            evidence_id=new_id("target_evidence"),
            target_id=derivative.target_id,
            source_role=derivative.source_role,
            derivative_id=derivative.derivative_id,
            derivative_artifact_id=derivative_artifact_id or derivative.derivative_id,
            backend=BASIC_PITCH_BACKEND,
            backend_version=backend_version,
            backend_config={
                "contract": _BASIC_PITCH_WORKER_CONTRACT,
                "external_python": self.external_python,
                "worker_sha256": worker_digest,
                "invocation_artifact_id": invocation_artifact_id,
                "model_path": str(manifest.get("model_path", "unknown")),
                "model_sha256": str(manifest.get("model_sha256", "unknown")),
                "input_asset_sha256": derivative.artifact_sha256,
                "onset_threshold": config.onset_threshold,
                "frame_threshold": config.frame_threshold,
                "minimum_note_length_ms": config.minimum_note_length_ms,
                "minimum_frequency_hz": (
                    config.minimum_frequency_hz
                    if config.minimum_frequency_hz is not None
                    else "none"
                ),
                "maximum_frequency_hz": (
                    config.maximum_frequency_hz
                    if config.maximum_frequency_hz is not None
                    else "none"
                ),
                "melodia_trick": config.melodia_trick,
                "candidate_pitch_policy": (
                    "integer decoded note; raw 3-bin/semitone contour retained"
                ),
            },
            onsets=tuple(onsets),
            note_candidates=tuple(candidates),
            activation_grid=activation_grid,
            warnings=tuple(warnings),
        )
        artifact_id = workspace.store_json_artifact(
            evidence,
            artifact_type="target.evidence.basic_pitch",
            analysis_run_id=run_id,
        )
        return evidence, artifact_id


def _validated_basic_pitch_npz(path: Path) -> dict[str, npt.NDArray[Any]]:
    required = {
        "onset",
        "note",
        "contour",
        "frame_times_seconds",
        "event_start_seconds",
        "event_end_seconds",
        "event_pitch_midi",
        "event_amplitude",
        "pitch_bend_offsets",
        "pitch_bend_values",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise RuntimeError(
                "Basic Pitch NPZ members differ from the safe worker contract: "
                f"{sorted(archive.files)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    onset = arrays["onset"]
    note = arrays["note"]
    contour = arrays["contour"]
    times = arrays["frame_times_seconds"]
    if onset.ndim != 2 or note.shape != onset.shape or onset.shape[1] != 88:
        raise RuntimeError("Basic Pitch note/onset grids must have synchronized [frames, 88] shape")
    if contour.ndim != 2 or contour.shape != (onset.shape[0], 264):
        raise RuntimeError("Basic Pitch contour grid must have synchronized [frames, 264] shape")
    if times.ndim != 1 or len(times) != onset.shape[0]:
        raise RuntimeError("Basic Pitch frame-time vector does not cover every activation frame")
    if np.any(np.diff(times) < 0):
        raise RuntimeError("Basic Pitch frame-time vector is not monotonic")
    for name in ("onset", "note", "contour", "frame_times_seconds"):
        if arrays[name].dtype.kind not in "fiu" or not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"Basic Pitch {name} is not a finite numeric array")
    for name in ("onset", "note", "contour"):
        values = arrays[name]
        if values.size and (float(np.min(values)) < -1e-6 or float(np.max(values)) > 1.000001):
            raise RuntimeError(f"Basic Pitch {name} activation lies outside [0, 1]")
    event_count = len(arrays["event_start_seconds"])
    for name in (
        "event_start_seconds",
        "event_end_seconds",
        "event_pitch_midi",
        "event_amplitude",
    ):
        value = arrays[name]
        if value.ndim != 1 or len(value) != event_count:
            raise RuntimeError("Basic Pitch note-event vectors are not synchronized")
        if value.dtype.kind not in "fiu" or not np.all(np.isfinite(value)):
            raise RuntimeError(f"Basic Pitch {name} is not a finite numeric vector")
    pitches = arrays["event_pitch_midi"]
    amplitudes = arrays["event_amplitude"]
    starts = arrays["event_start_seconds"]
    ends = arrays["event_end_seconds"]
    if np.any(ends <= starts):
        raise RuntimeError("Basic Pitch returned a non-positive note-event duration")
    if pitches.size and (int(np.min(pitches)) < 0 or int(np.max(pitches)) > 127):
        raise RuntimeError("Basic Pitch returned a pitch outside the MIDI note domain")
    if amplitudes.size and (
        float(np.min(amplitudes)) < -1e-6 or float(np.max(amplitudes)) > 1.000001
    ):
        raise RuntimeError("Basic Pitch returned a note amplitude outside [0, 1]")
    offsets = arrays["pitch_bend_offsets"]
    bends = arrays["pitch_bend_values"]
    if offsets.ndim != 1 or len(offsets) != event_count + 1 or offsets.dtype.kind not in "iu":
        raise RuntimeError("Basic Pitch pitch-bend offsets are invalid")
    if bends.ndim != 1 or bends.dtype.kind not in "fiu":
        raise RuntimeError("Basic Pitch pitch-bend evidence is invalid")
    if len(offsets) and (int(offsets[0]) != 0 or int(offsets[-1]) != len(bends)):
        raise RuntimeError("Basic Pitch pitch-bend offsets do not span their values")
    if np.any(np.diff(offsets) < 0):
        raise RuntimeError("Basic Pitch pitch-bend offsets are not monotonic")
    return arrays


def _seconds_to_sample(seconds: float, sample_rate: int, frame_count: int) -> int:
    rounded = int(
        (Decimal(str(seconds)) * Decimal(sample_rate)).quantize(Decimal(1), ROUND_HALF_UP)
    )
    return min(max(rounded, 0), frame_count)


def _unit_float(value: float, description: str) -> float:
    if not math.isfinite(value) or value < -1e-6 or value > 1.000001:
        raise RuntimeError(f"{description} is outside [0, 1]")
    return min(1.0, max(0.0, value))


def _basic_pitch_candidate_confidences(
    frame_times: npt.NDArray[Any],
    onset_grid: npt.NDArray[Any],
    note_grid: npt.NDArray[Any],
    *,
    start_seconds: float,
    end_seconds: float,
    pitch_midi: int,
    fallback: float,
) -> tuple[float, float]:
    pitch_bin = pitch_midi - 21
    if not 0 <= pitch_bin < onset_grid.shape[1] or len(frame_times) == 0:
        value = _unit_float(fallback, "Basic Pitch fallback confidence")
        return value, value
    start_index = int(np.argmin(np.abs(frame_times - start_seconds)))
    note_start = int(np.searchsorted(frame_times, start_seconds, side="left"))
    note_end = int(np.searchsorted(frame_times, end_seconds, side="right"))
    note_start = min(max(note_start, 0), len(frame_times) - 1)
    note_end = min(max(note_end, note_start + 1), len(frame_times))
    onset_confidence = _unit_float(
        float(onset_grid[start_index, pitch_bin]), "Basic Pitch onset confidence"
    )
    note_confidence = _unit_float(
        float(np.mean(note_grid[note_start:note_end, pitch_bin], dtype=np.float64)),
        "Basic Pitch note confidence",
    )
    return onset_confidence, note_confidence


class HTDemucsSubprocessSeparator:
    """Optional deterministic HTDemucs adapter using a separate Python environment."""

    def __init__(
        self,
        external_python: str | Path | None = None,
        *,
        model: str = "htdemucs",
        repo: str | Path | None = None,
        device: str = "cpu",
        shifts: int = 0,
        overlap: float = 0.25,
        split: bool = True,
        segment: int | None = None,
        jobs: int = 0,
        timeout_seconds: int = 7_200,
    ) -> None:
        self.external_python = _resolve_executable(external_python)
        self.model = model
        self.repo = Path(repo).resolve() if repo is not None else None
        self.device = device
        self.shifts = shifts
        self.overlap = overlap
        self.split = split
        self.segment = segment
        self.jobs = jobs
        self.timeout_seconds = timeout_seconds
        if not model:
            raise ValueError("Demucs model name cannot be empty")
        if shifts < 0 or jobs < 0:
            raise ValueError("Demucs shifts/jobs cannot be negative")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("Demucs overlap must be in [0, 1)")
        if segment is not None and segment <= 0:
            raise ValueError("Demucs segment must be positive")
        if timeout_seconds <= 0:
            raise ValueError("backend timeout must be positive")

    def available(self) -> bool:
        if self.repo is not None and not self.repo.is_dir():
            return False
        return (
            _external_python_probe(self.external_python, "demucs.api", "demucs").get("available")
            == "true"
        )

    def cache_identity(self) -> dict[str, Any]:
        worker = Path(__file__).with_name("_demucs_worker.py")
        with tempfile.TemporaryDirectory(prefix="music-locator-demucs-identity-") as temporary:
            output_manifest = Path(temporary) / "identity.json"
            arguments = [
                str(worker),
                "--identity-only",
                "--output-manifest",
                str(output_manifest),
                "--model",
                self.model,
                "--device",
                self.device,
                "--shifts",
                str(self.shifts),
                "--overlap",
                repr(self.overlap),
                "--jobs",
                str(self.jobs),
            ]
            if not self.split:
                arguments.append("--no-split")
            if self.segment is not None:
                arguments.extend(["--segment", str(self.segment)])
            if self.repo is not None:
                arguments.extend(["--repo", str(self.repo)])
            try:
                completed = subprocess.run(
                    [self.external_python, *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.SubprocessError) as error:
                model_identity: dict[str, Any] = {
                    "available": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            else:
                if completed.returncode == 0 and output_manifest.is_file():
                    parsed = json.loads(output_manifest.read_text(encoding="utf-8"))
                    model_identity = {
                        "available": True,
                        "model_state_sha256": str(parsed["model_state_sha256"]),
                        "model_class": str(parsed["model_class"]),
                        "demucs_version": str(parsed["demucs_version"]),
                        "torch_version": str(parsed["torch_version"]),
                        "torchaudio_version": str(parsed["torchaudio_version"]),
                        "worker_sha256": str(parsed["worker_sha256"]),
                    }
                else:
                    model_identity = {
                        "available": False,
                        "error": _bounded_log(
                            completed.stderr.strip() or completed.stdout.strip(), 2_000
                        ),
                    }
        return {
            "backend": HTDEMUCS_BACKEND,
            "contract": _HTDEMUCS_WORKER_CONTRACT,
            "adapter_sha256": TARGET_BACKENDS_CODE_SHA256,
            "external_python": self.external_python,
            "probe": _external_python_probe(self.external_python, "demucs.api", "demucs"),
            "worker_sha256": _worker_sha256(Path(__file__).with_name("_demucs_worker.py")),
            "model": self.model,
            "model_identity": model_identity,
            "repo": str(self.repo) if self.repo is not None else "demucs_default_remote",
            "repo_sha256": (
                _path_sha256(self.repo) if self.repo is not None else "remote_model_name"
            ),
            "device": self.device,
            "shifts": self.shifts,
            "overlap": self.overlap,
            "split": self.split,
            "segment": self.segment,
            "jobs": self.jobs,
        }

    def separate(
        self,
        workspace: Workspace,
        decoded_ref: DecodedTargetReference,
        *,
        run_id: str,
    ) -> tuple[SeparatedSource, ...]:
        manifest = decoded_ref.manifest
        stream = decoded_ref.decoded_stream
        if (
            manifest.decoded_asset_id != stream.asset_id
            or manifest.decoded_sample_rate != stream.sample_rate
            or manifest.decoded_channels != stream.channels
            or manifest.decoded_frame_count != stream.frame_count
        ):
            raise ValueError("decoded reference manifest and stream record disagree")
        reference_path = workspace.asset_path(stream.asset_id)
        info = sf.info(reference_path)
        if (
            info.frames != stream.frame_count
            or info.samplerate != stream.sample_rate
            or info.channels != stream.channels
            or info.subtype != "FLOAT"
        ):
            raise ValueError("HTDemucs requires the canonical FLOAT decoded reference")
        worker = Path(__file__).with_name("_demucs_worker.py")
        worker_digest = _worker_sha256(worker)

        with tempfile.TemporaryDirectory(prefix="music-locator-htdemucs-") as temporary_name:
            temporary = Path(temporary_name)
            input_path = temporary / "reference.wav"
            output_dir = temporary / "stems"
            output_manifest = temporary / "worker-manifest.json"
            shutil.copyfile(reference_path, input_path)
            arguments = [
                str(worker),
                "--input",
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--output-manifest",
                str(output_manifest),
                "--expected-sample-rate",
                str(stream.sample_rate),
                "--expected-channels",
                str(stream.channels),
                "--expected-frames",
                str(stream.frame_count),
                "--model",
                self.model,
                "--device",
                self.device,
                "--shifts",
                str(self.shifts),
                "--overlap",
                repr(self.overlap),
                "--jobs",
                str(self.jobs),
            ]
            if not self.split:
                arguments.append("--no-split")
            if self.segment is not None:
                arguments.extend(["--segment", str(self.segment)])
            if self.repo is not None:
                arguments.extend(["--repo", str(self.repo)])
            try:
                completed = subprocess.run(
                    [self.external_python, *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.SubprocessError) as error:
                invocation_artifact_id = workspace.store_json_artifact(
                    {
                        "schema_version": "0.0.2",
                        "contract": _HTDEMUCS_WORKER_CONTRACT,
                        "worker_sha256": worker_digest,
                        "executable": self.external_python,
                        "arguments": arguments,
                        "return_code": None,
                        "stdout": "",
                        "stderr": _bounded_log(f"{type(error).__name__}: {error}"),
                        "worker_manifest": None,
                    },
                    artifact_type="target.backend_invocation.htdemucs",
                    analysis_run_id=run_id,
                )
                raise RuntimeError(
                    "HTDemucs worker could not complete; invocation evidence "
                    f"{invocation_artifact_id}: {type(error).__name__}: {error}"
                ) from error
            worker_manifest: dict[str, Any] | None = None
            manifest_error: str | None = None
            if output_manifest.is_file():
                try:
                    parsed_manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
                    if isinstance(parsed_manifest, dict):
                        worker_manifest = parsed_manifest
                    else:
                        manifest_error = "worker manifest root is not an object"
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    manifest_error = f"{type(error).__name__}: {error}"
            invocation_artifact_id = workspace.store_json_artifact(
                {
                    "schema_version": "0.0.2",
                    "contract": _HTDEMUCS_WORKER_CONTRACT,
                    "worker_sha256": worker_digest,
                    "executable": self.external_python,
                    "arguments": arguments,
                    "return_code": completed.returncode,
                    "stdout": _bounded_log(completed.stdout),
                    "stderr": _bounded_log(completed.stderr),
                    "worker_manifest": worker_manifest,
                    "worker_manifest_error": manifest_error,
                },
                artifact_type="target.backend_invocation.htdemucs",
                analysis_run_id=run_id,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "HTDemucs worker failed; invocation evidence "
                    f"{invocation_artifact_id}: {_bounded_log(completed.stderr, 8_000)}"
                )
            if manifest_error is not None:
                raise RuntimeError(
                    "HTDemucs worker manifest is invalid; invocation evidence "
                    f"{invocation_artifact_id}: {manifest_error}"
                )
            if worker_manifest is None or worker_manifest.get("contract") != (
                _HTDEMUCS_WORKER_CONTRACT
            ):
                raise RuntimeError("HTDemucs worker did not produce the expected manifest")
            sources = self._store_stems(
                workspace,
                decoded_ref,
                output_dir,
                worker_manifest,
                invocation_artifact_id=invocation_artifact_id,
                run_id=run_id,
            )
        return sources

    def _store_stems(
        self,
        workspace: Workspace,
        decoded_ref: DecodedTargetReference,
        output_dir: Path,
        worker_manifest: dict[str, Any],
        *,
        invocation_artifact_id: str,
        run_id: str,
    ) -> tuple[SeparatedSource, ...]:
        stem_items = worker_manifest.get("stems")
        if not isinstance(stem_items, list) or not stem_items:
            raise RuntimeError("HTDemucs worker manifest contains no stems")
        roles: set[str] = set()
        output: list[SeparatedSource] = []
        reference_stream = decoded_ref.decoded_stream
        for raw_item in stem_items:
            if not isinstance(raw_item, dict):
                raise RuntimeError("HTDemucs stem manifest item is not an object")
            role = str(raw_item.get("role", ""))
            filename = str(raw_item.get("filename", ""))
            if (
                not role
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in role
                )
                or role in roles
            ):
                raise RuntimeError(f"HTDemucs returned an invalid or duplicate role: {role!r}")
            roles.add(role)
            if Path(filename).name != filename or not filename.endswith(".wav"):
                raise RuntimeError("HTDemucs returned an unsafe stem filename")
            stem_path = output_dir / filename
            if not stem_path.is_file():
                raise RuntimeError(f"HTDemucs stem is missing: {filename}")
            data = stem_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != str(raw_item.get("sha256", "")):
                raise RuntimeError(f"HTDemucs stem hash differs from manifest: {role}")
            stem_info = sf.info(stem_path)
            declared_rate = int(worker_manifest["output_sample_rate"])
            declared_channels = int(worker_manifest["output_channels"])
            declared_frames = int(worker_manifest["output_frames"])
            if (
                stem_info.format != "WAV"
                or stem_info.subtype != "FLOAT"
                or stem_info.samplerate != declared_rate
                or stem_info.channels != declared_channels
                or stem_info.frames != declared_frames
                or int(raw_item.get("frames", -1)) != declared_frames
            ):
                raise RuntimeError(f"HTDemucs FLOAT WAV conformance failed: {role}")
            samples, rate = sf.read(stem_path, dtype="float32", always_2d=True)
            if rate != declared_rate or samples.shape != (declared_frames, declared_channels):
                raise RuntimeError(f"HTDemucs decoded stem shape differs from manifest: {role}")
            if not np.all(np.isfinite(samples)):
                raise RuntimeError(f"HTDemucs stem contains non-finite samples: {role}")

            asset = workspace.store_asset(
                data,
                kind=AssetKind.TARGET_STEM,
                media_type="audio/wav",
                original_name=f"{decoded_ref.manifest.target_id}-{self.model}-{role}.wav",
            )
            stem_stream = _insert_audio_stream(
                workspace,
                asset_id=asset.asset_id,
                sample_rate=declared_rate,
                channels=declared_channels,
                frame_count=declared_frames,
                sample_format="FLOAT",
            )
            mapping = SampleTimeMap(
                source_stream_id=reference_stream.stream_id,
                derived_stream_id=stem_stream.stream_id,
                source_sample_rate=reference_stream.sample_rate,
                derived_sample_rate=stem_stream.sample_rate,
                segments=(
                    TimeMapSegment(
                        source_start_sample=0,
                        source_end_sample=reference_stream.frame_count,
                        derived_start_sample=0,
                        derived_end_sample=stem_stream.frame_count,
                    ),
                ),
            )
            map_artifact_id = workspace.store_json_artifact(
                mapping,
                artifact_type="time_map.target_reference_to_stem",
                analysis_run_id=run_id,
            )
            time_map_id = new_id("timemap")
            workspace.database.execute(
                """
                INSERT INTO time_maps(time_map_id, source_stream_id, derived_stream_id,
                                      artifact_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    time_map_id,
                    reference_stream.stream_id,
                    stem_stream.stream_id,
                    map_artifact_id,
                    utc_now(),
                ),
            )
            parameters: dict[str, int | float | str | bool] = {
                "backend": HTDEMUCS_BACKEND,
                "contract": _HTDEMUCS_WORKER_CONTRACT,
                "demucs_version": str(worker_manifest.get("demucs_version", "unknown")),
                "model": self.model,
                "model_state_sha256": str(worker_manifest.get("model_state_sha256", "unknown")),
                "device": self.device,
                "shifts": self.shifts,
                "overlap": self.overlap,
                "split": self.split,
                "jobs": self.jobs,
                "invocation_artifact_id": invocation_artifact_id,
                "worker_sha256": str(worker_manifest.get("worker_sha256", "unknown")),
                "channel_matrix_scope": "pre_model_channel_routing_only",
                "separator_is_observation_not_ground_truth": True,
            }
            if self.segment is not None:
                parameters["segment"] = self.segment
            derivative_record = DerivativeRecord(
                derivative_id=new_id("derivative"),
                source_stream_id=reference_stream.stream_id,
                derived_stream_id=stem_stream.stream_id,
                operation="htdemucs_nonlinear_source_hypothesis",
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
            derivative = AnalysisDerivativeEvidence(
                derivative_id=derivative_record.derivative_id,
                target_id=decoded_ref.manifest.target_id,
                source_role=role,
                source_asset_id=reference_stream.asset_id,
                source_stream_id=reference_stream.stream_id,
                derived_asset_id=asset.asset_id,
                derived_stream_id=stem_stream.stream_id,
                operation=derivative_record.operation,
                parameters=parameters,
                channel_matrix=_channel_routing_matrix(
                    reference_stream.channels, stem_stream.channels
                ),
                normalization="none",
                limiting="none",
                sample_time_map_artifact_id=map_artifact_id,
                source_sample_rate=reference_stream.sample_rate,
                derived_sample_rate=stem_stream.sample_rate,
                source_frame_count=reference_stream.frame_count,
                derived_frame_count=stem_stream.frame_count,
                artifact_sha256=asset.sha256,
            )
            derivative_artifact_id = workspace.store_json_artifact(
                derivative,
                artifact_type="target.analysis_derivative.stem",
                analysis_run_id=run_id,
            )
            output.append(
                SeparatedSource(
                    role=role,
                    asset=asset,
                    stream=stem_stream,
                    sample_map=mapping,
                    sample_map_artifact_id=map_artifact_id,
                    derivative=derivative,
                    derivative_artifact_id=derivative_artifact_id,
                    invocation_artifact_id=invocation_artifact_id,
                )
            )
        return tuple(output)


def _insert_audio_stream(
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


def _channel_routing_matrix(
    source_channels: int, derived_channels: int
) -> tuple[tuple[float, ...], ...]:
    if source_channels <= 0 or derived_channels <= 0:
        raise ValueError("channel counts must be positive")
    if source_channels == 1:
        return tuple((1.0,) for _ in range(derived_channels))
    if source_channels < derived_channels:
        raise ValueError("Demucs channel expansion is only defined for mono references")
    return tuple(
        tuple(1.0 if source == derived else 0.0 for source in range(source_channels))
        for derived in range(derived_channels)
    )
