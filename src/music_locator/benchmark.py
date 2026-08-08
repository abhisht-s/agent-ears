"""Frozen deterministic synthetic transformation benchmark."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import soundfile as sf
from pydantic import BaseModel, ConfigDict

from music_locator.audio import analyze_query, import_query
from music_locator.indexing import build_index
from music_locator.models import ResultStatus
from music_locator.search import search
from music_locator.storage import Workspace
from music_locator.targets import import_target

BENCHMARK_VERSION = "frozen-synthetic-v1"
BENCHMARK_SEED = 20_260_806
TICKS_PER_BEAT = 480
TEMPO_US_PER_BEAT = 500_000
BASE_NOTE_SECONDS = 0.58

TARGET_MELODIES: tuple[tuple[int, ...], ...] = (
    (
        60,
        62,
        65,
        64,
        69,
        67,
        71,
        66,
        68,
        63,
        70,
        65,
        61,
        67,
        64,
        72,
        69,
        62,
        66,
        71,
        63,
        68,
        65,
        70,
    ),
    (
        67,
        64,
        69,
        62,
        66,
        71,
        68,
        73,
        65,
        70,
        63,
        69,
        72,
        66,
        61,
        68,
        64,
        71,
        67,
        62,
        70,
        65,
        73,
        69,
    ),
    (
        55,
        60,
        58,
        63,
        57,
        65,
        61,
        67,
        59,
        64,
        56,
        62,
        68,
        60,
        66,
        57,
        63,
        69,
        61,
        65,
        58,
        67,
        62,
        70,
    ),
    (
        72,
        68,
        74,
        69,
        76,
        71,
        65,
        73,
        67,
        75,
        70,
        77,
        66,
        72,
        64,
        71,
        78,
        69,
        74,
        68,
        76,
        70,
        73,
        65,
    ),
    (
        62,
        69,
        64,
        71,
        65,
        60,
        68,
        63,
        72,
        66,
        70,
        61,
        67,
        73,
        64,
        69,
        62,
        71,
        65,
        68,
        60,
        70,
        63,
        72,
    ),
    (
        65,
        61,
        68,
        63,
        70,
        66,
        72,
        64,
        69,
        74,
        67,
        62,
        71,
        65,
        73,
        68,
        60,
        69,
        64,
        72,
        66,
        75,
        67,
        70,
    ),
)


class FrozenExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    example_id: str
    kind: Literal["positive", "absent", "insufficient"]
    expected_target_name: str | None
    source_target_index: int | None
    passage_start: int | None
    key_shift: int = 0
    octave_shift: int = 0
    tempo_scale: float = 1.0
    pitch_noise_cents: float = 0.0
    timing_drift: float = 0.0
    omit_index: int | None = None
    insert_after: int | None = None
    split_index: int | None = None
    merge_index: int | None = None
    sample_rate: int = 44_100


class FrozenManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_version: str
    seed: int
    targets: tuple[str, ...]
    examples: tuple[FrozenExample, ...]


@dataclass(frozen=True)
class SynthEvent:
    pitch: float
    duration_seconds: float


def frozen_manifest() -> FrozenManifest:
    examples: list[FrozenExample] = []
    variants: tuple[dict[str, Any], ...] = (
        {"key_shift": 5, "tempo_scale": 0.85},
        {"octave_shift": 12, "tempo_scale": 1.15, "pitch_noise_cents": 24.0},
        {"key_shift": -7, "timing_drift": 0.12, "pitch_noise_cents": 38.0},
        {"tempo_scale": 1.4, "omit_index": 4, "insert_after": 6},
        {"tempo_scale": 0.7, "split_index": 3, "merge_index": 7},
    )
    for target_index in range(len(TARGET_MELODIES)):
        for variant_index, variant in enumerate(variants):
            start = 2 + ((target_index * 3 + variant_index * 2) % 10)
            examples.append(
                FrozenExample(
                    example_id=f"positive-{target_index:02d}-{variant_index:02d}",
                    kind="positive",
                    expected_target_name=f"target-{target_index:02d}",
                    source_target_index=target_index,
                    passage_start=start,
                    sample_rate=48_000 if (target_index + variant_index) % 2 else 44_100,
                    **variant,
                )
            )
    for index in range(4):
        examples.append(
            FrozenExample(
                example_id=f"absent-{index:02d}",
                kind="absent",
                expected_target_name=None,
                source_target_index=None,
                passage_start=None,
                key_shift=index,
                sample_rate=44_100,
            )
        )
    examples.append(
        FrozenExample(
            example_id="insufficient-00",
            kind="insufficient",
            expected_target_name=None,
            source_target_index=0,
            passage_start=0,
            sample_rate=44_100,
        )
    )
    return FrozenManifest(
        benchmark_version=BENCHMARK_VERSION,
        seed=BENCHMARK_SEED,
        targets=tuple(f"target-{index:02d}" for index in range(len(TARGET_MELODIES))),
        examples=tuple(examples),
    )


def _target_payload(name: str, melody: tuple[int, ...]) -> dict[str, Any]:
    return {
        "schema_version": "0.0.1",
        "name": name,
        "ticks_per_beat": TICKS_PER_BEAT,
        "tempo_map": [{"tick": 0, "microseconds_per_beat": TEMPO_US_PER_BEAT}],
        "notes": [
            {
                "event_id": f"{name}-event-{index:03d}",
                "track_index": 0,
                "channel": 0,
                "pitch_midi": pitch,
                "velocity": 82 + (index % 4) * 8,
                "start_tick": index * TICKS_PER_BEAT,
                "end_tick": index * TICKS_PER_BEAT + 430,
            }
            for index, pitch in enumerate(melody)
        ],
    }


def _events_for_example(example: FrozenExample, rng: Random) -> list[SynthEvent]:
    if example.kind == "absent":
        bases = (48, 59, 50, 73, 52, 71, 54, 69, 56, 67)
        return [SynthEvent(float(pitch + example.key_shift), BASE_NOTE_SECONDS) for pitch in bases]
    if example.source_target_index is None or example.passage_start is None:
        raise ValueError("non-absent example lacks a source passage")
    length = 10
    pitches = list(
        TARGET_MELODIES[example.source_target_index][
            example.passage_start : example.passage_start + length
        ]
    )
    if example.kind == "insufficient":
        pitches = pitches[:2]
    events = [
        SynthEvent(
            pitch
            + example.key_shift
            + example.octave_shift
            + rng.gauss(0.0, example.pitch_noise_cents) / 100.0,
            BASE_NOTE_SECONDS
            * example.tempo_scale
            * (1.0 + rng.uniform(-example.timing_drift, example.timing_drift)),
        )
        for pitch in pitches
    ]
    if example.omit_index is not None and example.omit_index < len(events):
        events.pop(example.omit_index)
    if example.insert_after is not None and example.insert_after < len(events):
        inserted_pitch = events[example.insert_after].pitch + 3.0
        events.insert(
            example.insert_after + 1,
            SynthEvent(inserted_pitch, BASE_NOTE_SECONDS * example.tempo_scale * 0.7),
        )
    if example.split_index is not None and example.split_index < len(events):
        event = events[example.split_index]
        events[example.split_index : example.split_index + 1] = [
            SynthEvent(event.pitch, event.duration_seconds / 2),
            SynthEvent(event.pitch + 0.08, event.duration_seconds / 2),
        ]
    if example.merge_index is not None and example.merge_index + 1 < len(events):
        left = events[example.merge_index]
        right = events[example.merge_index + 1]
        events[example.merge_index : example.merge_index + 2] = [
            SynthEvent(
                (left.pitch + right.pitch) / 2,
                left.duration_seconds + right.duration_seconds,
            )
        ]
    return events


def _synthesize(events: list[SynthEvent], sample_rate: int, rng: Random) -> npt.NDArray[np.float32]:
    pieces: list[npt.NDArray[np.float32]] = []
    phase = 0.0
    for event_index, event in enumerate(events):
        frames = max(1, round(event.duration_seconds * sample_rate))
        gap_frames = min(frames // 5, round(0.035 * sample_rate))
        tone_frames = frames - gap_frames
        time = np.arange(tone_frames, dtype=np.float64) / sample_rate
        frequency = 440.0 * 2.0 ** ((event.pitch - 69.0) / 12.0)
        vibrato = 0.006 * np.sin(2.0 * np.pi * (5.0 + 0.1 * event_index) * time)
        instantaneous = 2.0 * np.pi * frequency * (1.0 + vibrato) / sample_rate
        phases = phase + np.cumsum(instantaneous)
        phase = float(phases[-1] % (2.0 * np.pi))
        signal = (
            np.sin(phases) + 0.28 * np.sin(2.0 * phases + 0.2) + 0.1 * np.sin(3.0 * phases + 0.4)
        )
        envelope = np.ones(tone_frames, dtype=np.float64)
        attack = min(tone_frames, round(0.025 * sample_rate))
        release = min(tone_frames, round(0.045 * sample_rate))
        if attack:
            envelope[:attack] *= np.linspace(0.0, 1.0, attack)
        if release:
            envelope[-release:] *= np.linspace(1.0, 0.0, release)
        noise = np.asarray([rng.gauss(0.0, 0.0025) for _ in range(tone_frames)], dtype=np.float64)
        pieces.append((0.21 * envelope * signal + noise).astype(np.float32))
        pieces.append(np.zeros(gap_frames, dtype=np.float32))
    return np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)


def generate_benchmark(destination: Path) -> FrozenManifest:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"benchmark destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    targets_dir = destination / "targets"
    queries_dir = destination / "queries"
    targets_dir.mkdir()
    queries_dir.mkdir()
    manifest = frozen_manifest()
    for index, melody in enumerate(TARGET_MELODIES):
        name = f"target-{index:02d}"
        (targets_dir / f"{name}.json").write_text(
            json.dumps(_target_payload(name, melody), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for example in manifest.examples:
        rng = Random(f"{BENCHMARK_SEED}:{example.example_id}")
        events = _events_for_example(example, rng)
        audio = _synthesize(events, example.sample_rate, rng)
        sf.write(
            queries_dir / f"{example.example_id}.wav", audio, example.sample_rate, subtype="PCM_16"
        )
    (destination / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run_benchmark(benchmark_dir: Path, workspace_path: Path, reports_dir: Path) -> dict[str, Any]:
    manifest = FrozenManifest.model_validate_json(
        (benchmark_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.benchmark_version != BENCHMARK_VERSION or manifest.seed != BENCHMARK_SEED:
        raise ValueError("benchmark manifest does not match the frozen suite identity")
    if workspace_path.exists():
        raise FileExistsError("benchmark workspace must not already exist")
    workspace = Workspace(workspace_path)
    workspace.initialize()
    target_ids_by_name: dict[str, str] = {}
    for path in sorted((benchmark_dir / "targets").glob("*.json")):
        target_id = import_target(workspace, path)
        target_ids_by_name[path.stem] = target_id
    index_id = build_index(workspace)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    positive_correct = 0
    positive_count = 0
    absent_abstained = 0
    absent_count = 0
    insufficient_typed = 0
    insufficient_count = 0
    localization_correct = 0
    for example in manifest.examples:
        query_path = benchmark_dir / "queries" / f"{example.example_id}.wav"
        query_id = import_query(workspace, query_path)
        analysis = analyze_query(workspace, query_id)
        result = search(workspace, query_id, index_id)
        expected_target_id = (
            target_ids_by_name.get(example.expected_target_name)
            if example.expected_target_name is not None
            else None
        )
        top1_correct = result.selected_target_id == expected_target_id
        localized = False
        if example.kind == "positive":
            positive_count += 1
            positive_correct += int(top1_correct)
            if result.target_range is not None and example.passage_start is not None:
                expected_tick = example.passage_start * TICKS_PER_BEAT
                localized = abs(result.target_range.start_tick - expected_tick) <= TICKS_PER_BEAT
                localization_correct += int(localized and top1_correct)
        elif example.kind == "absent":
            absent_count += 1
            abstained = result.status in {
                ResultStatus.NO_RELIABLE_MATCH,
                ResultStatus.AMBIGUOUS,
            }
            absent_abstained += int(abstained)
        else:
            insufficient_count += 1
            insufficient_typed += int(result.status == ResultStatus.INSUFFICIENT_QUERY)
        report = {
            "example": example.model_dump(mode="json"),
            "query_id": query_id,
            "analysis_run_id": analysis.analysis_run_id,
            "search_result": result.model_dump(mode="json"),
            "expected_target_id": expected_target_id,
            "top1_correct": top1_correct,
            "localized_within_one_note": localized,
        }
        reports.append(report)
        (reports_dir / f"{example.example_id}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    top1_rate = positive_correct / max(positive_count, 1)
    aggregate = {
        "benchmark_version": manifest.benchmark_version,
        "seed": manifest.seed,
        "positive_examples": positive_count,
        "positive_top1_correct": positive_correct,
        "positive_top1_rate": top1_rate,
        "localization_within_one_note_rate": localization_correct / max(positive_count, 1),
        "absent_examples": absent_count,
        "absent_abstained": absent_abstained,
        "absent_abstention_rate": absent_abstained / max(absent_count, 1),
        "insufficient_examples": insufficient_count,
        "insufficient_typed": insufficient_typed,
        "insufficient_typed_rate": insufficient_typed / max(insufficient_count, 1),
        "acceptance": {
            "top1_at_least_95_percent": top1_rate >= 0.95,
            "all_absent_abstained": absent_abstained == absent_count,
            "all_insufficient_typed": insufficient_typed == insufficient_count,
        },
    }
    (reports_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate


def clean_generated_benchmark(path: Path) -> None:
    """Explicit helper for tests only; CLI never deletes benchmark evidence."""

    shutil.rmtree(path)
