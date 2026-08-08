from __future__ import annotations

from pathlib import Path

import soundfile as sf

from music_locator.benchmark import BENCHMARK_SEED, generate_benchmark


def test_frozen_benchmark_generation_is_complete(tmp_path: Path) -> None:
    destination = tmp_path / "benchmark"
    manifest = generate_benchmark(destination)
    assert manifest.seed == BENCHMARK_SEED
    assert len(manifest.examples) == 35
    assert {example.kind for example in manifest.examples} == {
        "positive",
        "absent",
        "insufficient",
    }
    rates = {sf.info(path).samplerate for path in (destination / "queries").glob("*.wav")}
    assert rates == {44_100, 48_000}
