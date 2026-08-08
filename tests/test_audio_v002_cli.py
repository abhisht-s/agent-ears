from __future__ import annotations

import json
import shutil
from pathlib import Path
from random import Random

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from music_locator.benchmark import SynthEvent, _synthesize
from music_locator.cli import app

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for real-audio target decoding",
)


def _write_generated_phrase(
    path: Path,
    pitches: tuple[int, ...],
    *,
    stereo: bool,
    seed: str,
) -> None:
    signal = _synthesize(
        [SynthEvent(float(pitch), 0.52) for pitch in pitches],
        44_100,
        Random(seed),
    )
    audio = np.column_stack((signal, np.roll(signal, 11) * 0.92)) if stereo else signal
    sf.write(path, audio, 44_100, subtype="PCM_16")


def _json_invocation(runner: CliRunner, arguments: list[str]) -> dict[str, object]:
    invocation = runner.invoke(app, arguments)
    assert invocation.exit_code == 0, invocation.output
    payload = json.loads(invocation.output)
    assert isinstance(payload, dict)
    return payload


def test_real_audio_cli_vertical_slice_uses_native_sample_coordinates(
    tmp_path: Path,
) -> None:
    target_pitches = (60, 62, 65, 64, 69, 67, 71, 66, 68, 63, 70, 65)
    target_path = tmp_path / "generated-target.wav"
    query_path = tmp_path / "generated-hum.wav"
    workspace = tmp_path / "workspace"
    _write_generated_phrase(
        target_path,
        target_pitches,
        stereo=True,
        seed="audio-v002-cli-target",
    )
    _write_generated_phrase(
        query_path,
        tuple(pitch + 5 for pitch in target_pitches[:10]),
        stereo=False,
        seed="audio-v002-cli-query",
    )
    runner = CliRunner()

    imported = _json_invocation(
        runner,
        ["target", "import", str(target_path), "--workspace", str(workspace)],
    )
    target_id = str(imported["target_id"])

    analyzed = _json_invocation(
        runner,
        [
            "target",
            "analyze",
            target_id,
            "--no-basic-pitch",
            "--workspace",
            str(workspace),
        ],
    )
    assert analyzed["target_id"] == target_id
    assert int(analyzed["searchable_views"]) >= 1

    indexed = _json_invocation(
        runner,
        ["target", "index", target_id, "--workspace", str(workspace)],
    )
    index_id = str(indexed["index_id"])
    assert int(indexed["views"]) >= 1

    query = _json_invocation(
        runner,
        ["query", "import-analyze", str(query_path), "--workspace", str(workspace)],
    )
    query_id = str(query["query_id"])

    result = _json_invocation(
        runner,
        [
            "search",
            query_id,
            "--workspace",
            str(workspace),
            "--index-id",
            index_id,
            "--target-id",
            target_id,
        ],
    )

    assert result["selected_target_id"] == target_id
    assert result["selected_target_coordinate_kind"] == "decoded_samples"
    assert result["selected_target_view_id"] is not None
    assert result["selected_track_index"] is None
    assert result["selected_source_channel"] is None
    assert result["target_range"] is None
    assert result["strong_rhythmic_target_ticks"] == []

    sample_range = result["target_sample_range"]
    time_range = result["target_time_range"]
    assert isinstance(sample_range, dict)
    assert isinstance(time_range, dict)
    assert sample_range["sample_rate"] == 44_100
    assert 0 <= sample_range["start_sample"] < sample_range["end_sample"]
    assert time_range["start_microsecond"] < time_range["end_microsecond"]

    provenance = result["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["target_decode_manifest_artifact_id"] is not None
    assert provenance["target_time_map_artifact_id"] is not None
    assert provenance["target_analysis_artifact_id"] is not None
    assert provenance["target_evidence_artifact_id"] is not None
    assert provenance["target_view_id"] == result["selected_target_view_id"]

    for alternative in result["alternatives"]:
        assert alternative["coordinate_kind"] == "decoded_samples"
        assert alternative["target_range"] is None
        assert alternative["target_sample_range"] is not None

    verified = _json_invocation(
        runner,
        [
            "evidence",
            "verify",
            str(result["search_id"]),
            "--workspace",
            str(workspace),
        ],
    )
    assert verified["valid"] is True
