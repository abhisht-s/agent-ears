from __future__ import annotations

import json
from pathlib import Path
from random import Random

import soundfile as sf
from typer.testing import CliRunner

from music_locator.audio import analyze_query, import_query
from music_locator.benchmark import SynthEvent, _synthesize, _target_payload
from music_locator.cli import app
from music_locator.diagnostics import diagnose_search
from music_locator.indexing import build_index
from music_locator.search import search
from music_locator.storage import Workspace
from music_locator.targets import import_target


def _row_count(workspace: Workspace, table: str) -> int:
    row = workspace.database.fetchone(f"SELECT COUNT(*) AS count FROM {table}")
    assert row is not None
    return int(row["count"])


def test_search_diagnostic_replays_full_pipeline_without_mutating_workspace(
    tmp_path: Path,
) -> None:
    melody = (60, 62, 65, 64, 69, 67, 71, 66, 68, 63, 70, 65)
    target_path = tmp_path / "diagnostic-target.json"
    target_path.write_text(
        json.dumps(_target_payload("diagnostic-target", melody)),
        encoding="utf-8",
    )
    query_path = tmp_path / "diagnostic-query.wav"
    audio = _synthesize(
        [SynthEvent(float(pitch + 5), 0.58) for pitch in melody[:10]],
        44_100,
        Random("diagnostic-query"),
    )
    sf.write(query_path, audio, 44_100, subtype="PCM_16")

    workspace_path = tmp_path / "workspace"
    workspace = Workspace(workspace_path)
    import_target(workspace, target_path)
    index_id = build_index(workspace)
    query_id = import_query(workspace, query_path)
    analyze_query(workspace, query_id)
    result = search(workspace, query_id, index_id)
    counts_before = {
        table: _row_count(workspace, table)
        for table in ("artifacts", "analysis_runs", "searches", "alignments", "jobs")
    }

    report = diagnose_search(
        workspace,
        result.search_id,
        expected_target_start=0.0,
        expected_target_end=4.95,
    )

    assert report["search"]["search_id"] == result.search_id
    assert report["query_audio"]["source_stream"]["duration_seconds"] > 4.0
    assert report["query_audio"]["quality"]["voiced_fraction"] > 0.2
    assert report["query_perception"]["f0_contour_summary"]["voiced_frame_count"] > 0
    interpretations = report["query_perception"]["interpretations"]
    assert len(interpretations) == 4
    assert all(item["events"] for item in interpretations)
    assert all("interval_cents" in item["feature_sequences"] for item in interpretations)
    assert report["query_perception"]["interpretation_side_by_side"]

    candidates = report["candidate_generation"]["pre_dp_shortlist"]
    assert candidates
    assert candidates[0]["pre_alignment_rank"] == 1
    assert "matched_query_to_target_shingle_occurrences" in candidates[0]
    assert report["candidate_generation"]["query_shingle_posting_evidence"]

    alignments = report["persisted_top_alignments"]
    assert alignments
    assert alignments[0]["matrix_path_summary"]["matrix_cells_persisted"] is False
    assert alignments[0]["full_alignment_path"]
    contributions = alignments[0]["cost_contributions"]
    assert set(contributions["exclusive_raw_dp_cost_buckets"]) == {
        "pitch",
        "timing",
        "query_insertion",
        "target_insertion_deletion",
        "split_penalty",
        "merge_penalty",
        "octave_repair",
        "tempo_scaling",
    }
    assert abs(contributions["reconstruction_delta"]) < 1e-5

    expected = report["expected_passage"]
    assert expected["overlapping_voice_count"] >= 1
    assert expected["voices"][0]["forced_exact_window_alignment"]["full_alignment_path"]
    assert "comparison_to_selected_result" in expected["voices"][0][
        "forced_exact_window_alignment"
    ]
    assert report == diagnose_search(
        workspace,
        result.search_id,
        expected_target_start=0.0,
        expected_target_end=4.95,
    )
    assert counts_before == {
        table: _row_count(workspace, table)
        for table in ("artifacts", "analysis_runs", "searches", "alignments", "jobs")
    }

    output = tmp_path / "diagnostic.json"
    runner = CliRunner()
    invocation = runner.invoke(
        app,
        [
            "diagnose",
            "search",
            result.search_id,
            "--workspace",
            str(workspace_path),
            "--expected-target-start",
            "0.0",
            "--expected-target-end",
            "4.95",
            "--output",
            str(output),
        ],
    )
    assert invocation.exit_code == 0, invocation.output
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(invocation.output) == report
