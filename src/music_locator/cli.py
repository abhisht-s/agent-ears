"""Research CLI over the versioned local engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from music_locator.audio import analyze_query, import_query
from music_locator.benchmark import generate_benchmark, run_benchmark
from music_locator.diagnostics import diagnose_search
from music_locator.evidence import verify_search_evidence
from music_locator.indexing import build_index, latest_index_id, load_index
from music_locator.models import AudioTargetArtifact
from music_locator.render import render_match
from music_locator.search import export_search_result, load_search_result, search
from music_locator.storage import Workspace
from music_locator.target_audio import (
    TargetAnalysisOptions,
    analyze_audio_target,
    load_latest_target_bundle,
)
from music_locator.targets import import_target, list_targets

app = typer.Typer(no_args_is_help=True, help="Evidence-preserving melodic passage locator.")
target_app = typer.Typer(no_args_is_help=True, help="Import, analyze, and inspect targets.")
query_app = typer.Typer(no_args_is_help=True, help="Import and analyze WAV queries.")
index_app = typer.Typer(no_args_is_help=True, help="Build and inspect deterministic indexes.")
match_app = typer.Typer(no_args_is_help=True, help="Inspect and audition search results.")
evidence_app = typer.Typer(no_args_is_help=True, help="Verify durable provenance.")
benchmark_app = typer.Typer(no_args_is_help=True, help="Generate and run the frozen benchmark.")
diagnose_app = typer.Typer(no_args_is_help=True, help="Replay persisted search diagnostics.")
app.add_typer(target_app, name="target")
app.add_typer(query_app, name="query")
app.add_typer(index_app, name="index")
app.add_typer(match_app, name="match")
app.add_typer(evidence_app, name="evidence")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(diagnose_app, name="diagnose")

WorkspaceOption = Annotated[
    Path, typer.Option("--workspace", "-w", help="Durable local-engine workspace.")
]


def _workspace(path: Path) -> Workspace:
    workspace = Workspace(path)
    workspace.initialize()
    return workspace


@app.command("init")
def initialize(workspace: WorkspaceOption = Path(".music-locator")) -> None:
    current = _workspace(workspace)
    typer.echo(json.dumps({"workspace": str(current.root), "initialized": True}, indent=2))


@target_app.command("import")
def target_import(
    path: Path,
    workspace: WorkspaceOption = Path(".music-locator"),
    name: Annotated[str | None, typer.Option(help="Override the source name.")] = None,
) -> None:
    current = _workspace(workspace)
    job = current.create_job("target.import", {"path": str(path)})
    try:
        target_id = import_target(current, path, name=name)
        current.finish_job(job.job_id, result_entity_id=target_id)
    except Exception as error:
        current.finish_job(job.job_id, error=str(error))
        raise
    typer.echo(json.dumps({"target_id": target_id, "job_id": job.job_id}, indent=2))


@target_app.command("list")
def target_list(workspace: WorkspaceOption = Path(".music-locator")) -> None:
    current = _workspace(workspace)
    payload: list[dict[str, object]] = []
    for target, artifact_id in list_targets(current):
        item: dict[str, object] = {
            "target_id": target.target_id,
            "name": target.name,
            "kind": target.kind.value,
            "artifact_id": artifact_id,
        }
        if isinstance(target, AudioTargetArtifact):
            item.update(
                {
                    "original_name": target.original_name,
                    "source_sha256": target.source_sha256,
                    "notes": None,
                }
            )
            try:
                bundle, bundle_artifact_id, target_analysis_id = load_latest_target_bundle(
                    current, target.target_id
                )
            except ValueError:
                item["analysis_status"] = "not_analyzed"
            else:
                item.update(
                    {
                        "analysis_status": "ready",
                        "target_analysis_id": target_analysis_id,
                        "bundle_artifact_id": bundle_artifact_id,
                        "searchable_views": len(bundle.representation_artifact_ids),
                    }
                )
        else:
            item["notes"] = len(target.notes)
            item["analysis_status"] = "ready"
        payload.append(item)
    typer.echo(json.dumps(payload, indent=2))


@target_app.command("analyze")
def target_analyze(
    target_id: str,
    workspace: WorkspaceOption = Path(".music-locator"),
    basic_pitch: Annotated[
        bool,
        typer.Option("--basic-pitch/--no-basic-pitch", help="Use Basic Pitch when available."),
    ] = True,
    basic_pitch_python: Annotated[
        Path | None,
        typer.Option(help="Python executable in an isolated Basic Pitch environment."),
    ] = None,
    basic_pitch_model: Annotated[
        Path | None,
        typer.Option(help="Explicit Basic Pitch .onnx/.tflite/CoreML model path."),
    ] = None,
    separator: Annotated[
        str,
        typer.Option(help="Optional separator: none or htdemucs."),
    ] = "none",
    separator_python: Annotated[
        Path | None,
        typer.Option(help="Python executable in an isolated HTDemucs environment."),
    ] = None,
    selected_stream_index: Annotated[
        int | None,
        typer.Option(
            help="Container-global audio stream index; default chooses the default/first."
        ),
    ] = None,
) -> None:
    current = _workspace(workspace)
    arguments: dict[str, str | int | float | bool] = {
        "target_id": target_id,
        "basic_pitch": basic_pitch,
        "basic_pitch_python": str(basic_pitch_python) if basic_pitch_python else "auto",
        "basic_pitch_model": str(basic_pitch_model) if basic_pitch_model else "package_default",
        "separator": separator,
        "separator_python": str(separator_python) if separator_python else "auto",
        "selected_stream_index": (
            selected_stream_index if selected_stream_index is not None else "default"
        ),
    }
    job = current.create_job("target.analyze", arguments)
    try:
        result = analyze_audio_target(
            current,
            target_id,
            options=TargetAnalysisOptions(
                enable_basic_pitch=basic_pitch,
                basic_pitch_python=str(basic_pitch_python) if basic_pitch_python else None,
                basic_pitch_model_path=(
                    str(basic_pitch_model.resolve()) if basic_pitch_model else None
                ),
                separator=separator,
                separator_python=str(separator_python) if separator_python else None,
                selected_stream_index=selected_stream_index,
            ),
        )
        current.finish_job(job.job_id, result_entity_id=result.target_analysis_id)
    except Exception as error:
        current.finish_job(job.job_id, error=str(error))
        raise
    typer.echo(
        json.dumps(
            {
                "target_id": target_id,
                "target_analysis_id": result.target_analysis_id,
                "bundle_id": result.bundle.bundle_id,
                "bundle_artifact_id": result.bundle_artifact_id,
                "searchable_views": len(result.bundle.representation_artifact_ids),
                "cache_hit": result.cache_hit,
                "decode_cache_hit": result.decode_cache_hit,
                "warnings": result.bundle.warnings,
                "job_id": job.job_id,
            },
            indent=2,
        )
    )


@target_app.command("index")
def target_index(target_id: str, workspace: WorkspaceOption = Path(".music-locator")) -> None:
    current = _workspace(workspace)
    job = current.create_job("target.index", {"target_id": target_id})
    try:
        index_id = build_index(current, target_ids={target_id})
        current.finish_job(job.job_id, result_entity_id=index_id)
    except Exception as error:
        current.finish_job(job.job_id, error=str(error))
        raise
    snapshot, artifact_id = load_index(current, index_id)
    typer.echo(
        json.dumps(
            {
                "target_id": target_id,
                "index_id": index_id,
                "artifact_id": artifact_id,
                "views": len(snapshot.target_entries),
                "job_id": job.job_id,
            },
            indent=2,
        )
    )


@query_app.command("import")
def query_import(path: Path, workspace: WorkspaceOption = Path(".music-locator")) -> None:
    current = _workspace(workspace)
    job = current.create_job("query.import", {"path": str(path)})
    try:
        query_id = import_query(current, path)
        current.finish_job(job.job_id, result_entity_id=query_id)
    except Exception as error:
        current.finish_job(job.job_id, error=str(error))
        raise
    typer.echo(json.dumps({"query_id": query_id, "job_id": job.job_id}, indent=2))


@query_app.command("analyze")
def query_analyze(query_id: str, workspace: WorkspaceOption = Path(".music-locator")) -> None:
    current = _workspace(workspace)
    job = current.create_job("query.analyze", {"query_id": query_id})
    try:
        analysis = analyze_query(current, query_id)
        current.finish_job(job.job_id, result_entity_id=analysis.analysis_run_id)
    except Exception as error:
        current.finish_job(job.job_id, error=str(error))
        raise
    typer.echo(analysis.model_dump_json(indent=2))


@query_app.command("import-analyze")
def query_import_analyze(path: Path, workspace: WorkspaceOption = Path(".music-locator")) -> None:
    current = _workspace(workspace)
    query_id = import_query(current, path)
    analysis = analyze_query(current, query_id)
    typer.echo(
        json.dumps({"query_id": query_id, "analysis": analysis.model_dump(mode="json")}, indent=2)
    )


@index_app.command("build")
def index_build(
    workspace: WorkspaceOption = Path(".music-locator"),
    target_id: Annotated[
        str | None, typer.Option(help="Build an index containing only this target.")
    ] = None,
) -> None:
    current = _workspace(workspace)
    job = current.create_job(
        "index.build", {"target_id": target_id if target_id is not None else "all"}
    )
    try:
        index_id = build_index(current, target_ids={target_id} if target_id else None)
        current.finish_job(job.job_id, result_entity_id=index_id)
    except Exception as error:
        current.finish_job(job.job_id, error=str(error))
        raise
    snapshot, artifact_id = load_index(current, index_id)
    typer.echo(
        json.dumps(
            {
                "index_id": index_id,
                "artifact_id": artifact_id,
                "voices": len(snapshot.target_entries),
                "source_tracks": len(
                    {(entry.target_id, entry.track_index) for entry in snapshot.target_entries}
                ),
                "source_monophonic_voices": snapshot.parameters.get(
                    "source_monophonic_voice_count", 0
                ),
                "derived_voice_hypotheses": snapshot.parameters.get("derived_voice_count", 0),
                "shingle_keys": len(snapshot.shingles),
                "job_id": job.job_id,
            },
            indent=2,
        )
    )


@app.command("search")
def run_search(
    query_id: str,
    workspace: WorkspaceOption = Path(".music-locator"),
    index_id: Annotated[str | None, typer.Option(help="Defaults to latest index.")] = None,
    target_id: Annotated[
        str | None, typer.Option(help="Restrict retrieval to one target in the index.")
    ] = None,
) -> None:
    current = _workspace(workspace)
    selected_index_id = index_id or latest_index_id(current)
    job = current.create_job(
        "search",
        {
            "query_id": query_id,
            "index_id": selected_index_id,
            "target_id": target_id if target_id is not None else "all",
        },
    )
    try:
        result = search(current, query_id, selected_index_id, target_id=target_id)
        current.finish_job(job.job_id, result_entity_id=result.search_id)
    except Exception as error:
        current.finish_job(job.job_id, error=str(error))
        raise
    typer.echo(result.model_dump_json(indent=2))


@match_app.command("inspect")
def match_inspect(
    search_id: str,
    workspace: WorkspaceOption = Path(".music-locator"),
    output: Annotated[Path | None, typer.Option(help="Optional JSON export path.")] = None,
) -> None:
    current = _workspace(workspace)
    result, artifact_id = load_search_result(current, search_id)
    if output is not None:
        export_search_result(current, search_id, output)
    typer.echo(
        json.dumps(
            {"result_artifact_id": artifact_id, "result": result.model_dump(mode="json")},
            indent=2,
        )
    )


@match_app.command("render")
def match_render(
    search_id: str,
    output: Path,
    workspace: WorkspaceOption = Path(".music-locator"),
) -> None:
    current = _workspace(workspace)
    asset_id = render_match(current, search_id, output)
    typer.echo(json.dumps({"output": str(output), "render_asset_id": asset_id}, indent=2))


@evidence_app.command("verify")
def evidence_verify(search_id: str, workspace: WorkspaceOption = Path(".music-locator")) -> None:
    typer.echo(json.dumps(verify_search_evidence(_workspace(workspace), search_id), indent=2))


@diagnose_app.command("search")
def diagnose_search_command(
    search_id: str,
    workspace: WorkspaceOption = Path(".music-locator"),
    expected_target_start: Annotated[
        float | None,
        typer.Option(help="Known-correct target passage start in MIDI elapsed seconds."),
    ] = None,
    expected_target_end: Annotated[
        float | None,
        typer.Option(help="Known-correct target passage end in MIDI elapsed seconds."),
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Optional diagnostic JSON path.")] = None,
) -> None:
    """Explain perception, retrieval, and alignment for a completed search."""

    current = _workspace(workspace)
    try:
        report = diagnose_search(
            current,
            search_id,
            expected_target_start=expected_target_start,
            expected_target_end=expected_target_end,
        )
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.write_text(payload, encoding="utf-8")
    typer.echo(payload, nl=False)


@benchmark_app.command("generate")
def benchmark_generate(output: Path = Path("benchmark-generated/frozen-v1")) -> None:
    manifest = generate_benchmark(output)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "benchmark_version": manifest.benchmark_version,
                "examples": len(manifest.examples),
            },
            indent=2,
        )
    )


@benchmark_app.command("run")
def benchmark_run(
    benchmark_dir: Path,
    workspace: Path = Path(".benchmark-workspace"),
    reports: Path = Path("benchmark-results"),
) -> None:
    typer.echo(json.dumps(run_benchmark(benchmark_dir, workspace, reports), indent=2))


if __name__ == "__main__":
    app()
