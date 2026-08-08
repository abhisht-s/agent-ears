# Music Locator V0.0.1

An evidence-preserving local passage locator for 4–12 second monophonic hum,
non-lexical singing, and whistle queries against timed-note targets, including
transparent monophonic voice hypotheses derived from ordinary polyphonic MIDI.

This milestone deliberately excludes UI, MCP, DAW/video integration, beatboxing,
full-mix retrieval, source separation, learned embeddings, and vector databases.

## Development

```bash
uv sync --extra dev --no-editable
uv run --no-editable music-locator --help
uv run --no-editable pytest
uv run --no-editable ruff check .
uv run --no-editable mypy src
```

See `docs/milestone-1.md` for the command walkthrough and evidence model.
