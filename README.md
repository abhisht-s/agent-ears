# What agent-ears is:

agent-ears is an attempt to give AI agents a way to understand musical audio, not just spoken language.

Today, agents are already very good at understanding speech and text, but musical intent is still difficult to communicate naturally. A video editor may hum the exact part of a song they want to cut to. A producer may imitate a bassline, whistle a melody, or beatbox a rhythm instead of describing it in words.

agent-ears aims to turn those human musical expressions into structured, source-grounded understanding that an AI agent can use. 

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
