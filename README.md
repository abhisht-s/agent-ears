# agent-ears / Music Locator

agent-ears is an attempt to give AI agents a way to understand musical audio, not just spoken language.

Today, agents are already very good at understanding speech and text, but musical intent is still difficult to communicate naturally. A video editor may hum the exact part of a song they want to cut to. A producer may imitate a bassline, whistle a melody, or beatbox a rhythm instead of describing it in words.

agent-ears aims to turn those human musical expressions into structured,
source-grounded understanding that an AI agent can use.

## Music Locator V0.0.2

V0.0.2 extends the working V0.0.1 hum-to-MIDI locator to ordinary audio files.
Its first vertical slice is:

```text
encoded song or video
  -> pinned native-rate/stereo FFmpeg reference
  -> explicit analysis derivative
  -> continuous pYIN evidence + optional raw Basic Pitch evidence
  -> several traceable event/voice hypotheses
  -> V0.0.1 relative-pitch/rhythm retrieval and local alignment
  -> decoded sample range + original media time + provenance
```

Supported audio target extensions are `.mp3`, `.wav`, `.m4a`, `.aac`, `.aiff`,
`.aif`, `.mp4`, and `.mov`. MIDI and timed-note targets remain supported.

The production intermediate representation is not MIDI. Framewise pitch,
voicing, onset evidence, Basic Pitch activation grids, alternative event paths,
source roles, sample support, and time maps remain durable observations. Hard
event paths are non-authoritative matcher-facing hypotheses.

HTDemucs separation is implemented as an optional isolated worker; it is not
required to search an audio target. V0.0.2 does not yet implement beatboxing,
instrument-intent inference, embeddings, cloud services, MCP, DAW control, or
video-editor control.

The technical vertical slice is implemented and covered with generated-audio
tests. Product success is intentionally still pending the fresh blind-human-hum
test against `human-test/duro.mp3`; no answer timestamp is stored or used.

## Development

Python 3.12 and `ffmpeg`/`ffprobe` on `PATH` are required. A distributable build
must use an LGPL-compatible FFmpeg configuration; a local GPL-enabled Homebrew
build is suitable for development but must not be bundled as though it were LGPL.

```bash
uv sync --extra dev --no-editable --reinstall-package music-locator
PYTHONPATH=src .venv/bin/music-locator --help
PYTHONPATH=src .venv/bin/pytest
PYTHONPATH=src .venv/bin/ruff check .
PYTHONPATH=src .venv/bin/mypy src/music_locator
```

The base environment intentionally does not install Basic Pitch, TensorFlow,
ONNX Runtime, PyTorch, or Demucs. A reproducible local Basic Pitch worker can be
created separately:

```bash
uv venv --python 3.11.15 .backend-envs/basic-pitch
uv pip install --python .backend-envs/basic-pitch/bin/python \
  'basic-pitch[onnx]==0.4.0' 'setuptools==80.10.2' 'scikit-learn==1.5.1'
```

Resolve the installed ONNX model once and pass it explicitly during analysis:

```bash
BASIC_PITCH_MODEL="$PWD/.backend-envs/basic-pitch/lib/python3.11/site-packages/basic_pitch/saved_models/icassp_2022/nmp.onnx"
```

## Real-audio target flow

```bash
WORKSPACE=.audio-target-workspace

PYTHONPATH=src .venv/bin/music-locator target import human-test/duro.mp3 \
  --workspace "$WORKSPACE"

PYTHONPATH=src .venv/bin/music-locator target analyze TARGET_ID \
  --workspace "$WORKSPACE" \
  --basic-pitch-python .backend-envs/basic-pitch/bin/python \
  --basic-pitch-model "$BASIC_PITCH_MODEL" \
  --separator none

PYTHONPATH=src .venv/bin/music-locator target index TARGET_ID \
  --workspace "$WORKSPACE"

PYTHONPATH=src .venv/bin/music-locator query import-analyze fresh-hum.wav \
  --workspace "$WORKSPACE"

PYTHONPATH=src .venv/bin/music-locator search QUERY_ID \
  --workspace "$WORKSPACE" \
  --index-id INDEX_ID \
  --target-id TARGET_ID
```

The search JSON distinguishes MIDI ticks from decoded audio samples. An audio
selection reports the original filename, decoded-reference sample range and
rate, container/media microsecond range, selected extraction view and source
role, transformation and score evidence, alternatives, and accepted/ambiguous/
abstain semantics. It never fabricates PPQ, ticks, tracks, channels, or velocity.

See [milestone 1](docs/milestone-1.md) for the V0.0.1 evidence model and
[V0.0.2 real-audio target research](docs/V0.0.2_REAL_AUDIO_TARGET_RESEARCH.md)
for the architectural and ecosystem rationale.
