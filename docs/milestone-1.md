# Milestone 1: evidence-preserving melodic passage locator

## Scope

V0.0.1 accepts 4–12 second monophonic WAV queries and timed-note targets supplied
as Standard MIDI Files or the JSON interchange described below. Standard MIDI
Files may be polyphonic: the original source remains authoritative and unchanged,
while explicitly non-authoritative monophonic voice hypotheses are derived for
matching.
It retrieves and localizes passages under global transposition, octave displacement,
0.7–1.4x tempo changes, local timing drift, insertions, omissions, and split/merged
event hypotheses.

The milestone does not include a UI, MCP, Palmier, Ableton, beatboxing, full-mix
search, source separation, embeddings, or a vector database.

## Environment

The environment is locked to CPython 3.12.13 and exact dependency versions in
`uv.lock`.

```bash
uv sync --extra dev --no-editable
```

`--no-editable` is intentional. On the current uv-managed CPython/macOS toolchain,
editable path files are not processed reliably. Non-editable wheel installs avoid
that toolchain defect. `uv run --no-editable` keeps commands self-contained.

## End-to-end commands

Initialize a durable workspace:

```bash
uv run --no-editable music-locator init --workspace .music-locator
```

Import one or more authoritative targets:

```bash
uv run --no-editable music-locator target import song.mid --workspace .music-locator
uv run --no-editable music-locator target import melody.json --workspace .music-locator
uv run --no-editable music-locator target list --workspace .music-locator
```

Build a deterministic snapshot index:

```bash
uv run --no-editable music-locator index build --workspace .music-locator
```

Import and analyze a query (the combined command is convenient for local use):

```bash
uv run --no-editable music-locator query import-analyze hum.wav --workspace .music-locator
```

The output contains the `query_id`. Search the latest index:

```bash
uv run --no-editable music-locator search QUERY_ID --workspace .music-locator
```

Inspect, export, verify, and audition the result using its `search_id`:

```bash
uv run --no-editable music-locator match inspect SEARCH_ID --workspace .music-locator
uv run --no-editable music-locator match inspect SEARCH_ID --output result.json --workspace .music-locator
uv run --no-editable music-locator evidence verify SEARCH_ID --workspace .music-locator
uv run --no-editable music-locator match render SEARCH_ID matched-passage.wav --workspace .music-locator
```

Replay the persisted perception, pre-alignment retrieval shortlist, shingle
postings, and winning alignment paths without changing search state. With no
`--output`, the command prints deterministic, indented JSON for terminal reading:

```bash
uv run --no-editable music-locator diagnose search SEARCH_ID \
  --workspace .music-locator
```

Use `--output` to save the exact same human-readable JSON:

```bash
uv run --no-editable music-locator diagnose search SEARCH_ID \
  --workspace .music-locator --output diagnostic.json
```

After a listener identifies the known-correct passage in MIDI elapsed seconds,
compare every overlapping indexed voice against the selected result with a forced,
diagnostic-only alignment:

```bash
uv run --no-editable music-locator diagnose search SEARCH_ID \
  --workspace .music-locator \
  --expected-target-start 42.25 --expected-target-end 49.75 \
  --output diagnostic-with-ground-truth.json
```

The command is read-only with respect to the workspace. V0.0.1 persists the
winning DP backtrace and local costs, but not every matrix cell; the report states
that limitation and deterministically reconstructs the pitch, timing, insertion,
deletion, split, merge, and octave-repair cost buckets. The raw DP still derives
the tempo ratio after path selection, so its reconstructed raw-cost bucket is zero;
the structural final score separately and explicitly penalizes the absolute log
tempo deviation.

The audition render is a deliberately plain sine/harmonic synthesizer. It is
evidence for human checking, not a production MIDI renderer.

## Timed-note JSON interchange

Coordinates are integer MIDI ticks. Elapsed seconds are never canonical.

```json
{
  "schema_version": "0.0.1",
  "name": "lead melody",
  "ticks_per_beat": 480,
  "tempo_map": [
    {"tick": 0, "microseconds_per_beat": 500000}
  ],
  "notes": [
    {
      "event_id": "note-0001",
      "track_index": 0,
      "channel": 0,
      "pitch_midi": 64,
      "velocity": 96,
      "start_tick": 0,
      "end_tick": 430
    }
  ]
}
```

Already-monophonic pitched track/channel streams are indexed directly. For each
polyphonic pitched track/channel stream, indexing compares highest-at-onset,
lowest-at-onset, balanced pitch/timing continuity, and melody-salience continuity
paths. MIDI channel 10 percussion is not used for melodic extraction. Identical
event paths produced by multiple methods are coalesced while retaining all
supporting method names.

Derived paths select original note events rather than inventing notes. Overlapping
source durations may be shortened at the next selected onset solely in the indexed
hypothesis. Each indexed note records its original target note ID, raw MIDI note-on
and note-off event positions, source and indexed tick ranges, source track/channel,
and simultaneous alternatives. The original MIDI bytes, raw events, tempo map,
programs, channels, notes, and timing are not mutated.

## Evidence graph

The SQLite catalog runs in WAL mode with full synchronous commits. Bytes live in
read-only SHA-256 content-addressed paths. The database links:

```text
query source asset -> original audio stream
                   -> mono 22,050 Hz derivative stream
                   -> rational sample time map
                   -> quality, pYIN frames, voicing, onsets
                   -> selected interpretation + three alternatives
                   -> index snapshot + source/derived voice representations
                   -> search -> ranked alignment artifacts -> typed result
```

The verification command hash-checks the original query, analysis derivative,
target source, result, selected alignment, target representation, time map, pitch,
onset, analysis, interpretation, target, and index artifacts. It also validates
event indices, candidate-note-to-source-note provenance, raw MIDI event references,
coordinate endpoints, selected score, and tick/microsecond range consistency.

Result scores have the explicit label `uncalibrated_similarity`, set
`calibrated_probability` to `false`, and must never be presented as probabilities.

## Frozen benchmark

Generate the immutable suite definition and deterministic WAV fixtures:

```bash
uv run --no-editable music-locator benchmark generate --output benchmark-generated/frozen-v1
```

Run it in a fresh workspace (existing paths are rejected to avoid contaminating
results):

```bash
uv run --no-editable music-locator benchmark run benchmark-generated/frozen-v1 \
  --workspace .benchmark-workspace \
  --reports benchmark-results
```

The suite identity is `frozen-synthetic-v1`, seed `20260806`. It has 30 positive
queries across six targets, four absent queries, and one insufficient query. It
covers key and octave shifts, tempo scales from 0.7 to 1.4, pitch noise, timing
drift, insertions, omissions, and split/merged events. Every example gets a JSON
report; `aggregate.json` contains gates. `ambiguous` and `no_reliable_match` are
both abstentions because neither authorizes an edit.

## Quality commands

```bash
uv run --no-editable ruff check src tests
uv run --no-editable mypy src
uv run --no-editable pytest
```

## Known limitations

- pYIN is a CPU baseline and can be slow on first invocation due to numerical-kernel
  loading.
- Event reconciliation is heuristic; strong portamento, very legato singing, and
  repeated identical notes can yield alternate split/merged hypotheses.
- The decision score is intentionally uncalibrated. Thresholds are provisional and
  derived only from the development suite.
- Polyphonic voice extraction is heuristic. Inner voices, voice crossings, dense
  arpeggiation, sustain-pedal semantics, and melody handoffs between tracks may
  produce several close hypotheses; these remain typed ambiguity rather than being
  silently treated as ground truth.
- Audio stems and full mixes remain outside this milestone.
- MIDI audition rendering ignores instruments, controllers, sustain, bends, and
  expressive timing beyond note and tempo events.
- The synthetic suite is a transformation regression test, not evidence of human
  singer generalization.
