"""Transparent monophonic voice hypotheses derived from preserved target notes."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from music_locator.models import (
    CandidateNoteProvenance,
    TargetArtifact,
    TargetNoteEvent,
    VoiceHypothesis,
    VoiceQualityEvidence,
)

EXTRACTION_VERSION = "transparent_voice_paths_v1"
MIN_CANDIDATE_NOTES = 3
ParameterValue = int | float | str | bool
CoalescedVoice = tuple[
    str,
    tuple[TargetNoteEvent, ...],
    dict[str, ParameterValue],
    list[str],
]


@dataclass(frozen=True)
class ExtractedVoice:
    notes: tuple[TargetNoteEvent, ...]
    hypothesis: VoiceHypothesis


@dataclass(frozen=True)
class _PathState:
    notes: tuple[TargetNoteEvent, ...]
    cost: float
    last_ioi_ticks: int | None


def _is_monophonic(notes: Sequence[TargetNoteEvent]) -> bool:
    ordered = sorted(notes, key=lambda note: (note.start_tick, note.pitch_midi, note.end_tick))
    return all(right.start_tick >= left.end_tick for left, right in pairwise(ordered))


def _onset_groups(notes: Sequence[TargetNoteEvent]) -> list[tuple[TargetNoteEvent, ...]]:
    grouped: dict[int, list[TargetNoteEvent]] = defaultdict(list)
    for note in notes:
        grouped[note.start_tick].append(note)
    return [
        tuple(
            sorted(
                grouped[tick],
                key=lambda note: (note.pitch_midi, note.end_tick, note.event_id),
            )
        )
        for tick in sorted(grouped)
    ]


def _track_metadata(
    target: TargetArtifact,
) -> tuple[dict[int, str], dict[tuple[int, int], tuple[int, ...]]]:
    names: dict[int, str] = {}
    programs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for event in target.raw_events:
        if event.event_type == "track_name" and "name" in event.fields:
            names.setdefault(event.track_index, str(event.fields["name"]))
        if event.event_type == "program_change":
            channel = event.fields.get("channel")
            program = event.fields.get("program")
            if isinstance(channel, int) and isinstance(program, int):
                programs[(event.track_index, channel)].add(program)
    return names, {key: tuple(sorted(value)) for key, value in programs.items()}


def _edge_cost(
    previous: TargetNoteEvent,
    current: TargetNoteEvent,
    previous_ioi_ticks: int | None,
    ticks_per_beat: int,
    *,
    pitch_weight: float,
    timing_weight: float,
    duration_weight: float,
    overlap_weight: float,
    gap_weight: float,
) -> tuple[float, int]:
    leap = abs(current.pitch_midi - previous.pitch_midi)
    pitch_cost = pitch_weight * (leap + 0.65 * max(0, leap - 7))
    ioi = max(1, current.start_tick - previous.start_tick)
    gap_beats = max(0.0, (current.start_tick - previous.end_tick) / ticks_per_beat)
    overlap_beats = max(0.0, (previous.end_tick - current.start_tick) / ticks_per_beat)
    timing_cost = 0.0
    if previous_ioi_ticks is not None:
        timing_cost = timing_weight * abs(math.log(ioi / max(previous_ioi_ticks, 1)))
    previous_duration = max(1, previous.end_tick - previous.start_tick)
    current_duration = max(1, current.end_tick - current.start_tick)
    duration_cost = duration_weight * abs(math.log(current_duration / previous_duration))
    cost = (
        pitch_cost
        + timing_cost
        + duration_cost
        + overlap_weight * overlap_beats
        + gap_weight * max(0.0, gap_beats - 1.0)
    )
    return cost, ioi


def _continuity_path(
    groups: Sequence[tuple[TargetNoteEvent, ...]],
    ticks_per_beat: int,
    *,
    high_register_bias: float,
    max_skipped_groups: int,
    pitch_weight: float,
    timing_weight: float,
    duration_weight: float,
    overlap_weight: float,
    gap_weight: float,
    skip_penalty: float,
    selection_reward: float,
) -> tuple[TargetNoteEvent, ...]:
    if not groups:
        return ()
    all_notes = [note for group in groups for note in group]
    minimum_pitch = min(note.pitch_midi for note in all_notes)
    maximum_pitch = max(note.pitch_midi for note in all_notes)
    pitch_span = max(1, maximum_pitch - minimum_pitch)
    states: list[dict[str, _PathState]] = []
    for group_index, group in enumerate(groups):
        current_states: dict[str, _PathState] = {}
        for note in group:
            register_cost = high_register_bias * (maximum_pitch - note.pitch_midi) / pitch_span
            velocity_cost = 0.25 * (127 - note.velocity) / 127.0
            duration_beats = (note.end_tick - note.start_tick) / ticks_per_beat
            short_note_cost = 0.35 * max(0.0, 0.2 - duration_beats)
            node_cost = register_cost + velocity_cost + short_note_cost - selection_reward
            best = _PathState(
                notes=(note,),
                cost=group_index * skip_penalty + node_cost,
                last_ioi_ticks=None,
            )
            first_predecessor = max(0, group_index - max_skipped_groups - 1)
            for predecessor_index in range(first_predecessor, group_index):
                skipped = group_index - predecessor_index - 1
                for predecessor in states[predecessor_index].values():
                    edge_cost, ioi = _edge_cost(
                        predecessor.notes[-1],
                        note,
                        predecessor.last_ioi_ticks,
                        ticks_per_beat,
                        pitch_weight=pitch_weight,
                        timing_weight=timing_weight,
                        duration_weight=duration_weight,
                        overlap_weight=overlap_weight,
                        gap_weight=gap_weight,
                    )
                    candidate = _PathState(
                        notes=(*predecessor.notes, note),
                        cost=predecessor.cost + skipped * skip_penalty + edge_cost + node_cost,
                        last_ioi_ticks=ioi,
                    )
                    if candidate.cost < best.cost:
                        best = candidate
            current_states[note.event_id] = best
        states.append(current_states)
    viable: list[tuple[float, _PathState]] = []
    for group_index, group_states in enumerate(states):
        trailing_cost = (len(groups) - group_index - 1) * skip_penalty
        for state in group_states.values():
            if len(state.notes) >= MIN_CANDIDATE_NOTES:
                viable.append((state.cost + trailing_cost, state))
    if not viable:
        return ()
    return min(viable, key=lambda item: (item[0], -len(item[1].notes)))[1].notes


def _trim_to_monophonic(notes: Sequence[TargetNoteEvent]) -> tuple[TargetNoteEvent, ...]:
    ordered = sorted(notes, key=lambda note: (note.start_tick, note.pitch_midi, note.event_id))
    output: list[TargetNoteEvent] = []
    for index, note in enumerate(ordered):
        next_start = ordered[index + 1].start_tick if index + 1 < len(ordered) else note.end_tick
        indexed_end = min(note.end_tick, next_start)
        if indexed_end <= note.start_tick:
            continue
        output.append(note.model_copy(update={"end_tick": indexed_end}))
    return tuple(output)


def _voice_quality(
    indexed_notes: tuple[TargetNoteEvent, ...],
    source_notes: Sequence[TargetNoteEvent],
    groups: Sequence[tuple[TargetNoteEvent, ...]],
) -> VoiceQualityEvidence:
    source_by_id = {note.event_id: note for note in source_notes}
    ambiguous_count = 0
    truncated_count = 0
    retained_duration = 0
    source_duration = 0
    for note in indexed_notes:
        source = source_by_id[note.event_id]
        simultaneous = [
            other
            for other in source_notes
            if other.event_id != source.event_id
            and other.start_tick <= source.start_tick < other.end_tick
        ]
        ambiguous_count += int(bool(simultaneous))
        truncated_count += int(note.end_tick < source.end_tick)
        retained_duration += note.end_tick - note.start_tick
        source_duration += source.end_tick - source.start_tick
    leaps = [abs(right.pitch_midi - left.pitch_midi) for left, right in pairwise(indexed_notes)]
    mean_leap = sum(leaps) / max(len(leaps), 1)
    note_coverage = len(indexed_notes) / max(len(source_notes), 1)
    onset_coverage = len(indexed_notes) / max(len(groups), 1)
    ambiguity_fraction = ambiguous_count / max(len(indexed_notes), 1)
    truncation_fraction = truncated_count / max(len(indexed_notes), 1)
    duration_retention = retained_duration / max(source_duration, 1)
    quality = max(
        0.0,
        min(
            100.0,
            35.0
            + 22.0 * onset_coverage
            + 13.0 * min(1.0, len(indexed_notes) / 12.0)
            + 15.0 * (1.0 - min(1.0, mean_leap / 12.0))
            + 15.0 * duration_retention,
        ),
    )
    warnings: list[str] = []
    if ambiguity_fraction >= 0.5:
        warnings.append("many_notes_selected_from_simultaneous_choices")
    if truncation_fraction >= 0.35:
        warnings.append("many_source_durations_trimmed_to_make_voice_monophonic")
    if onset_coverage < 0.4:
        warnings.append("sparse_path_relative_to_source_onsets")
    return VoiceQualityEvidence(
        value=quality,
        components={
            "source_note_coverage": note_coverage,
            "source_onset_coverage": onset_coverage,
            "simultaneous_choice_fraction": ambiguity_fraction,
            "duration_truncation_fraction": truncation_fraction,
            "source_duration_retained_fraction": duration_retention,
            "mean_absolute_leap_semitones": mean_leap,
        },
        warnings=tuple(warnings),
    )


def _note_provenance(
    indexed_notes: tuple[TargetNoteEvent, ...], source_notes: Sequence[TargetNoteEvent]
) -> tuple[CandidateNoteProvenance, ...]:
    source_by_id = {note.event_id: note for note in source_notes}
    output: list[CandidateNoteProvenance] = []
    for index, indexed in enumerate(indexed_notes):
        source = source_by_id[indexed.event_id]
        simultaneous = tuple(
            sorted(
                other.event_id
                for other in source_notes
                if other.event_id != source.event_id
                and other.start_tick <= source.start_tick < other.end_tick
            )
        )
        output.append(
            CandidateNoteProvenance(
                candidate_event_index=index,
                source_event_id=source.event_id,
                source_track_index=source.track_index,
                source_channel=source.channel,
                source_start_tick=source.start_tick,
                source_end_tick=source.end_tick,
                indexed_start_tick=indexed.start_tick,
                indexed_end_tick=indexed.end_tick,
                source_note_on_raw_event_index=source.note_on_raw_event_index,
                source_note_off_raw_event_index=source.note_off_raw_event_index,
                simultaneous_source_event_ids=simultaneous,
            )
        )
    return tuple(output)


def _voice_id(
    target: TargetArtifact,
    track_index: int,
    channel: int,
    notes: Sequence[TargetNoteEvent],
) -> str:
    note_signature = "/".join(
        (
            f"{note.pitch_midi},{note.velocity},{note.start_tick},{note.end_tick},"
            f"{note.channel}"
        )
        for note in notes
    )
    signature = f"{target.source_sha256}|{track_index}|{channel}|{note_signature}"
    return "voice_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]


def extract_monophonic_voices(target: TargetArtifact) -> tuple[ExtractedVoice, ...]:
    """Return source voices and explicitly non-authoritative derived hypotheses.

    MIDI channel 10 (zero-based channel 9) is excluded. Every indexed note retains
    its source note ID and, for MIDI inputs, exact raw note-on/off event positions.
    """

    track_names, programs = _track_metadata(target)
    pitched_groups: dict[tuple[int, int], list[TargetNoteEvent]] = defaultdict(list)
    for note in target.notes:
        if note.channel != 9:
            pitched_groups[(note.track_index, note.channel)].append(note)
    output: list[ExtractedVoice] = []
    for (track_index, channel), unsorted_notes in sorted(pitched_groups.items()):
        source_notes = sorted(
            unsorted_notes,
            key=lambda note: (note.start_tick, note.pitch_midi, note.end_tick, note.event_id),
        )
        groups = _onset_groups(source_notes)
        methods: list[tuple[str, tuple[TargetNoteEvent, ...], dict[str, int | float | str | bool]]]
        if _is_monophonic(source_notes):
            methods = [("source_monophonic_v1", tuple(source_notes), {"duration_policy": "exact"})]
        else:
            top = tuple(group[-1] for group in groups)
            bottom = tuple(group[0] for group in groups)
            balanced_parameters: dict[str, int | float | str | bool] = {
                "max_skipped_onset_groups": 2,
                "pitch_weight": 0.24,
                "timing_weight": 0.72,
                "duration_weight": 0.42,
                "overlap_weight": 1.6,
                "gap_weight": 0.55,
                "skip_penalty": 0.85,
                "selection_reward": 3.8,
                "high_register_bias": 0.0,
            }
            lead_parameters: dict[str, int | float | str | bool] = {
                "max_skipped_onset_groups": 4,
                "pitch_weight": 0.2,
                "timing_weight": 0.88,
                "duration_weight": 0.6,
                "overlap_weight": 2.0,
                "gap_weight": 0.7,
                "skip_penalty": 0.72,
                "selection_reward": 3.6,
                "high_register_bias": 1.15,
            }
            balanced = _continuity_path(
                groups,
                target.ticks_per_beat,
                high_register_bias=0.0,
                max_skipped_groups=2,
                pitch_weight=0.24,
                timing_weight=0.72,
                duration_weight=0.42,
                overlap_weight=1.6,
                gap_weight=0.55,
                skip_penalty=0.85,
                selection_reward=3.8,
            )
            likely_lead = _continuity_path(
                groups,
                target.ticks_per_beat,
                high_register_bias=1.15,
                max_skipped_groups=4,
                pitch_weight=0.2,
                timing_weight=0.88,
                duration_weight=0.6,
                overlap_weight=2.0,
                gap_weight=0.7,
                skip_penalty=0.72,
                selection_reward=3.6,
            )
            methods = [
                (
                    "highest_at_onset_v1",
                    top,
                    {"onset_grouping_ticks": 0, "overlap_policy": "trim_at_next_onset"},
                ),
                (
                    "lowest_at_onset_v1",
                    bottom,
                    {"onset_grouping_ticks": 0, "overlap_policy": "trim_at_next_onset"},
                ),
                ("continuity_path_v1", balanced, balanced_parameters),
                ("melodic_salience_path_v1", likely_lead, lead_parameters),
            ]
        coalesced: dict[tuple[str, ...], CoalescedVoice] = {}
        for method, selected, parameters in methods:
            indexed_notes = _trim_to_monophonic(selected)
            if len(indexed_notes) < MIN_CANDIDATE_NOTES:
                continue
            signature = tuple(note.event_id for note in indexed_notes)
            if signature in coalesced:
                coalesced[signature][3].append(method)
            else:
                coalesced[signature] = (method, indexed_notes, parameters, [method])
        for _signature, (method, indexed_notes, parameters, supporting) in coalesced.items():
            direct = method == "source_monophonic_v1"
            voice_id = _voice_id(target, track_index, channel, indexed_notes)
            hypothesis = VoiceHypothesis(
                voice_id=voice_id,
                hypothesis_kind="source_monophonic" if direct else "derived_monophonic",
                authoritative=direct,
                source_track_index=track_index,
                source_channel=channel,
                source_track_name=track_names.get(track_index),
                source_programs=programs.get((track_index, channel), ()),
                extraction_method=method,
                supporting_methods=tuple(supporting),
                extraction_version=EXTRACTION_VERSION,
                parameters=parameters,
                quality=_voice_quality(indexed_notes, source_notes, groups),
                note_provenance=_note_provenance(indexed_notes, source_notes),
            )
            output.append(ExtractedVoice(notes=indexed_notes, hypothesis=hypothesis))
    return tuple(
        sorted(
            output,
            key=lambda voice: (
                voice.hypothesis.source_track_index,
                voice.hypothesis.source_channel,
                voice.hypothesis.extraction_method,
                voice.hypothesis.voice_id,
            ),
        )
    )
