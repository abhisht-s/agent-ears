"""Versioned domain contracts for durable evidence and agent-facing results."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["0.0.1"] = "0.0.1"
ANALYSIS_SAMPLE_RATE = 22_050


class StrictModel(BaseModel):
    """Base for persisted contracts: reject silent schema drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)


NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]


class ResultStatus(StrEnum):
    ACCEPTED = "accepted"
    AMBIGUOUS = "ambiguous"
    NO_RELIABLE_MATCH = "no_reliable_match"
    INSUFFICIENT_QUERY = "insufficient_query"
    UNSUPPORTED_TARGET = "unsupported_target"
    SOURCE_TIME_UNMAPPABLE = "source_time_unmappable"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AssetKind(StrEnum):
    QUERY_AUDIO = "query_audio"
    TARGET_MIDI = "target_midi"
    TARGET_NOTES = "target_notes"
    ANALYSIS_AUDIO = "analysis_audio"
    ARTIFACT = "artifact"
    RENDER = "render"


class TargetKind(StrEnum):
    MIDI = "midi"
    TIMED_NOTES = "timed_notes"


class Rational(StrictModel):
    numerator: int
    denominator: PositiveInt

    @model_validator(mode="after")
    def normalized(self) -> Rational:
        value = Fraction(self.numerator, self.denominator)
        if value.numerator != self.numerator or value.denominator != self.denominator:
            raise ValueError("rational values must be reduced")
        return self

    @classmethod
    def from_fraction(cls, value: Fraction) -> Rational:
        return cls(numerator=value.numerator, denominator=value.denominator)


class SampleRange(StrictModel):
    start_sample: NonNegativeInt
    end_sample: NonNegativeInt
    sample_rate: PositiveInt

    @model_validator(mode="after")
    def ordered(self) -> SampleRange:
        if self.end_sample < self.start_sample:
            raise ValueError("end_sample must be >= start_sample")
        return self


class TickRange(StrictModel):
    start_tick: NonNegativeInt
    end_tick: NonNegativeInt
    ticks_per_beat: PositiveInt

    @model_validator(mode="after")
    def ordered(self) -> TickRange:
        if self.end_tick < self.start_tick:
            raise ValueError("end_tick must be >= start_tick")
        return self


class MicrosecondRange(StrictModel):
    start_microsecond: NonNegativeInt
    end_microsecond: NonNegativeInt

    @model_validator(mode="after")
    def ordered(self) -> MicrosecondRange:
        if self.end_microsecond < self.start_microsecond:
            raise ValueError("end_microsecond must be >= start_microsecond")
        return self


class TimeMapSegment(StrictModel):
    source_start_sample: NonNegativeInt
    source_end_sample: PositiveInt
    derived_start_sample: NonNegativeInt
    derived_end_sample: PositiveInt

    @model_validator(mode="after")
    def nonempty(self) -> TimeMapSegment:
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("source segment must be non-empty")
        if self.derived_end_sample <= self.derived_start_sample:
            raise ValueError("derived segment must be non-empty")
        return self


def _round_fraction(value: Fraction) -> int:
    """Round a non-negative rational to the nearest integer, halves upward."""

    if value < 0:
        raise ValueError("sample coordinates cannot be negative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2 * remainder >= value.denominator)


class SampleTimeMap(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    source_stream_id: str
    derived_stream_id: str
    source_sample_rate: PositiveInt
    derived_sample_rate: PositiveInt
    segments: tuple[TimeMapSegment, ...]

    @model_validator(mode="after")
    def continuous(self) -> SampleTimeMap:
        if not self.segments:
            raise ValueError("at least one segment is required")
        for previous, current in zip(self.segments, self.segments[1:], strict=False):
            if previous.source_end_sample != current.source_start_sample:
                raise ValueError("source time-map segments must be contiguous")
            if previous.derived_end_sample != current.derived_start_sample:
                raise ValueError("derived time-map segments must be contiguous")
        return self

    def source_to_derived(self, sample: int) -> int:
        segment = self._source_segment(sample)
        source_offset = sample - segment.source_start_sample
        source_length = segment.source_end_sample - segment.source_start_sample
        derived_length = segment.derived_end_sample - segment.derived_start_sample
        mapped = Fraction(source_offset * derived_length, source_length)
        return segment.derived_start_sample + _round_fraction(mapped)

    def derived_to_source(self, sample: int) -> int:
        segment = self._derived_segment(sample)
        derived_offset = sample - segment.derived_start_sample
        derived_length = segment.derived_end_sample - segment.derived_start_sample
        source_length = segment.source_end_sample - segment.source_start_sample
        mapped = Fraction(derived_offset * source_length, derived_length)
        return segment.source_start_sample + _round_fraction(mapped)

    def _source_segment(self, sample: int) -> TimeMapSegment:
        if sample < 0:
            raise ValueError("sample must be non-negative")
        for segment in self.segments:
            if segment.source_start_sample <= sample <= segment.source_end_sample:
                return segment
        raise ValueError(f"source sample {sample} is outside the map")

    def _derived_segment(self, sample: int) -> TimeMapSegment:
        if sample < 0:
            raise ValueError("sample must be non-negative")
        for segment in self.segments:
            if segment.derived_start_sample <= sample <= segment.derived_end_sample:
                return segment
        raise ValueError(f"derived sample {sample} is outside the map")


class AssetRecord(StrictModel):
    asset_id: str
    kind: AssetKind
    sha256: str
    media_type: str
    original_name: str
    byte_length: NonNegativeInt
    blob_path: str
    created_at: str


class AudioStreamRecord(StrictModel):
    stream_id: str
    asset_id: str
    sample_rate: PositiveInt
    channels: PositiveInt
    frame_count: NonNegativeInt
    sample_format: str
    created_at: str


class DerivativeRecord(StrictModel):
    derivative_id: str
    source_stream_id: str
    derived_stream_id: str
    operation: str
    parameters: dict[str, int | float | str | bool]
    time_map_id: str
    created_at: str


class TempoChange(StrictModel):
    tick: NonNegativeInt
    microseconds_per_beat: PositiveInt


class RawMidiEvent(StrictModel):
    track_index: NonNegativeInt
    absolute_tick: NonNegativeInt
    delta_tick: NonNegativeInt
    event_type: str
    message_hex: str
    fields: dict[str, int | str | bool]


class TargetNoteEvent(StrictModel):
    event_id: str
    track_index: NonNegativeInt
    channel: NonNegativeInt
    pitch_midi: Annotated[int, Field(ge=0, le=127)]
    velocity: Annotated[int, Field(ge=0, le=127)]
    start_tick: NonNegativeInt
    end_tick: NonNegativeInt
    note_on_raw_event_index: NonNegativeInt | None = None
    note_off_raw_event_index: NonNegativeInt | None = None

    @model_validator(mode="after")
    def positive_duration(self) -> TargetNoteEvent:
        if self.end_tick <= self.start_tick:
            raise ValueError("target notes require positive duration")
        return self


class TimedNoteTargetInput(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    name: str
    ticks_per_beat: PositiveInt
    tempo_map: tuple[TempoChange, ...]
    notes: tuple[TargetNoteEvent, ...]


class TargetArtifact(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    target_id: str
    name: str
    kind: TargetKind
    asset_id: str
    ticks_per_beat: PositiveInt
    tempo_map: tuple[TempoChange, ...]
    raw_events: tuple[RawMidiEvent, ...]
    notes: tuple[TargetNoteEvent, ...]
    source_sha256: str


class QualityWarning(StrictModel):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str


class QualityReport(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    duration_samples: NonNegativeInt
    sample_rate: PositiveInt
    peak_amplitude: float
    rms_amplitude: float
    clipping_fraction: UnitFloat
    voiced_fraction: UnitFloat
    warnings: tuple[QualityWarning, ...]


class PitchFrame(StrictModel):
    frame_index: NonNegativeInt
    center_sample: NonNegativeInt
    f0_hz: float | None
    midi_cents: float | None
    voiced_probability: UnitFloat
    voiced: bool
    quality_weight: UnitFloat


class OnsetEvidence(StrictModel):
    sample: NonNegativeInt
    strength: float


class QueryEvent(StrictModel):
    event_id: str
    start_sample: NonNegativeInt
    end_sample: NonNegativeInt
    source_start_sample: NonNegativeInt
    source_end_sample: NonNegativeInt
    pitch_midi_cents: float
    pitch_deviation_cents: float
    voicing: UnitFloat
    onset_strength: float
    quality_weight: UnitFloat
    confident_frame_fraction: UnitFloat | None = None
    rms_amplitude: Annotated[float, Field(ge=0.0)] | None = None
    relative_rms: Annotated[float, Field(ge=0.0)] | None = None

    @model_validator(mode="after")
    def positive_duration(self) -> QueryEvent:
        if self.end_sample <= self.start_sample:
            raise ValueError("query events require positive duration")
        if self.source_end_sample <= self.source_start_sample:
            raise ValueError("source query events require positive duration")
        return self


class QueryRepresentation(StrictModel):
    relative_pitch_cents: tuple[float, ...]
    interval_cents: tuple[float, ...]
    contour_steps: tuple[Annotated[int, Field(ge=-1, le=1)], ...]
    onset_samples: tuple[NonNegativeInt, ...]
    log_ioi_ratios: tuple[float, ...]
    quality_weights: tuple[UnitFloat, ...]


class QueryEventUsabilityEvidence(StrictModel):
    source_event_id: str
    usable: bool
    reason_codes: tuple[str, ...]
    quality_weight: UnitFloat
    voicing_probability: UnitFloat
    pitch_deviation_cents: float
    onset_strength: float
    confident_frame_fraction: UnitFloat | None = None
    rms_amplitude: Annotated[float, Field(ge=0.0)] | None = None
    relative_rms: Annotated[float, Field(ge=0.0)] | None = None


class QueryInterpretation(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    interpretation_id: str
    query_id: str
    rank: Annotated[int, Field(ge=1, le=4)]
    selected: bool
    method: str
    method_parameters: dict[str, int | float | str | bool]
    events: tuple[QueryEvent, ...]
    representation: QueryRepresentation
    interpretation_score: float
    source_interpretation_id: str | None = None
    event_usability: tuple[QueryEventUsabilityEvidence, ...] = ()


class QueryAnalysisArtifact(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    query_id: str
    original_asset_id: str
    original_stream_id: str
    analysis_asset_id: str
    analysis_stream_id: str
    derivative_id: str
    time_map_id: str
    analysis_run_id: str
    quality: QualityReport
    pitch_frames_artifact_id: str
    onsets_artifact_id: str
    interpretation_ids: tuple[str, ...]
    selected_interpretation_id: str | None


class TargetIndexEntry(StrictModel):
    target_id: str
    track_index: NonNegativeInt
    event_count: NonNegativeInt
    representation_artifact_id: str
    voice_id: str = "legacy-source-track"
    source_channel: NonNegativeInt = 0


class VoiceQualityEvidence(StrictModel):
    value: Annotated[float, Field(ge=0.0, le=100.0)]
    label: Literal["uncalibrated_voice_quality"] = "uncalibrated_voice_quality"
    higher_is_better: Literal[True] = True
    calibrated_probability: Literal[False] = False
    components: dict[str, float]
    warnings: tuple[str, ...] = ()


class CandidateNoteProvenance(StrictModel):
    candidate_event_index: NonNegativeInt
    source_event_id: str
    source_track_index: NonNegativeInt
    source_channel: NonNegativeInt
    source_start_tick: NonNegativeInt
    source_end_tick: NonNegativeInt
    indexed_start_tick: NonNegativeInt
    indexed_end_tick: NonNegativeInt
    source_note_on_raw_event_index: NonNegativeInt | None = None
    source_note_off_raw_event_index: NonNegativeInt | None = None
    simultaneous_source_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_ranges(self) -> CandidateNoteProvenance:
        if self.source_end_tick <= self.source_start_tick:
            raise ValueError("source note provenance requires positive duration")
        if self.indexed_end_tick <= self.indexed_start_tick:
            raise ValueError("indexed note provenance requires positive duration")
        if self.indexed_start_tick != self.source_start_tick:
            raise ValueError("voice extraction may not move source note onsets")
        if self.indexed_end_tick > self.source_end_tick:
            raise ValueError("voice extraction may not extend source note durations")
        return self


class VoiceHypothesis(StrictModel):
    voice_id: str
    hypothesis_kind: Literal["source_monophonic", "derived_monophonic"]
    authoritative: bool
    source_track_index: NonNegativeInt
    source_channel: NonNegativeInt
    source_track_name: str | None = None
    source_programs: tuple[Annotated[int, Field(ge=0, le=127)], ...] = ()
    extraction_method: str
    supporting_methods: tuple[str, ...]
    extraction_version: str
    parameters: dict[str, int | float | str | bool]
    quality: VoiceQualityEvidence
    note_provenance: tuple[CandidateNoteProvenance, ...]


class VoiceHypothesisSummary(StrictModel):
    voice_id: str
    hypothesis_kind: Literal["source_monophonic", "derived_monophonic"]
    authoritative: bool
    source_track_index: NonNegativeInt
    source_channel: NonNegativeInt
    source_track_name: str | None = None
    source_programs: tuple[Annotated[int, Field(ge=0, le=127)], ...] = ()
    extraction_method: str
    supporting_methods: tuple[str, ...]
    extraction_version: str
    parameters: dict[str, int | float | str | bool]
    quality: VoiceQualityEvidence
    note_count: NonNegativeInt


class TargetTrackRepresentation(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    target_id: str
    track_index: NonNegativeInt
    event_ids: tuple[str, ...]
    pitch_cents: tuple[int, ...]
    relative_pitch_cents: tuple[float, ...]
    interval_cents: tuple[int, ...]
    contour_steps: tuple[Annotated[int, Field(ge=-1, le=1)], ...]
    onset_microseconds: tuple[NonNegativeInt, ...]
    log_ioi_milli: tuple[int, ...]
    start_ticks: tuple[NonNegativeInt, ...]
    end_ticks: tuple[NonNegativeInt, ...]
    velocities: tuple[Annotated[int, Field(ge=0, le=127)], ...]
    voice_hypothesis: VoiceHypothesis | None = None


class IndexSnapshot(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    index_id: str
    created_at: str
    target_entries: tuple[TargetIndexEntry, ...]
    shingles: dict[str, tuple[str, ...]]
    parameters: dict[str, int | float | str | bool]


class CandidateEvidence(StrictModel):
    target_id: str
    track_index: NonNegativeInt
    start_event_index: NonNegativeInt
    shingle_votes: NonNegativeInt
    matched_shingles: tuple[str, ...]
    weighted_shingle_score: Annotated[float, Field(ge=0.0)] = 0.0
    voice_id: str = "legacy-source-track"
    source_channel: NonNegativeInt = 0


class AlignmentPair(StrictModel):
    query_event_indices: tuple[NonNegativeInt, ...]
    target_event_indices: tuple[NonNegativeInt, ...]
    transition: Literal["match", "query_insertion", "target_insertion", "split", "merge"]
    pitch_residual_cents: float | None
    timing_residual_ppm: int | None
    local_cost: float


class TransformationEvidence(StrictModel):
    transposition_cents: int
    tempo_ratio_ppm: PositiveInt
    octave_repairs: NonNegativeInt
    query_insertions: NonNegativeInt
    target_insertions: NonNegativeInt
    splits: NonNegativeInt
    merges: NonNegativeInt


class ScoreEvidence(StrictModel):
    value: float
    label: Literal["uncalibrated_similarity"] = "uncalibrated_similarity"
    higher_is_better: Literal[True] = True
    calibrated_probability: Literal[False] = False
    components: dict[str, float]


class ProvenanceEvidence(StrictModel):
    query_asset_id: str
    query_asset_sha256: str
    query_analysis_artifact_id: str
    query_interpretation_id: str
    target_asset_id: str
    target_asset_sha256: str
    target_artifact_id: str
    index_id: str
    index_artifact_id: str
    search_id: str
    alignment_id: str | None
    target_representation_artifact_id: str | None = None
    target_voice_id: str | None = None
    search_interpretation_id: str | None = None
    search_interpretation_artifact_id: str | None = None


class AlignmentArtifact(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    alignment_id: str
    search_id: str
    target_id: str
    track_index: NonNegativeInt
    target_event_start_index: NonNegativeInt
    target_event_end_index: PositiveInt
    target_range: TickRange
    target_time_range: MicrosecondRange | None = None
    candidate: CandidateEvidence
    aligned_events: tuple[AlignmentPair, ...]
    transformations: TransformationEvidence
    score: ScoreEvidence
    contour_dtw_cost: float
    voice_hypothesis: VoiceHypothesisSummary | None = None
    query_interpretation_id: str | None = None
    query_interpretation_artifact_id: str | None = None


class MatchAlternative(StrictModel):
    rank: PositiveInt
    target_id: str
    target_name: str
    track_index: NonNegativeInt
    target_range: TickRange
    target_time_range: MicrosecondRange | None = None
    score: ScoreEvidence
    transformations: TransformationEvidence
    candidate: CandidateEvidence
    source_channel: NonNegativeInt = 0
    voice_hypothesis: VoiceHypothesisSummary | None = None
    query_interpretation_id: str | None = None


class SearchResult(StrictModel):
    schema_version: Literal["0.0.1"] = SCHEMA_VERSION
    search_id: str
    status: ResultStatus
    reason_codes: tuple[str, ...]
    status_explanation: str
    selected_target_id: str | None
    selected_target_name: str | None
    selected_track_index: NonNegativeInt | None
    target_range: TickRange | None
    target_time_range: MicrosecondRange | None = None
    query_source_range: SampleRange
    score: ScoreEvidence | None
    transformations: TransformationEvidence | None
    aligned_events: tuple[AlignmentPair, ...]
    strong_rhythmic_target_ticks: tuple[NonNegativeInt, ...]
    alternatives: tuple[MatchAlternative, ...]
    quality_warnings: tuple[QualityWarning, ...]
    provenance: ProvenanceEvidence | None
    selected_source_channel: NonNegativeInt | None = None
    selected_voice_hypothesis: VoiceHypothesisSummary | None = None
    selected_query_interpretation_id: str | None = None
    evaluated_query_interpretation_ids: tuple[str, ...] = ()
    usable_query_event_ids: tuple[str, ...] = ()
    excluded_query_event_ids: tuple[str, ...] = ()


class JobRecord(StrictModel):
    job_id: str
    operation: str
    status: JobStatus
    arguments: dict[str, str | int | float | bool]
    result_entity_id: str | None
    error: str | None
    created_at: str
    updated_at: str
