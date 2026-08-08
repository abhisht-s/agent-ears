"""Versioned domain contracts for durable evidence and agent-facing results."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

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
    TARGET_AUDIO = "target_audio"
    TARGET_AUDIO_REFERENCE = "target_audio_reference"
    TARGET_AUDIO_DERIVATIVE = "target_audio_derivative"
    TARGET_STEM = "target_stem"
    ANALYSIS_AUDIO = "analysis_audio"
    ARTIFACT = "artifact"
    RENDER = "render"


class TargetKind(StrEnum):
    MIDI = "midi"
    TIMED_NOTES = "timed_notes"
    AUDIO = "audio"


class TargetCoordinateKind(StrEnum):
    MIDI_TICKS = "midi_ticks"
    DECODED_SAMPLES = "decoded_samples"


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


class MediaMicrosecondRange(StrictModel):
    """Signed raw container/media PTS range for streams that begin before zero."""

    start_microsecond: int
    end_microsecond: int

    @model_validator(mode="after")
    def ordered(self) -> MediaMicrosecondRange:
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


class AudioTargetArtifact(StrictModel):
    """Immutable descriptor for an encoded real-audio target."""

    schema_version: Literal["0.0.2"] = "0.0.2"
    target_id: str
    name: str
    kind: Literal[TargetKind.AUDIO] = TargetKind.AUDIO
    asset_id: str
    original_name: str
    media_type: str
    source_sha256: str
    byte_length: NonNegativeInt


class DecoderInvocation(StrictModel):
    executable: str
    arguments: tuple[str, ...]
    version_line: str
    build_configuration: str
    return_code: int
    stderr: str


class DecodedReferenceManifest(StrictModel):
    """Pinned decoded-reference clock and the evidence needed to reproduce it."""

    schema_version: Literal["0.0.2"] = "0.0.2"
    target_id: str
    source_asset_id: str
    source_sha256: str
    source_byte_length: NonNegativeInt
    source_media_type: str
    source_original_name: str
    container_format_names: tuple[str, ...]
    selected_stream_index: NonNegativeInt
    codec_name: str
    codec_long_name: str | None = None
    source_sample_format: str | None = None
    source_sample_rate: PositiveInt
    source_channels: PositiveInt
    source_channel_layout: str | None = None
    media_time_base: Rational | None = None
    media_start_pts: int | None = None
    media_start_microseconds: int
    reported_duration_microseconds: int | None = None
    presentation_frame_count: NonNegativeInt | None = None
    canonicalization: str
    coded_sample_count: NonNegativeInt | None = None
    skip_samples: NonNegativeInt = 0
    discard_padding: NonNegativeInt = 0
    decoded_asset_id: str
    decoded_sha256: str
    decoded_media_type: Literal["audio/wav"] = "audio/wav"
    decoded_sample_rate: PositiveInt
    decoded_channels: PositiveInt
    decoded_channel_layout: str | None = None
    decoded_frame_count: NonNegativeInt
    decoded_sample_format: str
    ffprobe_artifact_id: str
    probe: DecoderInvocation
    decoder: DecoderInvocation
    warnings: tuple[str, ...] = ()


class TargetTimeMap(StrictModel):
    """Exact decoded-reference sample to container/media-time mapping."""

    schema_version: Literal["0.0.2"] = "0.0.2"
    target_id: str
    decoded_stream_id: str
    decoded_sample_rate: PositiveInt
    decoded_frame_count: NonNegativeInt
    media_time_base: Rational | None = None
    reference_zero_media_pts: int | None = None
    reference_zero_media_microseconds: int = 0
    rounding: Literal["nearest_half_up"] = "nearest_half_up"
    interpretation: str

    def reference_sample_to_media_microseconds(self, sample: int) -> int:
        if sample < 0 or sample > self.decoded_frame_count:
            raise ValueError("reference sample is outside the decoded stream")
        delta = _round_fraction(Fraction(sample * 1_000_000, self.decoded_sample_rate))
        return self.reference_zero_media_microseconds + delta

    def media_microseconds_to_reference_sample(self, microseconds: int) -> int:
        delta = microseconds - self.reference_zero_media_microseconds
        if delta < 0:
            raise ValueError("media time precedes the decoded reference")
        sample = _round_fraction(Fraction(delta * self.decoded_sample_rate, 1_000_000))
        if sample > self.decoded_frame_count:
            raise ValueError("media time is outside the decoded stream")
        return sample


class AnalysisDerivativeEvidence(StrictModel):
    schema_version: Literal["0.0.2"] = "0.0.2"
    derivative_id: str
    target_id: str
    source_role: str
    source_asset_id: str
    source_stream_id: str
    derived_asset_id: str
    derived_stream_id: str
    operation: str
    parameters: dict[str, int | float | str | bool]
    channel_matrix: tuple[tuple[float, ...], ...]
    normalization: Literal["none"] = "none"
    limiting: Literal["none"] = "none"
    sample_time_map_artifact_id: str
    source_sample_rate: PositiveInt
    derived_sample_rate: PositiveInt
    source_frame_count: NonNegativeInt
    derived_frame_count: NonNegativeInt
    artifact_sha256: str


class TargetPitchFrame(StrictModel):
    frame_id: str
    frame_index: NonNegativeInt
    derivative_start_sample: NonNegativeInt
    derivative_center_sample: NonNegativeInt
    derivative_end_sample: NonNegativeInt
    reference_start_sample: NonNegativeInt
    reference_center_sample: NonNegativeInt
    reference_end_sample: NonNegativeInt
    f0_hz: float | None
    pitch_midi_cents: float | None
    voiced_probability: UnitFloat
    voiced: bool
    backend: str

    @model_validator(mode="after")
    def ordered_support(self) -> TargetPitchFrame:
        if not (
            self.derivative_start_sample
            <= self.derivative_center_sample
            <= self.derivative_end_sample
        ):
            raise ValueError("derivative frame support is not ordered")
        if not (
            self.reference_start_sample <= self.reference_center_sample <= self.reference_end_sample
        ):
            raise ValueError("reference frame support is not ordered")
        return self


class TargetOnsetEvidence(StrictModel):
    onset_id: str
    derivative_sample: NonNegativeInt
    reference_sample: NonNegativeInt
    strength: float
    backend: str


class ActivationGridEvidence(StrictModel):
    backend: str
    backend_version: str
    raw_artifact_id: str
    raw_artifact_sha256: str
    arrays: dict[str, tuple[NonNegativeInt, ...]]
    dtype: str
    nominal_frames_per_second: float
    timing_interpretation: str


class TargetNoteCandidate(StrictModel):
    candidate_id: str
    pitch_midi_cents: float
    derivative_start_sample: NonNegativeInt
    derivative_end_sample: NonNegativeInt
    reference_start_sample: NonNegativeInt
    reference_end_sample: NonNegativeInt
    onset_confidence: UnitFloat
    note_confidence: UnitFloat
    amplitude: UnitFloat
    source_role: str
    backend: str
    supporting_evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def positive_support(self) -> TargetNoteCandidate:
        if self.derivative_end_sample <= self.derivative_start_sample:
            raise ValueError("note candidate derivative support must be positive")
        if self.reference_end_sample <= self.reference_start_sample:
            raise ValueError("note candidate reference support must be positive")
        return self


class TargetEventEvidenceArtifact(StrictModel):
    schema_version: Literal["0.0.2"] = "0.0.2"
    evidence_id: str
    target_id: str
    source_role: str
    derivative_id: str
    derivative_artifact_id: str
    backend: str
    backend_version: str
    backend_config: dict[str, int | float | str | bool]
    pitch_frames: tuple[TargetPitchFrame, ...] = ()
    onsets: tuple[TargetOnsetEvidence, ...] = ()
    note_candidates: tuple[TargetNoteCandidate, ...] = ()
    activation_grid: ActivationGridEvidence | None = None
    warnings: tuple[str, ...] = ()


class AudioVoiceHypothesisSummary(StrictModel):
    view_id: str
    source_role: str
    source_derivative_artifact_id: str
    evidence_artifact_id: str
    extraction_backend: str
    extraction_backend_version: str
    path_method: str
    path_rank: PositiveInt
    path_score: float
    authoritative: Literal[False] = False
    event_count: NonNegativeInt
    warnings: tuple[str, ...] = ()


class TargetRepresentationBundle(StrictModel):
    schema_version: Literal["0.0.2"] = "0.0.2"
    bundle_id: str
    target_id: str
    target_artifact_id: str
    decode_manifest_artifact_id: str
    target_time_map_artifact_id: str
    decoded_asset_id: str
    decoded_stream_id: str
    analysis_config: dict[str, JsonValue]
    cache_key: str
    derivative_artifact_ids: tuple[str, ...]
    evidence_artifact_ids: tuple[str, ...]
    backend_invocation_artifact_ids: tuple[str, ...] = ()
    representation_artifact_ids: tuple[str, ...]
    warnings: tuple[str, ...] = ()


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
    track_index: NonNegativeInt | None
    event_count: NonNegativeInt
    representation_artifact_id: str
    voice_id: str = "legacy-source-track"
    view_id: str | None = None
    source_channel: NonNegativeInt | None = 0


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
    schema_version: Literal["0.0.1", "0.0.2"] = SCHEMA_VERSION
    target_id: str
    track_index: NonNegativeInt | None
    view_id: str | None = None
    source_group_id: str | None = None
    event_ids: tuple[str, ...]
    pitch_cents: tuple[int, ...]
    relative_pitch_cents: tuple[float, ...]
    interval_cents: tuple[int, ...]
    contour_steps: tuple[Annotated[int, Field(ge=-1, le=1)], ...]
    onset_microseconds: tuple[NonNegativeInt, ...]
    log_ioi_milli: tuple[int, ...]
    coordinate_kind: TargetCoordinateKind = TargetCoordinateKind.MIDI_TICKS
    start_ticks: tuple[NonNegativeInt, ...] = ()
    end_ticks: tuple[NonNegativeInt, ...] = ()
    velocities: tuple[Annotated[int, Field(ge=0, le=127)], ...] = ()
    start_samples: tuple[NonNegativeInt, ...] = ()
    end_samples: tuple[NonNegativeInt, ...] = ()
    reference_sample_rate: PositiveInt | None = None
    reference_zero_media_microseconds: int | None = None
    event_confidences: tuple[UnitFloat, ...] = ()
    source_evidence_refs: tuple[str, ...] = ()
    voice_hypothesis: VoiceHypothesis | None = None
    audio_voice_hypothesis: AudioVoiceHypothesisSummary | None = None

    @model_validator(mode="after")
    def synchronized_events(self) -> TargetTrackRepresentation:
        count = len(self.event_ids)
        synchronized = (
            self.pitch_cents,
            self.relative_pitch_cents,
            self.onset_microseconds,
        )
        if any(len(values) != count for values in synchronized):
            raise ValueError("target representation event arrays must have equal lengths")
        if len(self.interval_cents) != max(0, count - 1):
            raise ValueError("target interval array has the wrong length")
        if len(self.contour_steps) != max(0, count - 1):
            raise ValueError("target contour array has the wrong length")
        if len(self.log_ioi_milli) != max(0, count - 1):
            raise ValueError("target rhythm array has the wrong length")
        if self.coordinate_kind == TargetCoordinateKind.MIDI_TICKS:
            if len(self.start_ticks) != count or len(self.end_ticks) != count:
                raise ValueError("MIDI representations require tick ranges")
            if self.start_samples or self.end_samples:
                raise ValueError("MIDI representations cannot carry decoded sample ranges")
            if self.audio_voice_hypothesis is not None:
                raise ValueError("MIDI representations cannot carry audio voice provenance")
            if (
                self.view_id is not None
                or self.start_samples
                or self.end_samples
                or self.reference_sample_rate is not None
                or self.reference_zero_media_microseconds is not None
                or self.event_confidences
                or self.source_evidence_refs
            ):
                raise ValueError("MIDI representations cannot carry audio evidence fields")
        else:
            if len(self.start_samples) != count or len(self.end_samples) != count:
                raise ValueError("audio representations require decoded sample ranges")
            if self.start_ticks or self.end_ticks:
                raise ValueError("audio representations cannot carry MIDI tick ranges")
            if self.velocities:
                raise ValueError("audio representations cannot carry MIDI velocity values")
            if self.reference_sample_rate is None:
                raise ValueError("audio representations require a reference sample rate")
            if self.voice_hypothesis is not None:
                raise ValueError("audio representations cannot carry MIDI voice provenance")
            if self.track_index is not None:
                raise ValueError("audio representations cannot carry a MIDI track index")
            if self.view_id is None or self.source_group_id is None:
                raise ValueError("audio representations require view and source-group identities")
            if self.audio_voice_hypothesis is None:
                raise ValueError("audio representations require an audio voice hypothesis")
            if self.audio_voice_hypothesis.view_id != self.view_id:
                raise ValueError("audio hypothesis and representation view IDs differ")
            if self.reference_zero_media_microseconds is None:
                raise ValueError("audio representations require their media-time origin")
            if len(self.event_confidences) != count:
                raise ValueError("audio confidence evidence must cover every event")
            if len(self.source_evidence_refs) != count:
                raise ValueError("audio source evidence must cover every event")
        if self.velocities and len(self.velocities) != count:
            raise ValueError("velocity evidence must cover every event")
        if self.event_confidences and len(self.event_confidences) != count:
            raise ValueError("confidence evidence must cover every event")
        if self.source_evidence_refs and len(self.source_evidence_refs) != count:
            raise ValueError("source evidence references must cover every event")
        if any(
            right < left
            for left, right in zip(
                self.onset_microseconds, self.onset_microseconds[1:], strict=False
            )
        ):
            raise ValueError("target event onsets must be monotonic")
        if self.coordinate_kind == TargetCoordinateKind.DECODED_SAMPLES:
            if any(
                end <= start
                for start, end in zip(self.start_samples, self.end_samples, strict=True)
            ):
                raise ValueError("audio event sample ranges must have positive duration")
            if any(
                right < left
                for left, right in zip(self.start_samples, self.start_samples[1:], strict=False)
            ):
                raise ValueError("audio event sample starts must be monotonic")
        return self


class IndexSnapshot(StrictModel):
    schema_version: Literal["0.0.1", "0.0.2"] = SCHEMA_VERSION
    index_id: str
    created_at: str
    target_entries: tuple[TargetIndexEntry, ...]
    shingles: dict[str, tuple[str, ...]]
    parameters: dict[str, int | float | str | bool]


class CandidateEvidence(StrictModel):
    target_id: str
    track_index: NonNegativeInt | None
    start_event_index: NonNegativeInt
    shingle_votes: NonNegativeInt
    matched_shingles: tuple[str, ...]
    weighted_shingle_score: Annotated[float, Field(ge=0.0)] = 0.0
    voice_id: str = "legacy-source-track"
    view_id: str | None = None
    source_channel: NonNegativeInt | None = 0


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
    target_decode_manifest_artifact_id: str | None = None
    target_time_map_artifact_id: str | None = None
    target_analysis_artifact_id: str | None = None
    target_evidence_artifact_id: str | None = None
    target_view_id: str | None = None


class AlignmentArtifact(StrictModel):
    schema_version: Literal["0.0.1", "0.0.2"] = SCHEMA_VERSION
    alignment_id: str
    search_id: str
    target_id: str
    track_index: NonNegativeInt | None
    target_event_start_index: NonNegativeInt
    target_event_end_index: PositiveInt
    coordinate_kind: TargetCoordinateKind = TargetCoordinateKind.MIDI_TICKS
    target_range: TickRange | None
    target_sample_range: SampleRange | None = None
    target_time_range: MicrosecondRange | MediaMicrosecondRange | None = None
    candidate: CandidateEvidence
    aligned_events: tuple[AlignmentPair, ...]
    transformations: TransformationEvidence
    score: ScoreEvidence
    contour_dtw_cost: float
    voice_hypothesis: VoiceHypothesisSummary | None = None
    audio_voice_hypothesis: AudioVoiceHypothesisSummary | None = None
    query_interpretation_id: str | None = None
    query_interpretation_artifact_id: str | None = None

    @model_validator(mode="after")
    def consistent_coordinate_evidence(self) -> AlignmentArtifact:
        if self.coordinate_kind == TargetCoordinateKind.MIDI_TICKS:
            if self.target_range is None or self.target_sample_range is not None:
                raise ValueError("MIDI alignment requires ticks and forbids decoded samples")
            if self.audio_voice_hypothesis is not None:
                raise ValueError("MIDI alignment cannot carry an audio hypothesis")
        else:
            if self.target_range is not None:
                raise ValueError("audio alignment cannot carry MIDI ticks")
            if self.target_sample_range is None or self.target_time_range is None:
                raise ValueError("audio alignment requires sample and media-time ranges")
            if self.track_index is not None or self.voice_hypothesis is not None:
                raise ValueError("audio alignment cannot carry MIDI voice coordinates")
            if self.audio_voice_hypothesis is None:
                raise ValueError("audio alignment requires an audio hypothesis")
        return self


class MatchAlternative(StrictModel):
    rank: PositiveInt
    target_id: str
    target_name: str
    track_index: NonNegativeInt | None
    coordinate_kind: TargetCoordinateKind = TargetCoordinateKind.MIDI_TICKS
    target_range: TickRange | None
    target_sample_range: SampleRange | None = None
    target_time_range: MicrosecondRange | MediaMicrosecondRange | None = None
    score: ScoreEvidence
    transformations: TransformationEvidence
    candidate: CandidateEvidence
    source_channel: NonNegativeInt | None = 0
    voice_hypothesis: VoiceHypothesisSummary | None = None
    audio_voice_hypothesis: AudioVoiceHypothesisSummary | None = None
    query_interpretation_id: str | None = None

    @model_validator(mode="after")
    def consistent_coordinate_evidence(self) -> MatchAlternative:
        if self.coordinate_kind == TargetCoordinateKind.MIDI_TICKS:
            if self.target_range is None or self.target_sample_range is not None:
                raise ValueError("MIDI alternative requires ticks and forbids decoded samples")
            if self.audio_voice_hypothesis is not None:
                raise ValueError("MIDI alternative cannot carry an audio hypothesis")
        else:
            if self.target_range is not None:
                raise ValueError("audio alternative cannot carry MIDI ticks")
            if self.target_sample_range is None or self.target_time_range is None:
                raise ValueError("audio alternative requires sample and media-time ranges")
            if self.track_index is not None or self.source_channel is not None:
                raise ValueError("audio alternative cannot carry MIDI track/channel coordinates")
            if self.voice_hypothesis is not None or self.audio_voice_hypothesis is None:
                raise ValueError("audio alternative requires only audio voice provenance")
        return self


class SearchResult(StrictModel):
    schema_version: Literal["0.0.1", "0.0.2"] = SCHEMA_VERSION
    search_id: str
    status: ResultStatus
    reason_codes: tuple[str, ...]
    status_explanation: str
    selected_target_id: str | None
    selected_target_name: str | None
    selected_track_index: NonNegativeInt | None
    target_range: TickRange | None
    target_sample_range: SampleRange | None = None
    target_time_range: MicrosecondRange | MediaMicrosecondRange | None = None
    query_source_range: SampleRange
    score: ScoreEvidence | None
    transformations: TransformationEvidence | None
    aligned_events: tuple[AlignmentPair, ...]
    strong_rhythmic_target_ticks: tuple[NonNegativeInt, ...]
    strong_rhythmic_target_samples: tuple[NonNegativeInt, ...] = ()
    alternatives: tuple[MatchAlternative, ...]
    quality_warnings: tuple[QualityWarning, ...]
    provenance: ProvenanceEvidence | None
    selected_source_channel: NonNegativeInt | None = None
    selected_voice_hypothesis: VoiceHypothesisSummary | None = None
    selected_audio_voice_hypothesis: AudioVoiceHypothesisSummary | None = None
    selected_target_coordinate_kind: TargetCoordinateKind | None = None
    selected_target_original_name: str | None = None
    selected_target_view_id: str | None = None
    selected_target_source_role: str | None = None
    selected_target_backend: str | None = None
    selected_query_interpretation_id: str | None = None
    evaluated_query_interpretation_ids: tuple[str, ...] = ()
    usable_query_event_ids: tuple[str, ...] = ()
    excluded_query_event_ids: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def infer_legacy_midi_coordinates(cls, value: Any) -> Any:
        if (
            isinstance(value, dict)
            and value.get("schema_version", "0.0.1") == "0.0.1"
            and value.get("selected_target_coordinate_kind") is None
            and value.get("target_range") is not None
            and value.get("target_sample_range") is None
        ):
            value = {**value, "selected_target_coordinate_kind": "midi_ticks"}
        return value

    @model_validator(mode="after")
    def consistent_selected_coordinates(self) -> SearchResult:
        coordinate = self.selected_target_coordinate_kind
        if coordinate is None:
            if any(
                value is not None
                for value in (
                    self.selected_target_id,
                    self.target_range,
                    self.target_sample_range,
                    self.target_time_range,
                    self.selected_voice_hypothesis,
                    self.selected_audio_voice_hypothesis,
                )
            ):
                raise ValueError("unselected result cannot carry target coordinate evidence")
            return self
        if self.selected_target_id is None or self.target_time_range is None:
            raise ValueError("selected result requires a target and elapsed media time")
        if coordinate == TargetCoordinateKind.MIDI_TICKS:
            if self.target_range is None or self.target_sample_range is not None:
                raise ValueError("MIDI result requires ticks and forbids decoded samples")
            if self.strong_rhythmic_target_samples or self.selected_audio_voice_hypothesis:
                raise ValueError("MIDI result cannot carry audio-only evidence")
        else:
            if self.target_range is not None or self.target_sample_range is None:
                raise ValueError("audio result requires decoded samples and forbids MIDI ticks")
            if self.strong_rhythmic_target_ticks:
                raise ValueError("audio result cannot carry rhythmic MIDI ticks")
            if self.selected_track_index is not None or self.selected_source_channel is not None:
                raise ValueError("audio result cannot carry MIDI track/channel coordinates")
            if self.selected_voice_hypothesis is not None:
                raise ValueError("audio result cannot carry MIDI voice provenance")
            if self.selected_audio_voice_hypothesis is None or self.selected_target_view_id is None:
                raise ValueError("audio result requires view and audio hypothesis provenance")
        return self


class JobRecord(StrictModel):
    job_id: str
    operation: str
    status: JobStatus
    arguments: dict[str, str | int | float | bool]
    result_entity_id: str | None
    error: str | None
    created_at: str
    updated_at: str
