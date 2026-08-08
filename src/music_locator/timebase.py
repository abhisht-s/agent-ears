"""Integer-only conversions between MIDI ticks and elapsed microseconds."""

from __future__ import annotations

from fractions import Fraction

from music_locator.models import TempoChange


def normalize_tempo_map(changes: tuple[TempoChange, ...]) -> tuple[TempoChange, ...]:
    """Sort and deduplicate tempo changes, ensuring a tick-zero default."""

    by_tick = {change.tick: change for change in changes}
    if 0 not in by_tick:
        by_tick[0] = TempoChange(tick=0, microseconds_per_beat=500_000)
    return tuple(by_tick[tick] for tick in sorted(by_tick))


def tick_to_microseconds(tick: int, ticks_per_beat: int, tempo_map: tuple[TempoChange, ...]) -> int:
    """Convert a tick to nearest integer microseconds without float time coordinates."""

    if tick < 0:
        raise ValueError("tick must be non-negative")
    changes = normalize_tempo_map(tempo_map)
    elapsed = Fraction(0)
    previous_tick = 0
    tempo = changes[0].microseconds_per_beat
    for change in changes[1:]:
        if change.tick >= tick:
            break
        elapsed += Fraction((change.tick - previous_tick) * tempo, ticks_per_beat)
        previous_tick = change.tick
        tempo = change.microseconds_per_beat
    elapsed += Fraction((tick - previous_tick) * tempo, ticks_per_beat)
    quotient, remainder = divmod(elapsed.numerator, elapsed.denominator)
    return quotient + int(2 * remainder >= elapsed.denominator)


def microseconds_to_tick(
    microseconds: int, ticks_per_beat: int, tempo_map: tuple[TempoChange, ...]
) -> int:
    """Invert a tempo map to nearest tick."""

    if microseconds < 0:
        raise ValueError("microseconds must be non-negative")
    changes = normalize_tempo_map(tempo_map)
    remaining = Fraction(microseconds)
    previous_tick = 0
    tempo = changes[0].microseconds_per_beat
    for change in changes[1:]:
        segment = Fraction((change.tick - previous_tick) * tempo, ticks_per_beat)
        if remaining <= segment:
            break
        remaining -= segment
        previous_tick = change.tick
        tempo = change.microseconds_per_beat
    tick_fraction = Fraction(previous_tick) + remaining * ticks_per_beat / tempo
    quotient, remainder = divmod(tick_fraction.numerator, tick_fraction.denominator)
    return quotient + int(2 * remainder >= tick_fraction.denominator)
