"""Tests for the native AED overlap segmenter binding."""

from __future__ import annotations

from dataclasses import replace
import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from omnivad import AedOverlapSegmenter, OmniAED
from omnivad.aed_overlap import (
    AED_KIND_MASK_MUSIC,
    AED_KIND_MASK_SINGING,
    AED_KIND_MASK_SPEECH,
    AedOnlineEvent,
)
from omnivad._binding import OmniAedOnlineEvent, OmniAedOnlineSegment, OmniAedOverlapConfig

SAMPLE_RATE = 16000


def _load_fixture(name: str = "hello_en.wav") -> np.ndarray:
    audio, sr = sf.read(Path("tests/data") / name, dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected 16kHz fixture, got {sr}Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


@lru_cache(maxsize=1)
def _primary_mp3_audio() -> np.ndarray:
    return _load_fixture("DQacCB9tDaw_16K_2mins.mp3")


@lru_cache(maxsize=1)
def _primary_mp3_non_overlap_speech_events() -> tuple[tuple[float, float], ...]:
    aed = OmniAED(
        speech_threshold=0.5,
        singing_threshold=0.5,
        music_threshold=0.5,
        max_speech_frames=20_000,
    )
    try:
        result = aed.detect(_primary_mp3_audio())
    finally:
        aed.close()
    return tuple(result["events"]["speech"])


def _crop_seconds(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    return audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]


def _real_speech_clip(index: int, duration: float = 1.8) -> np.ndarray:
    events = [
        event
        for event in _primary_mp3_non_overlap_speech_events()
        if event[1] - event[0] >= duration + 0.1
    ]
    assert len(events) > index, (
        f"Expected at least {index + 1} non-overlap AED speech events "
        f"lasting {duration + 0.1:.1f}s or longer in the primary MP3 fixture"
    )
    start, end = events[index]
    clip_start = max(start + 0.05, (start + end - duration) / 2)
    clip_start = min(clip_start, end - duration - 0.05)
    return _crop_seconds(_primary_mp3_audio(), clip_start, clip_start + duration)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _concat_audio(parts: list[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def _run_segmenter(
    audio: np.ndarray,
    step: int,
    *,
    hop_seconds: float = 0.5,
    overlap_seconds: float = 0.1,
    hard_split_pause_seconds: float = 0.2,
    max_chunk_seconds: float = 2.0,
    merge_gap_seconds: float = 0.2,
    music_gap_tolerance_seconds: float = 0.0,
):
    segmenter = AedOverlapSegmenter(
        hop_seconds=hop_seconds,
        overlap_seconds=overlap_seconds,
        hard_split_pause_seconds=hard_split_pause_seconds,
        max_chunk_seconds=max_chunk_seconds,
        merge_gap_seconds=merge_gap_seconds,
        music_gap_tolerance_seconds=music_gap_tolerance_seconds,
    )
    try:
        segments = []
        events = []
        for start in range(0, len(audio), step):
            result = segmenter.ingest(audio[start : start + step])
            event_offset = len(events)
            segments.extend(
                replace(segment, event_start_idx=segment.event_start_idx + event_offset) for segment in result.segments
            )
            events.extend(result.events)
        result = segmenter.flush()
        event_offset = len(events)
        segments.extend(
            replace(segment, event_start_idx=segment.event_start_idx + event_offset) for segment in result.segments
        )
        events.extend(result.events)
        return segments, events
    finally:
        segmenter.close()


def test_online_event_transcribable_uses_speech_or_singing_mask():
    speech = AedOnlineEvent(0.0, 1.0, "speech", AED_KIND_MASK_SPEECH, 0.9, 0.1, 0.0, 0.9)
    singing = AedOnlineEvent(0.0, 1.0, "singing", AED_KIND_MASK_SINGING, 0.1, 0.9, 0.0, 0.9)
    mixed_music_singing = AedOnlineEvent(
        0.0,
        1.0,
        "mixed",
        AED_KIND_MASK_MUSIC | AED_KIND_MASK_SINGING,
        0.0,
        0.8,
        0.9,
        0.9,
    )
    music = AedOnlineEvent(0.0, 1.0, "music", AED_KIND_MASK_MUSIC, 0.0, 0.0, 0.9, 0.9)
    silence = AedOnlineEvent(0.0, 1.0, "silence", 0, 0.0, 0.0, 0.0, 0.0)

    assert speech.is_transcribable
    assert singing.is_transcribable
    assert mixed_music_singing.is_transcribable
    assert not music.is_transcribable
    assert not silence.is_transcribable


def _run_silence_with_random_chunks(num_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    audio = np.zeros(num_samples, dtype=np.int16)
    segmenter = AedOverlapSegmenter(
        hop_seconds=0.5,
        overlap_seconds=0.1,
        hard_split_pause_seconds=0.2,
        max_chunk_seconds=2.0,
    )
    try:
        segments = []
        events = []
        pos = 0
        while pos < num_samples:
            step = int(rng.integers(1, min(5000, num_samples - pos) + 1))
            result = segmenter.ingest(audio[pos : pos + step])
            segments.extend(result.segments)
            events.extend(result.events)
            pos += step
        result = segmenter.flush()
        segments.extend(result.segments)
        events.extend(result.events)
        return segments, events
    finally:
        segmenter.close()


def test_abi_struct_sizes():
    assert ctypes.sizeof(OmniAedOverlapConfig) == 52
    assert ctypes.sizeof(OmniAedOnlineEvent) == 32
    assert ctypes.sizeof(OmniAedOnlineSegment) == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hop_seconds": 0.505, "overlap_seconds": 0.1},
        {"hop_seconds": 0.5, "overlap_seconds": 0.5},
        {"hop_seconds": 0.5, "overlap_seconds": -0.01},
        {"hop_seconds": 0.5, "edge_guard_seconds": 0.5},
        {"hop_seconds": 0.5, "edge_guard_seconds": 0.505},
        {"max_chunk_seconds": 0.0},
        {"speech_threshold": -0.1},
        {"music_threshold": 1.1},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(RuntimeError, match="invalid argument"):
        AedOverlapSegmenter(**kwargs)


def test_silence_returns_no_segments():
    segmenter = AedOverlapSegmenter(
        hop_seconds=0.5,
        overlap_seconds=0.1,
        hard_split_pause_seconds=0.2,
        max_chunk_seconds=2.0,
    )
    try:
        result = segmenter.ingest(np.zeros(1600, dtype=np.int16))
        assert result.segments == []
        assert result.events == []
        result = segmenter.flush()
        assert result.segments == []
        assert result.events == []
    finally:
        segmenter.close()


def test_empty_ingest_and_double_flush_are_empty():
    segmenter = AedOverlapSegmenter(
        hop_seconds=0.5,
        overlap_seconds=0.1,
        hard_split_pause_seconds=0.2,
        max_chunk_seconds=2.0,
    )
    try:
        assert segmenter.ingest(np.zeros(0, dtype=np.float32)).segments == []
        assert segmenter.flush().segments == []
        assert segmenter.flush().segments == []
    finally:
        segmenter.close()


@pytest.mark.parametrize("num_samples", [0, 1, 120, 399, 400, 500, 559, 560, 600, 1000, 16000, 24000, 32000])
def test_short_and_boundary_silence_lengths_do_not_error(num_samples):
    segments, events = _run_silence_with_random_chunks(num_samples, seed=num_samples + 17)
    assert segments == []
    assert events == []


def test_random_silence_lengths_up_to_two_seconds_do_not_error():
    rng = np.random.default_rng(12345)
    sample_counts = list(rng.integers(1, 16000, size=8))
    sample_counts.extend(rng.integers(16000, 32001, size=8))
    sample_counts.extend([120, 16000, 20000, 24000, 32000])

    for num_samples in sample_counts:
        segments, events = _run_silence_with_random_chunks(int(num_samples), seed=int(num_samples) + 99)
        assert segments == []
        assert events == []


def test_ingest_after_flush_is_rejected():
    segmenter = AedOverlapSegmenter(
        hop_seconds=0.5,
        overlap_seconds=0.1,
        hard_split_pause_seconds=0.2,
        max_chunk_seconds=2.0,
    )
    try:
        segmenter.flush()
        with pytest.raises(RuntimeError, match="invalid argument"):
            segmenter.ingest(np.zeros(1600, dtype=np.float32))
    finally:
        segmenter.close()


def test_rejects_non_mono_pcm_array():
    segmenter = AedOverlapSegmenter()
    try:
        with pytest.raises(ValueError, match="1D mono"):
            segmenter.ingest(np.zeros((1600, 2), dtype=np.float32))
    finally:
        segmenter.close()


def test_clone_has_fresh_state():
    audio = _load_fixture()
    segmenter = AedOverlapSegmenter(
        hop_seconds=0.5,
        overlap_seconds=0.1,
        hard_split_pause_seconds=0.2,
        max_chunk_seconds=2.0,
    )
    clone = segmenter.clone()
    try:
        assert segmenter.ingest(audio[:800]).segments == []
        original_final = segmenter.flush()
        clone_final = clone.ingest(audio[:800])
        assert clone_final.segments == []
        clone_final = clone.flush()
        assert clone_final == original_final
    finally:
        clone.close()
        segmenter.close()


def test_arbitrary_ingest_chunk_sizes_match():
    audio = _load_fixture()
    baseline_segments, baseline_events = _run_segmenter(audio, 800)

    for step in (1600, 5000):
        segments, events = _run_segmenter(audio, step)
        assert segments == baseline_segments
        assert events == baseline_events


def test_segments_are_monotonic_and_event_classes_are_visible():
    audio = _load_fixture()
    segments, events = _run_segmenter(audio, 1600)

    assert segments
    assert events
    for prev, cur in zip(segments, segments[1:]):
        assert prev.end <= cur.start
    for segment in segments:
        assert segment.end > segment.start
        assert segment.event_count > 0
    assert any(event.primary_kind == "speech" for event in events)


def test_force_split_clips_returned_events_to_segment_bounds():
    audio = _load_fixture()
    segments, events = _run_segmenter(
        audio,
        1600,
        hop_seconds=0.5,
        overlap_seconds=0.1,
        hard_split_pause_seconds=2.0,
        max_chunk_seconds=0.3,
    )

    assert segments
    assert len(segments) >= 2
    for segment in segments:
        assert segment.end - segment.start <= 0.301
        segment_events = events[segment.event_start_idx : segment.event_start_idx + segment.event_count]
        assert segment_events
        for event in segment_events:
            assert segment.start <= event.start
            assert event.end <= segment.end


def _run_gap_fixture(audio: np.ndarray, max_chunk_seconds: float):
    return _run_segmenter(
        audio,
        3200,
        hop_seconds=0.5,
        overlap_seconds=0.1,
        hard_split_pause_seconds=120.0,
        max_chunk_seconds=max_chunk_seconds,
        merge_gap_seconds=0.0,
        music_gap_tolerance_seconds=0.0,
    )


def test_real_mp3_derived_audio_prefers_longest_internal_gap_before_hard_boundary():
    max_chunk_seconds = 30.0
    segments, _events = _run_segmenter(
        _primary_mp3_audio(),
        32000,
        hop_seconds=2.0,
        overlap_seconds=0.25,
        hard_split_pause_seconds=120.0,
        max_chunk_seconds=max_chunk_seconds,
    )

    assert segments
    for segment in segments:
        assert segment.end - segment.start <= max_chunk_seconds + 1e-3

    boundary_pair = next(
        (
            (prev, cur)
            for prev, cur in zip(segments, segments[1:])
            if 40.0 <= prev.start <= 50.0 and prev.end >= 70.0
        ),
        None,
    )
    assert boundary_pair is not None, (
        "Expected the primary MP3 fixture to contain a long segment crossing "
        "the 30s max-chunk boundary"
    )
    prev, cur = boundary_pair
    hard_boundary = prev.start + max_chunk_seconds

    assert hard_boundary - prev.end > 0.5
    assert cur.start - prev.end > 1.0


def test_real_mp3_derived_composite_selects_inserted_longest_pause():
    audio = _concat_audio(
        [
            _real_speech_clip(0),
            _silence(0.4),
            _real_speech_clip(1),
            _silence(1.2),
            _real_speech_clip(2),
            _silence(0.6),
            _real_speech_clip(3),
        ]
    )

    segments, _events = _run_gap_fixture(audio, max_chunk_seconds=5.0)

    assert len(segments) >= 2
    first, second = segments[0], segments[1]
    assert first.end - first.start <= 5.001
    assert first.end == pytest.approx(4.0, abs=0.2)
    assert second.start == pytest.approx(5.2, abs=0.2)
    assert second.start - first.end > 1.0


def test_real_mp3_derived_composite_ignores_larger_pause_after_hard_boundary():
    audio = _concat_audio(
        [
            _real_speech_clip(0),
            _silence(0.45),
            _real_speech_clip(1, duration=2.6),
            _silence(1.4),
            _real_speech_clip(2),
        ]
    )

    segments, _events = _run_gap_fixture(audio, max_chunk_seconds=4.0)

    assert len(segments) >= 2
    first, second = segments[0], segments[1]
    assert first.end - first.start <= 4.001
    assert first.end == pytest.approx(1.8, abs=0.25)
    assert second.start == pytest.approx(2.25, abs=0.25)
    assert first.end < 3.0


def test_real_mp3_derived_composite_hard_splits_when_no_gap_precedes_boundary():
    audio = _real_speech_clip(0)

    segments, _events = _run_gap_fixture(audio, max_chunk_seconds=0.4)

    assert len(segments) >= 2
    first = segments[0]
    assert first.end - first.start <= 0.401
    assert first.end == pytest.approx(first.start + 0.4, abs=0.03)


def test_primary_mp3_fixture_is_deterministic_across_ingest_chunk_sizes():
    audio = _load_fixture("DQacCB9tDaw_16K_2mins.mp3")
    kwargs = {
        "hop_seconds": 2.0,
        "overlap_seconds": 0.25,
        "hard_split_pause_seconds": 2.0,
        "max_chunk_seconds": 30.0,
    }

    baseline_segments, baseline_events = _run_segmenter(audio, 7777, **kwargs)
    segments, events = _run_segmenter(audio, 32000, **kwargs)

    assert segments == baseline_segments
    assert events == baseline_events
    assert segments
    for prev, cur in zip(segments, segments[1:]):
        assert prev.end <= cur.start
    for segment in segments:
        assert segment.end - segment.start <= kwargs["max_chunk_seconds"] + 1e-3


def test_real_mp3_hard_split_preserves_transcribable_coverage():
    audio = _primary_mp3_audio()

    segments_large, events_large = _run_segmenter(
        audio,
        32000,
        hop_seconds=2.0,
        overlap_seconds=0.25,
        hard_split_pause_seconds=120.0,
        max_chunk_seconds=100.0,
    )

    segments_small, events_small = _run_segmenter(
        audio,
        32000,
        hop_seconds=2.0,
        overlap_seconds=0.25,
        hard_split_pause_seconds=120.0,
        max_chunk_seconds=1.0,
    )

    def transcribable_duration(events):
        return sum(e.end - e.start for e in events if e.is_transcribable)

    dur_large = transcribable_duration(events_large)
    dur_small = transcribable_duration(events_small)

    assert segments_large
    assert segments_small
    for segment in segments_small:
        assert segment.end - segment.start <= 1.001
    assert dur_small == pytest.approx(dur_large, abs=0.75)
    assert dur_small / dur_large > 0.99
