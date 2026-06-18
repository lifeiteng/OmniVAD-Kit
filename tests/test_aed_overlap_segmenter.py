"""Tests for the native AED overlap segmenter binding."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from omnivad import AedOverlapSegmenter
from omnivad._binding import OmniAedOnlineEvent, OmniAedOnlineSegment, OmniAedOverlapConfig


def _load_fixture(name: str = "hello_en.wav") -> np.ndarray:
    audio, sr = sf.read(Path("tests/data") / name, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16kHz fixture, got {sr}Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _run_segmenter(
    audio: np.ndarray,
    step: int,
    *,
    hop_seconds: float = 0.5,
    overlap_seconds: float = 0.1,
    hard_split_pause_seconds: float = 0.2,
    max_chunk_seconds: float = 2.0,
):
    segmenter = AedOverlapSegmenter(
        hop_seconds=hop_seconds,
        overlap_seconds=overlap_seconds,
        hard_split_pause_seconds=hard_split_pause_seconds,
        max_chunk_seconds=max_chunk_seconds,
    )
    try:
        segments = []
        events = []
        for start in range(0, len(audio), step):
            result = segmenter.ingest(audio[start : start + step])
            event_offset = len(events)
            segments.extend(
                replace(segment, event_start_idx=segment.event_start_idx + event_offset)
                for segment in result.segments
            )
            events.extend(result.events)
        result = segmenter.flush()
        event_offset = len(events)
        segments.extend(
            replace(segment, event_start_idx=segment.event_start_idx + event_offset)
            for segment in result.segments
        )
        events.extend(result.events)
        return segments, events
    finally:
        segmenter.close()


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
