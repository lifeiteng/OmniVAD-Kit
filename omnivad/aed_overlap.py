"""AED overlap segmenter bindings."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Union

import numpy as np

from omnivad._binding import (
    OmniAedOnlineEvent,
    OmniAedOnlineSegment,
    OmniAedOverlapConfig,
    _check,
    _lib,
    default_model_dir,
)

AED_EVENT_KINDS = {
    0: "silence",
    1: "speech",
    2: "singing",
    3: "music",
    4: "mixed",
}

AED_KIND_MASK_SPEECH = 1 << 0
AED_KIND_MASK_SINGING = 1 << 1
AED_KIND_MASK_MUSIC = 1 << 2
AED_KIND_MASK_TRANSCRIBABLE = AED_KIND_MASK_SPEECH | AED_KIND_MASK_SINGING


@dataclass(frozen=True)
class AedOnlineEvent:
    """A committed AED overlap event."""

    start: float
    end: float
    primary_kind: str
    kind_mask: int
    speech_confidence: float
    singing_confidence: float
    music_confidence: float
    confidence: float

    @property
    def is_transcribable(self) -> bool:
        """Return True when the event contains speech or singing."""
        return bool(self.kind_mask & AED_KIND_MASK_TRANSCRIBABLE)


@dataclass(frozen=True)
class AedOnlineSegment:
    """A committed transcribable segment."""

    start: float
    end: float
    event_start_idx: int
    event_count: int


@dataclass(frozen=True)
class AedOverlapResult:
    """Output from one ingest or flush call."""

    segments: list[AedOnlineSegment]
    events: list[AedOnlineEvent]


class AedOverlapSegmenter:
    """Pseudo-streaming AED segmenter backed by the native C implementation."""

    def __init__(
        self,
        model_path: str | None = None,
        *,
        hop_seconds: float = 2.0,
        overlap_seconds: float = 0.25,
        edge_guard_seconds: float = 0.0,
        hard_split_pause_seconds: float = 2.0,
        max_chunk_seconds: float = 60.0,
        hard_split_lookahead_seconds: float = 0.0,
        min_speech_seconds: float = 0.2,
        merge_gap_seconds: float = 0.2,
        music_gap_tolerance_seconds: float = 0.0,
        pad_start_seconds: float = 0.0,
        pad_end_seconds: float = 0.0,
        speech_threshold: float = 0.5,
        singing_threshold: float = 0.5,
        music_threshold: float = 0.5,
    ) -> None:
        self._handle = None
        if model_path is None:
            model_path = os.path.join(default_model_dir(), "aed.omnivad")

        cfg = OmniAedOverlapConfig(
            hop_ms=_seconds_to_ms(hop_seconds),
            overlap_ms=_seconds_to_ms(overlap_seconds),
            edge_guard_ms=_seconds_to_ms(edge_guard_seconds),
            hard_split_pause_ms=_seconds_to_ms(hard_split_pause_seconds),
            max_chunk_ms=_seconds_to_ms(max_chunk_seconds),
            min_speech_ms=_seconds_to_ms(min_speech_seconds),
            merge_gap_ms=_seconds_to_ms(merge_gap_seconds),
            music_gap_tolerance_ms=_seconds_to_ms(music_gap_tolerance_seconds),
            pad_start_ms=_seconds_to_ms(pad_start_seconds),
            pad_end_ms=_seconds_to_ms(pad_end_seconds),
            speech_threshold=float(speech_threshold),
            singing_threshold=float(singing_threshold),
            music_threshold=float(music_threshold),
            hard_split_lookahead_ms=_seconds_to_ms(hard_split_lookahead_seconds),
        )

        err = ctypes.c_int(0)
        self._handle = _lib.omni_aed_overlap_segmenter_create(
            model_path.encode("utf-8"),
            ctypes.byref(cfg),
            ctypes.byref(err),
        )
        if not self._handle:
            msg = _lib.omni_error_string(err.value).decode()
            raise RuntimeError(f"Failed to create AED overlap segmenter from {model_path}: {msg} ({err.value})")

    def clone(self) -> "AedOverlapSegmenter":
        """Create a new segmenter with the same model and config but fresh state."""
        err = ctypes.c_int(0)
        handle = _lib.omni_aed_overlap_segmenter_clone(self._handle, ctypes.byref(err))
        if not handle:
            msg = _lib.omni_error_string(err.value).decode()
            raise RuntimeError(f"Failed to clone AED overlap segmenter: {msg} ({err.value})")
        cloned = self.__class__.__new__(self.__class__)
        cloned._handle = handle
        return cloned

    def ingest(self, audio: Union[np.ndarray, list[float], list[int]]) -> AedOverlapResult:
        """Ingest one PCM chunk and return newly committed output."""
        data = np.asarray(audio)
        if data.ndim != 1:
            raise ValueError("audio must be a 1D mono PCM array")
        segments_ptr = ctypes.POINTER(OmniAedOnlineSegment)()
        segment_count = ctypes.c_int(0)
        events_ptr = ctypes.POINTER(OmniAedOnlineEvent)()
        event_count = ctypes.c_int(0)

        if data.dtype == np.int16:
            contiguous = np.ascontiguousarray(data, dtype=np.int16)
            ptr = contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
            fn = _lib.omni_aed_overlap_segmenter_ingest_int16
        else:
            contiguous = np.ascontiguousarray(data, dtype=np.float32)
            ptr = contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            fn = _lib.omni_aed_overlap_segmenter_ingest

        _check(
            fn(
                self._handle,
                ptr,
                len(contiguous),
                ctypes.byref(segments_ptr),
                ctypes.byref(segment_count),
                ctypes.byref(events_ptr),
                ctypes.byref(event_count),
            )
        )
        return _collect_result(segments_ptr, segment_count.value, events_ptr, event_count.value)

    def flush(self) -> AedOverlapResult:
        """Finalize the stream and return any pending segments."""
        segments_ptr = ctypes.POINTER(OmniAedOnlineSegment)()
        segment_count = ctypes.c_int(0)
        events_ptr = ctypes.POINTER(OmniAedOnlineEvent)()
        event_count = ctypes.c_int(0)
        _check(
            _lib.omni_aed_overlap_segmenter_flush(
                self._handle,
                ctypes.byref(segments_ptr),
                ctypes.byref(segment_count),
                ctypes.byref(events_ptr),
                ctypes.byref(event_count),
            )
        )
        return _collect_result(segments_ptr, segment_count.value, events_ptr, event_count.value)

    def reset(self) -> None:
        """Clear buffered audio and emitted state."""
        if self._handle:
            _lib.omni_aed_overlap_segmenter_reset(self._handle)

    def close(self) -> None:
        if self._handle:
            _lib.omni_aed_overlap_segmenter_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "AedOverlapSegmenter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _seconds_to_ms(value: float) -> int:
    return int(round(float(value) * 1000.0))


def _collect_result(
    segments_ptr: ctypes.POINTER(OmniAedOnlineSegment),
    segment_count: int,
    events_ptr: ctypes.POINTER(OmniAedOnlineEvent),
    event_count: int,
) -> AedOverlapResult:
    try:
        events = []
        for i in range(event_count):
            ev = events_ptr[i]
            events.append(
                AedOnlineEvent(
                    start=round(float(ev.start), 3),
                    end=round(float(ev.end), 3),
                    primary_kind=AED_EVENT_KINDS.get(int(ev.primary_kind), "unknown"),
                    kind_mask=int(ev.kind_mask),
                    speech_confidence=float(ev.speech_confidence),
                    singing_confidence=float(ev.singing_confidence),
                    music_confidence=float(ev.music_confidence),
                    confidence=float(ev.confidence),
                )
            )

        segments = []
        for i in range(segment_count):
            seg = segments_ptr[i]
            segments.append(
                AedOnlineSegment(
                    start=round(float(seg.start), 3),
                    end=round(float(seg.end), 3),
                    event_start_idx=int(seg.event_start_idx),
                    event_count=int(seg.event_count),
                )
            )
        return AedOverlapResult(segments=segments, events=events)
    finally:
        if segments_ptr:
            _lib.omni_free(segments_ptr)
        if events_ptr:
            _lib.omni_free(events_ptr)
