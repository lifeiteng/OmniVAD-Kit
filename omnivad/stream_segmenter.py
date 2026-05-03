"""Streaming VAD Segmenter — converts per-frame probabilities into segments online.

Pure-algorithm wrapper around the C ``omni_stream_segmenter_*`` family.
No model dependency — pair it with :class:`OmniStreamVAD` (or any prob source).

Phase 1 limitations:
  - merge_silence_frames must be 0 (causal smoothing only)
  - extend_speech_frames must be 0 (no lookahead extension)
  - max_speech_frames > 0 supported (force-split via min-prob search)

Typical usage::

    from omnivad import OmniStreamVAD, OmniStreamSegmenter

    vad = OmniStreamVAD()
    segmenter = OmniStreamSegmenter()
    total_samples = 0

    for chunk in audio_chunks:                 # 160-sample int16 chunks
        result = vad.process(chunk)
        total_samples += len(chunk)
        if result is None:
            continue
        for start, end in segmenter.process_frame(result.confidence):
            print(f"speech: {start:.2f}s -> {end:.2f}s")

    # End-of-stream: emit any in-progress segment
    for start, end in segmenter.flush(total_samples):
        print(f"speech (tail): {start:.2f}s -> {end:.2f}s")
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Tuple

import numpy as np

from omnivad._binding import OmniPostConfig, OmniSegment, _check, _lib


class OmniStreamSegmenter:
    """Streaming VAD segmenter — pure state machine over per-frame probabilities.

    Parameter naming aligns with :func:`merge_chunks` (the segment-packing
    utility) so the same concept uses the same name across both APIs:

    +---------------------+------------------------------+
    | OmniStreamSegmenter | merge_chunks (Chunker)       |
    +=====================+==============================+
    | min_speech_secs     | min_speech_secs              |
    | min_silence_secs    | min_silence_secs             |
    | max_chunk_secs      | max_chunk_secs               |
    +---------------------+------------------------------+

    Internally the C state machine still operates in 10ms frames; this
    wrapper does the seconds-to-frames conversion via ``round(secs / 0.01)``.

    Parameters
    ----------
    threshold : float
        Speech activation threshold (default: 0.4).
    smooth_window_size : int
        Causal moving-average window in frames (default: 5). Stays in frame
        units because it's an internal smoothing-kernel size, not a
        time-domain duration.
    min_speech_secs : float
        Minimum continuous speech duration to confirm START (default: 0.20).
    min_silence_secs : float
        Minimum continuous silence duration to emit END (default: 0.20).
        Equivalent to ``merge_chunks``'s ``min_silence_secs`` — gaps shorter
        than this don't terminate a segment.
    max_chunk_secs : float
        Force-split active segments longer than this at the
        lowest-probability point in the second half of the window
        (default: 30.0; matches Whisper input window). Set to 0 to disable.
        Equivalent to ``merge_chunks``'s ``max_chunk_secs``.

    Note: equivalents of ``merge_chunks``'s ``min_duration_off`` /
    ``pad_onset_secs`` / ``pad_offset_secs`` (i.e. OmniPostConfig's
    ``merge_silence_frames`` / ``extend_speech_frames``) are NOT supported
    here because they would require unbounded lookahead.
    """

    # Frame shift = 10ms; used for secs <-> frames conversion.
    _FRAME_SHIFT_SEC: float = 0.01

    def __init__(
        self,
        *,
        threshold: float = 0.4,
        smooth_window_size: int = 5,
        min_speech_secs: float = 0.20,
        min_silence_secs: float = 0.20,
        max_chunk_secs: float = 30.0,
    ):
        # set early so __del__ doesn't AttributeError if __init__ raises later
        self._handle = None

        def _to_frames(secs: float) -> int:
            return int(round(float(secs) / self._FRAME_SHIFT_SEC))

        cfg = OmniPostConfig()
        cfg.threshold = float(threshold)
        cfg.smooth_window_size = int(smooth_window_size)
        cfg.min_speech_frames = _to_frames(min_speech_secs)
        cfg.min_silence_frames = _to_frames(min_silence_secs)
        cfg.max_speech_frames = _to_frames(max_chunk_secs)
        cfg.merge_silence_frames = 0  # not supported in streaming
        cfg.extend_speech_frames = 0  # not supported in streaming

        err = ctypes.c_int(0)
        self._handle = _lib.omni_stream_segmenter_create(ctypes.byref(cfg), ctypes.byref(err))
        if not self._handle:
            msg = _lib.omni_error_string(err.value).decode()
            raise RuntimeError(f"Failed to create stream segmenter: {msg} ({err.value})")

    # --------------------------------------------------------------------- #
    #  Algorithm                                                             #
    # --------------------------------------------------------------------- #

    def process_frame(self, prob: float) -> List[Tuple[float, float]]:
        """Push one frame's raw probability. Returns 0+ completed segments."""
        if not self._handle:
            raise RuntimeError("OmniStreamSegmenter has been closed.")
        out_ptr = ctypes.POINTER(OmniSegment)()
        out_count = ctypes.c_int(0)
        rc = _lib.omni_stream_segmenter_process_frame(
            self._handle, float(prob), ctypes.byref(out_ptr), ctypes.byref(out_count)
        )
        _check(rc)
        return self._collect(out_ptr, out_count.value)

    def process_probs(self, probs) -> List[Tuple[float, float]]:
        """Push a batch of probabilities. Accepts numpy array or sequence of float."""
        if not self._handle:
            raise RuntimeError("OmniStreamSegmenter has been closed.")
        arr = np.ascontiguousarray(probs, dtype=np.float32)
        out_ptr = ctypes.POINTER(OmniSegment)()
        out_count = ctypes.c_int(0)
        rc = _lib.omni_stream_segmenter_process_probs(
            self._handle,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(arr),
            ctypes.byref(out_ptr),
            ctypes.byref(out_count),
        )
        _check(rc)
        return self._collect(out_ptr, out_count.value)

    def flush(self, total_samples_seen: int = 0) -> List[Tuple[float, float]]:
        """Flush in-progress segment (if any) at end-of-stream.

        Pass ``total_samples_seen`` (sum of ``len(chunk)`` across all
        ``process_frame`` calls upstream) to clamp the tail-segment end-time
        to the actual audio duration. Pass 0 to skip clamping (uses frame-rate
        end-time only: ``frames * 0.01 + 0.025``).
        """
        if not self._handle:
            raise RuntimeError("OmniStreamSegmenter has been closed.")
        out_ptr = ctypes.POINTER(OmniSegment)()
        out_count = ctypes.c_int(0)
        rc = _lib.omni_stream_segmenter_flush(
            self._handle,
            int(total_samples_seen),
            ctypes.byref(out_ptr),
            ctypes.byref(out_count),
        )
        _check(rc)
        return self._collect(out_ptr, out_count.value)

    # --------------------------------------------------------------------- #
    #  State queries                                                         #
    # --------------------------------------------------------------------- #

    @property
    def is_in_speech(self) -> bool:
        """Whether the segmenter is currently inside a confirmed speech segment."""
        if not self._handle:
            return False
        return bool(_lib.omni_stream_segmenter_is_in_speech(self._handle))

    @property
    def active_start(self) -> Optional[float]:
        """If in confirmed speech, return the start time (seconds) of the
        active segment; ``None`` otherwise."""
        if not self._handle:
            return None
        v = float(_lib.omni_stream_segmenter_get_active_start(self._handle))
        return None if v < 0.0 else v

    # --------------------------------------------------------------------- #
    #  Lifecycle                                                             #
    # --------------------------------------------------------------------- #

    def reset(self) -> None:
        """Reset internal state to fresh (state machine, buffers, frame counter)."""
        if self._handle:
            _lib.omni_stream_segmenter_reset(self._handle)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.omni_stream_segmenter_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # --------------------------------------------------------------------- #
    #  Internals                                                             #
    # --------------------------------------------------------------------- #

    def _collect(self, out_ptr, count: int) -> List[Tuple[float, float]]:
        result: List[Tuple[float, float]] = []
        try:
            for i in range(count):
                seg = out_ptr[i]
                result.append((float(seg.start), float(seg.end)))
        finally:
            if out_ptr:
                _lib.omni_free(out_ptr)
        return result
