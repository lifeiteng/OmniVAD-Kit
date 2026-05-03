"""Streaming Voice Activity Detection.

Wraps the C ``omni_stream_vad_*`` API. Bit-identical to upstream
FireRedStreamVad — process_one() returns per-frame events including
``is_speech_start`` / ``is_speech_end`` and the corresponding frame
indices, so a separate segmenter is not needed.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from omnivad._binding import (
    OMNI_ERR_NO_FRAMES,
    OmniStreamVadConfig,
    OmniStreamVadResult,
    _check,
    _lib,
    default_model_dir,
)
from omnivad.vad import _load_audio

_FRAME_PER_SECOND = 100  # 10ms hop


class StreamResult:
    """Per-frame result from streaming VAD.

    Attributes
    ----------
    confidence : float
        Raw model probability [0, 1].
    smoothed_prob : float
        Causal moving-average of confidence (window = smooth_window_size).
    is_speech : bool
        ``smoothed_prob >= threshold``.
    is_speech_start : bool
        True on the frame that confirms a new SPEECH segment.
    is_speech_end : bool
        True on the frame that confirms a SPEECH segment end.
    frame_idx : int
        1-based frame index of this frame (multiply by 0.01 for seconds).
    speech_start_frame : int
        1-based start frame of the segment when ``is_speech_start``; else -1.
    speech_end_frame : int
        1-based end frame of the segment when ``is_speech_end``; else -1.
    time : float
        Convenience: ``frame_idx * 0.01`` (seconds since stream start).
    """

    __slots__ = (
        "confidence",
        "smoothed_prob",
        "is_speech",
        "is_speech_start",
        "is_speech_end",
        "frame_idx",
        "speech_start_frame",
        "speech_end_frame",
        "time",
    )

    def __init__(self, result: OmniStreamVadResult):
        self.confidence: float = result.confidence
        self.smoothed_prob: float = result.smoothed_prob
        self.is_speech: bool = result.is_speech
        self.is_speech_start: bool = result.is_speech_start
        self.is_speech_end: bool = result.is_speech_end
        self.frame_idx: int = result.frame_idx
        self.speech_start_frame: int = result.speech_start_frame
        self.speech_end_frame: int = result.speech_end_frame
        self.time: float = result.frame_idx * 0.01

    def __repr__(self):
        events = []
        if self.is_speech_start:
            events.append(f"START@{self.speech_start_frame}")
        if self.is_speech_end:
            events.append(f"END@{self.speech_end_frame}")
        ev = " " + " ".join(events) if events else ""
        return (
            f"StreamResult(t={self.time:.3f}s, conf={self.confidence:.3f}, "
            f"smoothed={self.smoothed_prob:.3f}, speech={self.is_speech}{ev})"
        )


class OmniStreamVAD:
    """Streaming Voice Activity Detection — bit-identical to upstream FireRedStreamVad.

    Processes audio frame-by-frame (10ms chunks of 160 samples @ 16kHz)
    and emits per-frame ``StreamResult`` objects with both per-frame
    probabilities and segment-boundary events.

    Parameters
    ----------
    model_path : str, optional
        Path to .omnivad bundle file. Uses bundled model if not specified.
    threshold : float
        Speech activation threshold (default: 0.5).
    smooth_window_size : int
        Causal moving-average window in frames (default: 5).
    pad_start_frame : int
        Extend confirmed segment START backward by N frames (default: 5;
        clamped to >= smooth_window_size internally).
    min_speech_frame : int
        Minimum continuous speech frames to confirm START (default: 8 = 80ms).
    max_speech_frame : int
        Force-split when SPEECH-state count hits this (default: 2000 = 20s).
    min_silence_frame : int
        Minimum continuous silence frames to confirm END (default: 20 = 200ms).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        threshold: float = 0.5,
        smooth_window_size: int = 5,
        pad_start_frame: int = 5,
        min_speech_frame: int = 8,
        max_speech_frame: int = 2000,
        min_silence_frame: int = 20,
    ):
        self._handle = None  # set early so __del__ is safe if init raises

        if model_path is None:
            model_path = os.path.join(default_model_dir(), "stream-vad.omnivad")

        cfg = OmniStreamVadConfig(
            threshold=float(threshold),
            smooth_window_size=int(smooth_window_size),
            pad_start_frame=int(pad_start_frame),
            min_speech_frame=int(min_speech_frame),
            max_speech_frame=int(max_speech_frame),
            min_silence_frame=int(min_silence_frame),
        )

        err = ctypes.c_int(0)
        self._handle = _lib.omni_stream_vad_create(model_path.encode("utf-8"), ctypes.byref(cfg), ctypes.byref(err))
        if not self._handle:
            msg = _lib.omni_error_string(err.value).decode()
            raise RuntimeError(f"Failed to load stream VAD model from {model_path}: {msg} ({err.value})")

    def clone(self) -> "OmniStreamVAD":
        """Create a lightweight clone sharing model weights with fresh state."""
        if not self._handle:
            raise RuntimeError("Cannot clone a closed VAD instance.")
        err = ctypes.c_int(0)
        handle = _lib.omni_stream_vad_clone(self._handle, ctypes.byref(err))
        if not handle:
            msg = _lib.omni_error_string(err.value).decode()
            raise RuntimeError(f"Failed to clone stream VAD: {msg} ({err.value})")
        instance = type(self).__new__(type(self))
        instance._handle = handle
        return instance

    def process(self, pcm_chunk: np.ndarray) -> Optional[StreamResult]:
        """Process one audio chunk (160 int16 samples = 10ms @ 16kHz).

        Returns ``None`` until enough audio is accumulated (~25ms).
        Larger chunks emit only one frame per call (model output is
        single-frame); upstream callers should feed 160-sample chunks
        for 1:1 input-to-output frame mapping.
        """
        chunk = np.asarray(pcm_chunk)
        if chunk.dtype == np.float32:
            chunk = (chunk * 32768.0).clip(-32768, 32767).astype(np.int16)
        chunk = np.ascontiguousarray(chunk, dtype=np.int16)

        result = OmniStreamVadResult()
        ret = _lib.omni_stream_vad_process(
            self._handle,
            chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            len(chunk),
            ctypes.byref(result),
        )

        if ret == OMNI_ERR_NO_FRAMES:
            return None
        _check(ret)
        return StreamResult(result)

    def detect_full(self, audio: Union[str, Path, np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        """Process entire audio with the streaming model in batch mode.

        Returns raw per-frame speech probabilities as a numpy array.
        For segment-level output, use ``process()`` per frame and read
        the ``is_speech_start`` / ``is_speech_end`` events on the result.
        """
        data, fmt = _load_audio(audio, sample_rate)

        probs_ptr = ctypes.POINTER(ctypes.c_float)()
        num_frames = ctypes.c_int(0)

        if fmt == "int16":
            fn = _lib.omni_stream_vad_detect_full_int16
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        else:
            fn = _lib.omni_stream_vad_detect_full
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        _check(fn(self._handle, ptr, len(data), ctypes.byref(probs_ptr), ctypes.byref(num_frames)))

        result = np.ctypeslib.as_array(probs_ptr, shape=(num_frames.value,)).copy()
        _lib.omni_free(probs_ptr)
        return result

    def detect_segments(
        self, audio: Union[str, Path, np.ndarray], sample_rate: int = 16000
    ) -> List[Tuple[float, float]]:
        """Process entire audio and return ``[(start_sec, end_sec), ...]``.

        Equivalent to streaming the audio frame-by-frame through ``process()``
        and collecting all is_speech_start / is_speech_end events into segments.
        Open trailing segments are closed at the last processed frame.
        """
        data, fmt = _load_audio(audio, sample_rate)
        if fmt == "int16":
            pcm = data
        else:
            pcm = (data * 32768.0).clip(-32768, 32767).astype(np.int16)

        self.reset()
        chunk = 160
        starts: List[int] = []
        ends: List[int] = []
        last_frame = 0
        for offset in range(0, len(pcm) - chunk + 1, chunk):
            res = self.process(pcm[offset : offset + chunk])
            if res is None:
                continue
            last_frame = res.frame_idx
            if res.is_speech_start:
                starts.append(res.speech_start_frame)
            if res.is_speech_end:
                ends.append(res.speech_end_frame)

        # If trailing segment is open, close at last_frame.
        if len(starts) == len(ends) + 1 and last_frame > 0:
            ends.append(last_frame)
        return [(s / _FRAME_PER_SECOND, e / _FRAME_PER_SECOND) for s, e in zip(starts, ends)]

    @property
    def frame_offset(self) -> int:
        """Number of frames processed so far (multiply by 0.01s for elapsed audio)."""
        return _lib.omni_stream_vad_get_frame_offset(self._handle)

    def reset(self):
        """Reset all internal state (model cache, audio buffer, postprocessor)."""
        _lib.omni_stream_vad_reset(self._handle)

    def close(self):
        if getattr(self, "_handle", None):
            _lib.omni_stream_vad_destroy(self._handle)
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
