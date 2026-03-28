"""Streaming Voice Activity Detection."""

import ctypes
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from omnivad._binding import OMNI_ERR_NO_FRAMES, OmniVadStreamResult, _check, _lib, default_model_dir
from omnivad.vad import _load_audio


class StreamResult:
    """Result from processing one frame of streaming VAD."""

    __slots__ = ("confidence", "is_speech", "frame_offset", "time")

    def __init__(self, result: OmniVadStreamResult):
        self.confidence: float = result.confidence
        self.is_speech: bool = result.is_speech
        self.frame_offset: int = result.frame_offset
        self.time: float = result.frame_offset * 0.01

    def __repr__(self):
        return f"StreamResult(time={self.time:.3f}s, confidence={self.confidence:.3f}, is_speech={self.is_speech})"


class OmniStreamVAD:
    """Streaming Voice Activity Detection.

    Processes audio frame-by-frame (10ms chunks of 160 samples @ 16kHz).

    Parameters
    ----------
    model_path : str, optional
        Path to .omnivad bundle file. Uses bundled model if not specified.
    threshold : float
        Speech activation threshold (default: 0.5).
    """

    def __init__(self, model_path: str = None, *, threshold: float = 0.5):
        if model_path is None:
            model_path = os.path.join(default_model_dir(), "stream-vad.omnivad")

        self._handle = _lib.omni_vad_stream_create_from_bundle(model_path.encode("utf-8"), threshold)
        if not self._handle:
            raise RuntimeError(f"Failed to load stream VAD model from {model_path}")

    def process(self, pcm_chunk: np.ndarray) -> Optional[StreamResult]:
        """Process one audio chunk (160 int16 samples = 10ms @ 16kHz).

        Accepts int16 PCM or float32 (converted to int16 internally).
        Returns None until enough audio is accumulated (~25ms).
        """
        chunk = np.asarray(pcm_chunk)
        if chunk.dtype == np.float32:
            chunk = (chunk * 32768.0).clip(-32768, 32767).astype(np.int16)
        chunk = np.ascontiguousarray(chunk, dtype=np.int16)

        result = OmniVadStreamResult()
        ret = _lib.omni_vad_stream_process(
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
        """Process entire audio with streaming model (batch mode).

        Returns raw per-frame speech probabilities as numpy array.
        """
        data, fmt = _load_audio(audio, sample_rate)
        # stream_detect_full only has float (int16-range) C API
        if fmt == "i16":
            data = np.ascontiguousarray(data, dtype=np.float32)
        elif fmt == "f32":
            data = np.ascontiguousarray(data * 32768.0, dtype=np.float32)

        probs_ptr = ctypes.POINTER(ctypes.c_float)()
        num_frames = ctypes.c_int(0)

        _check(
            _lib.omni_vad_stream_detect_full(
                self._handle,
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(data),
                ctypes.byref(probs_ptr),
                ctypes.byref(num_frames),
            )
        )

        result = np.ctypeslib.as_array(probs_ptr, shape=(num_frames.value,)).copy()
        _lib.omni_free(probs_ptr)
        return result

    @property
    def frame_offset(self) -> int:
        """Current frame offset (each frame = 10ms)."""
        return _lib.omni_vad_stream_get_frame_offset(self._handle)

    def reset(self):
        """Reset all internal state (cache, audio buffer, frame offset)."""
        _lib.omni_vad_stream_reset(self._handle)

    def close(self):
        if self._handle:
            _lib.omni_vad_stream_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
