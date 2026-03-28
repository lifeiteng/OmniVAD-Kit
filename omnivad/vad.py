"""Non-streaming Voice Activity Detection."""

import ctypes
import os
from pathlib import Path
from typing import Union

import numpy as np

from omnivad._binding import OmniPostConfig, OmniSegment, _check, _lib, default_model_dir


class OmniVAD:
    """Non-streaming Voice Activity Detection.

    Processes entire audio at once and returns speech segments.
    Uses FireRedVAD's DFSMN model (~2MB, ~588K params, 100+ languages).

    Parameters
    ----------
    model_path : str, optional
        Path to .omnivad bundle file. Uses bundled model if not specified.
    threshold : float
        Speech activation threshold (default: 0.4).
    smooth_window_size : int
        Causal moving-average smoothing window in frames (default: 5).
    min_speech_frames : int
        Minimum consecutive speech frames to confirm (default: 20, = 200ms).
    max_speech_frames : int
        Force-split segments longer than this (default: 2000, = 20s).
    min_silence_frames : int
        Minimum silence frames to end speech (default: 20, = 200ms).
    merge_silence_frames : int
        Merge silence gaps shorter than this (default: 0, disabled).
    extend_speech_frames : int
        Extend each speech segment by N frames on each side (default: 0).
    """

    def __init__(
        self,
        model_path: str = None,
        *,
        threshold: float = 0.4,
        smooth_window_size: int = 5,
        min_speech_frames: int = 20,
        max_speech_frames: int = 2000,
        min_silence_frames: int = 20,
        merge_silence_frames: int = 0,
        extend_speech_frames: int = 0,
    ):
        if model_path is None:
            model_path = os.path.join(default_model_dir(), "vad.omnivad")

        err = ctypes.c_int(0)
        self._handle = _lib.omni_vad_create(model_path.encode("utf-8"), ctypes.byref(err))
        if not self._handle:
            msg = _lib.omni_error_string(err.value).decode()
            raise RuntimeError(f"Failed to load VAD model from {model_path}: {msg} ({err.value})")

        self._config = OmniPostConfig(
            threshold=threshold,
            smooth_window_size=smooth_window_size,
            min_speech_frames=min_speech_frames,
            min_silence_frames=min_silence_frames,
            max_speech_frames=max_speech_frames,
            merge_silence_frames=merge_silence_frames,
            extend_speech_frames=extend_speech_frames,
        )

    def detect(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        chunk_seconds: float = 0,
        overlap_seconds: float = 2.0,
        workers: int = 1,
    ) -> dict:
        """Detect speech segments in audio.

        Parameters
        ----------
        audio : str, Path, or numpy.ndarray
            Audio file path, or float32 mono PCM array (16kHz).
        sample_rate : int
            Only used for validation when audio is an array.
        chunk_seconds : float
            Split audio into chunks of this length (seconds). 0 = no chunking.
        overlap_seconds : float
            Overlap between consecutive chunks (seconds). Default: 2.0.
        workers : int
            Number of parallel threads for chunked processing (default: 1).
            The C library is thread-safe (each call creates its own Extractor).

        Returns
        -------
        dict
            ``{'duration': float, 'timestamps': [(start, end), ...]}``
        """
        _validate_chunking(chunk_seconds, overlap_seconds, workers)
        data, fmt = _load_audio(audio, sample_rate)
        duration = round(len(data) / 16000.0, 3)

        if chunk_seconds > 0 and len(data) > int(chunk_seconds * 16000):
            return self._detect_chunked(data, fmt, duration, chunk_seconds, overlap_seconds, workers)

        return {"duration": duration, "timestamps": self._detect_array(data, fmt)}

    def _detect_array(self, data: np.ndarray, fmt: str = "f32") -> list:
        """Run detection on a single audio array. Returns [(start, end), ...]."""
        segments_ptr = ctypes.POINTER(OmniSegment)()
        count = ctypes.c_int(0)

        if fmt == "int16":
            fn = _lib.omni_vad_detect_int16
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        else:
            fn = _lib.omni_vad_detect
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        _check(
            fn(
                self._handle,
                ptr,
                len(data),
                ctypes.byref(self._config),
                ctypes.byref(segments_ptr),
                ctypes.byref(count),
            )
        )

        timestamps = [(round(segments_ptr[i].start, 3), round(segments_ptr[i].end, 3)) for i in range(count.value)]

        if segments_ptr:
            _lib.omni_free(segments_ptr)

        return timestamps

    def _detect_chunked(self, data, fmt, duration, chunk_seconds, overlap_seconds, workers=1):
        """Process large audio in overlapping chunks with optional parallelism."""
        from omnivad._chunked import aggregate_segments, split_chunks

        chunk_samples = int(chunk_seconds * 16000)
        overlap_samples = int(overlap_seconds * 16000)
        chunks = split_chunks(len(data), chunk_samples, overlap_samples)

        def _process(chunk_range):
            start, end = chunk_range
            return (start / 16000.0, self._detect_array(data[start:end], fmt))

        if workers > 1 and len(chunks) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(workers, len(chunks))) as pool:
                chunk_results = list(pool.map(_process, chunks))
        else:
            chunk_results = [_process(c) for c in chunks]

        timestamps = aggregate_segments(chunk_results, overlap_seconds)
        return {"duration": duration, "timestamps": timestamps}

    def detect_probs(self, audio: Union[str, Path, np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        """Get per-frame speech probabilities.

        Returns
        -------
        numpy.ndarray
            Float32 array of per-frame speech probabilities.
        """
        data, fmt = _load_audio(audio, sample_rate)

        probs_ptr = ctypes.POINTER(ctypes.c_float)()
        num_frames = ctypes.c_int(0)

        if fmt == "int16":
            fn = _lib.omni_vad_detect_probs_int16
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        else:
            fn = _lib.omni_vad_detect_probs
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        _check(fn(self._handle, ptr, len(data), ctypes.byref(probs_ptr), ctypes.byref(num_frames)))

        result = np.ctypeslib.as_array(probs_ptr, shape=(num_frames.value,)).copy()
        _lib.omni_free(probs_ptr)
        return result

    def close(self):
        if self._handle:
            _lib.omni_vad_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _load_audio(audio: Union[str, Path, np.ndarray], sample_rate: int = 16000) -> tuple:
    """Load audio and detect format.

    Returns (data, format) where format is:
      "float32" — float32 in [-1.0, 1.0] (from soundfile, Web Audio, torch)
      "int16"   — int16 PCM (from raw WAV, microphone)

    Two formats only. C library handles conversion internally.
    """
    if isinstance(audio, (str, Path)):
        import soundfile as sf

        data, sr = sf.read(str(audio), dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sr}Hz. Please resample first.")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.ascontiguousarray(data, dtype=np.float32), "f32"

    _validate_array_sample_rate(sample_rate)
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return np.ascontiguousarray(audio, dtype=np.int16), "int16"
    # Any float input treated as [-1,1] normalized
    return np.ascontiguousarray(audio, dtype=np.float32), "f32"


def _validate_array_sample_rate(sample_rate: int) -> None:
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz. Please resample first.")


def _validate_chunking(chunk_seconds: float, overlap_seconds: float, workers: int) -> None:
    if chunk_seconds < 0:
        raise ValueError("chunk_seconds must be >= 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if chunk_seconds > 0 and overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")
