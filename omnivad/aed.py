"""Audio Event Detection (3-class: speech, singing, music)."""

import ctypes
import os
from pathlib import Path
from typing import Union

import numpy as np

from omnivad._binding import OmniAedPostConfig, OmniAedSegment, OmniPostConfig, _check, _lib, default_model_dir
from omnivad.vad import _load_audio, _validate_chunking

AED_CLASSES = {0: "speech", 1: "singing", 2: "music"}


class OmniAED:
    """Audio Event Detection: speech, singing, music classification.

    Processes entire audio and returns per-class event segments.
    Segments of different classes may overlap.

    Parameters
    ----------
    model_path : str, optional
        Path to .omnivad bundle file. Uses bundled model if not specified.
    speech_threshold : float
        Speech activation threshold (default: 0.4).
    singing_threshold : float
        Singing activation threshold (default: 0.5).
    music_threshold : float
        Music activation threshold (default: 0.5).
    smooth_window_size : int
        Smoothing window in frames (default: 5).
    min_speech_frames : int
        Min frames to confirm event (default: 20, = 200ms).
    max_speech_frames : int
        Force-split at this length (default: 3000, = 30s; matches Whisper).
    min_silence_frames : int
        Min silence to end event (default: 20, = 200ms).
    merge_silence_frames : int
        Merge silence gaps shorter than this (default: 0, disabled).
    extend_speech_frames : int
        Extend each segment by N frames on each side (default: 0).
    """

    def __init__(
        self,
        model_path: str = None,
        *,
        speech_threshold: float = 0.4,
        singing_threshold: float = 0.5,
        music_threshold: float = 0.5,
        smooth_window_size: int = 5,
        min_speech_frames: int = 20,
        max_speech_frames: int = 3000,
        min_silence_frames: int = 20,
        merge_silence_frames: int = 0,
        extend_speech_frames: int = 0,
    ):
        if model_path is None:
            model_path = os.path.join(default_model_dir(), "aed.omnivad")

        err = ctypes.c_int(0)
        self._handle = _lib.omni_aed_create(model_path.encode("utf-8"), ctypes.byref(err))
        if not self._handle:
            msg = _lib.omni_error_string(err.value).decode()
            raise RuntimeError(f"Failed to load AED model from {model_path}: {msg} ({err.value})")

        def _cfg(threshold):
            return OmniPostConfig(
                threshold=threshold,
                smooth_window_size=smooth_window_size,
                min_speech_frames=min_speech_frames,
                min_silence_frames=min_silence_frames,
                max_speech_frames=max_speech_frames,
                merge_silence_frames=merge_silence_frames,
                extend_speech_frames=extend_speech_frames,
            )

        self._config = OmniAedPostConfig(
            speech=_cfg(speech_threshold),
            singing=_cfg(singing_threshold),
            music=_cfg(music_threshold),
        )

    def detect(
        self,
        audio: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        chunk_seconds: float = 0,
        overlap_seconds: float = 2.0,
        workers: int = 1,
    ) -> dict:
        """Detect audio events (speech, singing, music).

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
            ``{'duration': float, 'events': {'speech': [...], 'singing': [...], 'music': [...]}}``
        """
        _validate_chunking(chunk_seconds, overlap_seconds, workers)
        data, fmt = _load_audio(audio, sample_rate)
        duration = round(len(data) / 16000.0, 3)

        if chunk_seconds > 0 and len(data) > int(chunk_seconds * 16000):
            return self._detect_chunked(data, fmt, duration, chunk_seconds, overlap_seconds, workers)

        return {"duration": duration, "events": self._detect_array(data, fmt)}

    def _detect_array(self, data: np.ndarray, fmt: str = "f32") -> dict:
        """Run detection on a single audio array. Returns events dict."""
        segments_ptr = ctypes.POINTER(OmniAedSegment)()
        count = ctypes.c_int(0)

        if fmt == "int16":
            fn = _lib.omni_aed_detect_int16
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        else:
            fn = _lib.omni_aed_detect
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

        events = {"speech": [], "singing": [], "music": []}
        for i in range(count.value):
            seg = segments_ptr[i]
            cls_name = AED_CLASSES.get(seg.cls, f"unknown_{seg.cls}")
            if cls_name in events:
                events[cls_name].append((round(seg.start, 3), round(seg.end, 3)))

        if segments_ptr:
            _lib.omni_free(segments_ptr)

        return events

    def _detect_chunked(self, data, fmt, duration, chunk_seconds, overlap_seconds, workers=1):
        """Process large audio in overlapping chunks with optional parallelism."""
        from omnivad._chunked import aggregate_aed_events, split_chunks

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

        events = aggregate_aed_events(chunk_results, overlap_seconds)
        return {"duration": duration, "events": events}

    def detect_probs(self, audio: Union[str, Path, np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        """Get per-frame probabilities for all 3 classes.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(num_frames, 3)`` — columns: speech, singing, music.
        """
        data, fmt = _load_audio(audio, sample_rate)

        probs_ptr = ctypes.POINTER(ctypes.c_float)()
        num_frames = ctypes.c_int(0)

        if fmt == "int16":
            fn = _lib.omni_aed_detect_probs_int16
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        else:
            fn = _lib.omni_aed_detect_probs
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        _check(fn(self._handle, ptr, len(data), ctypes.byref(probs_ptr), ctypes.byref(num_frames)))

        flat = np.ctypeslib.as_array(probs_ptr, shape=(num_frames.value * 3,)).copy()
        _lib.omni_free(probs_ptr)
        return flat.reshape(num_frames.value, 3)

    def close(self):
        if self._handle:
            _lib.omni_aed_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
