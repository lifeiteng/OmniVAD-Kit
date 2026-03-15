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

        self._handle = _lib.omni_vad_nonstream_create_from_bundle(model_path.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"Failed to load VAD model from {model_path}")

        self._config = OmniPostConfig(
            threshold=threshold,
            smooth_window_size=smooth_window_size,
            min_speech_frames=min_speech_frames,
            min_silence_frames=min_silence_frames,
            max_speech_frames=max_speech_frames,
            merge_silence_frames=merge_silence_frames,
            extend_speech_frames=extend_speech_frames,
        )

    def detect(self, audio: Union[str, Path, np.ndarray], sample_rate: int = 16000) -> dict:
        """Detect speech segments in audio.

        Parameters
        ----------
        audio : str, Path, or numpy.ndarray
            Audio file path, or float32 mono PCM array (16kHz).
        sample_rate : int
            Only used for validation when audio is an array.

        Returns
        -------
        dict
            ``{'duration': float, 'timestamps': [(start, end), ...]}``
        """
        data = _load_audio(audio, sample_rate)

        segments_ptr = ctypes.POINTER(OmniSegment)()
        count = ctypes.c_int(0)

        _check(
            _lib.omni_vad_nonstream_process(
                self._handle,
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                len(data),
                ctypes.byref(self._config),
                ctypes.byref(segments_ptr),
                ctypes.byref(count),
            )
        )

        timestamps = [(round(segments_ptr[i].start, 3), round(segments_ptr[i].end, 3)) for i in range(count.value)]

        if segments_ptr:
            _lib.omni_free(segments_ptr)

        return {"duration": round(len(data) / 16000.0, 3), "timestamps": timestamps}

    def detect_raw(self, audio: Union[str, Path, np.ndarray], sample_rate: int = 16000) -> np.ndarray:
        """Get raw frame-level speech probabilities.

        Returns
        -------
        numpy.ndarray
            Float32 array of per-frame speech probabilities.
        """
        data = _load_audio(audio, sample_rate)

        probs_ptr = ctypes.POINTER(ctypes.c_float)()
        num_frames = ctypes.c_int(0)

        _check(
            _lib.omni_vad_nonstream_process_raw(
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

    def close(self):
        if self._handle:
            _lib.omni_vad_nonstream_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _load_audio(audio: Union[str, Path, np.ndarray], sample_rate: int = 16000) -> np.ndarray:
    """Load audio from file path or validate numpy array.

    The native C library expects float samples in int16 range [-32768, 32767],
    matching the WavReader which casts int16 samples directly to float without
    normalization. soundfile returns normalized [-1, 1] floats, so we scale up.
    """
    if isinstance(audio, (str, Path)):
        import soundfile as sf

        data, sr = sf.read(str(audio), dtype="float32")
        if sr != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sr}Hz. Please resample first.")
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Scale from [-1, 1] to int16 range to match native WavReader
        data = data * 32768.0
        return np.ascontiguousarray(data, dtype=np.float32)
    # Assume raw arrays are already in int16 range (matching native convention)
    return np.ascontiguousarray(audio, dtype=np.float32)
