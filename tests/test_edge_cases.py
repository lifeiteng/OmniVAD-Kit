"""Edge-case tests for OmniVAD with tiny/empty audio inputs."""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from omnivad import OmniAED, OmniStreamVAD, OmniVAD
from omnivad._binding import (
    OMNI_ERR_INVALID_ARG,
    OMNI_ERR_LOAD_BUNDLE,
    OMNI_ERR_NO_FRAMES,
    OmniStreamVadResult,
    _lib,
    default_model_dir,
)


@pytest.fixture(scope="module")
def vad():
    return OmniVAD()


@pytest.fixture(scope="module")
def aed():
    return OmniAED()


@pytest.fixture(scope="module")
def svad():
    return OmniStreamVAD()


# --- VAD edge cases ---


@pytest.mark.parametrize("n_samples", [0, 1, 2, 10, 100, 159, 399])
def test_vad_tiny_audio(vad, n_samples):
    """Audio shorter than one frame (400 samples = 25ms) should raise."""
    audio = np.zeros(n_samples, dtype=np.float32)
    with pytest.raises(RuntimeError):
        vad.detect(audio)


@pytest.mark.parametrize("n_samples", [400, 500])
def test_vad_one_frame_ncnn_fail(vad, n_samples):
    """1 fbank frame is too few for DFSMN model — ncnn inference fails."""
    audio = np.zeros(n_samples, dtype=np.float32)
    with pytest.raises(RuntimeError, match="ncnn inference failed"):
        vad.detect(audio)


def test_vad_two_frames(vad):
    """600 samples (2 fbank frames) — minimum valid input for inference."""
    audio = np.zeros(600, dtype=np.float32)
    result = vad.detect(audio)
    assert result["duration"] > 0
    assert isinstance(result["timestamps"], list)


def test_vad_silence(vad):
    """1 second of silence — should return no speech."""
    audio = np.zeros(16000, dtype=np.float32)
    result = vad.detect(audio)
    assert result["duration"] == pytest.approx(1.0, abs=0.001)
    assert len(result["timestamps"]) == 0


def test_vad_raw_tiny(vad):
    """detect_probs on sub-frame audio should raise."""
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(RuntimeError):
        vad.detect_probs(audio)


def test_vad_raw_two_frames(vad):
    """detect_probs on 2+ frames should return probabilities."""
    audio = np.zeros(600, dtype=np.float32)
    probs = vad.detect_probs(audio)
    assert len(probs) >= 1
    assert probs.dtype == np.float32


def test_vad_rejects_non_16k_array(vad):
    """Array inputs must explicitly be 16kHz."""
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 16kHz audio"):
        vad.detect(audio, sample_rate=8000)


@pytest.mark.parametrize(
    ("chunk_seconds", "overlap_seconds", "workers"),
    [
        (5, 5, 1),
        (5, 6, 1),
        (5, -1, 1),
        (-1, 0, 1),
        (5, 1, 0),
    ],
)
def test_vad_rejects_invalid_chunk_config(vad, chunk_seconds, overlap_seconds, workers):
    """Chunking parameters must always produce forward progress."""
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError):
        vad.detect(
            audio,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            workers=workers,
        )


# --- AED edge cases ---


@pytest.mark.parametrize("n_samples", [0, 1, 2, 10, 100, 159, 399])
def test_aed_tiny_audio(aed, n_samples):
    """Audio shorter than one frame should raise."""
    audio = np.zeros(n_samples, dtype=np.float32)
    with pytest.raises(RuntimeError):
        aed.detect(audio)


@pytest.mark.parametrize("n_samples", [400, 500])
def test_aed_one_frame_ncnn_fail(aed, n_samples):
    """1 fbank frame is too few for DFSMN model — ncnn inference fails."""
    audio = np.zeros(n_samples, dtype=np.float32)
    with pytest.raises(RuntimeError, match="ncnn inference failed"):
        aed.detect(audio)


def test_aed_two_frames(aed):
    """600 samples (2 fbank frames) — minimum valid input for inference."""
    audio = np.zeros(600, dtype=np.float32)
    result = aed.detect(audio)
    assert result["duration"] > 0
    for cls in ("speech", "singing", "music"):
        assert isinstance(result["events"][cls], list)


def test_aed_silence(aed):
    """1 second of silence — should return no events."""
    audio = np.zeros(16000, dtype=np.float32)
    result = aed.detect(audio)
    assert result["duration"] == pytest.approx(1.0, abs=0.001)
    for cls in ("speech", "singing", "music"):
        assert len(result["events"][cls]) == 0


def test_aed_raw_tiny(aed):
    """detect_probs on sub-frame audio should raise."""
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(RuntimeError):
        aed.detect_probs(audio)


def test_aed_raw_two_frames(aed):
    """detect_probs on 2+ frames should return (N, 3) array."""
    audio = np.zeros(600, dtype=np.float32)
    probs = aed.detect_probs(audio)
    assert probs.ndim == 2
    assert probs.shape[1] == 3
    assert probs.dtype == np.float32


def test_aed_rejects_non_16k_array(aed):
    """Array inputs must explicitly be 16kHz."""
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 16kHz audio"):
        aed.detect(audio, sample_rate=8000)


@pytest.mark.parametrize(
    ("chunk_seconds", "overlap_seconds", "workers"),
    [
        (5, 5, 1),
        (5, 6, 1),
        (5, -1, 1),
        (-1, 0, 1),
        (5, 1, 0),
    ],
)
def test_aed_rejects_invalid_chunk_config(aed, chunk_seconds, overlap_seconds, workers):
    """Chunking parameters must always produce forward progress."""
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError):
        aed.detect(
            audio,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            workers=workers,
        )


# --- StreamVAD edge cases ---


def test_stream_vad_tiny_chunk(svad):
    """Chunks smaller than one analysis frame keep buffering and return None."""
    svad.reset()
    for n in (0, 1, 10, 100, 159):
        chunk = np.zeros(n, dtype=np.int16)
        result = svad.process(chunk)
        assert result is None


def test_stream_vad_exact_chunk(svad):
    """Exactly 160 samples (10ms) — may still return None initially."""
    svad.reset()
    chunk = np.zeros(160, dtype=np.int16)
    # First few frames return None while model accumulates context
    result = svad.process(chunk)
    # After enough frames, should return a result
    for _ in range(10):
        result = svad.process(chunk)
    assert result is not None
    assert hasattr(result, "confidence")
    assert hasattr(result, "is_speech")


def test_native_create_reports_missing_bundle():
    """Create should surface detailed load failures via out_error."""
    err = ctypes.c_int(0)
    handle = _lib.omni_vad_create(b"/tmp/omnivad-missing.omnivad", ctypes.byref(err))
    assert not handle
    assert err.value == OMNI_ERR_LOAD_BUNDLE


def test_native_stream_create_rejects_invalid_threshold():
    """Stream create should reject thresholds outside [0, 1]."""

    model_path = Path(default_model_dir()) / "stream-vad.omnivad"
    err = ctypes.c_int(0)
    cfg = _lib.omni_stream_vad_config_default()
    cfg.threshold = 1.5
    handle = _lib.omni_stream_vad_create(str(model_path).encode("utf-8"), ctypes.byref(cfg), ctypes.byref(err))
    assert not handle
    assert err.value == OMNI_ERR_INVALID_ARG


def test_native_stream_process_rejects_negative_samples():
    """Negative sample counts should fail with INVALID_ARG before inference."""
    model_path = Path(default_model_dir()) / "stream-vad.omnivad"
    err = ctypes.c_int(0)
    handle = _lib.omni_stream_vad_create(str(model_path).encode("utf-8"), None, ctypes.byref(err))
    assert handle
    assert err.value == 0

    result = OmniStreamVadResult()
    sample = (ctypes.c_int16 * 1)(0)
    try:
        ret = _lib.omni_stream_vad_process_int16(handle, sample, -1, ctypes.byref(result))
        assert ret == OMNI_ERR_INVALID_ARG

        ret = _lib.omni_stream_vad_process_int16(handle, sample, 1, ctypes.byref(result))
        assert ret == OMNI_ERR_NO_FRAMES
    finally:
        _lib.omni_stream_vad_destroy(handle)


def test_stream_vad_detect_full_tiny(svad):
    """detect_full on sub-frame audio should raise."""
    audio = np.zeros(100, dtype=np.float32)
    with pytest.raises(RuntimeError):
        svad.detect_full(audio)


def test_stream_vad_detect_full_silence(svad):
    """detect_full on 1s silence should return frame probabilities."""
    audio = np.zeros(16000, dtype=np.float32)
    probs = svad.detect_full(audio)
    assert len(probs) > 0
    assert probs.dtype == np.float32


def test_stream_vad_detect_full_rejects_non_16k_array(svad):
    """Array inputs must explicitly be 16kHz."""
    audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 16kHz audio"):
        svad.detect_full(audio, sample_rate=8000)
