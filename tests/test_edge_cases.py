"""Edge-case tests for OmniVAD with tiny/empty audio inputs."""

import numpy as np
import pytest

from omnivad import OmniAED, OmniStreamVAD, OmniVAD


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


# --- StreamVAD edge cases ---


def test_stream_vad_tiny_chunk(svad):
    """Chunks smaller than 160 samples — returns zero-confidence result."""
    svad.reset()
    for n in (0, 1, 10, 100, 159):
        chunk = np.zeros(n, dtype=np.int16)
        result = svad.process(chunk)
        assert result is not None
        assert result.confidence == pytest.approx(0.0)
        assert result.is_speech is False


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
