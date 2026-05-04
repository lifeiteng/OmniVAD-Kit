"""Cross-format equivalence: FP32 [-1,1] and int16 paths must agree.

Verifies the design contract that wrappers do zero scaling — both audio
formats are passed straight through to their matching C entries, where
the only scaling happens. Catches the entire 0.2.x-style class of latent
bugs where a wrapper-side conversion silently saturated the model.

Tolerance: int16 quantization adds ~1/65536 noise per sample. After
fbank + CMVN + DFSMN this propagates to <1e-3 in per-frame probability;
segment timestamps usually agree to the frame.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from omnivad import OmniAED, OmniStreamVAD, OmniVAD

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WAVS = [
    "hello_en.wav",
    "hello_zh.wav",
    "en_medium.wav",
    "zh_medium.wav",
    "event.wav",
]

PROB_TOL = 5e-3  # absolute tolerance per-frame probability
TIME_TOL = 0.02  # 2 frame slack on segment edges


def load_pair(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (float32 [-1,1], int16) views of the same audio."""
    f32, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16kHz, got {sr}")
    if f32.ndim > 1:
        f32 = f32.mean(axis=1)
    f32 = np.ascontiguousarray(f32, dtype=np.float32)
    i16 = (f32 * 32768.0).clip(-32768, 32767).astype(np.int16)
    return f32, i16


@pytest.fixture(scope="module")
def vad() -> OmniVAD:
    return OmniVAD()


@pytest.fixture(scope="module")
def aed() -> OmniAED:
    return OmniAED()


@pytest.fixture(scope="module")
def svad() -> OmniStreamVAD:
    return OmniStreamVAD()


# --- VAD ---


@pytest.mark.parametrize("wav", WAVS)
def test_vad_detect_probs_equiv(vad: OmniVAD, wav: str) -> None:
    f32, i16 = load_pair(os.path.join(DATA_DIR, wav))
    p_f32 = vad.detect_probs(f32)
    p_i16 = vad.detect_probs(i16)
    assert p_f32.shape == p_i16.shape
    assert np.max(np.abs(p_f32 - p_i16)) < PROB_TOL


@pytest.mark.parametrize("wav", WAVS)
def test_vad_detect_segments_equiv(vad: OmniVAD, wav: str) -> None:
    f32, i16 = load_pair(os.path.join(DATA_DIR, wav))
    r_f32 = vad.detect(f32)
    r_i16 = vad.detect(i16)
    assert len(r_f32["timestamps"]) == len(r_i16["timestamps"])
    for (s1, e1), (s2, e2) in zip(r_f32["timestamps"], r_i16["timestamps"]):
        assert abs(s1 - s2) < TIME_TOL
        assert abs(e1 - e2) < TIME_TOL


# --- AED ---


@pytest.mark.parametrize("wav", WAVS)
def test_aed_detect_probs_equiv(aed: OmniAED, wav: str) -> None:
    f32, i16 = load_pair(os.path.join(DATA_DIR, wav))
    p_f32 = aed.detect_probs(f32)
    p_i16 = aed.detect_probs(i16)
    assert p_f32.shape == p_i16.shape
    assert np.max(np.abs(p_f32 - p_i16)) < PROB_TOL


@pytest.mark.parametrize("wav", WAVS)
def test_aed_detect_segments_equiv(aed: OmniAED, wav: str) -> None:
    f32, i16 = load_pair(os.path.join(DATA_DIR, wav))
    r_f32 = aed.detect(f32)
    r_i16 = aed.detect(i16)
    for cls in ("speech", "singing", "music"):
        assert len(r_f32["events"][cls]) == len(r_i16["events"][cls]), cls
        for (s1, e1), (s2, e2) in zip(r_f32["events"][cls], r_i16["events"][cls]):
            assert abs(s1 - s2) < TIME_TOL, cls
            assert abs(e1 - e2) < TIME_TOL, cls


# --- Stream VAD ---


@pytest.mark.parametrize("wav", WAVS)
def test_stream_vad_detect_full_equiv(svad: OmniStreamVAD, wav: str) -> None:
    f32, i16 = load_pair(os.path.join(DATA_DIR, wav))
    p_f32 = svad.detect_full(f32)
    p_i16 = svad.detect_full(i16)
    assert p_f32.shape == p_i16.shape
    assert np.max(np.abs(p_f32 - p_i16)) < PROB_TOL


def _stream_probs(svad: OmniStreamVAD, audio: np.ndarray) -> np.ndarray:
    """Run process() over `audio` (float32 or int16), return per-frame confidence."""
    svad.reset()
    chunk = 160
    out: list[float] = []
    for offset in range(0, len(audio) - chunk + 1, chunk):
        res = svad.process(audio[offset : offset + chunk])
        if res is not None:
            out.append(res.confidence)
    return np.asarray(out, dtype=np.float32)


@pytest.mark.parametrize("wav", WAVS)
def test_stream_vad_process_equiv(svad: OmniStreamVAD, wav: str) -> None:
    """process() over the same audio in float32 vs int16 — frame probs match."""
    f32, i16 = load_pair(os.path.join(DATA_DIR, wav))
    seq_f32 = _stream_probs(svad, f32)
    seq_i16 = _stream_probs(svad, i16)
    assert seq_f32.shape == seq_i16.shape
    assert np.max(np.abs(seq_f32 - seq_i16)) < PROB_TOL


def test_stream_vad_process_rejects_unknown_dtype(svad: OmniStreamVAD) -> None:
    svad.reset()
    bad = np.zeros(160, dtype=np.float64)
    with pytest.raises(TypeError, match="float32"):
        svad.process(bad)
