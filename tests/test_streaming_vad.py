"""End-to-end test for OmniStreamingVAD on real audio.

Streams a known WAV file through the convenience wrapper and checks that
the resulting segments are reasonable (count > 0, bounded by audio duration).
"""

from __future__ import annotations

import os
import wave

import numpy as np
import pytest

from omnivad import OmniStreamingVAD

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_wav_int16(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path) as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    pcm = np.frombuffer(raw, dtype=np.int16)
    return pcm, sr


@pytest.fixture(scope="module")
def en_medium_audio():
    path = os.path.join(DATA_DIR, "en_medium.wav")
    if not os.path.exists(path):
        pytest.skip(f"test audio not found: {path}")
    return load_wav_int16(path)


def test_streaming_emits_segments(en_medium_audio):
    pcm, sr = en_medium_audio
    assert sr == 16000

    vad = OmniStreamingVAD()
    chunk_size = 160  # 10ms
    all_segments = []
    for i in range(0, len(pcm), chunk_size):
        all_segments.extend(vad.process(pcm[i : i + chunk_size]))
    all_segments.extend(vad.flush())
    vad.close()

    duration = len(pcm) / sr
    assert len(all_segments) > 0, "expected at least 1 speech segment"
    for start, end in all_segments:
        assert 0.0 <= start < end <= duration + 0.05, f"bad segment ({start}, {end})"


def test_streaming_chunk_size_invariance(en_medium_audio):
    """Same audio fed at chunk=160 vs chunk=1600 must produce same segments."""
    pcm, sr = en_medium_audio

    def run(chunk_size: int):
        vad = OmniStreamingVAD()
        out = []
        for i in range(0, len(pcm), chunk_size):
            out.extend(vad.process(pcm[i : i + chunk_size]))
        out.extend(vad.flush())
        vad.close()
        return out

    a = run(160)
    b = run(1600)
    assert len(a) == len(b)
    for (s1, e1), (s2, e2) in zip(a, b):
        assert abs(s1 - s2) < 0.05
        assert abs(e1 - e2) < 0.05


def test_state_query_during_speech(en_medium_audio):
    pcm, _ = en_medium_audio
    vad = OmniStreamingVAD()
    saw_speech = False
    for i in range(0, len(pcm), 160):
        vad.process(pcm[i : i + 160])
        if vad.is_in_speech:
            saw_speech = True
            assert vad.active_start is not None
            assert vad.active_start >= 0.0
    vad.close()
    assert saw_speech, "expected to be in speech at some point during en_medium.wav"


def test_reset_clears_segments(en_medium_audio):
    pcm, _ = en_medium_audio
    vad = OmniStreamingVAD()
    for i in range(0, 16000, 160):  # first 1s
        vad.process(pcm[i : i + 160])
    vad.reset()
    assert vad.total_samples_seen == 0
    assert not vad.is_in_speech
    vad.close()
