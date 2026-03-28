"""Determinism tests — repeated runs must produce identical results."""

import os

import numpy as np
import pytest

from omnivad import OmniAED, OmniStreamVAD, OmniVAD

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
AUDIO_FILES = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
REPEATS = 3


@pytest.fixture(scope="module")
def vad():
    return OmniVAD()


@pytest.fixture(scope="module")
def aed():
    return OmniAED()


@pytest.fixture(scope="module")
def svad():
    return OmniStreamVAD()


# --- VAD ---


@pytest.mark.parametrize("wav", AUDIO_FILES)
def test_vad_deterministic(vad, wav):
    """VAD.detect() on the same file N times → identical timestamps."""
    path = os.path.join(DATA_DIR, wav)
    results = [vad.detect(path) for _ in range(REPEATS)]
    for i in range(1, REPEATS):
        assert results[i]["timestamps"] == results[0]["timestamps"], f"run {i} differs from run 0"
        assert results[i]["duration"] == results[0]["duration"]


@pytest.mark.parametrize("wav", AUDIO_FILES)
def test_vad_raw_deterministic(vad, wav):
    """VAD.detect_raw() → identical frame probabilities."""
    path = os.path.join(DATA_DIR, wav)
    results = [vad.detect_raw(path) for _ in range(REPEATS)]
    for i in range(1, REPEATS):
        np.testing.assert_array_equal(results[i], results[0], err_msg=f"run {i} differs")


@pytest.mark.parametrize("wav", AUDIO_FILES)
def test_vad_chunked_deterministic(vad, wav):
    """VAD.detect() with chunking → identical timestamps across runs."""
    path = os.path.join(DATA_DIR, wav)
    results = [vad.detect(path, chunk_seconds=5, overlap_seconds=1) for _ in range(REPEATS)]
    for i in range(1, REPEATS):
        assert results[i]["timestamps"] == results[0]["timestamps"], f"run {i} differs from run 0"


# --- AED ---


@pytest.mark.parametrize("wav", AUDIO_FILES)
def test_aed_deterministic(aed, wav):
    """AED.detect() → identical events."""
    path = os.path.join(DATA_DIR, wav)
    results = [aed.detect(path) for _ in range(REPEATS)]
    for i in range(1, REPEATS):
        assert results[i]["events"] == results[0]["events"], f"run {i} differs from run 0"
        assert results[i]["duration"] == results[0]["duration"]


@pytest.mark.parametrize("wav", AUDIO_FILES)
def test_aed_raw_deterministic(aed, wav):
    """AED.detect_raw() → identical frame probabilities."""
    path = os.path.join(DATA_DIR, wav)
    results = [aed.detect_raw(path) for _ in range(REPEATS)]
    for i in range(1, REPEATS):
        np.testing.assert_array_equal(results[i], results[0], err_msg=f"run {i} differs")


@pytest.mark.parametrize("wav", AUDIO_FILES)
def test_aed_chunked_deterministic(aed, wav):
    """AED.detect() with chunking → identical events across runs."""
    path = os.path.join(DATA_DIR, wav)
    results = [aed.detect(path, chunk_seconds=5, overlap_seconds=1) for _ in range(REPEATS)]
    for i in range(1, REPEATS):
        assert results[i]["events"] == results[0]["events"], f"run {i} differs from run 0"


# --- StreamVAD ---


@pytest.mark.parametrize("wav", AUDIO_FILES)
def test_stream_vad_deterministic(svad, wav):
    """StreamVAD.detect_full() → identical frame probabilities."""
    path = os.path.join(DATA_DIR, wav)
    results = []
    for _ in range(REPEATS):
        svad.reset()
        results.append(svad.detect_full(path))
    for i in range(1, REPEATS):
        np.testing.assert_array_equal(results[i], results[0], err_msg=f"run {i} differs")
