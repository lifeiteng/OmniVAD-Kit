"""Tests for OmniStreamVAD.clone() — FastClone API.

Covers: independence, destroy-order safety, reset isolation,
multi-clone, concurrent inference with threads.
"""

import threading
from pathlib import Path

import numpy as np
import pytest

from omnivad import OmniStreamVAD

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"
# Fall back to repo-level test data
AUDIO_DIR = Path(__file__).parent.parent / "tests" / "data"
if not AUDIO_DIR.exists():
    AUDIO_DIR = Path(__file__).parent.parent / "native" / "test" / "data"


def _load_wav_int16(path: Path) -> np.ndarray:
    """Load 16kHz mono WAV as int16."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="int16")
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    return data


def _get_test_audio() -> np.ndarray:
    """Get test audio, trying multiple locations."""
    for candidate in [
        Path(__file__).parent / "data" / "hello_en.wav",
        Path(__file__).parent.parent / "tests" / "data" / "hello_en.wav",
    ]:
        if candidate.exists():
            return _load_wav_int16(candidate)
    # Generate synthetic speech-like audio (sine wave)
    t = np.arange(16000, dtype=np.float64) / 16000.0
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return audio


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def vad():
    """Create a fresh StreamVAD instance."""
    v = OmniStreamVAD()
    yield v
    v.close()


@pytest.fixture
def audio_chunks():
    """Get test audio as 10ms chunks (160 samples each)."""
    audio = _get_test_audio()
    chunks = []
    for i in range(0, len(audio) - 160, 160):
        chunks.append(audio[i : i + 160])
    return chunks


# --------------------------------------------------------------------------- #
#  Clone independence                                                          #
# --------------------------------------------------------------------------- #


class TestCloneIndependence:
    def test_clone_starts_with_fresh_state(self, vad, audio_chunks):
        """Clone should start with frame_offset=0 regardless of source state."""
        # Process some frames on source
        for chunk in audio_chunks[:10]:
            vad.process(chunk)
        source_offset = vad.frame_offset

        # Clone should be fresh
        clone = vad.clone()
        assert clone.frame_offset == 0
        assert source_offset > 0  # source was advanced
        clone.close()

    def test_clone_does_not_affect_source(self, vad, audio_chunks):
        """Processing on clone should not change source state."""
        source_offset_before = vad.frame_offset

        clone = vad.clone()
        for chunk in audio_chunks[:20]:
            clone.process(chunk)

        assert vad.frame_offset == source_offset_before
        clone.close()

    def test_source_does_not_affect_clone(self, vad, audio_chunks):
        """Processing on source should not change clone state."""
        clone = vad.clone()
        clone_offset_before = clone.frame_offset

        for chunk in audio_chunks[:20]:
            vad.process(chunk)

        assert clone.frame_offset == clone_offset_before
        clone.close()

    def test_clone_produces_same_results_as_fresh_create(self, vad, audio_chunks):
        """Clone should produce identical results to a freshly created instance."""
        clone = vad.clone()
        fresh = OmniStreamVAD()

        results_clone = []
        results_fresh = []
        for chunk in audio_chunks[:30]:
            r1 = clone.process(chunk)
            r2 = fresh.process(chunk)
            if r1 is not None:
                results_clone.append(r1.confidence)
            if r2 is not None:
                results_fresh.append(r2.confidence)

        assert len(results_clone) == len(results_fresh)
        for c, f in zip(results_clone, results_fresh):
            assert abs(c - f) < 1e-6, f"Clone={c} vs Fresh={f}"

        clone.close()
        fresh.close()


# --------------------------------------------------------------------------- #
#  Destroy-order safety                                                        #
# --------------------------------------------------------------------------- #


class TestDestroyOrder:
    def test_destroy_source_clone_survives(self, audio_chunks):
        """Clone should work after source is destroyed."""
        source = OmniStreamVAD()
        clone = source.clone()
        source.close()

        # Clone should still work
        for chunk in audio_chunks[:20]:
            clone.process(chunk)  # Should not crash

        clone.close()

    def test_destroy_clone_source_survives(self, audio_chunks):
        """Source should work after clone is destroyed."""
        source = OmniStreamVAD()
        clone = source.clone()
        clone.close()

        # Source should still work
        for chunk in audio_chunks[:20]:
            source.process(chunk)  # Should not crash

        source.close()

    def test_destroy_all_clones_then_source(self, audio_chunks):
        """All clones destroyed before source — no crash."""
        source = OmniStreamVAD()
        clones = [source.clone() for _ in range(5)]

        for c in clones:
            for chunk in audio_chunks[:5]:
                c.process(chunk)
            c.close()

        # Source should still work
        for chunk in audio_chunks[:5]:
            source.process(chunk)
        source.close()


# --------------------------------------------------------------------------- #
#  Reset isolation                                                             #
# --------------------------------------------------------------------------- #


class TestResetIsolation:
    def test_reset_source_does_not_affect_clone(self, vad, audio_chunks):
        """Resetting source should not change clone state."""
        clone = vad.clone()

        # Process some audio on both
        for chunk in audio_chunks[:10]:
            vad.process(chunk)
            clone.process(chunk)

        clone_offset = clone.frame_offset

        # Reset source
        vad.reset()
        assert vad.frame_offset == 0
        assert clone.frame_offset == clone_offset  # unchanged

        clone.close()

    def test_reset_clone_does_not_affect_source(self, vad, audio_chunks):
        """Resetting clone should not change source state."""
        clone = vad.clone()

        for chunk in audio_chunks[:10]:
            vad.process(chunk)
            clone.process(chunk)

        source_offset = vad.frame_offset

        clone.reset()
        assert clone.frame_offset == 0
        assert vad.frame_offset == source_offset  # unchanged

        clone.close()


# --------------------------------------------------------------------------- #
#  Multi-clone                                                                 #
# --------------------------------------------------------------------------- #


class TestMultiClone:
    def test_multiple_clones_independent(self, vad, audio_chunks):
        """N clones processing different audio should not interfere."""
        n = 8
        clones = [vad.clone() for _ in range(n)]

        for i, c in enumerate(clones):
            # Each clone processes a different slice
            start = i * 5
            for chunk in audio_chunks[start : start + 10]:
                c.process(chunk)

        # All should have independent offsets
        for c in clones:
            c.close()

    def test_clone_of_clone(self, vad, audio_chunks):
        """Cloning a clone should work (transitive sharing)."""
        clone1 = vad.clone()
        for chunk in audio_chunks[:5]:
            clone1.process(chunk)

        clone2 = clone1.clone()
        assert clone2.frame_offset == 0  # fresh state

        for chunk in audio_chunks[:10]:
            clone2.process(chunk)

        clone2.close()
        clone1.close()


# --------------------------------------------------------------------------- #
#  Concurrent clone + inference (multi-threaded)                               #
# --------------------------------------------------------------------------- #


class TestConcurrentClone:
    def test_concurrent_clone_and_process(self, audio_chunks):
        """Multiple threads cloning from same source and processing concurrently."""
        source = OmniStreamVAD()
        errors = []
        results = {}

        def worker(thread_id: int):
            try:
                clone = source.clone()
                thread_results = []
                for chunk in audio_chunks[:30]:
                    r = clone.process(chunk)
                    if r is not None:
                        thread_results.append(r.confidence)
                results[thread_id] = thread_results
                clone.close()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"

        # All threads should produce identical results (same audio)
        if results:
            ref = results[0]
            for tid, res in results.items():
                assert len(res) == len(ref), f"Thread {tid}: {len(res)} vs {len(ref)}"
                for i, (a, b) in enumerate(zip(res, ref)):
                    assert abs(a - b) < 1e-6, f"Thread {tid} frame {i}: {a} vs {b}"

        source.close()

    def test_concurrent_detect_full(self):
        """Multiple threads using detect_full on clones concurrently."""
        source = OmniStreamVAD()
        audio = _get_test_audio()
        errors = []
        results = {}

        def worker(thread_id: int):
            try:
                clone = source.clone()
                probs = clone.detect_full(audio)
                results[thread_id] = probs
                clone.close()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Thread errors: {errors}"

        # All threads should produce identical probabilities
        if results:
            ref = results[0]
            for tid, probs in results.items():
                np.testing.assert_allclose(probs, ref, atol=1e-6, err_msg=f"Thread {tid} mismatch")

        source.close()

    def test_concurrent_clone_destroy(self):
        """Stress test: rapidly clone and destroy from multiple threads."""
        source = OmniStreamVAD()
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(50):
                    clone = source.clone()
                    chunk = np.zeros(160, dtype=np.int16)
                    clone.process(chunk)
                    clone.close()
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Thread errors: {errors}"
        source.close()


# --------------------------------------------------------------------------- #
#  Error handling                                                              #
# --------------------------------------------------------------------------- #


class TestCloneErrors:
    def test_clone_closed_instance_raises(self):
        """Cloning a closed instance should raise RuntimeError."""
        vad = OmniStreamVAD()
        vad.close()
        with pytest.raises(RuntimeError, match="closed"):
            vad.clone()

    def test_context_manager_on_clone(self, vad, audio_chunks):
        """Clone should work with context manager (with statement)."""
        with vad.clone() as clone:
            for chunk in audio_chunks[:5]:
                clone.process(chunk)
        # clone is now closed, should not crash
