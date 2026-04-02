"""Thread-safety regression tests for OmniVAD-Kit.

Covers:
  1. Shared VAD handle concurrent detect_probs (positive)
  2. VAD chunked workers=4 vs workers=1 consistency
  3. Shared AED handle concurrent detect_probs (positive)
  4. AED chunked workers=4 vs workers=1 consistency
  5. StreamVAD contract: isolated handles stable, shared handle unsafe
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from omnivad import OmniAED, OmniStreamVAD, OmniVAD

DATA_DIR = Path(__file__).parent / "data"
THREADS = 4
REPEATS = 100
STREAM_CHUNK = 160  # 10ms @ 16kHz


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def wav_path() -> Path:
    return DATA_DIR / "hello_en.wav"


@pytest.fixture(scope="module")
def audio_f32(wav_path: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == 16000, f"Expected 16kHz, got {sr}"
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


@pytest.fixture(scope="module")
def audio_i16(audio_f32: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((audio_f32 * 32768.0).clip(-32768, 32767).astype(np.int16))


@pytest.fixture(scope="module")
def vad() -> OmniVAD:
    return OmniVAD()


@pytest.fixture(scope="module")
def aed() -> OmniAED:
    return OmniAED()


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _run_parallel(fn, threads: int = THREADS):
    """Run fn(worker_id) in parallel and return list of results."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(fn, range(threads)))


def _stream_sequence(svad: OmniStreamVAD, audio_i16: np.ndarray) -> list[tuple[int, float, bool]]:
    """Run StreamVAD frame-by-frame, return canonical sequence."""
    svad.reset()
    out = []
    for offset in range(0, len(audio_i16) - STREAM_CHUNK + 1, STREAM_CHUNK):
        result = svad.process(audio_i16[offset : offset + STREAM_CHUNK])
        if result is None:
            continue
        out.append((result.frame_offset, round(float(result.confidence), 6), bool(result.is_speech)))
    return out


# --------------------------------------------------------------------------- #
#  Test 1: Shared VAD handle concurrent detect_probs                           #
# --------------------------------------------------------------------------- #


def test_vad_shared_handle_concurrent_detect_probs(vad: OmniVAD, audio_f32: np.ndarray):
    """Shared OmniVAD handle: concurrent detect_probs must match serial baseline."""
    baseline = vad.detect_probs(audio_f32)

    def worker(_: int) -> np.ndarray:
        return vad.detect_probs(audio_f32)

    for _ in range(REPEATS):
        results = _run_parallel(worker)
        for got in results:
            np.testing.assert_array_equal(got, baseline)


# --------------------------------------------------------------------------- #
#  Test 2: VAD chunked workers consistency                                     #
# --------------------------------------------------------------------------- #


def test_vad_chunked_workers_consistency(vad: OmniVAD, wav_path: Path):
    """VAD detect(workers=4) must match detect(workers=1)."""
    baseline = vad.detect(wav_path, chunk_seconds=5, overlap_seconds=1, workers=1)

    for _ in range(REPEATS):
        got = vad.detect(wav_path, chunk_seconds=5, overlap_seconds=1, workers=THREADS)
        assert got["duration"] == baseline["duration"]
        assert got["timestamps"] == baseline["timestamps"]


# --------------------------------------------------------------------------- #
#  Test 3: Shared AED handle concurrent detect_probs                           #
# --------------------------------------------------------------------------- #


def test_aed_shared_handle_concurrent_detect_probs(aed: OmniAED, audio_f32: np.ndarray):
    """Shared OmniAED handle: concurrent detect_probs must match serial baseline."""
    baseline = aed.detect_probs(audio_f32)

    def worker(_: int) -> np.ndarray:
        return aed.detect_probs(audio_f32)

    for _ in range(REPEATS):
        results = _run_parallel(worker)
        for got in results:
            np.testing.assert_array_equal(got, baseline)


# --------------------------------------------------------------------------- #
#  Test 4: AED chunked workers consistency                                     #
# --------------------------------------------------------------------------- #


def test_aed_chunked_workers_consistency(aed: OmniAED, wav_path: Path):
    """AED detect(workers=4) must match detect(workers=1)."""
    baseline = aed.detect(wav_path, chunk_seconds=5, overlap_seconds=1, workers=1)

    for _ in range(REPEATS):
        got = aed.detect(wav_path, chunk_seconds=5, overlap_seconds=1, workers=THREADS)
        assert got["duration"] == baseline["duration"]
        assert got["events"] == baseline["events"]


# --------------------------------------------------------------------------- #
#  Test 5: StreamVAD thread-safety contract                                    #
# --------------------------------------------------------------------------- #


class TestStreamVadThreadSafety:
    """StreamVAD contract: isolated handles are safe, shared handle is not."""

    def test_isolated_handles_match_serial(self, audio_i16: np.ndarray):
        """Each thread creates its own OmniStreamVAD: results match serial baseline."""
        serial_svad = OmniStreamVAD()
        serial_seq = _stream_sequence(serial_svad, audio_i16)
        serial_svad.close()

        def worker(_: int) -> list[tuple[int, float, bool]]:
            local = OmniStreamVAD()
            try:
                return _stream_sequence(local, audio_i16)
            finally:
                local.close()

        results = _run_parallel(worker)
        for got in results:
            assert got == serial_seq, "Isolated StreamVAD handle produced different result"

    def test_shared_handle_is_unsafe(self):
        """Shared OmniStreamVAD handle under concurrent process: expect invariant breaks.

        This is a negative contract test run in a subprocess to isolate the
        intentional undefined behavior. Concurrent mutation of StreamVAD's
        internal std::deque causes:
          - macOS (libmalloc): silent heap corruption → data inconsistency
          - Linux (glibc ptmalloc2): heap integrity check → abort() / SIGABRT

        Both outcomes (crash or inconsistent results) prove unsafety.
        The subprocess approach prevents the crash from killing the test runner.
        """
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import sys
            from concurrent.futures import ThreadPoolExecutor
            import numpy as np
            import soundfile as sf
            from omnivad import OmniStreamVAD

            STREAM_CHUNK = 160
            THREADS = 4
            REPEATS = 20

            data, sr = sf.read("tests/data/hello_en.wav", dtype="float32")
            audio_i16 = np.ascontiguousarray((data * 32768.0).clip(-32768, 32767).astype(np.int16))

            # Serial baseline
            serial = OmniStreamVAD()
            serial.reset()
            serial_seq = []
            for offset in range(0, len(audio_i16) - STREAM_CHUNK + 1, STREAM_CHUNK):
                r = serial.process(audio_i16[offset:offset + STREAM_CHUNK])
                if r is not None:
                    serial_seq.append((r.frame_offset, round(float(r.confidence), 6)))
            serial.close()

            # Shared handle concurrent abuse
            shared = OmniStreamVAD()
            inconsistent = 0
            for _ in range(REPEATS):
                shared.reset()
                def worker(wid):
                    out = []
                    for off in range(wid * STREAM_CHUNK, len(audio_i16) - STREAM_CHUNK + 1, THREADS * STREAM_CHUNK):
                        r = shared.process(audio_i16[off:off + STREAM_CHUNK])
                        if r is not None:
                            out.append((r.frame_offset, round(float(r.confidence), 6)))
                    return out
                with ThreadPoolExecutor(max_workers=THREADS) as pool:
                    chunks = list(pool.map(worker, range(THREADS)))
                merged = sorted([x for c in chunks for x in c])
                if len(merged) != len(serial_seq) or merged != serial_seq:
                    inconsistent += 1
                    break  # one break is enough
            shared.close()

            # Exit 0 = inconsistent found (unsafe proven), 1 = no inconsistency
            sys.exit(0 if inconsistent > 0 else 1)
        """)

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            timeout=60,
        )

        # Crash (negative return code = signal) also proves unsafety
        crashed = result.returncode < 0
        inconsistency_found = result.returncode == 0

        if crashed:
            # e.g. -6 = SIGABRT on Linux (glibc heap corruption detection)
            return  # crash proves shared handle is unsafe
        if inconsistency_found:
            return  # data inconsistency proves shared handle is unsafe

        # returncode == 1: no inconsistency observed
        pytest.skip(
            "Shared OmniStreamVAD showed no invariant break in this run; inconclusive on this machine/scheduler"
        )
