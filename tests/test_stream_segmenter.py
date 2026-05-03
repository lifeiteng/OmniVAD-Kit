"""Tests for OmniStreamSegmenter — Python wrapper around the C streaming segmenter.

Mirrors native/test/test_stream_segmenter.cpp scenario-by-scenario so the
C and Python views of the algorithm stay bit-identical.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omnivad import OmniStreamSegmenter

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def make_probs(runs):
    """Build a flat float32 array from [(count, value), ...] runs."""
    out = []
    for n, v in runs:
        out.extend([float(v)] * int(n))
    return np.array(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
#  Config validation                                                           #
# --------------------------------------------------------------------------- #


def test_create_with_default_config():
    seg = OmniStreamSegmenter()
    assert not seg.is_in_speech
    assert seg.active_start is None
    seg.close()


def test_create_rejects_merge_silence():
    """merge_silence_frames is not exposed in Python API; the value is forced
    to 0 internally so callers can't pass it. Verify by introspecting that
    no such kwarg exists."""
    with pytest.raises(TypeError):
        OmniStreamSegmenter(merge_silence_frames=1)  # noqa


def test_create_rejects_extend_speech():
    with pytest.raises(TypeError):
        OmniStreamSegmenter(extend_speech_frames=1)  # noqa


# --------------------------------------------------------------------------- #
#  Algorithm tests                                                             #
# --------------------------------------------------------------------------- #


def test_all_silence_no_emit():
    seg = OmniStreamSegmenter()
    out = seg.process_probs(np.zeros(100, dtype=np.float32))
    assert out == []
    seg.close()


def test_all_speech_no_emit_until_flush():
    seg = OmniStreamSegmenter()
    out = seg.process_probs(np.ones(100, dtype=np.float32))
    assert out == []
    seg.close()


def test_short_pulse_no_emit():
    """5 silence + 10 speech (< min_speech=20) + 50 silence -> nothing."""
    seg = OmniStreamSegmenter()
    probs = make_probs([(5, 0), (10, 1), (50, 0)])
    assert seg.process_probs(probs) == []
    seg.close()


def test_one_clean_segment():
    """Mirrors C T4: 5 silence + 30 speech + 25 silence -> emit (0.01, 0.38)."""
    seg = OmniStreamSegmenter()
    probs = make_probs([(5, 0), (30, 1), (25, 0)])
    out = seg.process_probs(probs)
    assert len(out) == 1
    start, end = out[0]
    assert math.isclose(start, 0.01, abs_tol=1e-4)
    assert math.isclose(end, 0.38, abs_tol=1e-4)
    seg.close()


def test_two_segments():
    seg = OmniStreamSegmenter()
    probs = make_probs([(5, 0), (30, 1), (35, 0), (30, 1), (25, 0)])
    assert len(seg.process_probs(probs)) == 2
    seg.close()


def test_chunk_size_invariance():
    """process_frame N times == process_probs(N) == random small chunks."""
    probs = make_probs([(5, 0), (30, 1), (35, 0), (30, 1), (25, 0), (25, 1), (30, 0)])

    seg_a = OmniStreamSegmenter()
    res_a = seg_a.process_probs(probs)
    seg_a.close()

    seg_b = OmniStreamSegmenter()
    res_b = []
    for p in probs:
        res_b.extend(seg_b.process_frame(float(p)))
    seg_b.close()

    seg_c = OmniStreamSegmenter()
    res_c = []
    chunk_sizes = [1, 7, 3, 13, 1, 50, 100]
    idx = 0
    chunk_i = 0
    while idx < len(probs):
        n = min(chunk_sizes[chunk_i % len(chunk_sizes)], len(probs) - idx)
        res_c.extend(seg_c.process_probs(probs[idx : idx + n]))
        idx += n
        chunk_i += 1
    seg_c.close()

    assert res_a == res_b == res_c


# --------------------------------------------------------------------------- #
#  Force-split                                                                 #
# --------------------------------------------------------------------------- #


def test_force_split_continuous_speech():
    """Mirrors C T9: max_speech=20, 50 ones -> 3 splits."""
    seg = OmniStreamSegmenter(max_chunk_secs=0.20)
    out = seg.process_probs(np.ones(50, dtype=np.float32))
    assert len(out) == 3
    assert out[0] == pytest.approx((0.00, 0.10), abs=1e-4)
    assert out[1] == pytest.approx((0.11, 0.21), abs=1e-4)
    assert out[2] == pytest.approx((0.22, 0.32), abs=1e-4)
    seg.close()


def test_force_split_picks_min_prob():
    """Mirrors C T10: lowest prob in window wins."""
    seg = OmniStreamSegmenter(smooth_window_size=1, min_speech_secs=0.01, max_chunk_secs=0.20)
    probs = np.ones(25, dtype=np.float32)
    probs[15] = 0.5  # > threshold (0.4) but lowest in window
    out = seg.process_probs(probs)
    assert len(out) == 1
    assert math.isclose(out[0][1], 0.15, abs_tol=1e-4)
    seg.close()


def test_max_speech_zero_disables_split():
    seg = OmniStreamSegmenter(max_chunk_secs=0.0)
    out = seg.process_probs(np.ones(1000, dtype=np.float32))
    assert out == []  # awaits flush
    seg.close()


# --------------------------------------------------------------------------- #
#  Flush                                                                       #
# --------------------------------------------------------------------------- #


def test_flush_silence_only():
    seg = OmniStreamSegmenter()
    seg.process_probs(np.zeros(50, dtype=np.float32))
    assert seg.flush() == []
    seg.close()


def test_flush_unconfirmed_candidate():
    seg = OmniStreamSegmenter()
    seg.process_probs(np.ones(15, dtype=np.float32))  # < min_speech
    assert seg.flush() == []
    seg.close()


def test_flush_during_speech():
    """100 frames of 1.0 -> tail = 100*0.01 + 0.025 = 1.025."""
    seg = OmniStreamSegmenter()
    seg.process_probs(np.ones(100, dtype=np.float32))
    out = seg.flush(0)
    assert len(out) == 1
    assert math.isclose(out[0][0], 0.000, abs_tol=1e-4)
    assert math.isclose(out[0][1], 1.025, abs_tol=1e-4)
    seg.close()


def test_flush_clamps_to_wav_dur():
    """tail 1.025 clamped to wav_dur = 16000/16000 = 1.000."""
    seg = OmniStreamSegmenter()
    seg.process_probs(np.ones(100, dtype=np.float32))
    out = seg.flush(total_samples_seen=16000)
    assert math.isclose(out[0][1], 1.000, abs_tol=1e-4)
    seg.close()


def test_flush_during_possible_silence():
    """30 speech + 5 silence -> flush emits trailing (0, 0.375)."""
    seg = OmniStreamSegmenter()
    seg.process_probs(make_probs([(30, 1), (5, 0)]))
    out = seg.flush(0)
    assert len(out) == 1
    assert math.isclose(out[0][1], 0.375, abs_tol=1e-4)
    seg.close()


def test_flush_after_force_split():
    """T_flush6 mirror: 50 ones + max_speech=20 -> 3 splits + tail (0.33, 0.525)."""
    seg = OmniStreamSegmenter(max_chunk_secs=0.20)
    pre = seg.process_probs(np.ones(50, dtype=np.float32))
    assert len(pre) == 3
    tail = seg.flush(0)
    assert len(tail) == 1
    assert tail[0] == pytest.approx((0.33, 0.525), abs=1e-4)
    seg.close()


def test_flush_twice_idempotent():
    seg = OmniStreamSegmenter()
    seg.process_probs(np.ones(100, dtype=np.float32))
    seg.flush(0)
    assert seg.flush(0) == []
    seg.close()


# --------------------------------------------------------------------------- #
#  State queries                                                               #
# --------------------------------------------------------------------------- #


def test_is_in_speech_transitions():
    seg = OmniStreamSegmenter()
    assert not seg.is_in_speech
    seg.process_probs(np.zeros(5, dtype=np.float32))
    assert not seg.is_in_speech
    seg.process_probs(np.ones(40, dtype=np.float32))
    assert seg.is_in_speech
    assert seg.active_start is not None
    assert seg.active_start >= 0.0
    seg.close()


def test_reset_clears_state():
    seg = OmniStreamSegmenter()
    seg.process_probs(np.ones(50, dtype=np.float32))
    assert seg.is_in_speech
    seg.reset()
    assert not seg.is_in_speech
    assert seg.active_start is None
    seg.close()


# --------------------------------------------------------------------------- #
#  Context manager + lifecycle                                                 #
# --------------------------------------------------------------------------- #


def test_context_manager():
    with OmniStreamSegmenter() as seg:
        out = seg.process_probs(make_probs([(5, 0), (30, 1), (25, 0)]))
        assert len(out) == 1


def test_close_idempotent():
    seg = OmniStreamSegmenter()
    seg.close()
    seg.close()  # must not raise


def test_use_after_close_raises():
    seg = OmniStreamSegmenter()
    seg.close()
    with pytest.raises(RuntimeError):
        seg.process_frame(0.5)
