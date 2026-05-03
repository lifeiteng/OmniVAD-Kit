"""Tests for omnivad.merge_chunks (pure-algorithm chunking).

Mirrors native/test/test_chunking.cpp scenario-by-scenario so the C and
Python views of the same algorithm stay bit-identical. If either side's
output drifts, both this file and test_chunking.cpp must be updated.

Two layers exercised here:
  1. Public API ``merge_chunks`` — algorithm equivalence with C/TS.
  2. Binding layer (ctypes struct sizes, low-level guard codes, NaN
     handling, default-value drift between Python convenience defaults
     and C/TS ``default_chunk_config``).
"""

from __future__ import annotations

import ctypes
import math

import pytest

from omnivad import ChunkResult, default_chunk_config, merge_chunks
from omnivad._binding import (
    OMNI_ERR_INVALID_ARG,
    OMNI_ERR_NULL_POINTER,
    OmniChunk,
    OmniChunkConfig,
    OmniSegment,
    _lib,
)

# --------------------------------------------------------------------------- #
#  Defaults                                                                    #
# --------------------------------------------------------------------------- #


def test_default_config_matches_plan():
    cfg = default_chunk_config()
    assert cfg.max_chunk_secs == pytest.approx(30.0)
    assert math.isinf(cfg.max_gap_secs)
    assert cfg.pad_onset_secs == pytest.approx(0.04)
    assert cfg.pad_offset_secs == pytest.approx(0.04)
    assert cfg.min_speech_secs == pytest.approx(0.0)
    assert cfg.min_silence_secs == pytest.approx(0.20)


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #


def _check_chunks(actual: list[ChunkResult], expected: list[tuple[float, float, int, int]]):
    assert len(actual) == len(expected), f"chunk count: expected {len(expected)}, got {len(actual)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.start == pytest.approx(e[0], abs=1e-4), f"chunk[{i}].start"
        assert a.end == pytest.approx(e[1], abs=1e-4), f"chunk[{i}].end"
        assert a.seg_start_idx == e[2], f"chunk[{i}].seg_start_idx"
        assert a.seg_count == e[3], f"chunk[{i}].seg_count"


# --------------------------------------------------------------------------- #
#  Scenario 1: short audio fits in one chunk                                   #
# --------------------------------------------------------------------------- #


def test_short_audio_one_chunk():
    chunks = merge_chunks([(0.0, 5.0), (6.0, 10.0)], max_chunk_secs=30.0)
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


# --------------------------------------------------------------------------- #
#  Scenario 2: long audio multiple splits                                      #
# --------------------------------------------------------------------------- #


def test_long_audio_multiple_splits():
    chunks = merge_chunks(
        [(0.0, 10.0), (11.0, 20.0), (21.0, 30.0), (31.0, 40.0)],
        max_chunk_secs=20.0,
    )
    _check_chunks(chunks, [(0.0, 20.0, 0, 2), (21.0, 40.0, 2, 2)])


# --------------------------------------------------------------------------- #
#  Scenario 3: gap > max_gap_secs forces split                                      #
# --------------------------------------------------------------------------- #


def test_gap_split():
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)],
        max_chunk_secs=30.0,
        max_gap_secs=2.0,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1), (8.0, 10.0, 1, 1), (20.0, 25.0, 2, 1)])


# --------------------------------------------------------------------------- #
#  Scenario 4: single segment > max_chunk_secs -> equal hard-split                 #
# --------------------------------------------------------------------------- #


def test_single_segment_hard_split():
    chunks = merge_chunks([(0.0, 100.0)], max_chunk_secs=30.0)
    _check_chunks(
        chunks,
        [
            (0.0, 30.0, 0, 1),
            (30.0, 60.0, 0, 1),
            (60.0, 90.0, 0, 1),
            (90.0, 100.0, 0, 1),
        ],
    )


# --------------------------------------------------------------------------- #
#  Scenario 5: empty input -> zero chunks                                      #
# --------------------------------------------------------------------------- #


def test_empty_input():
    chunks = merge_chunks([], max_chunk_secs=30.0)
    assert chunks == []


# --------------------------------------------------------------------------- #
#  Scenario 6: min_speech_secs filters short segments                          #
# --------------------------------------------------------------------------- #


def test_min_speech_secs_filter():
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 5.0)],
        max_chunk_secs=30.0,
        min_speech_secs=0.5,
    )
    _check_chunks(chunks, [(1.0, 5.0, 0, 1)])


# --------------------------------------------------------------------------- #
#  Scenario 7: min_silence_secs merges close-by segments                       #
# --------------------------------------------------------------------------- #


def test_min_silence_secs_merge():
    chunks = merge_chunks(
        [(0.0, 5.0), (5.1, 10.0)],
        max_chunk_secs=30.0,
        min_silence_secs=0.5,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 1)])


# --------------------------------------------------------------------------- #
#  Scenario 8: pad_onset_secs / pad_offset_secs                                          #
# --------------------------------------------------------------------------- #


def test_pad_applied():
    chunks = merge_chunks(
        [(5.0, 10.0)],
        max_chunk_secs=30.0,
        pad_onset_secs=0.5,
        pad_offset_secs=0.5,
    )
    _check_chunks(chunks, [(4.5, 10.5, 0, 1)])


def test_pad_onset_secs_clamped_to_zero():
    chunks = merge_chunks(
        [(0.1, 5.0)],
        max_chunk_secs=30.0,
        pad_onset_secs=0.5,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1)])


# --------------------------------------------------------------------------- #
#  Boundary: max_chunk_secs <= 0 raises                                            #
# --------------------------------------------------------------------------- #


def test_invalid_max_chunk_secs_raises():
    with pytest.raises(RuntimeError, match=r"INVALID_ARG|invalid"):
        merge_chunks([(0.0, 5.0)], max_chunk_secs=0.0)
    with pytest.raises(RuntimeError, match=r"INVALID_ARG|invalid"):
        merge_chunks([(0.0, 5.0)], max_chunk_secs=-1.0)


# --------------------------------------------------------------------------- #
#  as_dict variant returns plain dicts                                         #
# --------------------------------------------------------------------------- #


def test_as_dict_returns_plain_dicts():
    chunks = merge_chunks([(0.0, 5.0)], max_chunk_secs=30.0, as_dict=True)
    assert chunks == [{"start": 0.0, "end": 5.0, "seg_start_idx": 0, "seg_count": 1}]


# --------------------------------------------------------------------------- #
#  ChunkResult.to_dict()                                                       #
# --------------------------------------------------------------------------- #


def test_chunk_result_to_dict():
    [c] = merge_chunks([(1.0, 2.0)], max_chunk_secs=30.0)
    assert c.to_dict() == {
        "start": 1.0,
        "end": 2.0,
        "seg_start_idx": 0,
        "seg_count": 1,
    }


# --------------------------------------------------------------------------- #
#  Binding layer: ABI struct sizes (must match C and TS hard-coded constants) #
# --------------------------------------------------------------------------- #


def test_abi_struct_sizes():
    """Sizes are hard-coded in packages/omnivad/src/wasm-binding.ts (SIZEOF_*).

    Any drift here would silently corrupt the ctypes view of the C structs and
    misalign the WASM heap layout — neither would surface as a Python-side
    crash, only as garbled output.

    OmniChunkConfig = 6 floats + 1 i32 mode = 28 bytes (no tail padding since
    alignof(struct) <= 4 on all current targets).
    """
    assert ctypes.sizeof(OmniSegment) == 8
    assert ctypes.sizeof(OmniChunk) == 16
    assert ctypes.sizeof(OmniChunkConfig) == 28


def test_abi_field_offsets():
    assert OmniChunk.start.offset == 0
    assert OmniChunk.end.offset == 4
    assert OmniChunk.seg_start_idx.offset == 8
    assert OmniChunk.seg_count.offset == 12
    assert OmniChunkConfig.max_chunk_secs.offset == 0
    assert OmniChunkConfig.max_gap_secs.offset == 4
    assert OmniChunkConfig.pad_onset_secs.offset == 8
    assert OmniChunkConfig.pad_offset_secs.offset == 12
    assert OmniChunkConfig.min_speech_secs.offset == 16
    assert OmniChunkConfig.min_silence_secs.offset == 20
    assert OmniChunkConfig.mode.offset == 24


# --------------------------------------------------------------------------- #
#  Python convenience-default vs C/TS canonical-default DRIFT                  #
# --------------------------------------------------------------------------- #


def test_python_convenience_defaults_differ_from_canonical():
    """Lock the (intentional?) drift between Python convenience defaults and
    the C/TS canonical defaults.

    ``merge_chunks(timestamps)`` (no kwargs) uses Python defaults
    ``pad=0, min_silence_secs=0`` — i.e. NO padding and NO silence pre-merge.

    ``default_chunk_config()`` (which the C and TS bindings expose as the
    ``DEFAULT_CHUNK_CONFIG``) uses ``pad=0.04, min_silence_secs=0.20``.

    Calling ``merge_chunks([...])`` in Python WILL produce different output
    from the equivalent TS call ``mergeChunks([...])`` (no options). This
    test pins the discrepancy so any future "fix" is intentional.
    """
    # Same input, two paths:
    via_python_defaults = merge_chunks([(0.0, 5.0), (5.1, 10.0)])
    canonical = default_chunk_config()
    via_canonical = merge_chunks(
        [(0.0, 5.0), (5.1, 10.0)],
        max_chunk_secs=canonical.max_chunk_secs,
        max_gap_secs=canonical.max_gap_secs,
        pad_onset_secs=canonical.pad_onset_secs,
        pad_offset_secs=canonical.pad_offset_secs,
        min_speech_secs=canonical.min_speech_secs,
        min_silence_secs=canonical.min_silence_secs,
    )

    # Python defaults: no padding, no merge → 2 segments survive as 2 segs in 1 chunk.
    assert via_python_defaults == [ChunkResult(start=0.0, end=10.0, seg_start_idx=0, seg_count=2)]

    # Canonical defaults: gap 0.1 < min_silence_secs 0.20 → pre-merged into 1 seg, then padded.
    assert len(via_canonical) == 1
    assert via_canonical[0].seg_count == 1  # pre-merged
    assert via_canonical[0].start == pytest.approx(0.0)  # 0 - 0.04 → clamped 0
    assert via_canonical[0].end == pytest.approx(10.04)  # 10 + 0.04


def test_default_chunk_config_matches_ts_constants():
    """default_chunk_config() must match wasm-binding.ts DEFAULT_CHUNK_CONFIG."""
    cfg = default_chunk_config()
    assert cfg.max_chunk_secs == pytest.approx(30.0)
    assert cfg.pad_onset_secs == pytest.approx(0.04)
    assert cfg.pad_offset_secs == pytest.approx(0.04)
    assert cfg.min_speech_secs == pytest.approx(0.0)
    assert cfg.min_silence_secs == pytest.approx(0.20)
    assert math.isinf(cfg.max_gap_secs)


# --------------------------------------------------------------------------- #
#  Low-level _lib guards (NULL pointers, NaN, num<0) — error-code passthrough #
# --------------------------------------------------------------------------- #


def _call_low_level(seg_ptr, n: int, cfg_ptr, out_ptr_ptr, out_count_ptr) -> int:
    """Direct ctypes call — bypasses merge_chunks() validation to test C guards."""
    return _lib.omni_merge_chunks(seg_ptr, n, cfg_ptr, out_ptr_ptr, out_count_ptr)


def test_low_level_null_out_chunks():
    cfg = default_chunk_config()
    seg_array = (OmniSegment * 1)()
    seg_array[0].start, seg_array[0].end = 0.0, 5.0
    out_count = ctypes.c_int(0)
    rc = _call_low_level(
        ctypes.cast(seg_array, ctypes.POINTER(OmniSegment)),
        1,
        ctypes.byref(cfg),
        None,  # NULL out_chunks
        ctypes.byref(out_count),
    )
    assert rc == OMNI_ERR_NULL_POINTER


def test_low_level_null_out_count():
    cfg = default_chunk_config()
    seg_array = (OmniSegment * 1)()
    seg_array[0].start, seg_array[0].end = 0.0, 5.0
    out_ptr = ctypes.POINTER(OmniChunk)()
    rc = _call_low_level(
        ctypes.cast(seg_array, ctypes.POINTER(OmniSegment)),
        1,
        ctypes.byref(cfg),
        ctypes.byref(out_ptr),
        None,  # NULL out_count
    )
    assert rc == OMNI_ERR_NULL_POINTER


def test_low_level_null_config():
    seg_array = (OmniSegment * 1)()
    seg_array[0].start, seg_array[0].end = 0.0, 5.0
    out_ptr = ctypes.POINTER(OmniChunk)()
    out_count = ctypes.c_int(0)
    rc = _call_low_level(
        ctypes.cast(seg_array, ctypes.POINTER(OmniSegment)),
        1,
        None,  # NULL config
        ctypes.byref(out_ptr),
        ctypes.byref(out_count),
    )
    assert rc == OMNI_ERR_NULL_POINTER


def test_low_level_null_segments_with_positive_n():
    cfg = default_chunk_config()
    out_ptr = ctypes.POINTER(OmniChunk)()
    out_count = ctypes.c_int(0)
    rc = _call_low_level(
        None,  # NULL segments
        5,  # but n > 0
        ctypes.byref(cfg),
        ctypes.byref(out_ptr),
        ctypes.byref(out_count),
    )
    assert rc == OMNI_ERR_NULL_POINTER


def test_low_level_negative_num_segments():
    cfg = default_chunk_config()
    seg_array = (OmniSegment * 1)()
    seg_array[0].start, seg_array[0].end = 0.0, 5.0
    out_ptr = ctypes.POINTER(OmniChunk)()
    out_count = ctypes.c_int(0)
    rc = _call_low_level(
        ctypes.cast(seg_array, ctypes.POINTER(OmniSegment)),
        -1,
        ctypes.byref(cfg),
        ctypes.byref(out_ptr),
        ctypes.byref(out_count),
    )
    assert rc == OMNI_ERR_INVALID_ARG


def test_max_chunk_secs_nan_rejected():
    """NaN > 0.0 is False → guard rejects."""
    with pytest.raises(RuntimeError, match=r"INVALID_ARG|invalid"):
        merge_chunks([(0.0, 5.0)], max_chunk_secs=float("nan"))


def test_max_chunk_secs_inf_accepted():
    """+Inf > 0 is True → accepted; everything fits in one chunk."""
    chunks = merge_chunks([(0.0, 5.0), (10.0, 20.0)], max_chunk_secs=float("inf"))
    _check_chunks(chunks, [(0.0, 20.0, 0, 2)])


# --------------------------------------------------------------------------- #
#  C↔Python algorithm equivalence — mirror new C scenarios 9-19                #
# --------------------------------------------------------------------------- #


def test_step1_before_step2_ordering():
    """Filter (drop short) MUST run before merge (close gaps).

    If reversed, middle short seg would pull the two halves into (0, 10).
    See test_chunking.cpp Scenario 9 for the full trace.
    """
    chunks = merge_chunks(
        [(0.0, 5.0), (5.4, 5.5), (5.6, 10.0)],
        max_chunk_secs=30.0,
        max_gap_secs=0.55,
        min_speech_secs=0.2,
        min_silence_secs=0.5,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1), (5.6, 10.0, 1, 1)])


def test_seg_idx_after_filter_and_merge():
    """seg_start_idx counts on POST-filter+merge view, not raw input."""
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 5.0), (5.1, 10.0), (20.0, 25.0)],
        max_chunk_secs=20.0,
        min_speech_secs=0.5,
        min_silence_secs=0.5,
    )
    _check_chunks(chunks, [(1.0, 10.0, 0, 1), (20.0, 25.0, 1, 1)])


def test_min_speech_secs_drops_all():
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 1.05)],
        max_chunk_secs=30.0,
        min_speech_secs=1.0,
    )
    assert chunks == []


def test_min_silence_secs_cascade_max_end():
    """Cascade-merge takes max(end) when next.end < cur.end."""
    chunks = merge_chunks(
        [(0.0, 10.0), (0.1, 5.0), (0.2, 8.0)],
        max_chunk_secs=30.0,
        min_silence_secs=0.5,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 1)])


def test_max_chunk_secs_equals_segment_dur():
    """`<=` boundary in Step 4: dur==max_chunk_secs NOT split."""
    chunks = merge_chunks([(0.0, 30.0)], max_chunk_secs=30.0)
    _check_chunks(chunks, [(0.0, 30.0, 0, 1)])


def test_max_gap_secs_equals_real_gap():
    """`>` strict boundary: gap==max_gap_secs NOT split."""
    chunks = merge_chunks(
        [(0.0, 5.0), (7.0, 10.0)],
        max_chunk_secs=30.0,
        max_gap_secs=2.0,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


def test_min_silence_secs_equals_real_gap():
    """`<` strict boundary: gap==min_silence_secs NOT merged."""
    chunks = merge_chunks(
        [(0.0, 5.0), (5.5, 10.0)],
        max_chunk_secs=30.0,
        min_silence_secs=0.5,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


def test_min_speech_secs_equals_segment_dur():
    """`>=` boundary: dur==min_speech_secs KEPT."""
    chunks = merge_chunks(
        [(0.0, 0.5), (1.0, 5.0)],
        max_chunk_secs=30.0,
        min_speech_secs=0.5,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 2)])


def test_pad_chunks_may_overlap():
    """Algorithm does not de-dupe overlap caused by padding."""
    chunks = merge_chunks(
        [(0.0, 5.0), (6.0, 10.0)],
        max_chunk_secs=30.0,
        max_gap_secs=0.5,
        pad_onset_secs=2.0,
        pad_offset_secs=2.0,
    )
    _check_chunks(chunks, [(0.0, 7.0, 0, 1), (4.0, 12.0, 1, 1)])


# --------------------------------------------------------------------------- #
#  LONGEST_GAP mode — mirror native/test/test_chunking.cpp scenarios LG1-LG10  #
# --------------------------------------------------------------------------- #


def test_lg_default_mode_is_greedy():
    """Backward compatibility: omitting `mode` must give GREEDY behaviour."""
    chunks = merge_chunks([(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)], max_chunk_secs=20.0)
    # GREEDY: cur=(0,5), accept (8,10) cur=(0,10) [10≤20], accept (20,25)
    # would_exceed (25-0=25>20) split → (0,10,0,2) + (20,25,2,1).
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1)])


def test_lg_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown chunking mode"):
        merge_chunks([(0.0, 5.0)], max_chunk_secs=30.0, mode="invalid")


def test_lg1_total_fits_single_chunk():
    chunks = merge_chunks(
        [(0.0, 5.0), (6.0, 10.0)],
        max_chunk_secs=30.0,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


def test_lg2_simple_cut_at_longest_gap():
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)],
        max_chunk_secs=20.0,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1)])


def test_lg3_recursive_splits():
    chunks = merge_chunks(
        [(0.0, 5.0), (7.0, 10.0), (20.0, 25.0), (40.0, 50.0)],
        max_chunk_secs=15.0,
        mode="longest_gap",
    )
    _check_chunks(
        chunks,
        [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1), (40.0, 50.0, 3, 1)],
    )


def test_lg4_single_seg_too_long_falls_back_to_hard_split():
    chunks = merge_chunks([(0.0, 100.0)], max_chunk_secs=30.0, mode="longest_gap")
    _check_chunks(
        chunks,
        [(0.0, 30.0, 0, 1), (30.0, 60.0, 0, 1), (60.0, 90.0, 0, 1), (90.0, 100.0, 0, 1)],
    )


def test_lg5_tie_break_leftmost_gap():
    chunks = merge_chunks(
        [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)],
        max_chunk_secs=10.0,
        mode="longest_gap",
    )
    _check_chunks(
        chunks,
        [(0.0, 5.0, 0, 1), (10.0, 15.0, 1, 1), (20.0, 25.0, 2, 1)],
    )


def test_lg6_max_gap_secs_honored_in_longest_gap_mode():
    """Both modes honor max_gap_secs; longest_gap cuts at the longest gap that
    violates the constraint."""
    chunks = merge_chunks(
        [(0.0, 5.0), (6.0, 10.0)],
        max_chunk_secs=30.0,
        max_gap_secs=0.1,  # gap=1.0 > 0.1 → must split
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1), (6.0, 10.0, 1, 1)])


def test_lg6b_max_gap_secs_inside_fitting_span():
    """Span fits max_chunk_secs, but max_gap_secs forces an internal split."""
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (15.0, 25.0)],
        max_chunk_secs=30.0,
        max_gap_secs=4.0,  # gap 5 > 4 → split at the longer gap
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (15.0, 25.0, 2, 1)])


def test_lg7_filter_then_merge_then_split():
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 5.0), (5.1, 10.0), (20.0, 30.0)],
        max_chunk_secs=15.0,
        min_speech_secs=0.5,
        min_silence_secs=0.5,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(1.0, 10.0, 0, 1), (20.0, 30.0, 1, 1)])


def test_lg8_pad_applied():
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 15.0)],
        max_chunk_secs=10.0,
        pad_onset_secs=0.5,
        pad_offset_secs=0.5,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 5.5, 0, 1), (7.5, 15.5, 1, 1)])


def test_lg9_empty_input():
    assert merge_chunks([], max_chunk_secs=30.0, mode="longest_gap") == []


def test_lg10_single_seg_fits():
    chunks = merge_chunks([(0.0, 30.0)], max_chunk_secs=30.0, mode="longest_gap")
    _check_chunks(chunks, [(0.0, 30.0, 0, 1)])


def test_lg_int_mode_accepted():
    """Integer mode value (advanced/internal use) also works."""
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)],
        max_chunk_secs=20.0,
        mode=1,  # OMNI_CHUNK_LONGEST_GAP
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1)])
