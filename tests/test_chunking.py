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
    assert cfg.chunk_size == pytest.approx(30.0)
    assert math.isinf(cfg.max_gap)
    assert cfg.pad_onset == pytest.approx(0.04)
    assert cfg.pad_offset == pytest.approx(0.04)
    assert cfg.min_duration_on == pytest.approx(0.0)
    assert cfg.min_duration_off == pytest.approx(0.24)


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
    chunks = merge_chunks([(0.0, 5.0), (6.0, 10.0)], chunk_size=30.0)
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


# --------------------------------------------------------------------------- #
#  Scenario 2: long audio multiple splits                                      #
# --------------------------------------------------------------------------- #


def test_long_audio_multiple_splits():
    chunks = merge_chunks(
        [(0.0, 10.0), (11.0, 20.0), (21.0, 30.0), (31.0, 40.0)],
        chunk_size=20.0,
    )
    _check_chunks(chunks, [(0.0, 20.0, 0, 2), (21.0, 40.0, 2, 2)])


# --------------------------------------------------------------------------- #
#  Scenario 3: gap > max_gap forces split                                      #
# --------------------------------------------------------------------------- #


def test_gap_split():
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)],
        chunk_size=30.0,
        max_gap=2.0,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1), (8.0, 10.0, 1, 1), (20.0, 25.0, 2, 1)])


# --------------------------------------------------------------------------- #
#  Scenario 4: single segment > chunk_size -> equal hard-split                 #
# --------------------------------------------------------------------------- #


def test_single_segment_hard_split():
    chunks = merge_chunks([(0.0, 100.0)], chunk_size=30.0)
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
    chunks = merge_chunks([], chunk_size=30.0)
    assert chunks == []


# --------------------------------------------------------------------------- #
#  Scenario 6: min_duration_on filters short segments                          #
# --------------------------------------------------------------------------- #


def test_min_duration_on_filter():
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 5.0)],
        chunk_size=30.0,
        min_duration_on=0.5,
    )
    _check_chunks(chunks, [(1.0, 5.0, 0, 1)])


# --------------------------------------------------------------------------- #
#  Scenario 7: min_duration_off merges close-by segments                       #
# --------------------------------------------------------------------------- #


def test_min_duration_off_merge():
    chunks = merge_chunks(
        [(0.0, 5.0), (5.1, 10.0)],
        chunk_size=30.0,
        min_duration_off=0.5,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 1)])


# --------------------------------------------------------------------------- #
#  Scenario 8: pad_onset / pad_offset                                          #
# --------------------------------------------------------------------------- #


def test_pad_applied():
    chunks = merge_chunks(
        [(5.0, 10.0)],
        chunk_size=30.0,
        pad_onset=0.5,
        pad_offset=0.5,
    )
    _check_chunks(chunks, [(4.5, 10.5, 0, 1)])


def test_pad_onset_clamped_to_zero():
    chunks = merge_chunks(
        [(0.1, 5.0)],
        chunk_size=30.0,
        pad_onset=0.5,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1)])


# --------------------------------------------------------------------------- #
#  Boundary: chunk_size <= 0 raises                                            #
# --------------------------------------------------------------------------- #


def test_invalid_chunk_size_raises():
    with pytest.raises(RuntimeError, match=r"INVALID_ARG|invalid"):
        merge_chunks([(0.0, 5.0)], chunk_size=0.0)
    with pytest.raises(RuntimeError, match=r"INVALID_ARG|invalid"):
        merge_chunks([(0.0, 5.0)], chunk_size=-1.0)


# --------------------------------------------------------------------------- #
#  as_dict variant returns plain dicts                                         #
# --------------------------------------------------------------------------- #


def test_as_dict_returns_plain_dicts():
    chunks = merge_chunks([(0.0, 5.0)], chunk_size=30.0, as_dict=True)
    assert chunks == [{"start": 0.0, "end": 5.0, "seg_start_idx": 0, "seg_count": 1}]


# --------------------------------------------------------------------------- #
#  ChunkResult.to_dict()                                                       #
# --------------------------------------------------------------------------- #


def test_chunk_result_to_dict():
    [c] = merge_chunks([(1.0, 2.0)], chunk_size=30.0)
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
    assert OmniChunkConfig.chunk_size.offset == 0
    assert OmniChunkConfig.max_gap.offset == 4
    assert OmniChunkConfig.pad_onset.offset == 8
    assert OmniChunkConfig.pad_offset.offset == 12
    assert OmniChunkConfig.min_duration_on.offset == 16
    assert OmniChunkConfig.min_duration_off.offset == 20
    assert OmniChunkConfig.mode.offset == 24


# --------------------------------------------------------------------------- #
#  Python convenience-default vs C/TS canonical-default DRIFT                  #
# --------------------------------------------------------------------------- #


def test_python_convenience_defaults_differ_from_canonical():
    """Lock the (intentional?) drift between Python convenience defaults and
    the C/TS canonical defaults.

    ``merge_chunks(timestamps)`` (no kwargs) uses Python defaults
    ``pad=0, min_duration_off=0`` — i.e. NO padding and NO silence pre-merge.

    ``default_chunk_config()`` (which the C and TS bindings expose as the
    ``DEFAULT_CHUNK_CONFIG``) uses ``pad=0.04, min_duration_off=0.24``.

    Calling ``merge_chunks([...])`` in Python WILL produce different output
    from the equivalent TS call ``mergeChunks([...])`` (no options). This
    test pins the discrepancy so any future "fix" is intentional.
    """
    # Same input, two paths:
    via_python_defaults = merge_chunks([(0.0, 5.0), (5.1, 10.0)])
    canonical = default_chunk_config()
    via_canonical = merge_chunks(
        [(0.0, 5.0), (5.1, 10.0)],
        chunk_size=canonical.chunk_size,
        max_gap=canonical.max_gap,
        pad_onset=canonical.pad_onset,
        pad_offset=canonical.pad_offset,
        min_duration_on=canonical.min_duration_on,
        min_duration_off=canonical.min_duration_off,
    )

    # Python defaults: no padding, no merge → 2 segments survive as 2 segs in 1 chunk.
    assert via_python_defaults == [ChunkResult(start=0.0, end=10.0, seg_start_idx=0, seg_count=2)]

    # Canonical defaults: gap 0.1 < min_off 0.24 → pre-merged into 1 seg, then padded.
    assert len(via_canonical) == 1
    assert via_canonical[0].seg_count == 1  # pre-merged
    assert via_canonical[0].start == pytest.approx(0.0)  # 0 - 0.04 → clamped 0
    assert via_canonical[0].end == pytest.approx(10.04)  # 10 + 0.04


def test_default_chunk_config_matches_ts_constants():
    """default_chunk_config() must match wasm-binding.ts DEFAULT_CHUNK_CONFIG."""
    cfg = default_chunk_config()
    assert cfg.chunk_size == pytest.approx(30.0)
    assert cfg.pad_onset == pytest.approx(0.04)
    assert cfg.pad_offset == pytest.approx(0.04)
    assert cfg.min_duration_on == pytest.approx(0.0)
    assert cfg.min_duration_off == pytest.approx(0.24)
    assert math.isinf(cfg.max_gap)


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


def test_chunk_size_nan_rejected():
    """NaN > 0.0 is False → guard rejects."""
    with pytest.raises(RuntimeError, match=r"INVALID_ARG|invalid"):
        merge_chunks([(0.0, 5.0)], chunk_size=float("nan"))


def test_chunk_size_inf_accepted():
    """+Inf > 0 is True → accepted; everything fits in one chunk."""
    chunks = merge_chunks([(0.0, 5.0), (10.0, 20.0)], chunk_size=float("inf"))
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
        chunk_size=30.0,
        max_gap=0.55,
        min_duration_on=0.2,
        min_duration_off=0.5,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1), (5.6, 10.0, 1, 1)])


def test_seg_idx_after_filter_and_merge():
    """seg_start_idx counts on POST-filter+merge view, not raw input."""
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 5.0), (5.1, 10.0), (20.0, 25.0)],
        chunk_size=20.0,
        min_duration_on=0.5,
        min_duration_off=0.5,
    )
    _check_chunks(chunks, [(1.0, 10.0, 0, 1), (20.0, 25.0, 1, 1)])


def test_min_duration_on_drops_all():
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 1.05)],
        chunk_size=30.0,
        min_duration_on=1.0,
    )
    assert chunks == []


def test_min_duration_off_cascade_max_end():
    """Cascade-merge takes max(end) when next.end < cur.end."""
    chunks = merge_chunks(
        [(0.0, 10.0), (0.1, 5.0), (0.2, 8.0)],
        chunk_size=30.0,
        min_duration_off=0.5,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 1)])


def test_chunk_size_equals_segment_dur():
    """`<=` boundary in Step 4: dur==chunk_size NOT split."""
    chunks = merge_chunks([(0.0, 30.0)], chunk_size=30.0)
    _check_chunks(chunks, [(0.0, 30.0, 0, 1)])


def test_max_gap_equals_real_gap():
    """`>` strict boundary: gap==max_gap NOT split."""
    chunks = merge_chunks(
        [(0.0, 5.0), (7.0, 10.0)],
        chunk_size=30.0,
        max_gap=2.0,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


def test_min_duration_off_equals_real_gap():
    """`<` strict boundary: gap==min_duration_off NOT merged."""
    chunks = merge_chunks(
        [(0.0, 5.0), (5.5, 10.0)],
        chunk_size=30.0,
        min_duration_off=0.5,
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


def test_min_duration_on_equals_segment_dur():
    """`>=` boundary: dur==min_duration_on KEPT."""
    chunks = merge_chunks(
        [(0.0, 0.5), (1.0, 5.0)],
        chunk_size=30.0,
        min_duration_on=0.5,
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 2)])


def test_pad_chunks_may_overlap():
    """Algorithm does not de-dupe overlap caused by padding."""
    chunks = merge_chunks(
        [(0.0, 5.0), (6.0, 10.0)],
        chunk_size=30.0,
        max_gap=0.5,
        pad_onset=2.0,
        pad_offset=2.0,
    )
    _check_chunks(chunks, [(0.0, 7.0, 0, 1), (4.0, 12.0, 1, 1)])


# --------------------------------------------------------------------------- #
#  LONGEST_GAP mode — mirror native/test/test_chunking.cpp scenarios LG1-LG10  #
# --------------------------------------------------------------------------- #


def test_lg_default_mode_is_greedy():
    """Backward compatibility: omitting `mode` must give GREEDY behaviour."""
    chunks = merge_chunks([(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)], chunk_size=20.0)
    # GREEDY: cur=(0,5), accept (8,10) cur=(0,10) [10≤20], accept (20,25)
    # would_exceed (25-0=25>20) split → (0,10,0,2) + (20,25,2,1).
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1)])


def test_lg_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown chunking mode"):
        merge_chunks([(0.0, 5.0)], chunk_size=30.0, mode="invalid")


def test_lg1_total_fits_single_chunk():
    chunks = merge_chunks(
        [(0.0, 5.0), (6.0, 10.0)],
        chunk_size=30.0,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2)])


def test_lg2_simple_cut_at_longest_gap():
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)],
        chunk_size=20.0,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1)])


def test_lg3_recursive_splits():
    chunks = merge_chunks(
        [(0.0, 5.0), (7.0, 10.0), (20.0, 25.0), (40.0, 50.0)],
        chunk_size=15.0,
        mode="longest_gap",
    )
    _check_chunks(
        chunks,
        [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1), (40.0, 50.0, 3, 1)],
    )


def test_lg4_single_seg_too_long_falls_back_to_hard_split():
    chunks = merge_chunks([(0.0, 100.0)], chunk_size=30.0, mode="longest_gap")
    _check_chunks(
        chunks,
        [(0.0, 30.0, 0, 1), (30.0, 60.0, 0, 1), (60.0, 90.0, 0, 1), (90.0, 100.0, 0, 1)],
    )


def test_lg5_tie_break_leftmost_gap():
    chunks = merge_chunks(
        [(0.0, 5.0), (10.0, 15.0), (20.0, 25.0)],
        chunk_size=10.0,
        mode="longest_gap",
    )
    _check_chunks(
        chunks,
        [(0.0, 5.0, 0, 1), (10.0, 15.0, 1, 1), (20.0, 25.0, 2, 1)],
    )


def test_lg6_max_gap_honored_in_longest_gap_mode():
    """Both modes honor max_gap; longest_gap cuts at the longest gap that
    violates the constraint."""
    chunks = merge_chunks(
        [(0.0, 5.0), (6.0, 10.0)],
        chunk_size=30.0,
        max_gap=0.1,  # gap=1.0 > 0.1 → must split
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 5.0, 0, 1), (6.0, 10.0, 1, 1)])


def test_lg6b_max_gap_inside_fitting_span():
    """Span fits chunk_size, but max_gap forces an internal split."""
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (15.0, 25.0)],
        chunk_size=30.0,
        max_gap=4.0,  # gap 5 > 4 → split at the longer gap
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (15.0, 25.0, 2, 1)])


def test_lg7_filter_then_merge_then_split():
    chunks = merge_chunks(
        [(0.0, 0.1), (1.0, 5.0), (5.1, 10.0), (20.0, 30.0)],
        chunk_size=15.0,
        min_duration_on=0.5,
        min_duration_off=0.5,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(1.0, 10.0, 0, 1), (20.0, 30.0, 1, 1)])


def test_lg8_pad_applied():
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 15.0)],
        chunk_size=10.0,
        pad_onset=0.5,
        pad_offset=0.5,
        mode="longest_gap",
    )
    _check_chunks(chunks, [(0.0, 5.5, 0, 1), (7.5, 15.5, 1, 1)])


def test_lg9_empty_input():
    assert merge_chunks([], chunk_size=30.0, mode="longest_gap") == []


def test_lg10_single_seg_fits():
    chunks = merge_chunks([(0.0, 30.0)], chunk_size=30.0, mode="longest_gap")
    _check_chunks(chunks, [(0.0, 30.0, 0, 1)])


def test_lg_int_mode_accepted():
    """Integer mode value (advanced/internal use) also works."""
    chunks = merge_chunks(
        [(0.0, 5.0), (8.0, 10.0), (20.0, 25.0)],
        chunk_size=20.0,
        mode=1,  # OMNI_CHUNK_LONGEST_GAP
    )
    _check_chunks(chunks, [(0.0, 10.0, 0, 2), (20.0, 25.0, 2, 1)])
