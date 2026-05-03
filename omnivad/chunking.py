"""Pure-algorithm chunking utility.

Wraps the C function ``omni_merge_chunks`` from libomnivad.

WhisperX-style binarize+merge, but the binarize half is skipped because
OmniVAD already returns binarized timestamps.

Public entry point::

    from omnivad import merge_chunks

    chunks = merge_chunks(
        timestamps=[(0.0, 5.0), (6.0, 10.0)],
        chunk_size=30.0,
        max_gap=2.0,
    )
    # [{"start": 0.0, "end": 10.0, "seg_start_idx": 0, "seg_count": 2}]
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from omnivad._binding import (
    OMNI_CHUNK_GREEDY,
    OMNI_CHUNK_LONGEST_GAP,
    OmniChunk,
    OmniChunkConfig,
    OmniSegment,
    _check,
    _lib,
)

__all__ = ["ChunkResult", "merge_chunks", "default_chunk_config"]

# Public mode aliases (string -> int) for ergonomic kwarg usage.
_MODE_ALIASES = {
    "greedy": OMNI_CHUNK_GREEDY,
    "longest_gap": OMNI_CHUNK_LONGEST_GAP,
}


@dataclass(frozen=True)
class ChunkResult:
    """A single chunk produced by :func:`merge_chunks`.

    Attributes
    ----------
    start : float
        Chunk start time (seconds), with ``pad_onset`` applied and clamped to ``>= 0``.
    end : float
        Chunk end time (seconds), with ``pad_offset`` applied.
    seg_start_idx : int
        Index of the first input segment included in this chunk. Refers to
        the *post-filter* segment list — segments dropped by ``min_duration_on``
        and pre-merged by ``min_duration_off`` are not counted.
    seg_count : int
        Number of input segments included in this chunk.
    """

    start: float
    end: float
    seg_start_idx: int
    seg_count: int

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "seg_start_idx": self.seg_start_idx,
            "seg_count": self.seg_count,
        }


def default_chunk_config() -> OmniChunkConfig:
    """Return the C-side default chunking config.

    Defaults::

        chunk_size       = 30.0      # seconds; matches Whisper input window
        max_gap          = math.inf  # disabled
        pad_onset        = 0.04
        pad_offset       = 0.04
        min_duration_on  = 0.0
        min_duration_off = 0.24
        mode             = OMNI_CHUNK_GREEDY
    """
    return _lib.omni_chunk_config_default()


def merge_chunks(
    timestamps: Sequence[tuple[float, float]] | Iterable[tuple[float, float]],
    chunk_size: float = 30.0,
    *,
    max_gap: float = math.inf,
    pad_onset: float = 0.0,
    pad_offset: float = 0.0,
    min_duration_on: float = 0.0,
    min_duration_off: float = 0.0,
    mode: str | int = "greedy",
    as_dict: bool = False,
) -> List[ChunkResult] | List[dict]:
    """Merge speech segments into duration-bounded chunks.

    Parameters
    ----------
    timestamps : sequence of (start, end)
        Input speech segments in seconds, sorted by ``start``. Typically the
        ``timestamps`` field of :meth:`OmniVAD.detect`'s return.
    chunk_size : float
        Hard upper bound on chunk duration (seconds). Must be > 0.
    max_gap : float
        Split if the gap between two adjacent segments exceeds this.
        Defaults to ``math.inf`` (no gap-based splitting).
        Honored by both ``'greedy'`` and ``'longest_gap'`` modes.
    pad_onset : float
        Extend each chunk start backward by this many seconds (clamped to >= 0).
    pad_offset : float
        Extend each chunk end forward by this many seconds.
    min_duration_on : float
        Drop input segments shorter than this many seconds.
    min_duration_off : float
        Pre-merge consecutive segments whose silence gap is shorter than this.
    mode : {'greedy', 'longest_gap'} or int
        Chunk packing strategy. Both modes honor ``chunk_size`` and
        ``max_gap`` as hard constraints — they only differ in WHERE to cut
        when forced to.

        - ``'greedy'`` (default) — sequential append; cuts at the FIRST
          point that violates a constraint. **Recommended for fixed-length
          -input ASR** like Whisper / whisperX (which pad to 30s anyway)
          — packs each chunk close to ``chunk_size`` to minimize wasted
          padding.
        - ``'longest_gap'`` — recursive split at the LONGEST internal
          pause until every chunk satisfies both constraints. Falls back
          to equal hard-split when a single segment alone exceeds
          ``chunk_size``. **Recommended for variable-length-input models**
          (forced alignment, TTS, encoder-style ASR) — splits at natural
          pauses; no fixed-length padding required, so chunks of unequal
          length are fine.
    as_dict : bool
        If True, return a list of plain dicts instead of :class:`ChunkResult`.

    Returns
    -------
    list of :class:`ChunkResult` (or dicts when ``as_dict=True``).
    """
    if isinstance(mode, str):
        try:
            mode_int = _MODE_ALIASES[mode]
        except KeyError:
            raise ValueError(f"Unknown chunking mode {mode!r}. Expected one of {sorted(_MODE_ALIASES)}.") from None
    else:
        mode_int = int(mode)

    seg_list = list(timestamps)
    n = len(seg_list)

    seg_array_t = OmniSegment * n if n > 0 else OmniSegment * 0
    seg_array = seg_array_t()
    for i, (s, e) in enumerate(seg_list):
        seg_array[i].start = float(s)
        seg_array[i].end = float(e)

    cfg = OmniChunkConfig(
        chunk_size=float(chunk_size),
        max_gap=float(max_gap),
        pad_onset=float(pad_onset),
        pad_offset=float(pad_offset),
        min_duration_on=float(min_duration_on),
        min_duration_off=float(min_duration_off),
        mode=mode_int,
    )

    out_ptr = ctypes.POINTER(OmniChunk)()
    out_count = ctypes.c_int(0)

    seg_ptr = ctypes.cast(seg_array, ctypes.POINTER(OmniSegment)) if n > 0 else None
    rc = _lib.omni_merge_chunks(
        seg_ptr,
        n,
        ctypes.byref(cfg),
        ctypes.byref(out_ptr),
        ctypes.byref(out_count),
    )
    _check(rc)

    count = out_count.value
    try:
        results: List[ChunkResult] = []
        for i in range(count):
            c = out_ptr[i]
            results.append(
                ChunkResult(
                    start=float(c.start),
                    end=float(c.end),
                    seg_start_idx=int(c.seg_start_idx),
                    seg_count=int(c.seg_count),
                )
            )
    finally:
        if out_ptr:
            _lib.omni_free(out_ptr)

    if as_dict:
        return [r.to_dict() for r in results]
    return results
