"""Chunked processing for large audio files with overlap and aggregation."""

import numpy as np


def split_chunks(total_samples, chunk_samples, overlap_samples):
    """Generate (start, end) sample indices for overlapping chunks.

    Returns list of (start_sample, end_sample) tuples.
    """
    chunks = []
    step = chunk_samples - overlap_samples
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append((start, end))
        if end >= total_samples:
            break
        start += step
    return chunks


def aggregate_segments(chunk_results, overlap_seconds, merge_gap=0.05):
    """Aggregate segments from overlapping chunks.

    Strategy: for each overlap region, use the midpoint as boundary.
    Chunk N contributes segments ending before midpoint, chunk N+1
    contributes segments starting after midpoint.
    Then merge segments closer than merge_gap.

    Parameters
    ----------
    chunk_results : list of (offset_seconds, segments)
        Each entry: offset is chunk start time, segments is [(start, end), ...]
        where start/end are relative to chunk start.
    overlap_seconds : float
        Overlap duration between consecutive chunks.
    merge_gap : float
        Merge segments with gap smaller than this (seconds).

    Returns
    -------
    list of (start, end) tuples in absolute time.
    """
    if not chunk_results:
        return []

    all_segments = []
    half_overlap = overlap_seconds / 2.0

    for i, (offset, segments) in enumerate(chunk_results):
        for start, end in segments:
            abs_start = offset + start
            abs_end = offset + end

            # First chunk: keep segments ending before midpoint of trailing overlap
            # Last chunk: keep segments starting after midpoint of leading overlap
            # Middle chunks: both constraints
            keep = True
            if i > 0:
                # Leading overlap with previous chunk
                leading_mid = offset + half_overlap
                if abs_end <= leading_mid:
                    keep = False
                elif abs_start < leading_mid:
                    abs_start = max(abs_start, leading_mid)
            if i < len(chunk_results) - 1:
                next_offset = chunk_results[i + 1][0]
                trailing_mid = next_offset + half_overlap
                if abs_start >= trailing_mid:
                    keep = False
                elif abs_end > trailing_mid:
                    abs_end = min(abs_end, trailing_mid)

            if keep and abs_end > abs_start:
                all_segments.append((round(abs_start, 3), round(abs_end, 3)))

    all_segments.sort()
    return _merge_close_segments(all_segments, merge_gap)


def _merge_close_segments(segments, merge_gap):
    """Merge segments that overlap or have a gap smaller than merge_gap."""
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def aggregate_aed_events(chunk_results, overlap_seconds, merge_gap=0.05):
    """Aggregate AED events from overlapping chunks.

    Parameters
    ----------
    chunk_results : list of (offset_seconds, events_dict)
        Each events_dict: {'speech': [...], 'singing': [...], 'music': [...]}

    Returns
    -------
    dict of {class_name: [(start, end), ...]}
    """
    per_class = {}
    for cls in ("speech", "singing", "music"):
        class_chunks = [(offset, events.get(cls, [])) for offset, events in chunk_results]
        per_class[cls] = aggregate_segments(class_chunks, overlap_seconds, merge_gap)
    return per_class
