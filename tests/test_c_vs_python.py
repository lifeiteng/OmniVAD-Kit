#!/usr/bin/env python3
"""
Compare omnivad (C via ctypes) results against Python reference data.
Validates that the C library produces timestamps matching the original
FireRedVAD Python implementation.

No external dependencies — uses bundled .omnivad models.
"""

import json
import os

import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REF_PATH = os.path.join(DATA_DIR, "reference_results.json")

# Tolerance: ncnn fp16/fp32 differences can shift timestamps by up to 80ms
# (observed on event.wav AED speech segment boundary)
TS_TOLERANCE = 0.1  # seconds


@pytest.fixture(scope="module")
def reference():
    with open(REF_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def vad():
    from omnivad import OmniVAD

    return OmniVAD()


@pytest.fixture(scope="module")
def aed():
    from omnivad import OmniAED

    return OmniAED()


def audio_ids(ref):
    """Return audio IDs that have both reference data and wav files."""
    ids = []
    for audio_id in ref:
        wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
        if os.path.exists(wav):
            ids.append(audio_id)
    return ids


def _load_ref():
    with open(REF_PATH) as f:
        return json.load(f)


_ref = _load_ref()
_ids = audio_ids(_ref)


@pytest.mark.parametrize("audio_id", _ids)
def test_vad_segment_count(vad, reference, audio_id):
    """VAD segment count should match reference (±1 tolerance)."""
    wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
    result = vad.detect(wav)
    ref_segs = reference[audio_id]["vad"]["timestamps"]
    assert abs(len(result["timestamps"]) - len(ref_segs)) <= 1, (
        f"segment count: got {len(result['timestamps'])}, ref {len(ref_segs)}"
    )


@pytest.mark.parametrize("audio_id", _ids)
def test_vad_timestamps(vad, reference, audio_id):
    """VAD timestamps should be within tolerance of reference."""
    wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
    result = vad.detect(wav)
    ref_segs = reference[audio_id]["vad"]["timestamps"]
    n = min(len(result["timestamps"]), len(ref_segs))
    for i in range(n):
        got_start, got_end = result["timestamps"][i]
        ref_start, ref_end = ref_segs[i]
        assert abs(got_start - ref_start) <= TS_TOLERANCE, f"seg[{i}] start: got {got_start}, ref {ref_start}"
        assert abs(got_end - ref_end) <= TS_TOLERANCE, f"seg[{i}] end: got {got_end}, ref {ref_end}"


@pytest.mark.parametrize("audio_id", _ids)
def test_aed_event_types(aed, reference, audio_id):
    """AED should detect the same event types as reference."""
    wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
    result = aed.detect(wav)
    ref_events = reference[audio_id]["aed"]["event_timestamps"]
    for cls in ("speech", "singing", "music"):
        ref_has = len(ref_events.get(cls, [])) > 0
        got_has = len(result["events"].get(cls, [])) > 0
        assert got_has == ref_has, (
            f"{cls}: got {'present' if got_has else 'absent'}, ref {'present' if ref_has else 'absent'}"
        )


@pytest.mark.parametrize("audio_id", _ids)
def test_aed_timestamps(aed, reference, audio_id):
    """AED timestamps should be within tolerance of reference."""
    wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
    result = aed.detect(wav)
    ref_events = reference[audio_id]["aed"]["event_timestamps"]
    for cls in ("speech", "singing", "music"):
        got = result["events"].get(cls, [])
        ref = ref_events.get(cls, [])
        n = min(len(got), len(ref))
        for i in range(n):
            assert abs(got[i][0] - ref[i][0]) <= TS_TOLERANCE, f"{cls}[{i}] start: got {got[i][0]}, ref {ref[i][0]}"
            assert abs(got[i][1] - ref[i][1]) <= TS_TOLERANCE, f"{cls}[{i}] end: got {got[i][1]}, ref {ref[i][1]}"


@pytest.mark.parametrize("audio_id", _ids)
def test_duration_matches(vad, reference, audio_id):
    """Reported duration should match reference."""
    wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
    result = vad.detect(wav)
    ref_dur = reference[audio_id]["vad"]["duration"]
    assert abs(result["duration"] - ref_dur) < 0.01, f"duration: got {result['duration']}, ref {ref_dur}"
