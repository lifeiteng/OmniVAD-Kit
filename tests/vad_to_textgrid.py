#!/usr/bin/env python3
"""
Run FireRedVAD (all 3 models) on audio and output TextGrid files.
Also benchmarks Python vs C (ncnn) speed with RTF.

Usage:
    python tests/vad_to_textgrid.py <audio.wav> [-o output.TextGrid]
    python tests/vad_to_textgrid.py tests/data/event.wav
    python tests/vad_to_textgrid.py tests/data/hello_en.wav -o hello.TextGrid

Output TextGrid has 4 tiers:
    1. VAD        — speech segments from non-stream VAD
    2. StreamVAD  — speech segments from stream VAD
    3. AED-Speech — speech events from AED
    4. AED-Events — all events (speech/singing/music) from AED
"""

import argparse
import os
import re
import subprocess
import sys
import time

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
BUILD_DIR = os.path.join(PROJECT_DIR, "native", "build")
FIREREDVAD_ROOT = os.path.join(PROJECT_DIR, "..", "FireRedVAD")
ONNX_DIR = os.path.join(FIREREDVAD_ROOT, "pretrained_models", "onnx_models")
STREAM_MODEL_DIR = os.path.join(FIREREDVAD_ROOT, "runtime", "convert", "out")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Python model dirs
MODEL_ROOT = os.path.join(FIREREDVAD_ROOT, "pretrained_models", "FireRedVAD")
VAD_DIR = os.path.join(MODEL_ROOT, "VAD")
STREAM_VAD_DIR = os.path.join(MODEL_ROOT, "Stream-VAD")
AED_DIR = os.path.join(MODEL_ROOT, "AED")


def write_textgrid(path, duration, tiers):
    """
    Write a Praat TextGrid file.
    tiers: list of (name, segments) where segments = [(start, end, label), ...]
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write("xmin = 0\n")
        f.write(f"xmax = {duration}\n")
        f.write("tiers? <exists>\n")
        f.write(f"size = {len(tiers)}\n")
        f.write("item []:\n")

        for i, (name, segments) in enumerate(tiers, 1):
            f.write(f"    item [{i}]:\n")
            f.write('        class = "IntervalTier"\n')
            f.write(f'        name = "{name}"\n')
            f.write("        xmin = 0\n")
            f.write(f"        xmax = {duration}\n")

            # Build full interval list (fill gaps with empty)
            intervals = []
            prev_end = 0.0
            for start, end, label in sorted(segments):
                if start > prev_end + 0.001:
                    intervals.append((prev_end, start, ""))
                intervals.append((start, end, label))
                prev_end = end
            if prev_end < duration - 0.001:
                intervals.append((prev_end, duration, ""))

            if not intervals:
                intervals = [(0, duration, "")]

            f.write(f"        intervals: size = {len(intervals)}\n")
            for j, (s, e, label) in enumerate(intervals, 1):
                f.write(f"        intervals [{j}]:\n")
                f.write(f"            xmin = {s}\n")
                f.write(f"            xmax = {e}\n")
                f.write(f'            text = "{label}"\n')


def run_python_vad(wav_path):
    """Run Python VAD and return (result, elapsed_seconds)."""
    sys.path.insert(0, FIREREDVAD_ROOT)
    from fireredvad import FireRedVad, FireRedVadConfig

    config = FireRedVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
    )
    vad = FireRedVad.from_pretrained(VAD_DIR, config)

    t0 = time.perf_counter()
    result, probs = vad.detect(wav_path)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_python_stream_vad(wav_path):
    """Run Python Stream VAD and return (result, elapsed_seconds)."""
    sys.path.insert(0, FIREREDVAD_ROOT)
    from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

    config = FireRedStreamVadConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=20,
    )
    svad = FireRedStreamVad.from_pretrained(STREAM_VAD_DIR, config)

    t0 = time.perf_counter()
    frame_results, result = svad.detect_full(wav_path)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_python_aed(wav_path):
    """Run Python AED and return (result, elapsed_seconds)."""
    sys.path.insert(0, FIREREDVAD_ROOT)
    from fireredvad import FireRedAed, FireRedAedConfig

    config = FireRedAedConfig(
        use_gpu=False,
        smooth_window_size=5,
        speech_threshold=0.4,
        singing_threshold=0.5,
        music_threshold=0.5,
        min_event_frame=20,
        max_event_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
    )
    aed = FireRedAed.from_pretrained(AED_DIR, config)

    t0 = time.perf_counter()
    result, probs = aed.detect(wav_path)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_c_vad(wav_path):
    """Run C non-stream VAD and return (segments, elapsed_seconds)."""
    t0 = time.perf_counter()
    r = subprocess.run(
        [
            os.path.join(BUILD_DIR, "test_nonstream_vad"),
            os.path.join(ONNX_DIR, "fireredvad_vad.ncnn.param"),
            os.path.join(ONNX_DIR, "fireredvad_vad.ncnn.bin"),
            os.path.join(DATA_DIR, "cmvn_means_vad.bin"),
            os.path.join(DATA_DIR, "cmvn_istd_vad.bin"),
            wav_path,
            "0.4",
            "5",
            "200",
            "200",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.perf_counter() - t0

    segments = []
    for line in r.stdout.split("\n"):
        m = re.search(r"\[\s*\d+\]\s+([\d.]+)\s+-\s+([\d.]+)", line)
        if m:
            segments.append((float(m.group(1)), float(m.group(2))))
    return segments, elapsed


def run_c_aed(wav_path):
    """Run C AED and return (events, elapsed_seconds)."""
    t0 = time.perf_counter()
    r = subprocess.run(
        [
            os.path.join(BUILD_DIR, "test_nonstream_aed"),
            os.path.join(ONNX_DIR, "fireredvad_aed.ncnn.param"),
            os.path.join(ONNX_DIR, "fireredvad_aed.ncnn.bin"),
            os.path.join(DATA_DIR, "cmvn_means_aed.bin"),
            os.path.join(DATA_DIR, "cmvn_istd_aed.bin"),
            wav_path,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.perf_counter() - t0

    events = {"speech": [], "singing": [], "music": []}
    for line in r.stdout.split("\n"):
        m = re.search(r"\[\s*\d+\]\s+(speech|singing|music)\s+([\d.]+)\s+-\s+([\d.]+)", line)
        if m:
            events[m.group(1)].append((float(m.group(2)), float(m.group(3))))
    return events, elapsed


def main():
    parser = argparse.ArgumentParser(description="FireRedVAD → TextGrid + RTF benchmark")
    parser.add_argument("audio", help="Input audio file (16kHz WAV)")
    parser.add_argument("-o", "--output", help="Output TextGrid path (default: <audio>.TextGrid)")
    args = parser.parse_args()

    wav_path = os.path.abspath(args.audio)
    if not os.path.exists(wav_path):
        print(f"Error: {wav_path} not found")
        sys.exit(1)

    output_path = args.output or wav_path.rsplit(".", 1)[0] + ".TextGrid"

    # Get duration
    import soundfile as sf

    info = sf.info(wav_path)
    duration = info.duration
    print(f"Audio: {wav_path}")
    print(f"Duration: {duration:.3f}s, Sample rate: {info.samplerate}Hz")
    print()

    # ── Run all models ──
    print("=" * 60)
    print("Running models...")
    print("=" * 60)

    # Python VAD
    print("\n[Python] Non-stream VAD...")
    py_vad_result, py_vad_time = run_python_vad(wav_path)
    py_vad_rtf = py_vad_time / duration
    print(f"  Segments: {len(py_vad_result['timestamps'])}")
    print(f"  Time: {py_vad_time:.4f}s, RTF: {py_vad_rtf:.4f}")

    # Python Stream VAD
    print("\n[Python] Stream VAD...")
    py_svad_result, py_svad_time = run_python_stream_vad(wav_path)
    py_svad_rtf = py_svad_time / duration
    print(f"  Segments: {len(py_svad_result['timestamps'])}")
    print(f"  Time: {py_svad_time:.4f}s, RTF: {py_svad_rtf:.4f}")

    # Python AED
    print("\n[Python] AED...")
    py_aed_result, py_aed_time = run_python_aed(wav_path)
    py_aed_rtf = py_aed_time / duration
    py_aed_events = py_aed_result["event2timestamps"]
    py_aed_ratios = py_aed_result.get("event2ratio", {})
    for event in ["speech", "singing", "music"]:
        ts = py_aed_events.get(event, [])
        ratio = py_aed_ratios.get(event, 0)
        if ts:
            total_dur = sum(e - s for s, e in ts)
            print(f"  {event}: {len(ts)} segments, {total_dur:.2f}s, ratio={ratio:.3f}")
            for i, (s, e) in enumerate(ts):
                print(f"    [{i + 1}] {s:.3f} - {e:.3f}  ({e - s:.3f}s)")
        else:
            print(f"  {event}: 0 segments")
    print(f"  Time: {py_aed_time:.4f}s, RTF: {py_aed_rtf:.6f}")

    # C VAD
    print("\n[C/ncnn] Non-stream VAD...")
    c_vad_segs, c_vad_time = run_c_vad(wav_path)
    c_vad_rtf = c_vad_time / duration
    print(f"  Segments: {len(c_vad_segs)}")
    for i, (s, e) in enumerate(c_vad_segs):
        print(f"    [{i + 1}] {s:.3f} - {e:.3f}  ({e - s:.3f}s)")
    print(f"  Time: {c_vad_time:.4f}s, RTF: {c_vad_rtf:.6f}")

    # C AED
    print("\n[C/ncnn] AED...")
    c_aed_events, c_aed_time = run_c_aed(wav_path)
    c_aed_rtf = c_aed_time / duration
    for event in ["speech", "singing", "music"]:
        ts = c_aed_events.get(event, [])
        if ts:
            total_dur = sum(e - s for s, e in ts)
            print(f"  {event}: {len(ts)} segments, {total_dur:.2f}s")
            for i, (s, e) in enumerate(ts):
                print(f"    [{i + 1}] {s:.3f} - {e:.3f}  ({e - s:.3f}s)")
        else:
            print(f"  {event}: 0 segments")
    print(f"  Time: {c_aed_time:.4f}s, RTF: {c_aed_rtf:.6f}")

    # ── RTF Summary ──
    print(f"\n{'=' * 70}")
    print("RTF Summary (Real-Time Factor, lower = faster)")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Python RTF':>14} {'C/ncnn RTF':>14} {'Speedup':>10}")
    print("-" * 64)
    print(f"{'Non-stream VAD':<20} {py_vad_rtf:>14.6f} {c_vad_rtf:>14.6f} {py_vad_rtf / c_vad_rtf:>9.1f}x")
    print(f"{'AED':<20} {py_aed_rtf:>14.6f} {c_aed_rtf:>14.6f} {py_aed_rtf / c_aed_rtf:>9.1f}x")
    print(f"{'Stream VAD':<20} {py_svad_rtf:>14.6f} {'N/A':>14} {'':>10}")
    print()

    # ── AED Side-by-side Comparison ──
    print(f"{'=' * 70}")
    print("AED Segment Comparison (Python vs C/ncnn)")
    print(f"{'=' * 70}")
    for event in ["speech", "singing", "music"]:
        py_ts = py_aed_events.get(event, [])
        c_ts = c_aed_events.get(event, [])
        print(f"\n  {event}:")
        max_rows = max(len(py_ts), len(c_ts))
        if max_rows == 0:
            print("    (none)")
            continue
        print(f"    {'#':>4}  {'Python':^25}  {'C/ncnn':^25}  {'Δstart':>8}  {'Δend':>8}")
        print(f"    {'':>4}  {'-' * 25}  {'-' * 25}  {'-' * 8}  {'-' * 8}")
        for i in range(max_rows):
            py_str = (
                f"{py_ts[i][0]:.3f} - {py_ts[i][1]:.3f} ({py_ts[i][1] - py_ts[i][0]:.3f}s)" if i < len(py_ts) else "—"
            )
            c_str = f"{c_ts[i][0]:.3f} - {c_ts[i][1]:.3f} ({c_ts[i][1] - c_ts[i][0]:.3f}s)" if i < len(c_ts) else "—"
            if i < len(py_ts) and i < len(c_ts):
                ds = abs(py_ts[i][0] - c_ts[i][0])
                de = abs(py_ts[i][1] - c_ts[i][1])
                print(f"    [{i + 1:>2}]  {py_str:<25}  {c_str:<25}  {ds:>7.3f}s  {de:>7.3f}s")
            else:
                print(f"    [{i + 1:>2}]  {py_str:<25}  {c_str:<25}")
    print()

    # ── Build TextGrid ──
    tiers = []

    # Tier 1: VAD
    vad_segs = [(s, e, "speech") for s, e in py_vad_result["timestamps"]]
    tiers.append(("VAD", vad_segs))

    # Tier 2: Stream VAD
    svad_segs = [(s, e, "speech") for s, e in py_svad_result["timestamps"]]
    tiers.append(("StreamVAD", svad_segs))

    # Tier 3-5: AED per class
    for event in ["speech", "singing", "music"]:
        segs = [(s, e, event) for s, e in py_aed_events.get(event, [])]
        tiers.append((f"AED-{event.capitalize()}", segs))

    # Tier 6: AED all events combined
    aed_all = []
    for event, timestamps in py_aed_events.items():
        for s, e in timestamps:
            aed_all.append((s, e, event))
    aed_all.sort()
    tiers.append(("AED-All", aed_all))

    write_textgrid(output_path, duration, tiers)
    print(f"TextGrid saved: {output_path}")


if __name__ == "__main__":
    main()
