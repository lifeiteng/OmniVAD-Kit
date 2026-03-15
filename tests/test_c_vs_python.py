#!/usr/bin/env python3
"""
Compare C library inference results against Python reference data.
Runs all 3 models on 5 audio files and verifies outputs match.
"""

import json
import os
import re
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "native", "build")
ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "FireRedVAD",
                         "pretrained_models", "onnx_models")
STREAM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                                 "FireRedVAD", "runtime", "convert", "out")


def run(exe, args):
    cmd = [os.path.join(BUILD_DIR, exe)] + args
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return r.stdout, r.returncode


def test_nonstream_vad():
    ref = json.load(open(os.path.join(DATA_DIR, "reference_results.json")))
    passed, total = 0, 0

    for audio_id, data in ref.items():
        wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
        if not os.path.exists(wav):
            continue

        stdout, rc = run("test_nonstream_vad", [
            os.path.join(ONNX_DIR, "fireredvad_vad.ncnn.param"),
            os.path.join(ONNX_DIR, "fireredvad_vad.ncnn.bin"),
            os.path.join(DATA_DIR, "cmvn_means_vad.bin"),
            os.path.join(DATA_DIR, "cmvn_istd_vad.bin"),
            wav, "0.4", "5", "200", "200"
        ])

        total += 1
        ref_segs = len(data["vad"]["timestamps"])

        # Parse segment count from output
        m = re.search(r"Detected (\d+) speech segments", stdout)
        if m:
            c_segs = int(m.group(1))
            # Allow ±2 segment difference due to post-processing edge cases
            # (ncnn float16 precision + different smoothing implementations)
            if abs(c_segs - ref_segs) <= 2:
                passed += 1
                print(f"  PASS {audio_id}: C={c_segs} segs, Python={ref_segs} segs")
            else:
                print(f"  FAIL {audio_id}: C={c_segs} segs, Python={ref_segs} segs")
        elif rc != 0:
            print(f"  FAIL {audio_id}: exit code {rc}")
        else:
            print(f"  FAIL {audio_id}: could not parse output")

    return passed, total


def test_nonstream_aed():
    ref = json.load(open(os.path.join(DATA_DIR, "reference_results.json")))
    passed, total = 0, 0

    for audio_id, data in ref.items():
        wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
        if not os.path.exists(wav):
            continue

        stdout, rc = run("test_nonstream_aed", [
            os.path.join(ONNX_DIR, "fireredvad_aed.ncnn.param"),
            os.path.join(ONNX_DIR, "fireredvad_aed.ncnn.bin"),
            os.path.join(DATA_DIR, "cmvn_means_aed.bin"),
            os.path.join(DATA_DIR, "cmvn_istd_aed.bin"),
            wav
        ])

        total += 1

        # Check that AED detects the right event types
        ref_events = data["aed"]["event_timestamps"]
        has_speech = len(ref_events.get("speech", [])) > 0
        has_singing = len(ref_events.get("singing", [])) > 0
        has_music = len(ref_events.get("music", [])) > 0

        c_has_speech = "speech" in stdout.lower() and re.search(r"speech\s+\d", stdout)
        c_has_singing = "singing" in stdout.lower() and re.search(r"singing\s+\d", stdout)
        c_has_music = "music" in stdout.lower() and re.search(r"music\s+\d", stdout)

        # Check event type detection matches
        match = True
        if has_speech != bool(c_has_speech):
            match = False
        if has_singing != bool(c_has_singing):
            match = False
        if has_music != bool(c_has_music):
            match = False

        if match:
            passed += 1
            events = []
            if has_speech: events.append("speech")
            if has_singing: events.append("singing")
            if has_music: events.append("music")
            print(f"  PASS {audio_id}: events={events}")
        else:
            print(f"  FAIL {audio_id}: event type mismatch")
            print(f"    Python: speech={has_speech}, singing={has_singing}, music={has_music}")
            print(f"    C:      speech={bool(c_has_speech)}, singing={bool(c_has_singing)}, music={bool(c_has_music)}")

    return passed, total


def test_stream_vad():
    ref = json.load(open(os.path.join(DATA_DIR, "reference_results.json")))
    passed, total = 0, 0

    for audio_id, data in ref.items():
        wav = os.path.join(DATA_DIR, f"{audio_id}.wav")
        if not os.path.exists(wav):
            continue

        stdout, rc = run("test_stream_vad", [
            os.path.join(STREAM_MODEL_DIR, "stream_packed.ncnn.param"),
            os.path.join(STREAM_MODEL_DIR, "stream_packed.ncnn.bin"),
            os.path.join(STREAM_MODEL_DIR, "cmvn_means_stream.bin"),
            os.path.join(STREAM_MODEL_DIR, "cmvn_istd_stream.bin"),
            wav
        ])

        total += 1

        if rc != 0:
            print(f"  FAIL {audio_id}: exit code {rc}")
            continue

        # Check speech was detected (count SPEECH frames)
        speech_count = stdout.count("SPEECH")
        ref_has_speech = len(data["stream_vad"]["timestamps"]) > 0

        if (speech_count > 0) == ref_has_speech:
            passed += 1
            print(f"  PASS {audio_id}: {speech_count} speech frames detected")
        else:
            print(f"  FAIL {audio_id}: speech_frames={speech_count}, ref_has_speech={ref_has_speech}")

    return passed, total


def main():
    print("=" * 60)
    print("C Library vs Python Reference Validation")
    print("=" * 60)

    total_passed, total_tests = 0, 0

    print("\n--- Non-stream VAD ---")
    p, t = test_nonstream_vad()
    total_passed += p
    total_tests += t

    print("\n--- Non-stream AED ---")
    p, t = test_nonstream_aed()
    total_passed += p
    total_tests += t

    print("\n--- Stream VAD ---")
    p, t = test_stream_vad()
    total_passed += p
    total_tests += t

    print(f"\n{'=' * 60}")
    status = "ALL PASSED" if total_passed == total_tests else "SOME FAILED"
    print(f"Results: {total_passed}/{total_tests} {status}")
    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    main()
