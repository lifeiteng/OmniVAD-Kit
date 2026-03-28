#!/usr/bin/env python3
"""
Simple VAD example — detect speech segments in an audio file.

Usage:
    python examples/simple_vad.py audio.wav
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "FireRedVAD"))

from fireredvad import FireRedVad, FireRedVadConfig


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio.wav>")
        sys.exit(1)

    wav_path = sys.argv[1]

    # Create VAD with default config
    config = FireRedVadConfig(
        speech_threshold=0.4,
        smooth_window_size=5,
        min_speech_frame=20,  # 200ms minimum speech
        max_speech_frame=2000,  # 20s maximum segment
        min_silence_frame=20,  # 200ms minimum silence to split
    )

    model_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "FireRedVAD", "pretrained_models", "FireRedVAD", "VAD"
    )
    vad = FireRedVad.from_pretrained(model_dir, config)

    # Detect
    result, probs = vad.detect(wav_path)

    # Print results
    print(f"Duration: {result['dur']}s")
    print(f"Speech segments: {len(result['timestamps'])}")
    for i, (start, end) in enumerate(result["timestamps"]):
        print(f"  [{i + 1}] {start:.3f}s - {end:.3f}s  ({end - start:.3f}s)")


if __name__ == "__main__":
    main()
