#!/usr/bin/env python3
"""
Simple AED example — detect speech, singing, and music in an audio file.

Usage:
    python examples/simple_aed.py audio.wav
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "FireRedVAD"))

from fireredvad import FireRedAed, FireRedAedConfig


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio.wav>")
        sys.exit(1)

    wav_path = sys.argv[1]

    config = FireRedAedConfig(
        speech_threshold=0.4,
        singing_threshold=0.5,
        music_threshold=0.5,
        smooth_window_size=5,
        min_event_frame=20,
        max_event_frame=2000,
        min_silence_frame=20,
    )

    model_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "FireRedVAD",
        "pretrained_models", "FireRedVAD", "AED"
    )
    aed = FireRedAed.from_pretrained(model_dir, config)

    result, probs = aed.detect(wav_path)

    print(f"Duration: {result['dur']}s")
    for event in ["speech", "singing", "music"]:
        segments = result["event2timestamps"].get(event, [])
        ratio = result["event2ratio"].get(event, 0)
        print(f"\n{event} ({ratio*100:.1f}% of audio):")
        if not segments:
            print("  (none)")
        for i, (start, end) in enumerate(segments):
            print(f"  [{i+1}] {start:.3f}s - {end:.3f}s  ({end-start:.3f}s)")


if __name__ == "__main__":
    main()
