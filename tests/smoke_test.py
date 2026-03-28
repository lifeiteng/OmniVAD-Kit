#!/usr/bin/env python3
"""Smoke test: verify omnivad installs correctly and produces valid output."""

import sys

audio_path = sys.argv[1]

# 1. Import all public classes
from omnivad import OmniAED, OmniStreamVAD, OmniVAD

print("import ok")

# 2. VAD
vad = OmniVAD()
r = vad.detect(audio_path)
assert r["duration"] > 0, f"VAD duration <= 0: {r}"
assert len(r["timestamps"]) > 0, f"VAD found no speech: {r}"
print(f"VAD: {len(r['timestamps'])} segments, duration={r['duration']}s")

# 3. AED
aed = OmniAED()
e = aed.detect(audio_path)
assert e["duration"] > 0, f"AED duration <= 0: {e}"
n_events = sum(len(v) for v in e["events"].values())
print(f"AED: {n_events} events")

# 4. StreamVAD — raw probabilities
svad = OmniStreamVAD()
probs = svad.detect_full(audio_path)
assert len(probs) > 0, "StreamVAD returned no frames"
print(f"StreamVAD: {len(probs)} frames")

# 5. CLI entry point
from omnivad.cli import main  # noqa: F401

print("smoke test passed")
