# OmniVAD

[![PyPI](https://img.shields.io/pypi/v/omnivad)](https://pypi.org/project/omnivad/)
[![npm](https://img.shields.io/npm/v/omnivad)](https://www.npmjs.com/package/omnivad)
[![License](https://img.shields.io/github/license/lifeiteng/OmniVAD-Kit)](LICENSE)

Cross-platform toolkit for [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) — SOTA voice activity detection and audio event detection.

**Three models, one toolkit, runs everywhere:**

| Model | What it does | Output |
|-------|-------------|--------|
| **VAD** | Speech detection (non-stream) | Speech timestamps |
| **Stream-VAD** | Real-time speech detection (frame-by-frame) | Per-frame speech probability |
| **AED** | Audio event detection (non-stream) | Speech / Singing / Music timestamps |

All models are based on DFSMN architecture, ~2.2MB each (~588K params), support 100+ languages.

## Packages

### Python (`omnivad/`)

PyPI package with native C bindings (ncnn). Models bundled in wheel.

```bash
pip install omnivad
```

**CLI:**

```bash
omnivad audio.wav                        # VAD + AED → audio.TextGrid
omnivad audio.wav -o out.json            # Output as JSON
omnivad audio.wav -o out.srt             # Output as SRT
omnivad audio.wav -o out.vtt             # Output as WebVTT
omnivad audio.wav -f srt                 # Format flag (textgrid/json/srt/vtt)
omnivad audio.wav -m vad                 # VAD only
omnivad audio.wav -m aed                 # AED only (speech/singing/music)
omnivad long.wav --chunk 600 --overlap 2 # Chunked processing for large audio
python -m omnivad audio.wav              # Also works
```

**Python API:**

```python
from omnivad import OmniVAD, OmniStreamVAD, OmniAED
import numpy as np

vad = OmniVAD()

# File path — auto-loads as float32 [-1,1]
result = vad.detect("audio.wav")
# {'duration': 2.24, 'timestamps': [(0.26, 1.82)]}

# Float32 array [-1.0, 1.0] — from soundfile, torchaudio, librosa
result = vad.detect(float32_array)

# Int16 array — from raw WAV, microphone PCM
result = vad.detect(np.array([...], dtype=np.int16))

# Large audio — chunked processing with overlap
# overlap_seconds must be smaller than chunk_seconds
result = vad.detect("long.wav", chunk_seconds=600, overlap_seconds=2)

# Stream VAD — real-time, feed 160 int16 samples (10ms) at a time
svad = OmniStreamVAD()
frame = None
while frame is None:
    frame = svad.process(pcm_160_int16)
# StreamResult(time=0.420s, confidence=0.95, is_speech=True)

# AED — speech + singing + music
aed = OmniAED()
events = aed.detect("audio.wav")
# {'duration': 22.0, 'events': {'speech': [...], 'singing': [...], 'music': [...]}}
```

**Platforms:** macOS (arm64/x86_64), Linux (x86_64/aarch64), Windows (x86_64)

### C/C++ Native Library (`native/`)

Unified C API with [ncnn](https://github.com/Tencent/ncnn) backend. Single header, single library.

```c
#include "omnivad.h"

// Stream VAD — real-time, 10ms per frame
OmniVadHandle vad = omni_vad_stream_create(param, bin, means, istd, 0.5f);
omni_vad_stream_process(vad, pcm_160_samples, 160, &result);
// result.confidence = 0.95, result.is_speech = true

// Non-stream VAD — whole audio to segments
OmniVadNonStreamHandle vad = omni_vad_nonstream_create(param, bin, means, istd);
omni_vad_nonstream_process(vad, audio, num_samples, &config, &segments, &count);
// segments[0] = { start: 0.44, end: 1.82 }

// AED — speech + singing + music detection
OmniAedNonStreamHandle aed = omni_aed_nonstream_create(param, bin, means, istd);
omni_aed_nonstream_process(aed, audio, num_samples, &config, &segments, &count);
// segments[0] = { start: 0.09, end: 12.32, cls: OMNI_AED_MUSIC }
```

**Build:**

```bash
# Prerequisites: cmake, ncnn (brew install ncnn)
cd native
cmake -B build && cmake --build build -j$(nproc)

# Test
./build/test_nonstream_vad model.param model.bin cmvn_means.bin cmvn_istd.bin audio.wav
```

**Platforms:** macOS (arm64/x86_64), Linux (x86_64/aarch64), Windows (x86_64), Android (armeabi-v7a/arm64-v8a)

### TypeScript/JavaScript (`packages/omnivad/`)

Works in both **browser** and **Node.js** via ncnn WebAssembly. **Zero dependencies**, models bundled.

```ts
import { OmniVAD, OmniStreamVAD, OmniAED } from 'omnivad';

// Non-stream VAD — models loaded automatically from bundled WASM
const vad = await OmniVAD.create();
const result = vad.detect(audioFloat32Array);  // Float32Array [-1.0, 1.0]
// { duration: 2.32, timestamps: [[0.44, 1.82]] }

// Also accepts Int16Array (raw PCM)
const result2 = vad.detect(pcmInt16Array);

// Stream VAD — frame-by-frame or full-audio batch mode
const svad = await OmniStreamVAD.create();
const frame = svad.processFrame(pcm160);  // null until enough audio is buffered
const full = svad.detectFull(audioFloat32Array);
// { probabilities: Float32Array(...), numFrames: 98, duration: 1.0 }

// AED — speech + singing + music
const aed = await OmniAED.create();
const events = aed.detect(audioFloat32Array);
// { duration: 22.0, events: { speech: [...], singing: [...], music: [...] }, ratios: { ... } }
```

**Build:**

```bash
cd packages/omnivad
pnpm install && pnpm build
# Output: dist/index.js + dist/index.cjs + dist/index.d.ts + dist/wasm/*
```

## Audio Input

High-level APIs accept 16kHz mono audio only.

- `OmniVAD` / `OmniAED` in Python and TypeScript accept normalized `float32`/`Float32Array` in `[-1, 1]` and `int16` / `Int16Array`.
- `OmniStreamVAD.process()` in Python accepts `int16` chunks and also converts normalized `float32` chunks internally.
- `OmniStreamVAD.processFrame()` in TypeScript expects `Int16Array` chunks.
- `OmniStreamVAD.detect_full()` / `detectFull()` accept full-audio buffers and handle normalization internally.
- The C API is slightly lower-level than the Python/TypeScript wrappers. For exact input contracts, use [`native/include/omnivad.h`](native/include/omnivad.h) as the source of truth.

## Audio Pipeline

```
16kHz PCM → Fbank (80-dim, 25ms window, 10ms shift) → CMVN → DFSMN → Sigmoid → Post-processing → Segments
                     Povey window                        μ/σ    ~2.2MB   [0,1]    4-state machine
                     pre-emphasis 0.97                                            merge/split/extend
```

## Model Files

Prebuilt `.omnivad` bundles used by the Python package, TypeScript package, and local examples are already included in this repo under `models/`.

You only need to download upstream FireRedVAD checkpoints if you want to re-export ONNX or regenerate the native assets yourself.

```bash
# Download upstream PyTorch models + export to ONNX
pip install fireredvad
python -m fireredvad.bin.export_onnx --all

# Or download pre-exported ONNX models directly
# fireredvad_vad.onnx              — Non-stream VAD (2.3MB)
# fireredvad_aed.onnx              — Non-stream AED (2.3MB)
# fireredvad_stream_vad_with_cache.onnx — Stream VAD (2.2MB)

# For C/ncnn: convert ONNX → ncnn with pnnx
pip install pnnx
pnnx fireredvad_vad.onnx "inputshape=[1,100,80]"
```

## Testing

```bash
# Run the full Python test suite
pip install -e ".[dev]"
pytest tests -v

# Utility scripts (not pytest — require external FireRedVAD models)
python tests/generate_reference.py            # Generate Python reference data
python tests/check_timestamp_accuracy.py      # Strict C vs Python comparison
python tests/vad_to_textgrid.py audio.wav     # Audio → TextGrid + RTF benchmark
```

**Accuracy (C/ncnn vs Python, 5 audio files × 3 models):**

| Model | Timestamp Δ | Probability Δ | Status |
|-------|------------|---------------|--------|
| VAD | ≤ 0.020s | ≤ 0.001 | Exact match |
| AED (singing/music) | ≤ 0.010s | ≤ 0.013 | Exact match |
| AED (speech) | ≤ 0.030s | ≤ 0.015 | Match (ncnn fp16 edge cases on `event.wav`) |
| Stream-VAD (detect_full) | ≤ 0.010s | ≤ 0.001 | Exact match |

## Project Structure

```
omnivad/
├── omnivad/                         # Python PyPI package
│   ├── __init__.py                  #   Public API: OmniVAD, OmniStreamVAD, OmniAED
│   ├── cli.py                       #   CLI entry point (omnivad command)
│   ├── _binding.py                  #   ctypes bindings to libomnivad
│   ├── vad.py                       #   OmniVAD (non-stream)
│   ├── stream_vad.py                #   OmniStreamVAD (real-time)
│   └── aed.py                       #   OmniAED (3-class)
├── native/                          # C/C++ library (ncnn backend)
│   ├── include/omnivad.h            #   Unified C API header
│   ├── src/omnivad.cpp              #   Core implementation
│   ├── frontend/                    #   Fbank/FFT/WAV (from FireRedVAD)
│   ├── test/                        #   4 test programs
│   └── CMakeLists.txt
├── packages/omnivad/                # TypeScript npm package
│   ├── src/
│   │   ├── vad.ts                   #   OmniVAD (non-stream)
│   │   ├── stream-vad.ts            #   OmniStreamVAD (real-time)
│   │   ├── aed.ts                   #   OmniAED (3-class)
│   │   ├── wasm-binding.ts          #   Emscripten/WASM bindings
│   │   ├── types.ts                 #   Public TypeScript types
│   │   ├── index.ts                 #   Package exports
│   │   └── wasm.d.ts                #   WASM module declarations
│   ├── package.json
│   └── tsconfig.json
└── tests/                           # Test suite
    ├── test_c_vs_python.py          #   Accuracy: omnivad vs Python reference
    ├── test_determinism.py          #   Repeated-run determinism
    ├── test_edge_cases.py           #   Edge cases: tiny/empty/silence inputs
    ├── smoke_test.py                #   CI smoke test (import + detect)
    ├── test_memory.sh               #   Native memory/leak checks
    ├── check_timestamp_accuracy.py  #   Strict C vs Python comparison (manual)
    ├── check_native.py              #   Native C binary validation (manual)
    ├── generate_reference.py        #   Generate Python reference data
    ├── vad_to_textgrid.py           #   Audio → TextGrid + RTF benchmark
    └── data/                        #   5 test audio files + reference JSON
```

## Performance

RTF (Real-Time Factor) on Apple M-series, lower = faster:

| Model | RTF | Speed |
|-------|-----|-------|
| VAD | ~0.003 | ~330x real-time |
| Stream-VAD | ~0.002 | ~500x real-time |
| AED | ~0.002 | ~500x real-time |

## Origin & Attribution

OmniVAD is a cross-platform deployment toolkit built on top of [**FireRedVAD**](https://github.com/FireRedTeam/FireRedVAD), developed by [Xiaohongshu (小红书)](https://www.xiaohongshu.com/). FireRedVAD provides high-quality Voice Activity Detection models and a lightweight Audio Event Detection model that can distinguish speech, singing, and music.

**Original paper:** [FireRedVAD (arXiv:2603.10420)](https://arxiv.org/abs/2603.10420)

**What FireRedVAD provides:** DFSMN-based models (~2.2MB each), Python inference code, PyTorch training, strong VAD benchmark results (FLEURS-VAD-102 F1: 97.57%).

**What OmniVAD adds:** Unified C API (ncnn backend) for native deployment, TypeScript/JavaScript npm package (ncnn WebAssembly) for browser and Node.js, cross-platform build system, comprehensive test suite with accuracy validation.

## License

Apache-2.0 — same as the upstream FireRedVAD.

## Credits

- [**FireRedVAD**](https://github.com/FireRedTeam/FireRedVAD) — Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu (Xiaohongshu)
- [ncnn](https://github.com/Tencent/ncnn) — Tencent
- [Emscripten](https://emscripten.org/) — WebAssembly toolchain
