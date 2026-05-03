# OmniVAD

[![PyPI](https://img.shields.io/pypi/v/omnivad)](https://pypi.org/project/omnivad/)
[![npm](https://img.shields.io/npm/v/omnivad)](https://www.npmjs.com/package/omnivad)
[![License](https://img.shields.io/github/license/lifeiteng/OmniVAD-Kit)](LICENSE)

**English** | [中文](README.zh.md)

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

# FastClone — share model weights, minimal memory per stream
clone = svad.clone()  # instant, ~0 memory overhead
clone.process(pcm_160_int16)  # fully independent state

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

int err = OMNI_OK;

// VAD — whole audio to speech segments
OmniVadHandle vad = omni_vad_create("vad.omnivad", &err);
omni_vad_detect_int16(vad, pcm, num_samples, &config, &segments, &count);
// segments[0] = { start: 0.44, end: 1.82 }

// Stream VAD — real-time, 10ms per frame
OmniStreamVadHandle svad = omni_stream_vad_create("stream-vad.omnivad", 0.5f, &err);
omni_stream_vad_process(svad, pcm_160_samples, 160, &result);
// result.confidence = 0.95, result.is_speech = true

// FastClone — share model weights across streams
OmniStreamVadHandle clone = omni_stream_vad_clone(svad, &err);
omni_stream_vad_process(clone, other_pcm, 160, &result);  // independent state

// AED — speech + singing + music detection
OmniAedHandle aed = omni_aed_create("aed.omnivad", &err);
omni_aed_detect_int16(aed, pcm, num_samples, &config, &segments, &count);
// segments[0] = { start: 0.09, end: 12.32, cls: OMNI_AED_MUSIC }
```

**Build:**

```bash
# Prerequisites: cmake, ncnn (brew install ncnn)
cd native
cmake -B build && cmake --build build -j$(nproc)

# Test
./build/test_all ../models/ audio.wav
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

## Thread Safety

| Component | Shared handle | Independent handles | Notes |
|-----------|:---:|:---:|-------|
| **OmniVAD** | **Safe** | **Safe** | `ncnn::Net` is read-only; each call creates a local `Fbank` and `Extractor` |
| **OmniAED** | **Safe** | **Safe** | Same architecture as VAD |
| **OmniStreamVAD** | **Unsafe** | **Safe** | Mutable internal state (`audio_buffer`, `cache`, `frame_offset`) |

**Guidelines:**

- `OmniVAD` and `OmniAED` instances can be safely shared across threads for concurrent inference. The Python `workers` parameter in `detect(..., workers=N)` already uses this pattern.
- `OmniStreamVAD` instances must **not** be shared across threads. Create one instance per thread for parallel streaming.
- Handle creation (`omni_*_create`) should be done sequentially — ncnn's model loading is not designed for highly concurrent initialization.
- Never call `close()` / `destroy()` on a handle while another thread is using it.

**Running thread-safety tests:**

```bash
# Python
pytest tests/test_thread_safety.py -v

# C++ (requires ncnn)
./native/build/test_thread_safety models/ tests/data/hello_en.wav [threads] [repeats]
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

## Chunking — `merge_chunks` / `mergeChunks`

After VAD produces a list of speech `(start, end)` segments, the chunking
utility groups them into duration-bounded chunks suitable for downstream
ASR / forced alignment / TTS. It is a **pure function** with no model
dependency — Python uses `ctypes`, TypeScript uses Emscripten WASM, and
C calls the native function directly. All three bindings share a single C
implementation in `native/src/chunking.cpp`.

```python
from omnivad import merge_chunks
chunks = merge_chunks(timestamps, max_chunk_secs=30.0, mode="greedy")
```

```ts
import { mergeChunks } from "omnivad";
const chunks = await mergeChunks(timestamps, { maxChunkSecs: 30.0, mode: "longest_gap" });
```

### Pipeline (5 steps; Steps 1–2 and 4–5 are shared by both modes)

```
input (sorted segments)
  │
  ├─ Step 1: drop segments with duration < min_speech_secs
  │
  ├─ Step 2: pre-merge consecutive segments with gap < min_silence_secs
  │          (cascades; takes max(end) on overlap)
  │
  ├─ Step 3: pack into chunks  ─┬─ mode = "greedy"
  │                              │     sequential append; split when next
  │                              │     would exceed max_chunk_secs OR gap > max_gap_secs
  │                              │
  │                              └─ mode = "longest_gap"
  │                                    recursive split at the longest gap
  │                                    until every chunk's span ≤ max_chunk_secs
  │
  ├─ Step 4: equal hard-split any chunk still longer than max_chunk_secs
  │          (only triggers when a single segment alone exceeds max_chunk_secs)
  │
  └─ Step 5: apply pad_onset_secs (clamped to ≥ 0) and pad_offset_secs
             output chunks: (start, end, seg_start_idx, seg_count)
```

### Mode comparison

| Property | `greedy` (default) | `longest_gap` |
|---|---|---|
| Strategy | Sequential append until next overflow | Recursive split at longest internal gap until each chunk fits `max_chunk_secs` |
| Honors `max_chunk_secs` | **Yes** — hard upper bound | **Yes** — recursion stops when chunk span ≤ `max_chunk_secs` |
| Boundary location | First overflow point | Longest pause inside the over-long span |
| Honors `max_gap_secs` | **Yes** — split at first `gap > max_gap_secs` | **Yes** — recursion also stops only when no internal gap exceeds `max_gap_secs` |
| Single seg > `max_chunk_secs` | Step 4 equal hard-split | Same — Step 4 fallback |
| Determinism | Deterministic | Deterministic; **leftmost** wins on tie |
| Recommended for | **Whisper / whisperX-style ASR** (fixed-length input, padded to 30s) | **Variable-length-input models** — forced alignment, TTS, encoder-style ASR. Splits at natural pauses; no fixed-length padding required. |

Example with the same input, both modes (`max_chunk_secs=20`):

```
Input (max_chunk_secs = 20):
  seg 0 = (0, 5)
  seg 1 = (8, 10)     gap from seg 0 = 3
  seg 2 = (20, 25)    gap from seg 1 = 10   ← longer

greedy
  start cur = (0, 5)
  accept seg 1            → cur = (0, 10)   [length 10 ≤ 20 ✓]
  next seg 2 would_exceed:  25 - 0 = 25 > 20  → SPLIT
  chunks: [(0, 10, 0, 2), (20, 25, 2, 1)]

longest_gap
  span = 25 > 20            → must split
  longest gap = 10 at idx 1 → cut between seg 1 and seg 2
    left  = [seg 0, seg 1]  span = 10 ≤ 20 ✓ → keep
    right = [seg 2]         span = 5  ≤ 20 ✓ → keep
  chunks: [(0, 10, 0, 2), (20, 25, 2, 1)]
```

(In this minimal example both modes happen to agree. They diverge whenever
the longest gap is not the **first** overflow point.)

### `seg_start_idx` / `seg_count` semantics

These index into the **post-Step-1+Step-2** view of the input — segments
dropped by `min_speech_secs` and pre-merged by `min_silence_secs` are
NOT in the indexing space. Both modes follow this convention.

### Defaults

`omni_chunk_config_default()` (C / `default_chunk_config()` Python /
`DEFAULT_CHUNK_CONFIG` TS) returns:

| field | default | source |
|---|---|---|
| `max_chunk_secs` | `30.0` | seconds; matches Whisper's 30s input window |
| `max_gap_secs` | `INFINITY` | disabled |
| `pad_onset_secs` / `pad_offset_secs` | `0.04` / `0.04` | |
| `min_speech_secs` | `0.0` | pairs with VAD `min_speech_frames` |
| `min_silence_secs` | `0.20` | matches VAD `min_silence_frames=20` @ 10ms shift |
| `mode` | `OMNI_CHUNK_GREEDY` | backward-compatible |

> **Heads-up — Python convenience defaults differ.** The Python kwargs of
> `merge_chunks(...)` use zeros for `pad_onset_secs`, `pad_offset_secs`,
> `min_silence_secs` (so the simplest call gives raw output). To match
> the canonical defaults, use the values returned by `default_chunk_config()`.
> See `tests/test_chunking.py::test_python_convenience_defaults_differ_from_canonical`.

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
