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

# Stream VAD — real-time, feed 160 samples (10ms) at a time
# Accepts float32 in [-1, 1] (Web Audio, soundfile, torch) or int16 PCM
svad = OmniStreamVAD()
frame = None
while frame is None:
    frame = svad.process(pcm_160)  # np.float32 or np.int16
# StreamResult(time=0.420s, confidence=0.95, is_speech=True)

# FastClone — share model weights, minimal memory per stream
clone = svad.clone()  # instant, ~0 memory overhead
clone.process(pcm_160)  # fully independent state

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
// Two entries: omni_stream_vad_process (float [-1,1]), _int16 (int16 PCM)
OmniStreamVadHandle svad = omni_stream_vad_create("stream-vad.omnivad", 0.5f, &err);
omni_stream_vad_process(svad, float_160_samples, 160, &result);   // FP32
omni_stream_vad_process_int16(svad, pcm_160_samples, 160, &result); // int16

// FastClone — share model weights across streams
OmniStreamVadHandle clone = omni_stream_vad_clone(svad, &err);
omni_stream_vad_process_int16(clone, other_pcm, 160, &result);  // independent state

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
// processFrame() accepts Float32Array [-1, 1] or Int16Array — dispatch by dtype
const frame = svad.processFrame(float32_160);  // null until enough audio is buffered
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

High-level APIs accept 16kHz mono audio only. Two formats, same convention
across all 3 model types and all 3 layers (C / Python / TypeScript):

- `float32` / `Float32Array` in `[-1, 1]` (Web Audio, soundfile, torch)
- `int16` / `Int16Array` PCM (WAV, microphone)

Wrappers dispatch by dtype to the matching C entry — **never scale or
convert in Python/JS**. All scaling lives in the C library: the `f32`
entry multiplies by 32768.0f, the `_int16` entry casts to float.

| Method | FP32 entry | int16 entry |
|---|---|---|
| `OmniVAD.detect / detect_probs` | `omni_vad_detect[_probs]` | `omni_vad_detect[_probs]_int16` |
| `OmniAED.detect / detect_probs` | `omni_aed_detect[_probs]` | `omni_aed_detect[_probs]_int16` |
| `OmniStreamVAD.process` | `omni_stream_vad_process` | `omni_stream_vad_process_int16` |
| `OmniStreamVAD.detect_full` | `omni_stream_vad_detect_full` | `omni_stream_vad_detect_full_int16` |

For exact contracts see [`native/include/omnivad.h`](native/include/omnivad.h).

## Audio Pipeline

```
16kHz PCM → Fbank (80-dim, 25ms window, 10ms shift) → CMVN → DFSMN → Sigmoid → Post-processing → Segments
                     Povey window                        μ/σ    ~2.2MB   [0,1]    4-state machine
                     pre-emphasis 0.97                                            merge/split/extend
```

## Streaming VAD — `OmniStreamVAD`

For long audio (live streams, hours-long recordings, real-time captioning),
`OmniStreamVAD` processes audio frame-by-frame and emits **segment-boundary
events on the same call** that confirms the boundary — bit-identical to
upstream [FireRedVAD's `FireRedStreamVad`](https://github.com/FireRedTeam/FireRedVAD/blob/main/fireredvad/stream_vad.py).

Each successful `process()` call returns a result with both per-frame
probabilities AND segment-boundary flags:

| Field | Meaning |
|-------|---------|
| `confidence` | raw model probability `[0, 1]` |
| `smoothed_prob` | causal moving-average over `smooth_window_size` frames |
| `is_speech` | `smoothed_prob >= threshold` |
| `is_speech_start` | `True` on the frame that confirms a new SPEECH segment |
| `is_speech_end` | `True` on the frame that confirms a SPEECH segment end |
| `frame_idx` | 1-based frame index (multiply by 0.01 for seconds) |
| `speech_start_frame` | 1-based segment start (when `is_speech_start`) |
| `speech_end_frame` | 1-based segment end (when `is_speech_end`) |

### Configuration (defaults match upstream FireRedVAD)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `threshold` | `0.5` | Speech activation threshold |
| `smooth_window_size` | `5` | Causal moving-average window (frames) |
| `pad_start_frame` | `5` | Extend confirmed segment START backward by N frames |
| `min_speech_frame` | `8` | Min continuous speech frames to confirm START (~80ms) |
| `max_speech_frame` | `2000` | Force-split when SPEECH-state count hits this (~20s) |
| `min_silence_frame` | `20` | Min continuous silence frames to confirm END (~200ms) |

### Python

```python
from omnivad import OmniStreamVAD
import numpy as np

vad = OmniStreamVAD()                              # upstream defaults
pcm = np.fromfile("speech.pcm", dtype=np.int16)

for i in range(0, len(pcm), 160):                  # 10ms chunks
    result = vad.process(pcm[i : i + 160])
    if result is None:
        continue
    if result.is_speech_start:
        print(f"START @ {result.speech_start_frame * 0.01:.2f}s")
    if result.is_speech_end:
        print(f"END   @ {result.speech_end_frame * 0.01:.2f}s")

# Or get [(start_sec, end_sec), ...] in one call:
segments = OmniStreamVAD().detect_segments("speech.wav")
```

### TypeScript

```typescript
import { OmniStreamVAD } from "omnivad";

const vad = await OmniStreamVAD.create();
for (let i = 0; i + 160 <= pcm.length; i += 160) {
    const result = vad.processFrame(pcm.subarray(i, i + 160));
    if (!result) continue;
    if (result.isSpeechStart) {
        console.log(`START @ ${(result.speechStartFrame * 0.01).toFixed(2)}s`);
    }
    if (result.isSpeechEnd) {
        console.log(`END   @ ${(result.speechEndFrame * 0.01).toFixed(2)}s`);
    }
}
```

### Pairing with `merge_chunks`

`OmniStreamVAD` emits raw VAD segments. To pack them into Whisper-sized
30s chunks for downstream ASR, feed the emitted `[start, end]` pairs to
`merge_chunks` (see next section).

## AED Overlap Segmenter — `AedOverlapSegmenter` / `OmniAEDOverlapSegmenter`

A **pseudo-streaming whole-window AED** segmenter: feed audio chunk by chunk
and it commits transcribable segments (speech / singing) and per-window events
as soon as they are decided. It runs the AED model on overlapping windows, so
late audio can refine a boundary before it is committed. Backed by a single C
implementation (`omni_aed_overlap_segmenter_*`); Python uses `ctypes`,
TypeScript uses Emscripten WASM, and Rust wraps the same C API.

Each `ingest()` / `flush()` returns `{ segments, events }` for what was newly
committed by that call. Events carry `is_transcribable` (speech **or** singing)
and confidences for all three classes; segments are duration-bounded
transcribable chunks referencing the events they cover.

### Configuration (defaults match `omni_aed_overlap_config_default()`)

| Parameter (Python `_seconds` / TS `Secs`) | Default | Meaning |
|-------------------------------------------|---------|---------|
| `hop` | `2.0` | AED window advance per step |
| `overlap` | `0.25` | Overlap retained between adjacent windows |
| `edge_guard` | `0.0` | Drop probabilities within this margin of a window edge |
| `hard_split_pause` | `2.0` | Force a segment boundary after a silence pause this long |
| `max_chunk` | `60.0` | Hard upper bound on a transcribable chunk |
| `min_speech` | `0.2` | Drop committed events shorter than this |
| `merge_gap` | `0.2` | Merge adjacent same-kind events across gaps shorter than this |
| `music_gap_tolerance` | `0.0` | Tolerate music gaps up to this when extending a music run |
| `pad_start` / `pad_end` | `0.0` | Pad committed segment start / end |
| `speech_threshold` / `singing_threshold` / `music_threshold` | `0.5` | Per-class thresholds |

### Python

```python
import numpy as np
from omnivad import AedOverlapSegmenter

seg = AedOverlapSegmenter(hop_seconds=2.0, overlap_seconds=0.25)
pcm = np.fromfile("podcast.pcm", dtype=np.int16)

for i in range(0, len(pcm), 32000):                 # 2s chunks
    out = seg.ingest(pcm[i : i + 32000])
    for s in out.segments:
        print(f"transcribable {s.start:.2f}–{s.end:.2f}s")
out = seg.flush()                                    # final partial window
seg.close()
```

### TypeScript

```typescript
import { OmniAEDOverlapSegmenter } from "omnivad";

const seg = await OmniAEDOverlapSegmenter.create({ hopSecs: 2.0, overlapSecs: 0.25 });
for (let i = 0; i + 32000 <= pcm.length; i += 32000) {     // 2s chunks
    const { segments } = seg.ingest(pcm.subarray(i, i + 32000));
    for (const s of segments) {
        console.log(`transcribable ${s.start.toFixed(2)}–${s.end.toFixed(2)}s`);
    }
}
const tail = seg.flush();                                   // final partial window
seg.dispose();
```

### Rust

```rust
use omnivad::AedOverlapSegmenter;

let mut seg = AedOverlapSegmenter::from_bundle_path("models/aed.omnivad", Default::default())?;
for chunk in pcm.chunks(32_000) {
    let out = seg.ingest_i16(chunk)?;
    for s in &out.segments {
        println!("transcribable {:.2}–{:.2}s", s.start, s.end);
    }
}
let _tail = seg.flush()?;
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

### Whisper / WhisperX-style ASR pipeline

`OmniVAD` (whole-audio, batch) + `merge_chunks(mode="greedy")` is the
1:1 equivalent of WhisperX's `Binarize(max_duration=chunk_size)` +
greedy packing. Use this recipe when feeding chunks into Whisper-family
ASR models that expect a fixed 30s input window:

```python
from omnivad import OmniVAD, merge_chunks

vad = OmniVAD()                              # threshold=0.4 default — safer for Whisper
result = vad.detect("long-audio.wav")        # whole-audio batch VAD

chunks = merge_chunks(
    timestamps=result["timestamps"],
    max_chunk_secs=30.0,                     # Whisper's input window
    mode="greedy",                           # WhisperX behavior
    pad_onset_secs=0.04,
    pad_offset_secs=0.04,
    min_silence_secs=0.20,                   # matches VAD min_silence_frames=20
)
# Each chunk: { start, end, seg_start_idx, seg_count }
# Slice the audio at [start, end] and feed each slice to Whisper.
```

Notes:

- Keep the default `threshold=0.4`. Whisper tolerates extra padding silence
  but is sensitive to clipped word edges (raising to 0.5 risks dropping
  weak word-initial/final consonants and triggering hallucinations).
- Do **not** use `mode="longest_gap"` here — that mode targets
  variable-length-input models (forced alignment, TTS), not WhisperX.
- For very long audio (>1 hour), pass `chunk_seconds=600, overlap_seconds=2`
  to `vad.detect(...)` to limit peak memory.

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

## Local Development

This section covers building OmniVAD from source and consuming the in-tree
build from another project on the same machine — the loop you want when
hacking on the C/C++ core, the Python wrapper, or the TS bindings.

### Prerequisites

| Target | Required | Notes |
|--------|----------|-------|
| Python wheel | Python 3.10+, CMake 3.15+, a C++14 toolchain | `pip install -e .` runs scikit-build-core, which **fetches ncnn automatically** via CMake `FetchContent`. |
| Standalone C/C++ library | CMake 3.15+, a pre-installed ncnn (`brew install ncnn` or build from source) | `native/CMakeLists.txt` does **not** fetch ncnn — set `-DNCNN_ROOT=...` if it isn't on the default search path. |
| TypeScript bundle | Node 18+, [pnpm](https://pnpm.io/) | Builds `dist/index.{js,cjs,d.ts}` only — does **not** rebuild the WASM. |
| WASM module | [emsdk](https://emscripten.org/docs/getting_started/downloads.html) (any recent version) | Required only when you change C/C++ code and need a fresh `dist/wasm/omnivad.wasm`. |

### Build the Python package (editable install)

```bash
pip install -e ".[dev]"
```

What this produces:

- `omnivad/libomnivad.{dylib,so,dll}` — the shared library actually loaded
  at runtime by `omnivad/_binding.py`.
- `omnivad/models/*.omnivad` — bundled model files (copied by CMake `install(...)`).
- An editable entry in your environment's `site-packages` pointing back at
  the source tree.

When you change **C/C++ code** in `native/`, re-run `pip install -e .` to
relink the dylib. (CMake's incremental build means this is fast.) Pure
Python edits don't need a reinstall.

### Build the TypeScript package

```bash
cd packages/omnivad
pnpm install
pnpm build          # tsup → dist/index.{js,cjs,d.ts}
pnpm typecheck      # tsc --noEmit
```

This step **does not** rebuild the WASM — it consumes whatever's already in
`dist/wasm/`. If you only edited TS, you're done.

### Build the WASM module (when you change C/C++)

```bash
EMSDK=/path/to/emsdk packages/omnivad/wasm/build.sh
```

The script writes `omnivad.{js,cjs,wasm}` directly into
`packages/omnivad/dist/wasm/`. After this, re-run `pnpm build` only if you
also changed TS.

> The `EMSDK` env var must point at your emsdk root (the directory that
> contains `emsdk_env.sh` and `upstream/emscripten/`). The script aborts
> with a clear error if it's missing.

### Consume the in-tree build from another repo

#### Python — `pip install -e <path>`

```bash
# In the target project's venv:
pip install -e /abs/path/to/OmniVAD-Kit          # editable, picks up your edits
# or, isolated wheel:
pip install /abs/path/to/OmniVAD-Kit             # builds and installs a fresh wheel
```

`pip install -e` is what you want for the dev loop — re-running it after a
C/C++ edit relinks the dylib in place; pure Python edits are picked up
without reinstalling.

#### TypeScript — three options, pick by use case

| Option | Command | When to use |
|--------|---------|-------------|
| **A. Tarball (closest to npm)** | `cd packages/omnivad && pnpm pack`<br>then in target: `pnpm add /abs/path/omnivad-0.2.8.tgz` | Verifying what real consumers will install. Clean, no symlink quirks. |
| **B. `file:` protocol** | In target `package.json`: `"omnivad": "file:../OmniVAD-Kit/packages/omnivad"` | In-tree monorepo-style consumption. Re-run `pnpm install` to pick up rebuilds. |
| **C. Global link** | `cd packages/omnivad && pnpm link --global`<br>then in target: `pnpm link --global omnivad` | Fast iteration across many projects. Watch for peer/hoist quirks. |

For all three, **rebuild before testing**:

```bash
cd packages/omnivad
pnpm build                                       # if only TS changed
EMSDK=/path/to/emsdk wasm/build.sh && pnpm build # if C/C++ changed
```

### Full rebuild after a C/C++ change (cheat sheet)

```bash
# From the repo root:
pip install -e .                                       # Python dylib
EMSDK=/path/to/emsdk packages/omnivad/wasm/build.sh    # WASM (.wasm + glue)
( cd packages/omnivad && pnpm build )                  # TS bundle
```

### Standalone C/C++ build (for native tests / embedding)

```bash
cd native
cmake -B build -DNCNN_ROOT=/path/to/ncnn   # only if ncnn isn't auto-discovered
cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
./build/test_all ../models ../tests/data/hello_en.wav
```

This is independent from the Python wheel build — the wheel uses CMake
`FetchContent` to pull a pinned ncnn, while `native/` expects a
pre-installed one.

### Lint / format

```bash
ruff check --fix . && ruff format .                    # Python (line-length 120)
( cd packages/omnivad && pnpm typecheck )              # TypeScript
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
