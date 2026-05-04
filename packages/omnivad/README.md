# omnivad

[![npm](https://img.shields.io/npm/v/omnivad)](https://www.npmjs.com/package/omnivad)
[![npm bundle size](https://img.shields.io/bundlephobia/min/omnivad)](https://bundlephobia.com/package/omnivad)
[![license](https://img.shields.io/npm/l/omnivad)](https://github.com/lifeiteng/OmniVAD-Kit/blob/main/LICENSE)

Cross-platform Voice Activity Detection and Audio Event Detection via WebAssembly.
Runs in **browsers, Web Workers, and Node.js** with a single API. Zero runtime
dependencies. Built on [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
from Xiaohongshu (DFSMN architecture, ~2.2 MB per model).

## What's in the box

| Class | Use case | Output |
|-------|----------|--------|
| **`OmniVAD`** | Whole-audio voice activity detection | `[start, end]` timestamps |
| **`OmniStreamVAD`** | Real-time, frame-by-frame VAD with segment-boundary events | per-frame probability + start/end events |
| **`OmniAED`** | Audio event detection (3-class) | `speech` / `singing` / `music` timestamps |
| **`mergeChunks`** | Pack VAD output into Whisper-style 30 s chunks | `{ start, end, segStartIdx, segCount }[]` |

All four share one WASM module (~2.2 MB SIMD-enabled), one C implementation,
and a single bundle (~24 KB JS, ESM + CJS + types).

## Install

```bash
pnpm add omnivad     # or: npm install omnivad / yarn add omnivad
```

Models are served from jsDelivr by default (zero config). For air-gapped or
custom deployments, pass `modelUrl` or pre-loaded `modelData`.

## Quickstart — whole-audio VAD

```ts
import { OmniVAD } from "omnivad";

const vad = await OmniVAD.create();

// Float32Array in [-1, 1] (Web Audio, decodeAudioData) or Int16Array (raw PCM)
const result = vad.detect(audioFloat32);
// { duration: 12.4, timestamps: [[0.35, 4.8], [5.1, 12.4]] }
```

## Streaming VAD — real-time, frame-by-frame

`OmniStreamVAD` processes 10 ms frames (160 samples @ 16 kHz) and emits
segment-boundary events on the same call that confirms the boundary —
bit-identical to upstream FireRedVAD's `FireRedStreamVad`.

`processFrame()` accepts `Float32Array` in `[-1, 1]` (Web Audio,
`AudioWorkletProcessor`, decoded WebRTC tracks) or `Int16Array` PCM
(WAV / microphone). Dispatch is by dtype — no scaling in JS.

```ts
import { OmniStreamVAD } from "omnivad";

const vad = await OmniStreamVAD.create();

// Float32Array [-1, 1] from Web Audio:
for (let i = 0; i + 160 <= floatPcm.length; i += 160) {
  const r = vad.processFrame(floatPcm.subarray(i, i + 160));
  if (!r) continue;
  if (r.isSpeechStart) console.log(`START @ ${(r.speechStartFrame * 0.01).toFixed(2)}s`);
  if (r.isSpeechEnd)   console.log(`END   @ ${(r.speechEndFrame   * 0.01).toFixed(2)}s`);
}

// Or Int16Array PCM from a WAV file — same call, same result:
for (let i = 0; i + 160 <= int16Pcm.length; i += 160) {
  vad.processFrame(int16Pcm.subarray(i, i + 160));
}
```

`processFrame()` returns `{ confidence, smoothedProb, isSpeech, isSpeechStart,
isSpeechEnd, frameIdx, speechStartFrame, speechEndFrame }` — every field comes
straight from the C state machine.

## Audio Event Detection — speech / singing / music

```ts
import { OmniAED } from "omnivad";

const aed = await OmniAED.create();
const events = aed.detect(audioFloat32);
// { duration: 22.0,
//   events: { speech: [[...]], singing: [[...]], music: [[...]] },
//   ratios: { speech: 0.41, singing: 0.0, music: 0.59 } }
```

## Whisper / WhisperX-style chunking

`OmniVAD` + `mergeChunks(mode: "greedy")` is the 1:1 equivalent of WhisperX's
`Binarize(max_duration=chunk_size)` + greedy packing. Use this recipe when
feeding chunks into Whisper-family ASR models that expect a fixed 30 s window:

```ts
import { OmniVAD, mergeChunks } from "omnivad";

const vad = await OmniVAD.create();                 // threshold=0.4 default — safer for Whisper
const result = vad.detect(audioFloat32);

const chunks = await mergeChunks(result.timestamps, {
  maxChunkSecs:    30.0,                            // Whisper input window
  mode:            "greedy",                        // WhisperX behavior
  padOnsetSecs:    0.04,
  padOffsetSecs:   0.04,
  minSilenceSecs:  0.20,
});
// Slice the audio at [chunk.start, chunk.end] and feed each slice to Whisper.
```

A second mode `"longest_gap"` exists for variable-length-input models
(forced alignment, TTS) — see the GitHub README for the comparison table.

## Multi-stream concurrency

`OmniStreamVAD` instances have mutable per-stream state and **must not** be
shared across concurrent streams. Use `clone()` to spin up a fresh instance
that shares the underlying model weights but has its own state — instant,
near-zero memory overhead per stream.

```ts
const base = await OmniStreamVAD.create();
const streamA = base.clone();
const streamB = base.clone();
// Process two independent audio sessions in parallel.
```

## Models and CDN

By default, models are fetched from jsDelivr:

```
https://cdn.jsdelivr.net/npm/omnivad@<version>/models/{vad,stream-vad,aed}.omnivad
```

Override per call when you need to host them yourself or pre-bundle:

```ts
const vad = await OmniVAD.create({
  modelUrl: "https://your-cdn/vad.omnivad",   // or
  modelData: arrayBufferYouAlreadyHave,
});
```

In Node.js, models are read from the installed package (`omnivad/models/`) — no
network access required at runtime.

## Performance

Real-Time Factor (lower = faster) on Apple M-series:

| Model | RTF | Speed |
|-------|-----|-------|
| VAD | ~0.003 | ~330× real-time |
| Streaming VAD | ~0.002 | ~500× real-time |
| AED | ~0.002 | ~500× real-time |

WASM is built with SIMD enabled and ncnn fp16 weights.

## Accuracy

Verified bit-identical to upstream PyTorch reference on 5 audio files × 3
models — see the [accuracy table](https://github.com/lifeiteng/OmniVAD-Kit#testing)
in the main repo.

## Browser, Worker, Node — same API

The package detects its runtime and loads the right glue:

- **Browsers (main thread)** — classic-script injection of the Emscripten glue
  (works around `MODULARIZE=1` IIFE issues with `import()`).
- **Web Workers / ServiceWorkers** — same path via `importScripts`.
- **Node.js (≥ 18)** — `createRequire` + local CJS resolution. No bundler
  config needed.

## See also

- Full documentation, accuracy tables, C/C++ API, Python package, native build:
  [GitHub repository](https://github.com/lifeiteng/OmniVAD-Kit)
- [中文 README](https://github.com/lifeiteng/OmniVAD-Kit/blob/main/README.zh.md)
- [Local development guide](https://github.com/lifeiteng/OmniVAD-Kit#local-development)

## Credits

- [**FireRedVAD**](https://github.com/FireRedTeam/FireRedVAD) — Kaituo Xu,
  Wenpeng Li, Kai Huang, Kun Liu (Xiaohongshu). Source models, DFSMN
  architecture, training pipeline.
- [ncnn](https://github.com/Tencent/ncnn) — Tencent. Inference backend.
- [Emscripten](https://emscripten.org/) — WebAssembly toolchain.

## License

Apache-2.0 — same as upstream FireRedVAD.
