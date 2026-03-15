# OmniVAD

FireRedVAD cross-platform toolkit — browser + desktop (Windows/Linux/macOS).

## Architecture

```
Audio PCM (16kHz int16) → Fbank (80-dim) → CMVN → DFSMN inference → Sigmoid → Post-processing
```

Three models, same DFSMN arch (~2.2MB each, ~588K params):

| Model | Input | Output | Mode |
|-------|-------|--------|------|
| VAD | feat [B,T,80] | probs [B,T,1] | non-stream |
| Stream-VAD | feat [1,1,80] + cache [1,1024,19] | probs [1,1,1] + cache [1,1024,19] | stream (frame-by-frame) |
| AED | feat [B,T,80] | probs [B,T,3] (speech/singing/music) | non-stream |

## Deliverables

### P0: Unified C API + native library

Extend FireRedVAD's ncnn C API (currently stream-VAD only) to cover all 3 models.

**Files:**
- `native/include/fireredvad.h` — unified C API header
- `native/src/fireredvad.cpp` — implementation (ncnn backend)
- `native/CMakeLists.txt` — build config
- `native/build.sh` — host build script

**API surface:**
```c
// Stream VAD (existing, relocated)
FireredVADHandle firered_vad_create(params, model);
int  firered_vad_process(handle, pcm160, result);
void firered_vad_reset(handle);
void firered_vad_destroy(handle);

// Non-stream VAD (new)
int firered_vad_detect(params, model, pcm_data, num_samples, results, num_results);

// Non-stream AED (new)
int firered_aed_detect(params, model, pcm_data, num_samples, results, num_results);
```

**Build targets:**
- macOS: x86_64 + arm64 (Universal Binary)
- Linux: x86_64
- Windows: x86_64 (MSVC)

### P1: TypeScript/JavaScript npm package

Single npm package works in browser + Node.js.

**Stack:**
- ONNX Runtime Web — model inference (WASM backend, ~3MB)
- Pure TypeScript fbank — feature extraction (port from C++ `fbank.h`)
- ONNX models from `pretrained_models/onnx_models/`

**Package:** `@�fireredvad/core` or `fireredvad`

**Files:**
```
packages/omnivad/
├── src/
│   ├── index.ts          — public exports
│   ├── vad.ts            — FireRedVAD class (non-stream)
│   ├── stream-vad.ts     — FireRedStreamVAD class
│   ├── aed.ts            — FireRedAED class
│   ├── fbank.ts          — 80-dim mel fbank (pure TS)
│   ├── fft.ts            — radix-2 FFT
│   ├── cmvn.ts           — CMVN normalization
│   ├── post-process.ts   — smoothing + thresholding + segmentation
│   └── types.ts          — shared types
├── package.json
└── tsconfig.json
```

**API:**
```ts
import { FireRedVAD, FireRedStreamVAD, FireRedAED } from 'fireredvad';

// Non-stream VAD
const vad = await FireRedVAD.create({ modelUrl: '...' });
const result = await vad.detect(audioBuffer);
// => { duration: 2.32, timestamps: [[0.44, 1.82]] }

// Stream VAD (real-time)
const svad = await FireRedStreamVAD.create({ modelUrl: '...' });
for (const chunk of audioChunks) {
  const frame = await svad.processFrame(chunk);
  // => { confidence: 0.87, isSpeech: true, frameOffset: 42 }
}

// AED (3-class)
const aed = await FireRedAED.create({ modelUrl: '...' });
const events = await aed.detect(audioBuffer);
// => { events: { speech: [[0.4,3.5]], singing: [[1.8,20.0]], music: [[0.1,22.0]] } }
```

**Fbank implementation (port from C++):**
- Povey window (not Hanning): `pow(0.5 - 0.5*cos(2*PI*i/(N-1)), 0.85)`
- Pre-emphasis with streaming state: `y[n] = x[n] - 0.97*x[n-1]`
- 80-dim mel filterbank, 20Hz-8000Hz
- FFT size = next power of 2 from frame_length (400 → 512)
- DC offset removal
- Log mel energy (floor 1e-20)

### P2: Integration (future)

- Tauri plugin
- CDN hosting for models
- npm publish
- Web demo page

## Project structure

```
omnivad/
├── PLAN.md
├── README.md
├── .gitignore
├── native/                  # C/C++ core (ncnn)
│   ├── include/
│   ├── src/
│   ├── frontend/            # fbank.h, fft.h/cc, wav.h (from FireRedVAD)
│   ├── CMakeLists.txt
│   └── build.sh
├── packages/
│   └── fireredvad/          # npm package (TypeScript)
│       ├── src/
│       ├── test/
│       ├── package.json
│       └── tsconfig.json
└── models/                  # Symlink or copy of ONNX models
```

## Key decisions

1. **Browser inference: ONNX Runtime Web** (not ncnn WASM) — more mature, WebGPU support
2. **Fbank: pure TypeScript** (not WASM) — simpler toolchain, ~200 lines of code, sufficient perf
3. **No separate Node.js binding** — WASM package covers both browser and Node.js
4. **CMVN data: baked into npm package** — tiny (80 floats × 2 = 640 bytes)
