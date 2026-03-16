# CLAUDE.md

## Project Overview

OmniVAD-Kit is a cross-platform VAD/AED toolkit based on FireRedVAD. Three components:

| Component | Path | Tech | Purpose |
|-----------|------|------|---------|
| C/C++ native | `native/` | ncnn, CMake | Core library, unified C API |
| Python package | `omnivad/` | ctypes, scikit-build-core | PyPI package with native bindings |
| TypeScript/JS | `packages/omnivad/` | ONNX Runtime Web, tsup | npm package for browser + Node.js |

Three models (~2.2MB each, DFSMN architecture): VAD, Stream-VAD, AED (speech/singing/music).

## Build & Test Commands

### Python package (PyPI: `omnivad`)
```bash
pip install -e .                    # Install (triggers C++ build via scikit-build-core)
pip install -e ".[dev]"             # Install with dev deps
pytest                              # Run tests
ruff check --fix . && ruff format . # Lint
```

### C/C++ native library
```bash
# Requires ncnn installed (brew install ncnn / build from source)
cmake -B native/build -S native -DNCNN_ROOT=/path/to/ncnn
cmake --build native/build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

# Run tests
./native/build/test_all models/ tests/data/speech.wav
```

### TypeScript package (npm: `omnivad`)
```bash
cd packages/omnivad
pnpm install
pnpm build       # tsup → dist/ (ESM + CJS + .d.ts)
npx tsc --noEmit # Typecheck
```

## Architecture

```
16kHz PCM → Fbank (80-dim) → CMVN → DFSMN → Sigmoid → Post-processing → Segments
```

- Python bindings (`omnivad/_binding.py`) use ctypes to load `libomnivad.so/dylib/dll`
- Models are bundled as `.omnivad` files in `models/` (packed ncnn param+bin+cmvn)
- Root `CMakeLists.txt` = Python wheel build (FetchContent ncnn)
- `native/CMakeLists.txt` = Standalone native build (expects pre-installed ncnn)

## Key Files

- `native/include/omnivad.h` — Unified C API header (all 3 models)
- `native/src/omnivad.cpp` — Implementation (~1100 lines)
- `omnivad/__init__.py` — Python public API: OmniVAD, OmniStreamVAD, OmniAED
- `omnivad/_binding.py` — ctypes bindings
- `packages/omnivad/src/index.ts` — TypeScript public API

## CI Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `publish.yml` | `v*` tag | Build wheels (cibuildwheel) + publish to PyPI |
| `native.yml` | `native/**` push | Build C++ on Ubuntu + macOS |
| `memory-check.yml` | `native/**` push | Valgrind memory leak check |
| `typescript.yml` | `packages/**` push | TypeScript typecheck + build |

## Conventions

- Code comments and README: English
- Python: ruff (line-length 120, py310)
- TypeScript: strict mode, ESM + CJS dual output
- C++: C++14, C API exported
- ncnn built with `-DNCNN_OPENMP=OFF -DNCNN_VULKAN=OFF`
