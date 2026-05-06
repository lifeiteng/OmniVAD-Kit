# Browser bench: main thread vs Web Worker

Self-contained browser harness for measuring OmniVAD streaming-VAD performance
on your machine. Reports:

- **RTF** (real-time factor) — median compute time / audio duration
- **Compute ms** — median wall time per run
- **Main-thread stall ms** — accumulated frame budget overrun during the
  bench window. This is the actual UX signal: how many ms the UI was
  unresponsive.

Two execution paths are compared on the same audio + model:

| Path        | Where compute runs       | Expected stall              |
|-------------|--------------------------|-----------------------------|
| Main thread | UI thread, blocking      | ≈ compute time × repeats    |
| Web Worker  | Background thread        | near 0                      |

If main-thread RTF is acceptable but stall is high, **just move VAD into a
Worker** — same compute time, responsive UI. If RTF itself is too high,
look at INT8 quantization or a smaller model instead.

## Run

The wasm + model files have to be served over HTTP (file:// blocks
cross-origin fetches). Serve from the **repo root**:

```bash
cd /path/to/OmniVAD-Kit
python3 -m http.server 8000
# then open http://localhost:8000/examples/browser-bench/index.html
```

Click **Run bench**. First run takes ~1 s extra to compile the wasm and fetch
the 1.1 MB model from the local filesystem.

## What it loads

| Resource                       | Path                                        |
|--------------------------------|---------------------------------------------|
| `OmniStreamVAD` ESM            | `/packages/omnivad/dist/index.js`           |
| WASM binary                    | `/packages/omnivad/dist/wasm/omnivad.wasm`  |
| Model                          | `/models/stream-vad.omnivad`                |
| Test audio (zh, 18 s)          | `/tests/data/zh_medium.wav`                 |

The bench passes `modelUrl` explicitly so it never falls back to the jsDelivr
CDN — you're measuring the build in `dist/`, not whatever's published on npm.

## Reading the numbers

Compare main-thread `RTF` to worker `RTF`. They should be **within a few
percent** of each other — the Worker doesn't make compute faster, only
non-blocking. The interesting delta is `Main-thread stall`: thousands of
ms on main, near zero in worker. That single number tells you whether
moving to a Worker solves the user-perceived problem.

## Worker form

The bench uses a **module worker** (`new Worker(url, {type:"module"})`)
that imports `OmniStreamVAD` directly. This relies on the loader fallback
in `wasm-binding.ts:loadScript()` — when `importScripts` is unavailable
(module worker), it fetches the Emscripten glue and runs it via
`new Function`. CSP without `'unsafe-eval'` will block this fallback;
in that case fall back to a classic worker that calls the wasm `ccall`
directly (see `wasm/bench-stream.cjs` for that pattern).
