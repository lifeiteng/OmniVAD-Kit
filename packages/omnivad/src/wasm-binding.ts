/**
 * Low-level WASM binding for omnivad C API.
 * Loads the Emscripten module and provides typed wrappers.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type EmscriptenModule = any;

let _module: EmscriptenModule | null = null;
let _loading: Promise<EmscriptenModule> | null = null;

/**
 * Browser-side loader for Emscripten's IIFE glue
 * (`var createOmniVAD = (() => {...})()`). The glue isn't an ES module —
 * dynamic `import()` yields an empty namespace, so we have to inject the
 * script and read `globalThis.createOmniVAD` afterwards. See initWasm().
 *
 * Three execution contexts are supported:
 *   1. DOM (main thread)  — classic <script> tag.
 *   2. Classic worker     — synchronous `importScripts(url)`.
 *   3. Module worker      — `importScripts` is unavailable by spec, so
 *                           fall back to fetch + `new Function` eval and
 *                           manually republish `createOmniVAD` to
 *                           globalThis (the IIFE's top-level `var` becomes
 *                           function-scoped inside `new Function`'s body).
 */
function loadScript(url: string): Promise<void> {
  if (typeof globalThis.document === "undefined") {
    // Worker context.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const g = globalThis as any;
    // Try classic-worker importScripts first. The function exists on
    // both classic AND module worker globals, but module workers throw
    // synchronously on call: "Module scripts don't support
    // importScripts()". Fall through to fetch+eval on any failure.
    if (typeof g.importScripts === "function") {
      try {
        g.importScripts(url);
        return Promise.resolve();
      } catch {
        // fall through
      }
    }
    // Module worker (or classic worker where importScripts was blocked).
    // Fetch the glue and run it via `new Function`. CSP without
    // `'unsafe-eval'` will block this — same CSP profile as importScripts,
    // so callers facing strict CSP need a custom loader either way.
    return (async () => {
      const resp = await fetch(url);
      if (!resp.ok) {
        throw new Error(`Failed to fetch omnivad glue from ${url}: ${resp.status}`);
      }
      const code = await resp.text();
      // eslint-disable-next-line no-new-func
      new Function(
        "globalThis",
        `${code}\nglobalThis.createOmniVAD = createOmniVAD;`,
      )(globalThis);
    })();
  }
  // DOM (main thread) — classic script tag injection.
  return new Promise<void>((resolve, reject) => {
    const s = globalThis.document!.createElement("script");
    s.src = url;
    s.async = true;
    s.crossOrigin = "anonymous";
    s.onload = () => resolve();
    s.onerror = () =>
      reject(new Error(`Failed to load omnivad glue script: ${url}`));
    globalThis.document!.head.appendChild(s);
  });
}

/** Post-processing config matching C struct OmniPostConfig (7 x i32/float, 28 bytes) */
export interface PostConfig {
  threshold: number;
  smoothWindowSize: number;
  minSpeechFrames: number;
  minSilenceFrames: number;
  maxSpeechFrames: number;
  mergeSilenceFrames: number;
  extendSpeechFrames: number;
}

const SIZEOF_POST_CONFIG = 28; // 7 * 4 bytes
const SIZEOF_AED_POST_CONFIG = 3 * SIZEOF_POST_CONFIG; // 84 bytes
const SIZEOF_SEGMENT = 8; // start(f32) + end(f32)
const SIZEOF_AED_SEGMENT = 16; // start(f32) + end(f32) + cls(i32) + confidence(f32)
const SIZEOF_CHUNK_CONFIG = 28; // 6 floats + 1 i32 mode
const SIZEOF_CHUNK = 16; // start(f32) + end(f32) + seg_start_idx(i32) + seg_count(i32)

/** Re-exported for ABI self-tests. */
export const _SIZEOF_CHUNK_CONFIG = SIZEOF_CHUNK_CONFIG;
export const _SIZEOF_CHUNK = SIZEOF_CHUNK;
const OMNI_ERR_NO_FRAMES = -7;

// Build-time injected by tsup (see tsup.config.ts → define). Source of
// truth is package.json version; the runtime constant can't drift.
declare const __OMNIVAD_VERSION__: string;

/** Package version — used to construct default CDN URLs. */
export const VERSION = __OMNIVAD_VERSION__;

/** Default CDN base for model files (jsDelivr serves npm package contents). */
export const DEFAULT_CDN_BASE = `https://cdn.jsdelivr.net/npm/omnivad@${VERSION}/models`;

/** Model filenames keyed by type. */
export const MODEL_FILES = {
  vad: "vad.omnivad",
  "stream-vad": "stream-vad.omnivad",
  aed: "aed.omnivad",
} as const;

export type ModelType = keyof typeof MODEL_FILES;

/**
 * Initialize the WASM module. Call once before using any other functions.
 * Safe to call multiple times (returns cached module).
 */
export async function initWasm(
  wasmLocator?: (filename: string) => string,
): Promise<EmscriptenModule> {
  if (_module) return _module;
  if (_loading) return _loading;

  _loading = (async () => {
    // Dynamic import of the Emscripten glue
    let createOmniVAD: (opts?: Record<string, unknown>) => Promise<EmscriptenModule>;
    let defaultLocateFile: ((filename: string) => string) | undefined;

    if (typeof globalThis.process?.versions?.node === "string") {
      // Node.js: use require for .cjs (avoids ESM detection issues)
      const { createRequire } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ "module");
      const { dirname, join } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ "path");
      const req = createRequire(import.meta.url);
      const gluePath = req.resolve("../dist/wasm/omnivad.cjs");
      const wasmDir = dirname(gluePath);
      createOmniVAD = req(gluePath);
      defaultLocateFile = (filename: string) => join(wasmDir, filename);
    } else {
      // Browser. We can't `await import()` the glue: Emscripten with
      // MODULARIZE=1 (no EXPORT_ES6=1) emits an IIFE
      //   var createOmniVAD = (() => { ... })();
      // which has zero ES exports, so dynamic import yields an empty
      // module record and `mod.default || mod` is the empty namespace
      // object → `createOmniVAD()` throws "is not a function".
      //
      // Solution: classic <script> injection, then read the global the
      // IIFE wrote to (`globalThis.createOmniVAD`). Caches via the global
      // so repeat callers don't re-fetch the script.
      let glueUrlStr: string;
      if (wasmLocator) {
        glueUrlStr = wasmLocator("omnivad.js");
      } else {
        // Native ESM resolution path (when consumed without a bundler).
        glueUrlStr = new URL("../dist/wasm/omnivad.js", import.meta.url).href;
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const g = globalThis as any;
      let factory: ((opts?: Record<string, unknown>) => Promise<EmscriptenModule>) | undefined =
        g.createOmniVAD;
      if (typeof factory !== "function") {
        await loadScript(glueUrlStr);
        factory = g.createOmniVAD;
      }
      if (typeof factory !== "function") {
        throw new Error(
          `omnivad.js loaded from ${glueUrlStr} but globalThis.createOmniVAD is missing`,
        );
      }
      createOmniVAD = factory;
      // glueUrlStr may already be absolute (when wasmLocator returns a
      // full URL) or relative (native ESM mode). new URL("./", abs) is
      // safe; new URL("./", path-relative) throws — guard with a base.
      const baseHref =
        typeof globalThis.location !== "undefined"
          ? globalThis.location.href
          : "file:///";
      const absGlue = new URL(glueUrlStr, baseHref);
      const wasmBaseUrl = new URL("./", absGlue);
      defaultLocateFile = (filename: string) => new URL(filename, wasmBaseUrl).toString();
    }

    const opts: Record<string, unknown> = {};
    const locateFile = wasmLocator ?? defaultLocateFile;
    if (locateFile) {
      opts.locateFile = (path: string) => locateFile(path);
    }

    _module = await createOmniVAD(opts);
    return _module!;
  })();

  return _loading;
}

/**
 * Load a model file as ArrayBuffer.
 *
 * Resolution order:
 *   1. modelData (ArrayBuffer) — use directly
 *   2. modelUrl (string/URL) — fetch from that URL
 *   3. Node.js — read from npm package's models/ directory
 *   4. Browser — fetch from jsDelivr CDN
 */
export async function loadModel(
  modelType: ModelType,
  modelUrl?: string | URL,
  modelData?: ArrayBuffer,
): Promise<ArrayBuffer> {
  if (modelData) return modelData;

  if (modelUrl) {
    const resp = await fetch(modelUrl.toString());
    if (!resp.ok) throw new Error(`Failed to fetch model from ${modelUrl}: ${resp.status}`);
    return resp.arrayBuffer();
  }

  const filename = MODEL_FILES[modelType];

  if (typeof globalThis.process?.versions?.node === "string") {
    // Node.js: read from package's models/ directory
    const { createRequire } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ "module");
    const { dirname, join } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ "path");
    const { readFile } = await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ "fs/promises");
    const req = createRequire(import.meta.url);
    const pkgDir = dirname(req.resolve("../package.json"));
    const modelPath = join(pkgDir, "models", filename);
    const buf = await readFile(modelPath);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }

  // Browser: fetch from CDN
  const url = `${DEFAULT_CDN_BASE}/${filename}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch model from ${url}: ${resp.status}`);
  return resp.arrayBuffer();
}

/** Get the initialized WASM module (throws if not initialized) */
export function getModule(): EmscriptenModule {
  if (!_module) throw new Error("WASM not initialized. Call initWasm() first.");
  return _module;
}

function readNativeError(M: EmscriptenModule, code: number): string {
  const msg = M.ccall("omni_error_string", "string", ["number"], [code]);
  return msg ? `${msg} (${code})` : `error ${code}`;
}

/** Shared helper: call a C create function with out_error, throw on failure. */
function createModel(
  M: EmscriptenModule,
  fnName: string,
  argTypes: string[],
  args: unknown[],
  label: string,
): number {
  const errPtr = M._malloc(4);
  try {
    const handle = M.ccall(fnName, "number", [...argTypes, "number"], [...args, errPtr]);
    if (!handle) {
      const err = M.getValue(errPtr, "i32");
      throw new Error(`Failed to create ${label} model: ${readNativeError(M, err)}`);
    }
    return handle;
  } finally {
    M._free(errPtr);
  }
}

// -------------------------------------------------------------------------- //
//  Memory helpers                                                             //
// -------------------------------------------------------------------------- //

/**
 * Audio format: two types only. Same convention across all 3 model types.
 *   "f32"   — float* in [-1.0, 1.0] (Web Audio, soundfile, torch)
 *   "int16" — int16_t* PCM (WAV files, microphones)
 */
export type AudioFormat = "f32" | "int16";

/** Copy Float32Array audio into WASM heap, returns pointer. Caller must free. */
export function copyAudioToHeap(M: EmscriptenModule, audio: Float32Array): number {
  const ptr = M._malloc(audio.length * 4);
  const heap = new Float32Array(M.HEAPU8.buffer, ptr, audio.length);
  heap.set(audio);
  return ptr;
}

/** Copy Int16Array audio into WASM heap, returns pointer. Caller must free. */
export function copyInt16ToHeap(M: EmscriptenModule, audio: Int16Array): number {
  const ptr = M._malloc(audio.length * 2);
  const heap = new Int16Array(M.HEAPU8.buffer, ptr, audio.length);
  heap.set(audio);
  return ptr;
}

/**
 * Audio format dispatch: copy `audio` to WASM heap using the correct
 * integer/float layout for its dtype, and return the matching format
 * tag for downstream C-entry routing.
 *
 * Wrappers MUST go through this helper — never scale or cast in JS.
 * All scaling lives in the C entries (the f32 entry multiplies by
 * 32768.0f, the int16 entry casts to float).
 */
export function dispatchAudio(
  M: EmscriptenModule,
  audio: Float32Array | Int16Array,
): { ptr: number; length: number; format: AudioFormat } {
  if (audio instanceof Float32Array) {
    return { ptr: copyAudioToHeap(M, audio), length: audio.length, format: "f32" };
  }
  if (audio instanceof Int16Array) {
    return { ptr: copyInt16ToHeap(M, audio), length: audio.length, format: "int16" };
  }
  throw new TypeError(
    `unsupported audio dtype; expected Float32Array in [-1, 1] or Int16Array`,
  );
}

/** Write PostConfig struct to WASM heap at ptr */
export function writePostConfig(M: EmscriptenModule, ptr: number, cfg: PostConfig): void {
  M.setValue(ptr + 0, cfg.threshold, "float");
  M.setValue(ptr + 4, cfg.smoothWindowSize, "i32");
  M.setValue(ptr + 8, cfg.minSpeechFrames, "i32");
  M.setValue(ptr + 12, cfg.minSilenceFrames, "i32");
  M.setValue(ptr + 16, cfg.maxSpeechFrames, "i32");
  M.setValue(ptr + 20, cfg.mergeSilenceFrames, "i32");
  M.setValue(ptr + 24, cfg.extendSpeechFrames, "i32");
}

export const DEFAULT_VAD_CONFIG: PostConfig = {
  threshold: 0.4,
  smoothWindowSize: 5,
  minSpeechFrames: 20,
  minSilenceFrames: 20,
  maxSpeechFrames: 3000,
  mergeSilenceFrames: 0,
  extendSpeechFrames: 0,
};

// -------------------------------------------------------------------------- //
//  Chunking (pure-algorithm, mirrors omnivad.h's omni_merge_chunks)           //
// -------------------------------------------------------------------------- //

/** OmniChunkMode enum values (must match native/include/omnivad.h). */
export const OMNI_CHUNK_GREEDY = 0;
export const OMNI_CHUNK_LONGEST_GAP = 1;

/**
 * Chunking strategy:
 * - "greedy" — sequential append. Recommended for fixed-length-input ASR
 *              (Whisper / whisperX, which pad to 30s anyway).
 * - "longest_gap" — recursive split at longest pause; falls back to hard-split
 *                   when a single segment exceeds maxChunkSecs. Recommended for
 *                   variable-length-input models (forced alignment, TTS,
 *                   encoder-style ASR); no fixed-length padding required.
 */
export type ChunkMode = "greedy" | "longest_gap";

/** Configuration for omni_merge_chunks (matches C struct OmniChunkConfig, 28 bytes) */
export interface ChunkConfig {
  maxChunkSecs: number;        // hard upper bound on chunk duration (seconds), > 0
  maxGapSecs: number;           // split if gap > this. Infinity disables. Honored by both modes.
  padOnsetSecs: number;         // extend chunk start backward (clamped >= 0)
  padOffsetSecs: number;        // extend chunk end forward
  minSpeechSecs: number;    // drop input segments shorter than this; pairs with VAD minSpeechFrames
  minSilenceSecs: number;   // pre-merge gaps shorter than this; pairs with VAD minSilenceFrames
  mode: ChunkMode;          // packing strategy (default "greedy")
}

/**
 * Default chunk config. Mirrors C-side omni_chunk_config_default(); kept in
 * TS so callers don't need a roundtrip into WASM just to read defaults.
 *
 * Defaults: max_chunk_secs matches Whisper's 30s input window.
 */
export const DEFAULT_CHUNK_CONFIG: ChunkConfig = {
  maxChunkSecs: 30.0,
  maxGapSecs: Infinity,
  padOnsetSecs: 0.04,
  padOffsetSecs: 0.04,
  minSpeechSecs: 0.0,
  minSilenceSecs: 0.20,  // matches VAD minSilenceFrames=20 @ 10ms shift
  mode: "greedy",
};

function modeToInt(m: ChunkMode): number {
  switch (m) {
    case "greedy": return OMNI_CHUNK_GREEDY;
    case "longest_gap": return OMNI_CHUNK_LONGEST_GAP;
    default: throw new Error(`Unknown chunking mode: ${String(m)}`);
  }
}

/** Write ChunkConfig struct to WASM heap at ptr (must be SIZEOF_CHUNK_CONFIG bytes). */
export function writeChunkConfig(M: EmscriptenModule, ptr: number, cfg: ChunkConfig): void {
  M.setValue(ptr + 0,  cfg.maxChunkSecs,       "float");
  M.setValue(ptr + 4,  cfg.maxGapSecs,          "float");
  M.setValue(ptr + 8,  cfg.padOnsetSecs,        "float");
  M.setValue(ptr + 12, cfg.padOffsetSecs,       "float");
  M.setValue(ptr + 16, cfg.minSpeechSecs,   "float");
  M.setValue(ptr + 20, cfg.minSilenceSecs,  "float");
  M.setValue(ptr + 24, modeToInt(cfg.mode), "i32");
}

export interface ChunkRecord {
  start: number;
  end: number;
  segStartIdx: number;
  segCount: number;
}

/**
 * Call omni_merge_chunks via the WASM module.
 *
 * @param segments  array of [start, end] pairs, sorted by start (caller's contract)
 * @param config    chunking configuration
 * @returns array of ChunkRecord. On C error, throws.
 */
export function chunkMerge(
  M: EmscriptenModule,
  segments: Array<[number, number]>,
  config: ChunkConfig,
): ChunkRecord[] {
  const numSegments = segments.length;

  const segPtr = numSegments > 0 ? M._malloc(numSegments * SIZEOF_SEGMENT) : 0;
  const cfgPtr = M._malloc(SIZEOF_CHUNK_CONFIG);
  const outPtrPtr = M._malloc(4); // pointer to OmniChunk*
  const outCountPtr = M._malloc(4);

  try {
    for (let i = 0; i < numSegments; i++) {
      const base = segPtr + i * SIZEOF_SEGMENT;
      M.setValue(base + 0, segments[i][0], "float");
      M.setValue(base + 4, segments[i][1], "float");
    }
    writeChunkConfig(M, cfgPtr, config);
    M.setValue(outPtrPtr, 0, "i32");
    M.setValue(outCountPtr, 0, "i32");

    const rc = M.ccall(
      "omni_merge_chunks",
      "number",
      ["number", "number", "number", "number", "number"],
      [segPtr, numSegments, cfgPtr, outPtrPtr, outCountPtr],
    );
    if (rc !== 0) {
      throw new Error(`omni_merge_chunks failed: ${readNativeError(M, rc)}`);
    }

    const count = M.getValue(outCountPtr, "i32");
    const chunkPtr = M.getValue(outPtrPtr, "i32");
    const chunks: ChunkRecord[] = [];
    for (let i = 0; i < count; i++) {
      const base = chunkPtr + i * SIZEOF_CHUNK;
      chunks.push({
        start:       M.getValue(base + 0, "float"),
        end:         M.getValue(base + 4, "float"),
        segStartIdx: M.getValue(base + 8, "i32"),
        segCount:    M.getValue(base + 12, "i32"),
      });
    }
    if (chunkPtr) M._free(chunkPtr);
    return chunks;
  } finally {
    if (segPtr) M._free(segPtr);
    M._free(cfgPtr);
    M._free(outPtrPtr);
    M._free(outCountPtr);
  }
}

// -------------------------------------------------------------------------- //
//  Non-stream VAD                                                             //
// -------------------------------------------------------------------------- //

export function vadCreate(M: EmscriptenModule, modelBuffer: ArrayBuffer): number {
  const bytes = new Uint8Array(modelBuffer);
  const ptr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, ptr);
  try {
    return createModel(M, "omni_vad_create_from_buffer", ["number", "number"], [ptr, bytes.length], "VAD");
  } finally {
    M._free(ptr);
  }
}

function readSegments(M: EmscriptenModule, segPtrPtr: number, countPtr: number): Array<[number, number]> {
  const count = M.getValue(countPtr, "i32");
  const segPtr = M.getValue(segPtrPtr, "i32");
  const segments: Array<[number, number]> = [];
  for (let i = 0; i < count; i++) {
    const base = segPtr + i * SIZEOF_SEGMENT;
    segments.push([
      Math.round(M.getValue(base, "float") * 1000) / 1000,
      Math.round(M.getValue(base + 4, "float") * 1000) / 1000,
    ]);
  }
  if (segPtr) M._free(segPtr);
  return segments;
}

export function vadDetect(
  M: EmscriptenModule,
  handle: number,
  audioPtr: number,
  numSamples: number,
  cfg: PostConfig,
  format: AudioFormat = "f32",
): Array<[number, number]> {
  const cfgPtr = M._malloc(SIZEOF_POST_CONFIG);
  const segPtrPtr = M._malloc(4);
  const countPtr = M._malloc(4);
  const fn = format === "int16" ? "omni_vad_detect_int16" : "omni_vad_detect";

  try {
    writePostConfig(M, cfgPtr, cfg);
    const ret = M.ccall(
      fn,
      "number",
      ["number", "number", "number", "number", "number", "number"],
      [handle, audioPtr, numSamples, cfgPtr, segPtrPtr, countPtr],
    );
    if (ret !== 0) throw new Error(`VAD detect failed: ${ret}`);
    return readSegments(M, segPtrPtr, countPtr);
  } finally {
    M._free(cfgPtr);
    M._free(segPtrPtr);
    M._free(countPtr);
  }
}

export function vadDestroy(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_vad_destroy", null, ["number"], [handle]);
}

// -------------------------------------------------------------------------- //
//  Non-stream AED                                                             //
// -------------------------------------------------------------------------- //

const AED_CLASSES: Record<number, string> = { 0: "speech", 1: "singing", 2: "music" };

export function aedCreate(M: EmscriptenModule, modelBuffer: ArrayBuffer): number {
  const bytes = new Uint8Array(modelBuffer);
  const ptr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, ptr);
  try {
    return createModel(M, "omni_aed_create_from_buffer", ["number", "number"], [ptr, bytes.length], "AED");
  } finally {
    M._free(ptr);
  }
}

export interface AedPostConfig {
  speech: PostConfig;
  singing: PostConfig;
  music: PostConfig;
}

export function aedDetect(
  M: EmscriptenModule,
  handle: number,
  audioPtr: number,
  numSamples: number,
  cfg: AedPostConfig,
  format: AudioFormat = "f32",
): Record<string, Array<[number, number]>> {
  const cfgPtr = M._malloc(SIZEOF_AED_POST_CONFIG);
  const segPtrPtr = M._malloc(4);
  const countPtr = M._malloc(4);
  const fn = format === "int16" ? "omni_aed_detect_int16" : "omni_aed_detect";

  try {
    writePostConfig(M, cfgPtr, cfg.speech);
    writePostConfig(M, cfgPtr + SIZEOF_POST_CONFIG, cfg.singing);
    writePostConfig(M, cfgPtr + 2 * SIZEOF_POST_CONFIG, cfg.music);

    const ret = M.ccall(
      fn,
      "number",
      ["number", "number", "number", "number", "number", "number"],
      [handle, audioPtr, numSamples, cfgPtr, segPtrPtr, countPtr],
    );
    if (ret !== 0) throw new Error(`AED detect failed: ${ret}`);

    const count = M.getValue(countPtr, "i32");
    const segPtr = M.getValue(segPtrPtr, "i32");
    const events: Record<string, Array<[number, number]>> = {
      speech: [],
      singing: [],
      music: [],
    };
    for (let i = 0; i < count; i++) {
      const base = segPtr + i * SIZEOF_AED_SEGMENT;
      const cls = M.getValue(base + 8, "i32");
      const name = AED_CLASSES[cls];
      if (name && events[name]) {
        events[name].push([
          Math.round(M.getValue(base, "float") * 1000) / 1000,
          Math.round(M.getValue(base + 4, "float") * 1000) / 1000,
        ]);
      }
    }
    if (segPtr) M._free(segPtr);
    return events;
  } finally {
    M._free(cfgPtr);
    M._free(segPtrPtr);
    M._free(countPtr);
  }
}

export function aedDestroy(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_aed_destroy", null, ["number"], [handle]);
}

// -------------------------------------------------------------------------- //
//  Stream VAD                                                                 //
// -------------------------------------------------------------------------- //

/** OmniStreamVadConfig — bit-identical to upstream FireRedStreamVadConfig. */
export interface StreamVadConfig {
  threshold: number;
  smoothWindowSize: number;
  padStartFrame: number;
  minSpeechFrame: number;
  maxSpeechFrame: number;
  minSilenceFrame: number;
}

/** Defaults match upstream FireRedStreamVadConfig. */
export const DEFAULT_STREAM_VAD_CONFIG: StreamVadConfig = {
  threshold: 0.5,
  smoothWindowSize: 5,
  padStartFrame: 5,
  minSpeechFrame: 8,
  maxSpeechFrame: 2000,
  minSilenceFrame: 20,
};

const SIZEOF_STREAM_VAD_CONFIG = 24; // float + 5 i32

function writeStreamVadConfig(M: EmscriptenModule, ptr: number, cfg: StreamVadConfig): void {
  M.setValue(ptr + 0,  cfg.threshold,        "float");
  M.setValue(ptr + 4,  cfg.smoothWindowSize, "i32");
  M.setValue(ptr + 8,  cfg.padStartFrame,    "i32");
  M.setValue(ptr + 12, cfg.minSpeechFrame,   "i32");
  M.setValue(ptr + 16, cfg.maxSpeechFrame,   "i32");
  M.setValue(ptr + 20, cfg.minSilenceFrame,  "i32");
}

export function streamVadCreate(
  M: EmscriptenModule,
  modelBuffer: ArrayBuffer,
  config: Partial<StreamVadConfig> = {},
): number {
  // Skip undefined overrides — callers often relay optional kwargs that may
  // be `undefined`, and spreading those would clobber the upstream defaults.
  const overrides = Object.fromEntries(
    Object.entries(config).filter(([, v]) => v !== undefined),
  );
  const cfg: StreamVadConfig = { ...DEFAULT_STREAM_VAD_CONFIG, ...overrides };
  const bytes = new Uint8Array(modelBuffer);
  const dataPtr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, dataPtr);
  const cfgPtr = M._malloc(SIZEOF_STREAM_VAD_CONFIG);
  try {
    writeStreamVadConfig(M, cfgPtr, cfg);
    return createModel(
      M,
      "omni_stream_vad_create_from_buffer",
      ["number", "number", "number"],
      [dataPtr, bytes.length, cfgPtr],
      "StreamVAD",
    );
  } finally {
    M._free(dataPtr);
    M._free(cfgPtr);
  }
}

/** Per-frame result from streaming VAD. Bit-identical to upstream
 *  StreamVadFrameResult: includes segment-boundary events. */
export interface StreamVadResult {
  confidence: number;
  smoothedProb: number;
  isSpeech: boolean;
  isSpeechStart: boolean;
  isSpeechEnd: boolean;
  frameIdx: number;
  speechStartFrame: number;
  speechEndFrame: number;
}

const SIZEOF_STREAM_VAD_RESULT = 24; // 2*float + 3*bool + 1pad + 3*i32

/**
 * Process one chunk of audio (160 samples = 10ms). Returns null if buffering.
 *
 * Caller must have already copied audio to `audioPtr` via `dispatchAudio()`
 * (or `copyAudioToHeap` / `copyInt16ToHeap`); `format` selects the C entry.
 */
export function streamVadProcess(
  M: EmscriptenModule,
  handle: number,
  audioPtr: number,
  numSamples: number,
  format: AudioFormat = "f32",
): StreamVadResult | null {
  const resultPtr = M._malloc(SIZEOF_STREAM_VAD_RESULT);
  const fn = format === "int16" ? "omni_stream_vad_process_int16" : "omni_stream_vad_process";
  try {
    const ret = M.ccall(
      fn,
      "number",
      ["number", "number", "number", "number"],
      [handle, audioPtr, numSamples, resultPtr],
    );
    if (ret === OMNI_ERR_NO_FRAMES) return null;
    if (ret !== 0) throw new Error(`StreamVAD process failed: ${ret}`);
    return {
      confidence:       M.getValue(resultPtr + 0,  "float"),
      smoothedProb:     M.getValue(resultPtr + 4,  "float"),
      isSpeech:         M.getValue(resultPtr + 8,  "i8") !== 0,
      isSpeechStart:    M.getValue(resultPtr + 9,  "i8") !== 0,
      isSpeechEnd:      M.getValue(resultPtr + 10, "i8") !== 0,
      frameIdx:         M.getValue(resultPtr + 12, "i32"),
      speechStartFrame: M.getValue(resultPtr + 16, "i32"),
      speechEndFrame:   M.getValue(resultPtr + 20, "i32"),
    };
  } finally {
    M._free(resultPtr);
  }
}

/**
 * Batch mode: run the full audio through the streaming model at once and
 * return raw per-frame probabilities. `format` selects the C entry; the
 * caller must have copied `audioPtr` with the matching layout.
 */
export function streamVadDetectFull(
  M: EmscriptenModule,
  handle: number,
  audioPtr: number,
  numSamples: number,
  format: AudioFormat = "f32",
): { probabilities: Float32Array; numFrames: number } {
  const probsPtrPtr = M._malloc(4);
  const framesPtr = M._malloc(4);
  const fn =
    format === "int16" ? "omni_stream_vad_detect_full_int16" : "omni_stream_vad_detect_full";
  try {
    const ret = M.ccall(
      fn,
      "number",
      ["number", "number", "number", "number", "number"],
      [handle, audioPtr, numSamples, probsPtrPtr, framesPtr],
    );
    if (ret !== 0) throw new Error(`StreamVAD detectFull failed: ${ret}`);

    const numFrames = M.getValue(framesPtr, "i32");
    const probsPtr = M.getValue(probsPtrPtr, "i32");
    const probabilities = probsPtr
      ? new Float32Array(new Float32Array(M.HEAPU8.buffer, probsPtr, numFrames))
      : new Float32Array(0);
    if (probsPtr) M._free(probsPtr);
    return { probabilities, numFrames };
  } finally {
    M._free(probsPtrPtr);
    M._free(framesPtr);
  }
}

/** Clone a stream VAD handle (shares model weights, fresh per-instance state). */
export function streamVadClone(M: EmscriptenModule, handle: number): number {
  const errPtr = M._malloc(4);
  try {
    const newHandle = M.ccall(
      "omni_stream_vad_clone",
      "number",
      ["number", "number"],
      [handle, errPtr],
    );
    if (!newHandle) {
      const err = M.getValue(errPtr, "i32");
      throw new Error(`StreamVAD clone failed: ${readNativeError(M, err)}`);
    }
    return newHandle;
  } finally {
    M._free(errPtr);
  }
}

export function streamVadReset(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_stream_vad_reset", null, ["number"], [handle]);
}

export function streamVadDestroy(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_stream_vad_destroy", null, ["number"], [handle]);
}

// -------------------------------------------------------------------------- //
//  AED overlap segmenter (pseudo-streaming whole-window AED)                  //
// -------------------------------------------------------------------------- //

/** OmniAedEventKind values (must match native/include/omnivad.h). */
export const AED_EVENT_KINDS: Record<number, string> = {
  0: "silence",
  1: "speech",
  2: "singing",
  3: "music",
  4: "mixed",
};

export const AED_KIND_MASK_SPEECH = 1 << 0;
export const AED_KIND_MASK_SINGING = 1 << 1;
export const AED_KIND_MASK_MUSIC = 1 << 2;
/** Speech | Singing — an event is transcribable when it contains either. */
export const AED_KIND_MASK_TRANSCRIBABLE = AED_KIND_MASK_SPEECH | AED_KIND_MASK_SINGING;

/** Resolved AED overlap config (seconds; mirrors public AEDOverlapConfig). */
export interface AedOverlapConfig {
  hopSecs: number;
  overlapSecs: number;
  edgeGuardSecs: number;
  hardSplitPauseSecs: number;
  maxChunkSecs: number;
  hardSplitLookaheadSecs: number;
  minSpeechSecs: number;
  mergeGapSecs: number;
  musicGapToleranceSecs: number;
  padStartSecs: number;
  padEndSecs: number;
  speechThreshold: number;
  singingThreshold: number;
  musicThreshold: number;
}

/** Defaults mirror native omni_aed_overlap_config_default(). */
export const DEFAULT_AED_OVERLAP_CONFIG: AedOverlapConfig = {
  hopSecs: 2.0,
  overlapSecs: 0.25,
  edgeGuardSecs: 0.0,
  hardSplitPauseSecs: 2.0,
  maxChunkSecs: 60.0,
  hardSplitLookaheadSecs: 0.0,
  minSpeechSecs: 0.2,
  mergeGapSecs: 0.2,
  musicGapToleranceSecs: 0.0,
  padStartSecs: 0.0,
  padEndSecs: 0.0,
  speechThreshold: 0.5,
  singingThreshold: 0.5,
  musicThreshold: 0.5,
};

const SIZEOF_AED_OVERLAP_CONFIG = 56; // 10 i32 (ms) + 3 float + trailing lookahead i32
const SIZEOF_AED_ONLINE_EVENT = 32; // 2 f32 + i32 kind + u32 mask + 4 f32
const SIZEOF_AED_ONLINE_SEGMENT = 16; // 2 f32 + 2 i32

/** Re-exported for ABI self-tests. */
export const _SIZEOF_AED_OVERLAP_CONFIG = SIZEOF_AED_OVERLAP_CONFIG;
export const _SIZEOF_AED_ONLINE_EVENT = SIZEOF_AED_ONLINE_EVENT;
export const _SIZEOF_AED_ONLINE_SEGMENT = SIZEOF_AED_ONLINE_SEGMENT;

function secondsToMs(v: number): number {
  return Math.round(v * 1000);
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

function writeAedOverlapConfig(M: EmscriptenModule, ptr: number, cfg: AedOverlapConfig): void {
  M.setValue(ptr + 0,  secondsToMs(cfg.hopSecs),               "i32");
  M.setValue(ptr + 4,  secondsToMs(cfg.overlapSecs),           "i32");
  M.setValue(ptr + 8,  secondsToMs(cfg.edgeGuardSecs),         "i32");
  M.setValue(ptr + 12, secondsToMs(cfg.hardSplitPauseSecs),    "i32");
  M.setValue(ptr + 16, secondsToMs(cfg.maxChunkSecs),          "i32");
  M.setValue(ptr + 20, secondsToMs(cfg.minSpeechSecs),         "i32");
  M.setValue(ptr + 24, secondsToMs(cfg.mergeGapSecs),          "i32");
  M.setValue(ptr + 28, secondsToMs(cfg.musicGapToleranceSecs), "i32");
  M.setValue(ptr + 32, secondsToMs(cfg.padStartSecs),          "i32");
  M.setValue(ptr + 36, secondsToMs(cfg.padEndSecs),            "i32");
  M.setValue(ptr + 40, cfg.speechThreshold,                    "float");
  M.setValue(ptr + 44, cfg.singingThreshold,                   "float");
  M.setValue(ptr + 48, cfg.musicThreshold,                     "float");
  M.setValue(ptr + 52, secondsToMs(cfg.hardSplitLookaheadSecs), "i32");
}

/** One committed event (binding-level; mirrors public AEDOverlapEvent). */
export interface AedOverlapEventRecord {
  start: number;
  end: number;
  primaryKind: string;
  kindMask: number;
  speechConfidence: number;
  singingConfidence: number;
  musicConfidence: number;
  confidence: number;
  isTranscribable: boolean;
}

/** One committed transcribable segment (binding-level). */
export interface AedOverlapSegmentRecord {
  start: number;
  end: number;
  eventStartIdx: number;
  eventCount: number;
}

export interface AedOverlapResultRecord {
  segments: AedOverlapSegmentRecord[];
  events: AedOverlapEventRecord[];
}

export function aedOverlapCreate(
  M: EmscriptenModule,
  modelBuffer: ArrayBuffer,
  config: AedOverlapConfig,
): number {
  const bytes = new Uint8Array(modelBuffer);
  const dataPtr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, dataPtr);
  const cfgPtr = M._malloc(SIZEOF_AED_OVERLAP_CONFIG);
  try {
    writeAedOverlapConfig(M, cfgPtr, config);
    return createModel(
      M,
      "omni_aed_overlap_segmenter_create_from_buffer",
      ["number", "number", "number"],
      [dataPtr, bytes.length, cfgPtr],
      "AEDOverlapSegmenter",
    );
  } finally {
    M._free(dataPtr);
    M._free(cfgPtr);
  }
}

export function aedOverlapClone(M: EmscriptenModule, handle: number): number {
  const errPtr = M._malloc(4);
  try {
    const newHandle = M.ccall(
      "omni_aed_overlap_segmenter_clone",
      "number",
      ["number", "number"],
      [handle, errPtr],
    );
    if (!newHandle) {
      const err = M.getValue(errPtr, "i32");
      throw new Error(`AEDOverlapSegmenter clone failed: ${readNativeError(M, err)}`);
    }
    return newHandle;
  } finally {
    M._free(errPtr);
  }
}

/** Read the two output arrays produced by ingest/flush and free them. */
function readAedOverlapResult(
  M: EmscriptenModule,
  segPtrPtr: number,
  segCountPtr: number,
  evPtrPtr: number,
  evCountPtr: number,
): AedOverlapResultRecord {
  const evCount = M.getValue(evCountPtr, "i32");
  const evPtr = M.getValue(evPtrPtr, "i32");
  const events: AedOverlapEventRecord[] = [];
  for (let i = 0; i < evCount; i++) {
    const base = evPtr + i * SIZEOF_AED_ONLINE_EVENT;
    const kindMask = M.getValue(base + 12, "i32") >>> 0;
    events.push({
      start: round3(M.getValue(base + 0, "float")),
      end: round3(M.getValue(base + 4, "float")),
      primaryKind: AED_EVENT_KINDS[M.getValue(base + 8, "i32")] ?? "unknown",
      kindMask,
      speechConfidence: M.getValue(base + 16, "float"),
      singingConfidence: M.getValue(base + 20, "float"),
      musicConfidence: M.getValue(base + 24, "float"),
      confidence: M.getValue(base + 28, "float"),
      isTranscribable: (kindMask & AED_KIND_MASK_TRANSCRIBABLE) !== 0,
    });
  }

  const segCount = M.getValue(segCountPtr, "i32");
  const segPtr = M.getValue(segPtrPtr, "i32");
  const segments: AedOverlapSegmentRecord[] = [];
  for (let i = 0; i < segCount; i++) {
    const base = segPtr + i * SIZEOF_AED_ONLINE_SEGMENT;
    segments.push({
      start: round3(M.getValue(base + 0, "float")),
      end: round3(M.getValue(base + 4, "float")),
      eventStartIdx: M.getValue(base + 8, "i32"),
      eventCount: M.getValue(base + 12, "i32"),
    });
  }

  if (evPtr) M._free(evPtr);
  if (segPtr) M._free(segPtr);
  return { segments, events };
}

/**
 * Ingest one PCM chunk and return newly committed output.
 *
 * Caller must have already copied audio to `audioPtr` via `dispatchAudio()`;
 * `format` selects the C entry.
 */
export function aedOverlapIngest(
  M: EmscriptenModule,
  handle: number,
  audioPtr: number,
  numSamples: number,
  format: AudioFormat = "f32",
): AedOverlapResultRecord {
  const segPtrPtr = M._malloc(4);
  const segCountPtr = M._malloc(4);
  const evPtrPtr = M._malloc(4);
  const evCountPtr = M._malloc(4);
  const fn =
    format === "int16"
      ? "omni_aed_overlap_segmenter_ingest_int16"
      : "omni_aed_overlap_segmenter_ingest";
  try {
    M.setValue(segPtrPtr, 0, "i32");
    M.setValue(evPtrPtr, 0, "i32");
    const ret = M.ccall(
      fn,
      "number",
      ["number", "number", "number", "number", "number", "number", "number"],
      [handle, audioPtr, numSamples, segPtrPtr, segCountPtr, evPtrPtr, evCountPtr],
    );
    if (ret !== 0) throw new Error(`AEDOverlapSegmenter ingest failed: ${readNativeError(M, ret)}`);
    return readAedOverlapResult(M, segPtrPtr, segCountPtr, evPtrPtr, evCountPtr);
  } finally {
    M._free(segPtrPtr);
    M._free(segCountPtr);
    M._free(evPtrPtr);
    M._free(evCountPtr);
  }
}

/** Finalize the stream and return any pending segments. */
export function aedOverlapFlush(M: EmscriptenModule, handle: number): AedOverlapResultRecord {
  const segPtrPtr = M._malloc(4);
  const segCountPtr = M._malloc(4);
  const evPtrPtr = M._malloc(4);
  const evCountPtr = M._malloc(4);
  try {
    M.setValue(segPtrPtr, 0, "i32");
    M.setValue(evPtrPtr, 0, "i32");
    const ret = M.ccall(
      "omni_aed_overlap_segmenter_flush",
      "number",
      ["number", "number", "number", "number", "number"],
      [handle, segPtrPtr, segCountPtr, evPtrPtr, evCountPtr],
    );
    if (ret !== 0) throw new Error(`AEDOverlapSegmenter flush failed: ${readNativeError(M, ret)}`);
    return readAedOverlapResult(M, segPtrPtr, segCountPtr, evPtrPtr, evCountPtr);
  } finally {
    M._free(segPtrPtr);
    M._free(segCountPtr);
    M._free(evPtrPtr);
    M._free(evCountPtr);
  }
}

export function aedOverlapReset(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_aed_overlap_segmenter_reset", null, ["number"], [handle]);
}

export function aedOverlapDestroy(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_aed_overlap_segmenter_destroy", null, ["number"], [handle]);
}
