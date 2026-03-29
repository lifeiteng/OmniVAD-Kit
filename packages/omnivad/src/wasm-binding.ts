/**
 * Low-level WASM binding for omnivad C API.
 * Loads the Emscripten module and provides typed wrappers.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type EmscriptenModule = any;

let _module: EmscriptenModule | null = null;
let _loading: Promise<EmscriptenModule> | null = null;

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
const OMNI_ERR_NO_FRAMES = -7;

/** Package version — used to construct default CDN URLs. */
export const VERSION = "0.2.1";

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
      const { createRequire } = await import(/* webpackIgnore: true */ "module");
      const { dirname, join } = await import("path");
      const req = createRequire(import.meta.url);
      const gluePath = req.resolve("../dist/wasm/omnivad.cjs");
      const wasmDir = dirname(gluePath);
      createOmniVAD = req(gluePath);
      defaultLocateFile = (filename: string) => join(wasmDir, filename);
    } else {
      // Browser: dynamic import
      const glueUrl = new URL("../dist/wasm/omnivad.js", import.meta.url);
      const mod = await import(/* webpackIgnore: true */ glueUrl.href);
      createOmniVAD = mod.default || mod;
      const wasmBaseUrl = new URL("./", glueUrl);
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
    const { createRequire } = await import(/* webpackIgnore: true */ "module");
    const { dirname, join } = await import("path");
    const { readFile } = await import("fs/promises");
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

/** Copy Float32Array audio into WASM heap, returns pointer. Caller must free. */
export function copyAudioToHeap(M: EmscriptenModule, audio: Float32Array): number {
  const ptr = M._malloc(audio.length * 4);
  const heap = new Float32Array(M.HEAPU8.buffer, ptr, audio.length);
  heap.set(audio);
  return ptr;
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
  maxSpeechFrames: 2000,
  mergeSilenceFrames: 0,
  extendSpeechFrames: 0,
};

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

/**
 * Audio format: two types only.
 *   "f32"   — float* in [-1.0, 1.0] (Web Audio API, soundfile, torch)
 *   "int16" — int16_t* PCM (WAV files, microphones)
 */
export type AudioFormat = "f32" | "int16";

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

export function streamVadCreate(
  M: EmscriptenModule,
  modelBuffer: ArrayBuffer,
  threshold = 0.5,
): number {
  const bytes = new Uint8Array(modelBuffer);
  const ptr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, ptr);
  try {
    return createModel(
      M,
      "omni_stream_vad_create_from_buffer",
      ["number", "number", "number"],
      [ptr, bytes.length, threshold],
      "StreamVAD",
    );
  } finally {
    M._free(ptr);
  }
}

export interface StreamVadResult {
  confidence: number;
  isSpeech: boolean;
  frameOffset: number;
}

/** Process one chunk of int16 PCM (160 samples = 10ms). Returns null if buffering. */
export function streamVadProcess(
  M: EmscriptenModule,
  handle: number,
  pcm16Ptr: number,
  numSamples: number,
): StreamVadResult | null {
  // OmniStreamVadResult: { float confidence, bool is_speech, int frame_offset } = 12 bytes
  const resultPtr = M._malloc(12);
  try {
    const ret = M.ccall(
      "omni_stream_vad_process",
      "number",
      ["number", "number", "number", "number"],
      [handle, pcm16Ptr, numSamples, resultPtr],
    );
    if (ret === OMNI_ERR_NO_FRAMES) return null;
    if (ret !== 0) throw new Error(`StreamVAD process failed: ${ret}`);
    return {
      confidence: M.getValue(resultPtr, "float"),
      isSpeech: M.getValue(resultPtr + 4, "i8") !== 0,
      frameOffset: M.getValue(resultPtr + 8, "i32"),
    };
  } finally {
    M._free(resultPtr);
  }
}

export function streamVadReset(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_stream_vad_reset", null, ["number"], [handle]);
}

export function streamVadDestroy(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_stream_vad_destroy", null, ["number"], [handle]);
}
