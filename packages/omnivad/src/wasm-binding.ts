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

    if (typeof globalThis.process?.versions?.node === "string") {
      // Node.js: use require for .cjs (avoids ESM detection issues)
      const { createRequire } = await import(/* webpackIgnore: true */ "module");
      const req = createRequire(import.meta.url);
      createOmniVAD = req("../dist/wasm/omnivad.cjs");
    } else {
      // Browser: dynamic import
      // @ts-expect-error Emscripten-generated module, no type declarations
      const mod = await import(/* webpackIgnore: true */ "../dist/wasm/omnivad.js");
      createOmniVAD = mod.default || mod;
    }

    const opts: Record<string, unknown> = {};
    if (wasmLocator) {
      opts.locateFile = (path: string) => wasmLocator(path);
    }

    _module = await createOmniVAD(opts);
    return _module!;
  })();

  return _loading;
}

/** Get the initialized WASM module (throws if not initialized) */
export function getModule(): EmscriptenModule {
  if (!_module) throw new Error("WASM not initialized. Call initWasm() first.");
  return _module;
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

export function vadCreate(M: EmscriptenModule, bundlePath = "models/vad.omnivad"): number {
  const handle = M.ccall(
    "omni_vad_nonstream_create_from_bundle",
    "number",
    ["string"],
    [bundlePath],
  );
  if (!handle) throw new Error("Failed to create VAD model");
  return handle;
}

/**
 * Audio format for C API selection.
 *   "int16_range" — float* in [-32768, 32767] (default, existing API)
 *   "i16"         — int16_t* PCM
 *   "f32"         — float* in [-1.0, 1.0] (Web Audio API format)
 */
export type AudioFormat = "int16_range" | "i16" | "f32";

const VAD_PROCESS_FN: Record<AudioFormat, string> = {
  int16_range: "omni_vad_nonstream_process",
  i16: "omni_vad_nonstream_process_i16",
  f32: "omni_vad_nonstream_process_f32",
};

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
  format: AudioFormat = "int16_range",
): Array<[number, number]> {
  const cfgPtr = M._malloc(SIZEOF_POST_CONFIG);
  const segPtrPtr = M._malloc(4);
  const countPtr = M._malloc(4);

  try {
    writePostConfig(M, cfgPtr, cfg);
    const ret = M.ccall(
      VAD_PROCESS_FN[format],
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
  M.ccall("omni_vad_nonstream_destroy", null, ["number"], [handle]);
}

// -------------------------------------------------------------------------- //
//  Non-stream AED                                                             //
// -------------------------------------------------------------------------- //

const AED_CLASSES: Record<number, string> = { 0: "speech", 1: "singing", 2: "music" };

export function aedCreate(M: EmscriptenModule, bundlePath = "models/aed.omnivad"): number {
  const handle = M.ccall(
    "omni_aed_nonstream_create_from_bundle",
    "number",
    ["string"],
    [bundlePath],
  );
  if (!handle) throw new Error("Failed to create AED model");
  return handle;
}

export interface AedPostConfig {
  speech: PostConfig;
  singing: PostConfig;
  music: PostConfig;
}

const AED_PROCESS_FN: Record<AudioFormat, string> = {
  int16_range: "omni_aed_nonstream_process",
  i16: "omni_aed_nonstream_process_i16",
  f32: "omni_aed_nonstream_process_f32",
};

export function aedDetect(
  M: EmscriptenModule,
  handle: number,
  audioPtr: number,
  numSamples: number,
  cfg: AedPostConfig,
  format: AudioFormat = "int16_range",
): Record<string, Array<[number, number]>> {
  const cfgPtr = M._malloc(SIZEOF_AED_POST_CONFIG);
  const segPtrPtr = M._malloc(4);
  const countPtr = M._malloc(4);

  try {
    writePostConfig(M, cfgPtr, cfg.speech);
    writePostConfig(M, cfgPtr + SIZEOF_POST_CONFIG, cfg.singing);
    writePostConfig(M, cfgPtr + 2 * SIZEOF_POST_CONFIG, cfg.music);

    const ret = M.ccall(
      AED_PROCESS_FN[format],
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
  M.ccall("omni_aed_nonstream_destroy", null, ["number"], [handle]);
}

// -------------------------------------------------------------------------- //
//  Stream VAD                                                                 //
// -------------------------------------------------------------------------- //

export function streamVadCreate(
  M: EmscriptenModule,
  threshold = 0.5,
  bundlePath = "models/stream-vad.omnivad",
): number {
  const handle = M.ccall(
    "omni_vad_stream_create_from_bundle",
    "number",
    ["string", "number"],
    [bundlePath, threshold],
  );
  if (!handle) throw new Error("Failed to create StreamVAD model");
  return handle;
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
  // OmniVadStreamResult: { float confidence, bool is_speech, int frame_offset } = 12 bytes
  const resultPtr = M._malloc(12);
  try {
    const ret = M.ccall(
      "omni_vad_stream_process",
      "number",
      ["number", "number", "number", "number"],
      [handle, pcm16Ptr, numSamples, resultPtr],
    );
    if (ret === -6) return null; // OMNI_ERR_NO_FRAMES
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
  M.ccall("omni_vad_stream_reset", null, ["number"], [handle]);
}

export function streamVadDestroy(M: EmscriptenModule, handle: number): void {
  M.ccall("omni_vad_stream_destroy", null, ["number"], [handle]);
}
