/**
 * Non-streaming Voice Activity Detection (WASM/ncnn backend).
 *
 * Audio format:
 *   - Int16Array: raw 16-bit PCM (most efficient, zero conversion)
 *   - Float32Array in [-1.0, 1.0]: normalized audio (Web Audio API format)
 *   - Float32Array in [-32768, 32767]: int16-range float (legacy/internal)
 *
 * Format is auto-detected: if max(abs(values)) <= 1.0, treated as normalized float.
 */

import type { VADConfig, VADResult } from "./types.js";
import {
  initWasm,
  getModule,
  copyAudioToHeap,
  vadCreate,
  vadDetect,
  vadDestroy,
  DEFAULT_VAD_CONFIG,
  type AudioFormat,
  type PostConfig,
} from "./wasm-binding.js";

const SAMPLE_RATE = 16000;

export class OmniVAD {
  private handle: number;
  private config: PostConfig;

  private constructor(handle: number, config: PostConfig) {
    this.handle = handle;
    this.config = config;
  }

  /**
   * Create a new OmniVAD instance.
   * Initializes WASM and loads the bundled ncnn model.
   */
  static async create(options: VADConfig = {}): Promise<OmniVAD> {
    await initWasm();
    const M = getModule();
    const handle = vadCreate(M);
    const config: PostConfig = {
      threshold: options.speechThreshold ?? DEFAULT_VAD_CONFIG.threshold,
      smoothWindowSize: options.smoothWindowSize ?? DEFAULT_VAD_CONFIG.smoothWindowSize,
      minSpeechFrames: options.minSpeechFrames ?? DEFAULT_VAD_CONFIG.minSpeechFrames,
      minSilenceFrames: options.minSilenceFrames ?? DEFAULT_VAD_CONFIG.minSilenceFrames,
      maxSpeechFrames: options.maxSpeechFrames ?? DEFAULT_VAD_CONFIG.maxSpeechFrames,
      mergeSilenceFrames: options.mergeSilenceFrames ?? DEFAULT_VAD_CONFIG.mergeSilenceFrames,
      extendSpeechFrames: options.extendSpeechFrames ?? DEFAULT_VAD_CONFIG.extendSpeechFrames,
    };
    return new OmniVAD(handle, config);
  }

  /**
   * Detect speech segments in audio.
   *
   * Accepts Int16Array (PCM), Float32Array [-1,1] (Web Audio), or
   * Float32Array [-32768,32767] (legacy). Format is auto-detected.
   */
  detect(audio: Float32Array | Int16Array): VADResult {
    const M = getModule();
    const { ptr, length, format } = prepareAudio(M, audio);

    try {
      const timestamps = vadDetect(M, this.handle, ptr, length, this.config, format);
      return {
        duration: Math.round((length / SAMPLE_RATE) * 1000) / 1000,
        timestamps,
      };
    } finally {
      M._free(ptr);
    }
  }

  /** Release native resources. */
  dispose(): void {
    if (this.handle) {
      vadDestroy(getModule(), this.handle);
      this.handle = 0;
    }
  }
}

/** Copy audio to WASM heap with format detection. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function prepareAudio(M: any, audio: Float32Array | Int16Array): { ptr: number; length: number; format: AudioFormat } {
  if (audio instanceof Int16Array) {
    // Int16 PCM → copy as int16, use _i16 C API
    const ptr = M._malloc(audio.length * 2);
    const heap = new Int16Array(M.HEAPU8.buffer, ptr, audio.length);
    heap.set(audio);
    return { ptr, length: audio.length, format: "i16" };
  }

  // Float32Array — detect range
  const format = detectFloatFormat(audio);
  const ptr = M._malloc(audio.length * 4);
  const heap = new Float32Array(M.HEAPU8.buffer, ptr, audio.length);
  heap.set(audio);
  return { ptr, length: audio.length, format };
}

function detectFloatFormat(audio: Float32Array): AudioFormat {
  // Sample up to 1000 values to detect range
  const step = Math.max(1, Math.floor(audio.length / 1000));
  let maxAbs = 0;
  for (let i = 0; i < audio.length; i += step) {
    const v = Math.abs(audio[i]);
    if (v > maxAbs) maxAbs = v;
  }
  // If max value <= 1.0, it's normalized [-1,1]; otherwise it's int16-range
  return maxAbs <= 1.0 ? "f32" : "int16_range";
}
