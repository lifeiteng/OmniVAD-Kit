/**
 * Non-streaming Voice Activity Detection (WASM/ncnn backend).
 *
 * Audio format:
 *   - Int16Array: raw 16-bit PCM, converted to normalized float internally
 *   - Float32Array in [-1.0, 1.0]: normalized audio (Web Audio API format)
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
   * Accepts Int16Array (PCM) or normalized Float32Array in [-1, 1].
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

/** Copy audio to WASM heap as normalized float audio. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function prepareAudio(M: any, audio: Float32Array | Int16Array): { ptr: number; length: number; format: AudioFormat } {
  const f32 = audio instanceof Int16Array ? int16ToNormalizedFloat32(audio) : audio;
  const ptr = copyAudioToHeap(M, f32);
  return { ptr, length: f32.length, format: "f32" };
}

function int16ToNormalizedFloat32(i16: Int16Array): Float32Array {
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
}
