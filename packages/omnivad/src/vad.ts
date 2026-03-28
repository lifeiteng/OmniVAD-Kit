/**
 * Non-streaming Voice Activity Detection (WASM/ncnn backend).
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
   * @param audio - Float32Array (int16 range) or Int16Array of 16kHz mono PCM
   */
  detect(audio: Float32Array | Int16Array): VADResult {
    const M = getModule();
    const f32 = audio instanceof Int16Array ? int16ToFloat32(audio) : audio;
    const audioPtr = copyAudioToHeap(M, f32);

    try {
      const timestamps = vadDetect(M, this.handle, audioPtr, f32.length, this.config);
      return {
        duration: Math.round((f32.length / SAMPLE_RATE) * 1000) / 1000,
        timestamps,
      };
    } finally {
      M._free(audioPtr);
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

function int16ToFloat32(i16: Int16Array): Float32Array {
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i];
  return f32;
}
