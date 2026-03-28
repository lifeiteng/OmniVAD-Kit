/**
 * Audio Event Detection: speech, singing, music (WASM/ncnn backend).
 */

import type { AEDConfig, AEDResult } from "./types.js";
import {
  initWasm,
  getModule,
  copyAudioToHeap,
  aedCreate,
  aedDetect,
  aedDestroy,
  DEFAULT_VAD_CONFIG,
  type AedPostConfig,
} from "./wasm-binding.js";

const SAMPLE_RATE = 16000;

export class OmniAED {
  private handle: number;
  private config: AedPostConfig;

  private constructor(handle: number, config: AedPostConfig) {
    this.handle = handle;
    this.config = config;
  }

  /**
   * Create a new OmniAED instance.
   * Initializes WASM and loads the bundled ncnn model.
   */
  static async create(options: AEDConfig = {}): Promise<OmniAED> {
    await initWasm();
    const M = getModule();
    const handle = aedCreate(M);

    const base = {
      smoothWindowSize: options.smoothWindowSize ?? DEFAULT_VAD_CONFIG.smoothWindowSize,
      minSpeechFrames: options.minSpeechFrames ?? DEFAULT_VAD_CONFIG.minSpeechFrames,
      minSilenceFrames: options.minSilenceFrames ?? DEFAULT_VAD_CONFIG.minSilenceFrames,
      maxSpeechFrames: options.maxSpeechFrames ?? DEFAULT_VAD_CONFIG.maxSpeechFrames,
      mergeSilenceFrames: options.mergeSilenceFrames ?? DEFAULT_VAD_CONFIG.mergeSilenceFrames,
      extendSpeechFrames: options.extendSpeechFrames ?? DEFAULT_VAD_CONFIG.extendSpeechFrames,
    };

    const config: AedPostConfig = {
      speech: { ...base, threshold: options.speechThreshold ?? 0.4 },
      singing: { ...base, threshold: options.singingThreshold ?? 0.5 },
      music: { ...base, threshold: options.musicThreshold ?? 0.5 },
    };
    return new OmniAED(handle, config);
  }

  /**
   * Detect audio events (speech, singing, music).
   * @param audio - Float32Array (int16 range) or Int16Array of 16kHz mono PCM
   */
  detect(audio: Float32Array | Int16Array): AEDResult {
    const M = getModule();
    const f32 = audio instanceof Int16Array ? int16ToFloat32(audio) : audio;
    const audioPtr = copyAudioToHeap(M, f32);

    try {
      const events = aedDetect(M, this.handle, audioPtr, f32.length, this.config);
      return {
        duration: Math.round((f32.length / SAMPLE_RATE) * 1000) / 1000,
        events,
        ratios: {},
      };
    } finally {
      M._free(audioPtr);
    }
  }

  /** Release native resources. */
  dispose(): void {
    if (this.handle) {
      aedDestroy(getModule(), this.handle);
      this.handle = 0;
    }
  }
}

function int16ToFloat32(i16: Int16Array): Float32Array {
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i];
  return f32;
}
