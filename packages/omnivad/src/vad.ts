/**
 * Non-streaming Voice Activity Detection (WASM/ncnn backend).
 *
 * Audio format: two types only. Wrappers dispatch by dtype to the matching
 * C entry — never scale or cast in JS.
 *   - Float32Array in [-1.0, 1.0] (Web Audio, soundfile, torch)
 *   - Int16Array (raw 16-bit PCM from WAV / microphone)
 */

import type { VADConfig, VADResult } from "./types.js";
import {
  initWasm,
  getModule,
  dispatchAudio,
  loadModel,
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
   * Loads model from CDN (browser), local package (Node.js), or custom source.
   */
  static async create(options: VADConfig = {}): Promise<OmniVAD> {
    await initWasm();
    const M = getModule();
    const modelBuffer = await loadModel("vad", options.modelUrl, options.modelData);
    const handle = vadCreate(M, modelBuffer);
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
    const { ptr, length, format } = dispatchAudio(M, audio);

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
