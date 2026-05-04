/**
 * Audio Event Detection: speech, singing, music (WASM/ncnn backend).
 *
 * Audio format: same as OmniVAD — Float32Array in [-1, 1] or Int16Array PCM.
 * Wrappers dispatch by dtype; all scaling lives in the C entries.
 */

import type { AEDConfig, AEDResult } from "./types.js";
import {
  initWasm,
  getModule,
  dispatchAudio,
  loadModel,
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
   * Loads model from CDN (browser), local package (Node.js), or custom source.
   */
  static async create(options: AEDConfig = {}): Promise<OmniAED> {
    await initWasm();
    const M = getModule();
    const modelBuffer = await loadModel("aed", options.modelUrl, options.modelData);
    const handle = aedCreate(M, modelBuffer);

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
   *
   * Accepts Int16Array (PCM) or normalized Float32Array in [-1, 1].
   */
  detect(audio: Float32Array | Int16Array): AEDResult {
    const M = getModule();
    const { ptr, length, format } = dispatchAudio(M, audio);
    const duration = Math.round((length / SAMPLE_RATE) * 1000) / 1000;

    try {
      const events = aedDetect(M, this.handle, ptr, length, this.config, format);
      return {
        duration,
        events,
        ratios: computeCoverageRatios(events, duration),
      };
    } finally {
      M._free(ptr);
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

function computeCoverageRatios(
  events: Record<string, Array<[number, number]>>,
  duration: number,
): Record<string, number> {
  const ratios: Record<string, number> = {
    speech: 0,
    singing: 0,
    music: 0,
  };

  if (duration <= 0) return ratios;

  for (const cls of Object.keys(ratios)) {
    const covered = (events[cls] ?? []).reduce((sum, [start, end]) => sum + Math.max(0, end - start), 0);
    ratios[cls] = Math.round(Math.min(1, covered / duration) * 1e6) / 1e6;
  }

  return ratios;
}
