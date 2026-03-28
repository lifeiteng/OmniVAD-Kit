/**
 * Audio Event Detection: speech, singing, music (WASM/ncnn backend).
 *
 * Audio format: same as OmniVAD — Int16Array, Float32Array [-1,1], or Float32Array int16-range.
 * Auto-detected.
 */

import type { AEDConfig, AEDResult } from "./types.js";
import {
  initWasm,
  getModule,
  aedCreate,
  aedDetect,
  aedDestroy,
  DEFAULT_VAD_CONFIG,
  type AudioFormat,
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
   *
   * Accepts Int16Array (PCM), Float32Array [-1,1] (Web Audio), or
   * Float32Array [-32768,32767] (legacy). Format is auto-detected.
   */
  detect(audio: Float32Array | Int16Array): AEDResult {
    const M = getModule();
    const { ptr, length, format } = prepareAudio(M, audio);

    try {
      const events = aedDetect(M, this.handle, ptr, length, this.config, format);
      return {
        duration: Math.round((length / SAMPLE_RATE) * 1000) / 1000,
        events,
        ratios: {},
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function prepareAudio(M: any, audio: Float32Array | Int16Array): { ptr: number; length: number; format: AudioFormat } {
  if (audio instanceof Int16Array) {
    const ptr = M._malloc(audio.length * 2);
    const heap = new Int16Array(M.HEAPU8.buffer, ptr, audio.length);
    heap.set(audio);
    return { ptr, length: audio.length, format: "i16" };
  }

  const format = detectFloatFormat(audio);
  const ptr = M._malloc(audio.length * 4);
  const heap = new Float32Array(M.HEAPU8.buffer, ptr, audio.length);
  heap.set(audio);
  return { ptr, length: audio.length, format };
}

function detectFloatFormat(audio: Float32Array): AudioFormat {
  const step = Math.max(1, Math.floor(audio.length / 1000));
  let maxAbs = 0;
  for (let i = 0; i < audio.length; i += step) {
    const v = Math.abs(audio[i]);
    if (v > maxAbs) maxAbs = v;
  }
  return maxAbs <= 1.0 ? "f32" : "int16_range";
}
