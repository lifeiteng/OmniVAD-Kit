/**
 * Audio Event Detection: speech, singing, music (WASM/ncnn backend).
 *
 * Audio format: same as OmniVAD — Int16Array or normalized Float32Array [-1, 1].
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
   * Accepts Int16Array (PCM) or normalized Float32Array in [-1, 1].
   */
  detect(audio: Float32Array | Int16Array): AEDResult {
    const M = getModule();
    const { ptr, length, format } = prepareAudio(M, audio);
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function prepareAudio(M: any, audio: Float32Array | Int16Array): { ptr: number; length: number; format: AudioFormat } {
  const f32 = audio instanceof Int16Array ? int16ToNormalizedFloat32(audio) : audio;
  const ptr = M._malloc(f32.length * 4);
  const heap = new Float32Array(M.HEAPU8.buffer, ptr, f32.length);
  heap.set(f32);
  return { ptr, length: f32.length, format: "f32" };
}

function int16ToNormalizedFloat32(i16: Int16Array): Float32Array {
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
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
