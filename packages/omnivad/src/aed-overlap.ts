/**
 * AED overlap segmenter: pseudo-streaming whole-window AED that commits
 * transcribable segments (speech/singing) as audio is fed in chunk by chunk.
 *
 * Mirrors the native C `omni_aed_overlap_segmenter_*` API and the Python
 * `AedOverlapSegmenter`. Audio format is the same as the other model classes —
 * Float32Array in [-1, 1] or Int16Array PCM; wrappers dispatch by dtype and
 * all scaling lives in the C entries.
 */

import type {
  AEDOverlapConfig,
  AEDOverlapEvent,
  AEDOverlapResult,
  AEDOverlapSegment,
} from "./types.js";
import {
  initWasm,
  getModule,
  dispatchAudio,
  loadModel,
  aedOverlapCreate,
  aedOverlapClone,
  aedOverlapIngest,
  aedOverlapFlush,
  aedOverlapReset,
  aedOverlapDestroy,
  DEFAULT_AED_OVERLAP_CONFIG,
  type AedOverlapConfig as ResolvedConfig,
  type AedOverlapResultRecord,
} from "./wasm-binding.js";

/** Resolve a public config (all-optional, seconds) against native defaults. */
function resolveConfig(options: AEDOverlapConfig): ResolvedConfig {
  return {
    hopSecs: options.hopSecs ?? DEFAULT_AED_OVERLAP_CONFIG.hopSecs,
    overlapSecs: options.overlapSecs ?? DEFAULT_AED_OVERLAP_CONFIG.overlapSecs,
    edgeGuardSecs: options.edgeGuardSecs ?? DEFAULT_AED_OVERLAP_CONFIG.edgeGuardSecs,
    hardSplitPauseSecs: options.hardSplitPauseSecs ?? DEFAULT_AED_OVERLAP_CONFIG.hardSplitPauseSecs,
    maxChunkSecs: options.maxChunkSecs ?? DEFAULT_AED_OVERLAP_CONFIG.maxChunkSecs,
    hardSplitLookaheadSecs:
      options.hardSplitLookaheadSecs ?? DEFAULT_AED_OVERLAP_CONFIG.hardSplitLookaheadSecs,
    minSpeechSecs: options.minSpeechSecs ?? DEFAULT_AED_OVERLAP_CONFIG.minSpeechSecs,
    mergeGapSecs: options.mergeGapSecs ?? DEFAULT_AED_OVERLAP_CONFIG.mergeGapSecs,
    musicGapToleranceSecs:
      options.musicGapToleranceSecs ?? DEFAULT_AED_OVERLAP_CONFIG.musicGapToleranceSecs,
    padStartSecs: options.padStartSecs ?? DEFAULT_AED_OVERLAP_CONFIG.padStartSecs,
    padEndSecs: options.padEndSecs ?? DEFAULT_AED_OVERLAP_CONFIG.padEndSecs,
    speechThreshold: options.speechThreshold ?? DEFAULT_AED_OVERLAP_CONFIG.speechThreshold,
    singingThreshold: options.singingThreshold ?? DEFAULT_AED_OVERLAP_CONFIG.singingThreshold,
    musicThreshold: options.musicThreshold ?? DEFAULT_AED_OVERLAP_CONFIG.musicThreshold,
  };
}

/** Binding records and public types share the same shape; cast directly. */
function toResult(record: AedOverlapResultRecord): AEDOverlapResult {
  return {
    segments: record.segments as AEDOverlapSegment[],
    events: record.events as AEDOverlapEvent[],
  };
}

export class OmniAEDOverlapSegmenter {
  private handle: number;

  private constructor(handle: number) {
    this.handle = handle;
  }

  /**
   * Create a new AED overlap segmenter.
   * Loads the AED model from CDN (browser), local package (Node.js), or a
   * custom source.
   */
  static async create(options: AEDOverlapConfig = {}): Promise<OmniAEDOverlapSegmenter> {
    await initWasm();
    const M = getModule();
    const modelBuffer = await loadModel("aed", options.modelUrl, options.modelData);
    const handle = aedOverlapCreate(M, modelBuffer, resolveConfig(options));
    return new OmniAEDOverlapSegmenter(handle);
  }

  /**
   * Ingest one PCM chunk and return newly committed output.
   *
   * Accepts Int16Array (PCM) or normalized Float32Array in [-1, 1].
   */
  ingest(audio: Float32Array | Int16Array): AEDOverlapResult {
    const M = getModule();
    const { ptr, length, format } = dispatchAudio(M, audio);
    try {
      return toResult(aedOverlapIngest(M, this.handle, ptr, length, format));
    } finally {
      M._free(ptr);
    }
  }

  /** Finalize the stream and return any pending segments. */
  flush(): AEDOverlapResult {
    return toResult(aedOverlapFlush(getModule(), this.handle));
  }

  /** Clear buffered audio and emitted state, keeping the loaded model. */
  reset(): void {
    aedOverlapReset(getModule(), this.handle);
  }

  /** Create a new segmenter with the same model/config and fresh state. */
  clone(): OmniAEDOverlapSegmenter {
    const newHandle = aedOverlapClone(getModule(), this.handle);
    return new OmniAEDOverlapSegmenter(newHandle);
  }

  /** Release native resources. */
  dispose(): void {
    if (this.handle) {
      aedOverlapDestroy(getModule(), this.handle);
      this.handle = 0;
    }
  }
}
