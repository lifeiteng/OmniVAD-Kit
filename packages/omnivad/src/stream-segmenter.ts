/**
 * OmniStreamSegmenter — pure-algorithm streaming VAD post-processor.
 *
 * Converts per-frame VAD probabilities into [start, end] speech segments
 * online. No model dependency — pair with OmniStreamVAD or any other prob source.
 *
 * Phase 1 supports causal post-processing steps 1-4 + 7 (force-split).
 * mergeSilenceFrames and extendSpeechFrames are not exposed here because
 * they would require unbounded lookahead.
 *
 * Typical usage:
 *
 *   const vad = await OmniStreamVAD.create();
 *   const segmenter = await OmniStreamSegmenter.create();
 *   let totalSamples = 0;
 *
 *   for (const chunk of audioChunks) {       // 160-sample Int16Array chunks
 *     const result = vad.processFrame(chunk);
 *     totalSamples += chunk.length;
 *     if (!result) continue;
 *     for (const [start, end] of segmenter.processFrame(result.confidence)) {
 *       console.log(`speech: ${start.toFixed(2)}s -> ${end.toFixed(2)}s`);
 *     }
 *   }
 *
 *   for (const [start, end] of segmenter.flush(totalSamples)) {
 *     console.log(`speech (tail): ${start.toFixed(2)}s -> ${end.toFixed(2)}s`);
 *   }
 */

import {
  getModule,
  initWasm,
  streamSegmenterCreate,
  streamSegmenterDestroy,
  streamSegmenterFlush,
  streamSegmenterGetActiveStart,
  streamSegmenterIsInSpeech,
  streamSegmenterProcessFrame,
  streamSegmenterProcessProbs,
  streamSegmenterReset,
  type PostConfig,
} from "./wasm-binding.js";

/** Configuration for OmniStreamSegmenter. Mirrors C OmniPostConfig but
 *  hides mergeSilenceFrames and extendSpeechFrames (forced to 0). */
export interface StreamSegmenterConfig {
  /** Speech activation threshold. Default: 0.4. */
  threshold?: number;
  /** Causal moving-average window in frames. Default: 5. */
  smoothWindowSize?: number;
  /** Min continuous speech frames to confirm START. Default: 20 = 200ms. */
  minSpeechFrames?: number;
  /** Min continuous silence frames to emit END. Default: 20 = 200ms. */
  minSilenceFrames?: number;
  /** Force-split active segments longer than this. Default: 3000 = 30s.
   *  Set to 0 to disable force-split. */
  maxSpeechFrames?: number;
}

const DEFAULTS: Required<StreamSegmenterConfig> = {
  threshold:        0.4,
  smoothWindowSize: 5,
  minSpeechFrames:  20,
  minSilenceFrames: 20,
  maxSpeechFrames:  3000,
};

/** A completed speech segment, in seconds. */
export type StreamSegment = { start: number; end: number };

export class OmniStreamSegmenter {
  private handle: number;
  private closed: boolean = false;

  private constructor(handle: number) {
    this.handle = handle;
  }

  /** Create a new streaming segmenter. Lazily initializes WASM if needed. */
  static async create(options: StreamSegmenterConfig = {}): Promise<OmniStreamSegmenter> {
    await initWasm();
    const M = getModule();
    const cfg: PostConfig = {
      threshold:           options.threshold        ?? DEFAULTS.threshold,
      smoothWindowSize:    options.smoothWindowSize ?? DEFAULTS.smoothWindowSize,
      minSpeechFrames:     options.minSpeechFrames  ?? DEFAULTS.minSpeechFrames,
      minSilenceFrames:    options.minSilenceFrames ?? DEFAULTS.minSilenceFrames,
      maxSpeechFrames:     options.maxSpeechFrames  ?? DEFAULTS.maxSpeechFrames,
      mergeSilenceFrames:  0,
      extendSpeechFrames:  0,
    };
    const handle = streamSegmenterCreate(M, cfg);
    return new OmniStreamSegmenter(handle);
  }

  /** Push one frame's raw probability. Returns 0+ completed segments. */
  processFrame(prob: number): StreamSegment[] {
    this.assertOpen();
    const M = getModule();
    return streamSegmenterProcessFrame(M, this.handle, prob).map(([s, e]) => ({ start: s, end: e }));
  }

  /** Push a batch of probabilities. */
  processProbs(probs: Float32Array | number[]): StreamSegment[] {
    this.assertOpen();
    const M = getModule();
    const arr = probs instanceof Float32Array ? probs : Float32Array.from(probs);
    return streamSegmenterProcessProbs(M, this.handle, arr).map(([s, e]) => ({ start: s, end: e }));
  }

  /** Flush in-progress segment at end-of-stream.
   *  @param totalSamplesSeen  cumulative int16 samples fed to upstream VAD
   *                           (0 to skip wav-duration clamping). */
  flush(totalSamplesSeen: number = 0): StreamSegment[] {
    this.assertOpen();
    const M = getModule();
    return streamSegmenterFlush(M, this.handle, totalSamplesSeen).map(([s, e]) => ({ start: s, end: e }));
  }

  /** Whether currently inside a confirmed speech segment. */
  get isInSpeech(): boolean {
    if (this.closed) return false;
    const M = getModule();
    return streamSegmenterIsInSpeech(M, this.handle);
  }

  /** Active segment's start time (seconds), or null if not in speech. */
  get activeStart(): number | null {
    if (this.closed) return null;
    const M = getModule();
    const v = streamSegmenterGetActiveStart(M, this.handle);
    return v < 0 ? null : v;
  }

  /** Reset all internal state to fresh. */
  reset(): void {
    this.assertOpen();
    const M = getModule();
    streamSegmenterReset(M, this.handle);
  }

  /** Destroy and free resources. */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    const M = getModule();
    streamSegmenterDestroy(M, this.handle);
  }

  private assertOpen(): void {
    if (this.closed) throw new Error("OmniStreamSegmenter has been closed.");
  }
}
