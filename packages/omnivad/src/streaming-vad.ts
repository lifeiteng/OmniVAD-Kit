/**
 * High-level streaming VAD: PCM chunks in, completed segments out.
 *
 * Wraps OmniStreamVAD (model inference) and OmniStreamSegmenter (state machine)
 * into a single object so callers don't have to glue them together manually.
 *
 * Typical usage:
 *
 *   const vad = await OmniStreamingVAD.create();
 *   for (const chunk of pcmChunks) {                // Int16Array, any length
 *     for (const { start, end } of vad.process(chunk)) {
 *       console.log(`speech: ${start.toFixed(2)}s -> ${end.toFixed(2)}s`);
 *     }
 *   }
 *   for (const { start, end } of vad.flush()) {
 *     console.log(`speech (tail): ${start.toFixed(2)}s -> ${end.toFixed(2)}s`);
 *   }
 *
 * If you also need per-frame confidence, use OmniStreamVAD + OmniStreamSegmenter
 * directly to share inference output across multiple consumers.
 */

import type { ModelSource } from "./types.js";
import { OmniStreamVAD } from "./stream-vad.js";
import {
  OmniStreamSegmenter,
  type StreamSegment,
  type StreamSegmenterConfig,
} from "./stream-segmenter.js";

/** Combined config: ModelSource + StreamSegmenterConfig (segmenter applies
 *  the same threshold to its smoothed-prob comparison). */
export interface StreamingVADConfig extends ModelSource, StreamSegmenterConfig {
  /** Override the threshold passed to the underlying stream VAD. Defaults
   *  to whatever the segmenter uses (default 0.4) so the VAD's is_speech
   *  bit is consistent with the segmenter's view. */
  vadThreshold?: number;
}

export class OmniStreamingVAD {
  /** omni_stream_vad_process emits at most one frame per call regardless of
   *  input size, so we split incoming chunks into 160-sample sub-frames
   *  internally. (Codex flagged this gotcha during the design review.) */
  private static readonly STREAM_VAD_HOP = 160; // 10ms @ 16kHz

  private readonly vad: OmniStreamVAD;
  private readonly segmenter: OmniStreamSegmenter;
  private totalSamplesSeen = 0;
  private closed = false;

  private constructor(vad: OmniStreamVAD, segmenter: OmniStreamSegmenter) {
    this.vad = vad;
    this.segmenter = segmenter;
  }

  static async create(options: StreamingVADConfig = {}): Promise<OmniStreamingVAD> {
    const segmenterThreshold = options.threshold ?? 0.4;
    const vadThreshold       = options.vadThreshold ?? segmenterThreshold;

    const vad = await OmniStreamVAD.create({
      modelUrl:        options.modelUrl,
      modelData:       options.modelData,
      speechThreshold: vadThreshold,
    });
    const segmenter = await OmniStreamSegmenter.create({
      threshold:        segmenterThreshold,
      smoothWindowSize: options.smoothWindowSize,
      minSpeechFrames:  options.minSpeechFrames,
      minSilenceFrames: options.minSilenceFrames,
      maxSpeechFrames:  options.maxSpeechFrames,
    });
    return new OmniStreamingVAD(vad, segmenter);
  }

  /** Push one PCM chunk of arbitrary length. Returns 0+ completed segments. */
  process(pcmChunk: Int16Array): StreamSegment[] {
    if (this.closed) throw new Error("OmniStreamingVAD has been closed.");
    this.totalSamplesSeen += pcmChunk.length;

    const out: StreamSegment[] = [];
    const hop = OmniStreamingVAD.STREAM_VAD_HOP;
    for (let offset = 0; offset < pcmChunk.length; offset += hop) {
      const sub = pcmChunk.subarray(offset, offset + hop);
      const result = this.vad.processFrame(sub);
      if (result) {
        for (const seg of this.segmenter.processFrame(result.confidence)) out.push(seg);
      }
    }
    return out;
  }

  /** Emit any in-progress segment at end-of-stream. */
  flush(): StreamSegment[] {
    if (this.closed) throw new Error("OmniStreamingVAD has been closed.");
    return this.segmenter.flush(this.totalSamplesSeen);
  }

  // ---- State queries -----------------------------------------------------

  get isInSpeech(): boolean {
    return !this.closed && this.segmenter.isInSpeech;
  }

  get activeStart(): number | null {
    return this.closed ? null : this.segmenter.activeStart;
  }

  get totalSamples(): number {
    return this.totalSamplesSeen;
  }

  // ---- Lifecycle ---------------------------------------------------------

  reset(): void {
    if (this.closed) return;
    this.segmenter.reset();
    // OmniStreamVAD does not expose reset on the JS side; recreating is the
    // only safe path for a full reset of model state. For the typical
    // long-audio use-case, reset on the segmenter alone is sufficient.
    this.totalSamplesSeen = 0;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.segmenter.close();
    // OmniStreamVAD doesn't expose explicit close in the public API; it's
    // owned by the GC. We hold no other resources to release here.
  }
}
