// OmniVAD TypeScript types

/** Result from non-streaming VAD detection */
export interface VADResult {
  /** Audio duration in seconds */
  duration: number;
  /** Array of [start, end] timestamp pairs in seconds */
  timestamps: [number, number][];
}

/** Result from Audio Event Detection (3-class) */
export interface AEDResult {
  /** Audio duration in seconds */
  duration: number;
  /** Events keyed by type ("speech", "singing", "music") with timestamp pairs */
  events: Record<string, [number, number][]>;
  /** Detected duration coverage ratio for each event type */
  ratios: Record<string, number>;
}

/** Per-frame result from streaming VAD */
export interface StreamVADFrameResult {
  /** Raw probability from model output */
  confidence: number;
  /** Currently identical to confidence; reserved for future smoothing */
  smoothedConfidence: number;
  /** Whether current frame is classified as speech */
  isSpeech: boolean;
  /** 1-based frame index of the emitted frame */
  frameIndex: number;
  /** True when speech becomes active at this frame */
  isSpeechStart: boolean;
  /** True when speech ends on the previous frame */
  isSpeechEnd: boolean;
  /** Start frame of the active or just-finished speech segment */
  speechStartFrame: number;
  /** End frame of the just-finished speech segment, or 0 if not ending */
  speechEndFrame: number;
}

/** Full-audio streaming-model output */
export interface StreamVADFullResult {
  /** Per-frame speech probabilities */
  probabilities: Float32Array;
  /** Number of emitted frames */
  numFrames: number;
  /** Audio duration in seconds */
  duration: number;
}

/** Model source options shared by all model types. */
export interface ModelSource {
  /** URL to fetch the .omnivad model from (overrides default CDN). */
  modelUrl?: string | URL;
  /** Pre-loaded model data (skips fetch entirely). */
  modelData?: ArrayBuffer;
}

/** Configuration for non-streaming VAD */
export interface VADConfig extends ModelSource {
  /** Speech probability threshold (default: 0.4) */
  speechThreshold?: number;
  /** Smoothing window size in frames (default: 5) */
  smoothWindowSize?: number;
  /** Minimum speech segment length in frames (default: 20) */
  minSpeechFrames?: number;
  /** Maximum speech segment length in frames before splitting (default: 2000 = 20s) */
  maxSpeechFrames?: number;
  /** Minimum silence segment length in frames for state machine (default: 20) */
  minSilenceFrames?: number;
  /** Merge silence segments shorter than this (default: 0 = disabled) */
  mergeSilenceFrames?: number;
  /** Extend speech segments by this many frames on each side (default: 0) */
  extendSpeechFrames?: number;
}

/** Configuration for Audio Event Detection */
export interface AEDConfig extends VADConfig {
  /** Singing probability threshold (default: 0.5) */
  singingThreshold?: number;
  /** Music probability threshold (default: 0.5) */
  musicThreshold?: number;
}

/** Configuration for streaming VAD */
export interface StreamVADConfig extends ModelSource {
  /** Speech probability threshold (default: 0.5) */
  speechThreshold?: number;
}

/**
 * Chunk packing strategy. Both modes honor `chunkSize` and `maxGap` as
 * hard constraints — they only differ in WHERE the cut lands.
 *
 * - `"greedy"` — sequential append; cuts at the first point that violates
 *   a constraint. Recommended for **fixed-length-input ASR** like Whisper /
 *   whisperX (which pad to 30s anyway).
 * - `"longest_gap"` — recursive split at the longest internal pause until
 *   every chunk satisfies both constraints. Falls back to equal hard-split
 *   when a single segment exceeds `chunkSize`. Recommended for
 *   **variable-length-input models** (forced alignment, TTS, encoder-style
 *   ASR) — splits at natural pauses, no fixed-length padding required.
 */
export type ChunkMode = "greedy" | "longest_gap";

/**
 * Configuration for {@link mergeChunks}. Mirrors C struct OmniChunkConfig.
 * All fields are optional in the public API; defaults match
 * {@link DEFAULT_CHUNK_CONFIG}.
 */
export interface ChunkOptions {
  /** Hard upper bound on chunk duration in seconds. Must be > 0. Default: 30. */
  chunkSize?: number;
  /** Split if the gap between adjacent segments exceeds this. Pass `Infinity`
   *  to disable. Default: `Infinity`. Honored by both modes. */
  maxGap?: number;
  /** Extend each chunk start backward by this many seconds (clamped to >= 0).
   *  Default: 0.04. */
  padOnset?: number;
  /** Extend each chunk end forward by this many seconds. Default: 0.04. */
  padOffset?: number;
  /** Drop input segments shorter than this many seconds. Default: 0.0. */
  minDurationOn?: number;
  /** Pre-merge consecutive segments whose silence gap is shorter than this.
   *  Default: 0.24. */
  minDurationOff?: number;
  /** Packing strategy. Default: `"greedy"`. */
  mode?: ChunkMode;
}

/** A single chunk emitted by {@link mergeChunks}. */
export interface ChunkResult {
  /** Chunk start time (seconds), with `padOnset` applied (clamped to >= 0). */
  start: number;
  /** Chunk end time (seconds), with `padOffset` applied. */
  end: number;
  /** Index of the first input segment included in this chunk. Refers to the
   *  *post-filter* segment list — segments dropped by `minDurationOn` and
   *  pre-merged by `minDurationOff` are not counted. */
  segStartIdx: number;
  /** Number of input segments included in this chunk. */
  segCount: number;
}
