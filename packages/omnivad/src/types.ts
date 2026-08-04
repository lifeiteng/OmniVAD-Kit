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

/** Per-frame result from streaming VAD.
 *
 *  Bit-identical to upstream FireRedVAD's StreamVadFrameResult: every
 *  successful processFrame() call carries both per-frame probabilities
 *  AND segment-boundary events (no external segmenter needed). */
export interface StreamVADFrameResult {
  /** Raw probability from model output [0, 1] */
  confidence: number;
  /** Causal moving-average of confidence (window = smoothWindowSize) */
  smoothedProb: number;
  /** smoothedProb >= threshold */
  isSpeech: boolean;
  /** 1-based frame index of the emitted frame */
  frameIndex: number;
  /** True on the frame that confirms a new SPEECH segment */
  isSpeechStart: boolean;
  /** True on the frame that confirms a SPEECH segment end */
  isSpeechEnd: boolean;
  /** 1-based start frame of the segment when isSpeechStart, else -1 */
  speechStartFrame: number;
  /** 1-based end frame of the segment when isSpeechEnd, else -1 */
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
  /** Maximum speech segment length in frames before splitting (default: 3000 = 30s; matches Whisper) */
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

/** Configuration for streaming VAD.
 *
 *  Bit-identical to upstream FireRedStreamVadConfig — every parameter
 *  has the same name (without the speech_ prefix) and the same default. */
export interface StreamVADConfig extends ModelSource {
  /** Speech activation threshold [0, 1] (default: 0.5). */
  threshold?: number;
  /** Causal moving-average window in frames (default: 5). */
  smoothWindowSize?: number;
  /** Extend confirmed segment START backward by N frames (default: 5;
   *  clamped to >= smoothWindowSize internally). */
  padStartFrame?: number;
  /** Min continuous speech frames to confirm START (default: 8 = 80ms). */
  minSpeechFrame?: number;
  /** Force-split when SPEECH-state count hits this (default: 2000 = 20s). */
  maxSpeechFrame?: number;
  /** Min continuous silence frames to confirm END (default: 20 = 200ms). */
  minSilenceFrame?: number;
}

/**
 * Chunk packing strategy. Both modes honor `maxChunkSecs` and `maxGapSecs` as
 * hard constraints — they only differ in WHERE the cut lands.
 *
 * - `"greedy"` — sequential append; cuts at the first point that violates
 *   a constraint. Recommended for **fixed-length-input ASR** like Whisper /
 *   whisperX (which pad to 30s anyway).
 * - `"longest_gap"` — recursive split at the longest internal pause until
 *   every chunk satisfies both constraints. Falls back to equal hard-split
 *   when a single segment exceeds `maxChunkSecs`. Recommended for
 *   **variable-length-input models** (forced alignment, TTS, encoder-style
 *   ASR) — splits at natural pauses, no fixed-length padding required.
 *   **NOTE: This is NOT how WhisperX packs chunks** — WhisperX uses greedy
 *   packing (`Binarize(max_duration=...)` + sequential append). For
 *   WhisperX-equivalent behavior pass `mode: "greedy"` (the default).
 */
export type ChunkMode = "greedy" | "longest_gap";

/**
 * Configuration for {@link mergeChunks}. Mirrors C struct OmniChunkConfig.
 * All fields are optional in the public API; defaults match
 * {@link DEFAULT_CHUNK_CONFIG}.
 */
export interface ChunkOptions {
  /** Hard upper bound on chunk duration in seconds. Must be > 0. Default: 30. */
  maxChunkSecs?: number;
  /** Split if the gap between adjacent segments exceeds this. Pass `Infinity`
   *  to disable. Default: `Infinity`. Honored by both modes. */
  maxGapSecs?: number;
  /** Extend each chunk start backward by this many seconds (clamped to >= 0).
   *  Default: 0.04. */
  padOnsetSecs?: number;
  /** Extend each chunk end forward by this many seconds. Default: 0.04. */
  padOffsetSecs?: number;
  /** Drop input segments shorter than this many seconds. Default: 0.0.
   *  Pairs with VAD `minSpeechFrames` (frame-domain equivalent). */
  minSpeechSecs?: number;
  /** Pre-merge consecutive segments whose silence gap is shorter than this.
   *  Default: 0.20 (matches VAD `minSilenceFrames=20` @ 10ms frame shift). */
  minSilenceSecs?: number;
  /** Packing strategy. Default: `"greedy"`. */
  mode?: ChunkMode;
}

/**
 * Configuration for {@link OmniAEDOverlapSegmenter}. Mirrors the native C
 * struct `OmniAedOverlapConfig` (all durations are seconds in the public API
 * and converted to milliseconds at the C boundary). All fields are optional;
 * defaults match the native `omni_aed_overlap_config_default()`.
 */
export interface AEDOverlapConfig extends ModelSource {
  /** AED window advance per ingest step, in seconds. Default: 2.0. */
  hopSecs?: number;
  /** Overlap retained between adjacent windows, in seconds. Default: 0.25. */
  overlapSecs?: number;
  /** Drop probabilities within this margin of a window edge, in seconds. Default: 0.0. */
  edgeGuardSecs?: number;
  /** Force a transcribable-segment boundary after a silence pause this long, in seconds. Default: 2.0. */
  hardSplitPauseSecs?: number;
  /** Regular hard-split boundary for a transcribable chunk, in seconds. Default: 60.0. */
  maxChunkSecs?: number;
  /** Delay committing maxChunkSecs by this much audio so a shorter final tail can stay merged. Default: 0.0. */
  hardSplitLookaheadSecs?: number;
  /** Drop committed events shorter than this, in seconds. Default: 0.2. */
  minSpeechSecs?: number;
  /** Merge adjacent same-kind events separated by a gap shorter than this, in seconds. Default: 0.2. */
  mergeGapSecs?: number;
  /** Tolerate music gaps up to this long when extending a music run, in seconds. Default: 0.0. */
  musicGapToleranceSecs?: number;
  /** Pad each committed segment start backward by this many seconds. Default: 0.0. */
  padStartSecs?: number;
  /** Pad each committed segment end forward by this many seconds. Default: 0.0. */
  padEndSecs?: number;
  /** Speech probability threshold (default: 0.5). */
  speechThreshold?: number;
  /** Singing probability threshold (default: 0.5). */
  singingThreshold?: number;
  /** Music probability threshold (default: 0.5). */
  musicThreshold?: number;
}

/** A committed AED event from {@link OmniAEDOverlapSegmenter}. Timestamps are seconds. */
export interface AEDOverlapEvent {
  /** Event start time in seconds. */
  start: number;
  /** Event end time in seconds. */
  end: number;
  /** Primary kind: "silence" | "speech" | "singing" | "music" | "mixed". */
  primaryKind: string;
  /** Bitmask of present kinds (speech=1, singing=2, music=4). */
  kindMask: number;
  /** Speech confidence over the event window. */
  speechConfidence: number;
  /** Singing confidence over the event window. */
  singingConfidence: number;
  /** Music confidence over the event window. */
  musicConfidence: number;
  /** Confidence of the primary kind. */
  confidence: number;
  /** True when the event contains speech or singing (i.e. is transcribable). */
  isTranscribable: boolean;
}

/** A committed transcribable chunk from {@link OmniAEDOverlapSegmenter}. */
export interface AEDOverlapSegment {
  /** Segment start time in seconds (with `padStartSecs` applied). */
  start: number;
  /** Segment end time in seconds (with `padEndSecs` applied). */
  end: number;
  /** Index of the first event covered by this segment (within the same result). */
  eventStartIdx: number;
  /** Number of events covered by this segment. */
  eventCount: number;
}

/** Output from one {@link OmniAEDOverlapSegmenter.ingest} or `flush` call. */
export interface AEDOverlapResult {
  /** Newly committed transcribable segments. */
  segments: AEDOverlapSegment[];
  /** Newly committed events. */
  events: AEDOverlapEvent[];
}

/** A single chunk emitted by {@link mergeChunks}. */
export interface ChunkResult {
  /** Chunk start time (seconds), with `padOnsetSecs` applied (clamped to >= 0). */
  start: number;
  /** Chunk end time (seconds), with `padOffsetSecs` applied. */
  end: number;
  /** Index of the first input segment included in this chunk. Refers to the
   *  *post-filter* segment list — segments dropped by `minSpeechSecs` and
   *  pre-merged by `minSilenceSecs` are not counted. */
  segStartIdx: number;
  /** Number of input segments included in this chunk. */
  segCount: number;
}
