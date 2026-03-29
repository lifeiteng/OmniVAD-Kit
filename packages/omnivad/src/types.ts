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
