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
  /** Raw ratio of frames above threshold for each event type */
  ratios: Record<string, number>;
}

/** Per-frame result from streaming VAD */
export interface StreamVADFrameResult {
  /** Raw probability from model output */
  confidence: number;
  /** Smoothed probability after moving average */
  smoothedConfidence: number;
  /** Whether current frame is classified as speech (after post-processing) */
  isSpeech: boolean;
  /** 1-based frame index */
  frameIndex: number;
  /** True when a speech segment starts at this frame */
  isSpeechStart: boolean;
  /** True when a speech segment ends at this frame */
  isSpeechEnd: boolean;
  /** 1-based frame index of speech start */
  speechStartFrame: number;
  /** 1-based frame index of speech end */
  speechEndFrame: number;
}

/** Configuration for non-streaming VAD */
export interface VADConfig {
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
export interface StreamVADConfig {
  /** Speech probability threshold (default: 0.5) */
  speechThreshold?: number;
}
