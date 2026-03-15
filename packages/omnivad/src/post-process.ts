// Post-processing for VAD and AED
// Ported from:
//   - fireredvad/core/vad_postprocessor.py (non-streaming)
//   - fireredvad/core/stream_vad_postprocessor.py (streaming)

import type { StreamVADFrameResult } from "./types.js";

// Constants matching fireredvad/core/constants.py
const FRAME_SHIFT_S = 0.01; // 10ms
const FRAME_LENGTH_S = 0.025; // 25ms
const FRAME_PER_SECONDS = 100; // 1000 / 10

// --------------------------------------------------------------------------
// Non-streaming VAD post-processor
// --------------------------------------------------------------------------

enum VadState {
  SILENCE = 0,
  POSSIBLE_SPEECH = 1,
  SPEECH = 2,
  POSSIBLE_SILENCE = 3,
}

export class VadPostProcessor {
  private readonly smoothWindowSize: number;
  private readonly probThreshold: number;
  private readonly minSpeechFrame: number;
  private readonly maxSpeechFrame: number;
  private readonly minSilenceFrame: number;
  private readonly mergeSilenceFrame: number;
  private readonly extendSpeechFrame: number;

  constructor(
    smoothWindowSize: number,
    probThreshold: number,
    minSpeechFrame: number,
    maxSpeechFrame: number,
    minSilenceFrame: number,
    mergeSilenceFrame: number,
    extendSpeechFrame: number,
  ) {
    this.smoothWindowSize = Math.max(1, smoothWindowSize);
    this.probThreshold = probThreshold;
    this.minSpeechFrame = minSpeechFrame;
    this.maxSpeechFrame = maxSpeechFrame;
    this.minSilenceFrame = minSilenceFrame;
    this.mergeSilenceFrame = mergeSilenceFrame;
    this.extendSpeechFrame = extendSpeechFrame;
  }

  /**
   * Full post-processing pipeline: smooth -> threshold -> state machine ->
   * fix start -> merge silence -> extend -> split long.
   *
   * @param rawProbs - Raw probabilities from model output
   * @returns Binary decision array (0=silence, 1=speech)
   */
  process(rawProbs: number[]): number[] {
    if (rawProbs.length === 0) {
      return [];
    }

    const smoothedProbs = this.smoothProb(rawProbs);
    const binaryPreds = this.applyThreshold(smoothedProbs);
    const decisions = this.smoothPredsWithStateMachine(binaryPreds);
    const fixedDecisions = this.fixSmoothWindowStart(decisions);
    const mergedDecisions = this.mergeShortSilenceSegments(fixedDecisions);
    const extendedDecisions = this.extendSpeechSegments(mergedDecisions);
    const finalDecisions = this.splitLongSpeechSegments(
      extendedDecisions,
      rawProbs,
    );

    return finalDecisions;
  }

  /**
   * Convert binary decisions to timestamp segments.
   *
   * @param decisions - Binary decision array
   * @param wavDur - Optional audio duration to cap the end time
   * @returns Array of [start, end] timestamp pairs in seconds
   */
  decisionToSegment(
    decisions: number[],
    wavDur?: number,
  ): [number, number][] {
    const segments: [number, number][] = [];
    let speechStart: number | null = null;

    for (let t = 0; t < decisions.length; ++t) {
      if (decisions[t] === 1 && speechStart === null) {
        speechStart = t;
      } else if (decisions[t] === 0 && speechStart !== null) {
        const start = speechStart * FRAME_SHIFT_S;
        const end = t * FRAME_SHIFT_S;
        segments.push([
          Math.round(start * 1000) / 1000,
          Math.round(end * 1000) / 1000,
        ]);
        speechStart = null;
      }
    }

    if (speechStart !== null) {
      let endTime = decisions.length * FRAME_SHIFT_S + FRAME_LENGTH_S;
      if (wavDur !== undefined) {
        endTime = Math.min(endTime, wavDur);
      }
      const start = speechStart * FRAME_SHIFT_S;
      segments.push([
        Math.round(start * 1000) / 1000,
        Math.round(endTime * 1000) / 1000,
      ]);
    }

    return segments;
  }

  /**
   * Smooth probabilities using convolution with 'full' mode,
   * then fix boundary frames with cumulative average.
   * Matches the Python np.convolve(probs, kernel, 'full')[:len(probs)] behavior.
   */
  private smoothProb(probs: number[]): number[] {
    if (this.smoothWindowSize <= 1) {
      return probs.slice();
    }

    const n = probs.length;
    const w = this.smoothWindowSize;

    // np.convolve(probs, ones(w)/w, 'full')[:n]
    const smoothed = new Array<number>(n);
    for (let i = 0; i < n; ++i) {
      let sum = 0;
      let count = 0;
      for (let j = 0; j < w; ++j) {
        const idx = i - j;
        if (idx >= 0 && idx < n) {
          sum += probs[idx];
          count++;
        }
      }
      smoothed[i] = sum / w; // divide by window size, not count (matches 'full' mode)
    }

    // Fix boundary: first (w-1) frames use cumulative average
    const boundary = Math.min(w - 1, n);
    for (let i = 0; i < boundary; ++i) {
      let sum = 0;
      for (let j = 0; j <= i; ++j) {
        sum += probs[j];
      }
      smoothed[i] = sum / (i + 1);
    }

    return smoothed;
  }

  /**
   * Apply threshold to get binary predictions.
   */
  private applyThreshold(probs: number[]): number[] {
    return probs.map((p) => (p >= this.probThreshold ? 1 : 0));
  }

  /**
   * State machine for smoothing binary predictions.
   * Transitions are constrained by minSpeechFrame and minSilenceFrame.
   */
  private smoothPredsWithStateMachine(binaryPreds: number[]): number[] {
    if (this.minSpeechFrame <= 0 && this.minSilenceFrame <= 0) {
      return binaryPreds.slice();
    }

    const decisions = new Array<number>(binaryPreds.length).fill(0);
    let state = VadState.SILENCE;
    let speechStart = -1;
    let silenceStart = -1;

    for (let t = 0; t < binaryPreds.length; ++t) {
      const isSpeech = binaryPreds[t];

      if (state === VadState.SILENCE) {
        if (isSpeech) {
          state = VadState.POSSIBLE_SPEECH;
          speechStart = t;
        }
      } else if (state === VadState.POSSIBLE_SPEECH) {
        if (isSpeech) {
          if (t - speechStart >= this.minSpeechFrame) {
            state = VadState.SPEECH;
            for (let k = speechStart; k < t; ++k) {
              decisions[k] = 1;
            }
          }
        } else {
          state = VadState.SILENCE;
          speechStart = -1;
        }
      } else if (state === VadState.SPEECH) {
        if (!isSpeech) {
          state = VadState.POSSIBLE_SILENCE;
          silenceStart = t;
        }
      } else if (state === VadState.POSSIBLE_SILENCE) {
        if (!isSpeech) {
          if (t - silenceStart >= this.minSilenceFrame) {
            state = VadState.SILENCE;
            speechStart = -1;
          }
        } else {
          state = VadState.SPEECH;
          silenceStart = -1;
        }
      }

      // Current frame decision
      if (state === VadState.SPEECH || state === VadState.POSSIBLE_SILENCE) {
        decisions[t] = 1;
      } else {
        decisions[t] = 0;
      }
    }

    return decisions;
  }

  /**
   * Fix the start of speech segments by extending backwards
   * by smoothWindowSize frames.
   */
  private fixSmoothWindowStart(decisions: number[]): number[] {
    const newDecisions = decisions.slice();
    for (let t = 1; t < decisions.length; ++t) {
      if (decisions[t - 1] === 0 && decisions[t] === 1) {
        const start = Math.max(0, t - this.smoothWindowSize);
        for (let k = start; k < t; ++k) {
          newDecisions[k] = 1;
        }
      }
    }
    return newDecisions;
  }

  /**
   * Merge short silence segments (shorter than mergeSilenceFrame) between
   * speech segments.
   */
  private mergeShortSilenceSegments(decisions: number[]): number[] {
    if (this.mergeSilenceFrame <= 0) {
      return decisions.slice();
    }

    const newDecisions = decisions.slice();
    let silenceStart: number | null = null;

    for (let t = 0; t < decisions.length; ++t) {
      if (
        t > 0 &&
        decisions[t - 1] === 1 &&
        decisions[t] === 0 &&
        silenceStart === null
      ) {
        silenceStart = t;
      } else if (
        t > 0 &&
        decisions[t - 1] === 0 &&
        decisions[t] === 1 &&
        silenceStart !== null
      ) {
        const silenceFrame = t - silenceStart;
        if (silenceFrame < this.mergeSilenceFrame) {
          for (let k = silenceStart; k < t; ++k) {
            newDecisions[k] = 1;
          }
        }
        silenceStart = null;
      }
    }

    return newDecisions;
  }

  /**
   * Extend speech segments by N frames on each side using convolution.
   * Matches Python: np.convolve(decisions, ones(2*N+1), 'same') > 0
   */
  private extendSpeechSegments(decisions: number[]): number[] {
    if (this.extendSpeechFrame <= 0) {
      return decisions.slice();
    }

    const n = decisions.length;
    const kernelSize = 2 * this.extendSpeechFrame + 1;
    const extended = new Array<number>(n);

    // Convolution with mode='same'
    const halfKernel = Math.floor(kernelSize / 2);
    for (let i = 0; i < n; ++i) {
      let sum = 0;
      for (let j = 0; j < kernelSize; ++j) {
        const idx = i - halfKernel + j;
        if (idx >= 0 && idx < n) {
          sum += decisions[idx];
        }
      }
      extended[i] = sum > 0 ? 1 : 0;
    }

    return extended;
  }

  /**
   * Split long speech segments by finding the minimum probability point
   * in the second half of each over-long segment.
   */
  private splitLongSpeechSegments(
    decisions: number[],
    probs: number[],
  ): number[] {
    const newDecisions = decisions.slice();
    const segments = this.decisionToSegment(decisions);

    for (const [startS, endS] of segments) {
      const startFrame = Math.floor(startS / FRAME_SHIFT_S);
      const endFrame = Math.floor(endS / FRAME_SHIFT_S);
      const durFrames = endFrame - startFrame;

      if (durFrames > this.maxSpeechFrame) {
        const segmentProbs = probs.slice(startFrame, endFrame);
        const splitPoints = this.findSplitPoints(segmentProbs);
        for (const splitPoint of splitPoints) {
          const splitFrame = startFrame + splitPoint;
          newDecisions[splitFrame] = 0;
        }
      }
    }

    return newDecisions;
  }

  /**
   * Find split points within a long segment by locating minimum
   * probability positions in sliding windows.
   */
  private findSplitPoints(probs: number[]): number[] {
    const splitPoints: number[] = [];
    const length = probs.length;
    let start = 0;

    while (start < length) {
      if (length - start <= this.maxSpeechFrame) {
        break;
      }
      const windowStart = Math.floor(start + this.maxSpeechFrame / 2);
      const windowEnd = Math.floor(start + this.maxSpeechFrame);
      const windowProbs = probs.slice(windowStart, windowEnd);

      // Find index of minimum probability in window
      let minIdx = 0;
      let minVal = windowProbs[0];
      for (let i = 1; i < windowProbs.length; ++i) {
        if (windowProbs[i] < minVal) {
          minVal = windowProbs[i];
          minIdx = i;
        }
      }

      const globalMinIdx = windowStart + minIdx;
      splitPoints.push(globalMinIdx);
      start = globalMinIdx + 1;
    }

    return splitPoints;
  }
}

// --------------------------------------------------------------------------
// Streaming VAD post-processor
// --------------------------------------------------------------------------

enum StreamVadState {
  SILENCE = 0,
  POSSIBLE_SPEECH = 1,
  SPEECH = 2,
  POSSIBLE_SILENCE = 3,
}

export class StreamVadPostProcessor {
  smoothWindowSize: number;
  speechThreshold: number;
  padStartFrame: number;
  minSpeechFrame: number;
  maxSpeechFrame: number;
  minSilenceFrame: number;

  private frameCnt: number;
  private smoothWindow: number[];
  private smoothWindowSum: number;
  private state: StreamVadState;
  private speechCnt: number;
  private silenceCnt: number;
  private hitMaxSpeech: boolean;
  private lastSpeechStartFrame: number;
  private lastSpeechEndFrame: number;

  constructor(
    smoothWindowSize: number,
    speechThreshold: number,
    padStartFrame: number,
    minSpeechFrame: number,
    maxSpeechFrame: number,
    minSilenceFrame: number,
  ) {
    this.smoothWindowSize = Math.max(1, smoothWindowSize);
    this.speechThreshold = speechThreshold;
    this.padStartFrame = Math.max(this.smoothWindowSize, padStartFrame);
    this.minSpeechFrame = minSpeechFrame;
    this.maxSpeechFrame = maxSpeechFrame;
    this.minSilenceFrame = minSilenceFrame;

    // Initialize state
    this.frameCnt = 0;
    this.smoothWindow = [];
    this.smoothWindowSum = 0.0;
    this.state = StreamVadState.SILENCE;
    this.speechCnt = 0;
    this.silenceCnt = 0;
    this.hitMaxSpeech = false;
    this.lastSpeechStartFrame = -1;
    this.lastSpeechEndFrame = -1;
  }

  /**
   * Reset all internal state for a new audio stream.
   */
  reset(): void {
    this.frameCnt = 0;
    this.smoothWindow = [];
    this.smoothWindowSum = 0.0;
    this.state = StreamVadState.SILENCE;
    this.speechCnt = 0;
    this.silenceCnt = 0;
    this.hitMaxSpeech = false;
    this.lastSpeechStartFrame = -1;
    this.lastSpeechEndFrame = -1;
  }

  /**
   * Process one frame's raw probability.
   *
   * @param rawProb - Raw probability from model (0..1)
   * @returns StreamVADFrameResult with speech detection info
   */
  processOneFrame(rawProb: number): StreamVADFrameResult {
    this.frameCnt += 1;

    const smoothedProb = this.smoothProb(rawProb);
    const isSpeech = smoothedProb >= this.speechThreshold;

    const result: StreamVADFrameResult = {
      confidence: Math.round(rawProb * 1000) / 1000,
      smoothedConfidence: Math.round(smoothedProb * 1000) / 1000,
      isSpeech,
      frameIndex: this.frameCnt,
      isSpeechStart: false,
      isSpeechEnd: false,
      speechStartFrame: -1,
      speechEndFrame: -1,
    };

    this.stateTransition(isSpeech, result);

    return result;
  }

  /**
   * Smooth a single probability value using a running window average.
   */
  private smoothProb(prob: number): number {
    if (this.smoothWindowSize <= 1) {
      return prob;
    }

    this.smoothWindow.push(prob);
    this.smoothWindowSum += prob;

    if (this.smoothWindow.length > this.smoothWindowSize) {
      const left = this.smoothWindow.shift()!;
      this.smoothWindowSum -= left;
    }

    return this.smoothWindowSum / this.smoothWindow.length;
  }

  /**
   * State machine for streaming VAD.
   * Modifies the result object in-place with speech start/end events.
   */
  private stateTransition(
    isSpeech: boolean,
    result: StreamVADFrameResult,
  ): void {
    if (this.hitMaxSpeech) {
      result.isSpeechStart = true;
      result.speechStartFrame = this.frameCnt;
      this.lastSpeechStartFrame = result.speechStartFrame;
      this.hitMaxSpeech = false;
    }

    if (this.state === StreamVadState.SILENCE) {
      if (isSpeech) {
        this.state = StreamVadState.POSSIBLE_SPEECH;
        this.speechCnt += 1;
      } else {
        this.silenceCnt += 1;
        this.speechCnt = 0;
      }
    } else if (this.state === StreamVadState.POSSIBLE_SPEECH) {
      if (isSpeech) {
        this.speechCnt += 1;
        if (this.speechCnt >= this.minSpeechFrame) {
          this.state = StreamVadState.SPEECH;
          result.isSpeechStart = true;
          result.speechStartFrame = Math.max(
            1,
            this.frameCnt - this.speechCnt + 1 - this.padStartFrame,
            this.lastSpeechEndFrame + 1,
          );
          this.lastSpeechStartFrame = result.speechStartFrame;
          this.silenceCnt = 0;
        }
      } else {
        this.state = StreamVadState.SILENCE;
        this.silenceCnt = 1;
        this.speechCnt = 0;
      }
    } else if (this.state === StreamVadState.SPEECH) {
      this.speechCnt += 1;
      if (isSpeech) {
        this.silenceCnt = 0;
        if (this.speechCnt >= this.maxSpeechFrame) {
          this.hitMaxSpeech = true;
          this.speechCnt = 0;
          result.isSpeechEnd = true;
          result.speechEndFrame = this.frameCnt;
          result.speechStartFrame = this.lastSpeechStartFrame;
          this.lastSpeechStartFrame = -1;
          this.lastSpeechEndFrame = result.speechEndFrame;
        }
      } else {
        this.state = StreamVadState.POSSIBLE_SILENCE;
        this.silenceCnt += 1;
      }
    } else if (this.state === StreamVadState.POSSIBLE_SILENCE) {
      this.speechCnt += 1;
      if (isSpeech) {
        this.state = StreamVadState.SPEECH;
        this.silenceCnt = 0;
        if (this.speechCnt >= this.maxSpeechFrame) {
          this.hitMaxSpeech = true;
          this.speechCnt = 0;
          result.isSpeechEnd = true;
          result.speechEndFrame = this.frameCnt;
          result.speechStartFrame = this.lastSpeechStartFrame;
          this.lastSpeechStartFrame = -1;
          this.lastSpeechEndFrame = result.speechEndFrame;
        }
      } else {
        this.silenceCnt += 1;
        if (this.silenceCnt >= this.minSilenceFrame) {
          this.state = StreamVadState.SILENCE;
          result.isSpeechEnd = true;
          result.speechEndFrame = this.frameCnt;
          result.speechStartFrame = this.lastSpeechStartFrame;
          this.lastSpeechEndFrame = result.speechEndFrame;
          this.lastSpeechStartFrame = -1;
          this.speechCnt = 0;
        }
      }
    }
  }
}

/**
 * Convert streaming VAD frame results into timestamp segments.
 * Matches upstream FireRedStreamVad.results_to_timestamps() from Python.
 *
 * @param results - Array of StreamVADFrameResult from streaming processing
 * @returns Array of [start, end] timestamp pairs in seconds
 */
export function streamResultsToTimestamps(
  results: StreamVADFrameResult[],
): [number, number][] {
  // Sort by frame index
  const sorted = [...results].sort((a, b) => a.frameIndex - b.frameIndex);

  const frameTimestamps: [number, number][] = [];
  let start = -1;
  let end = -1;

  for (const r of sorted) {
    if (r.isSpeechStart) {
      if (start !== -1) {
        // Previous start without end — log warning but continue
      }
      start = Math.max(0, r.speechStartFrame - 1);
      end = -1;
    } else if (r.isSpeechEnd) {
      end = Math.max(0, r.speechEndFrame - 1);
      frameTimestamps.push([start, end]);
      start = -1;
      end = -1;
    }
  }

  // Handle unterminated speech segment
  if (start !== -1) {
    end = sorted[sorted.length - 1].frameIndex - 1;
    frameTimestamps.push([start, end]);
  }

  // Convert frame indices to seconds
  const timestamps: [number, number][] = [];
  for (const [s, e] of frameTimestamps) {
    timestamps.push([s / FRAME_PER_SECONDS, e / FRAME_PER_SECONDS]);
  }

  return timestamps;
}
