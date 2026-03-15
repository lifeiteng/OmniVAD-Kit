// OmniStreamVAD — Streaming Voice Activity Detection
// Uses ONNX Runtime Web for model inference with cache state.
//
// Model: fireredvad_stream_vad_with_cache.onnx
// Input: feat [1, time, 80], caches_in [8, 1, 128, 19]
// Output: probs [1, time, 1], caches_out [8, 1, 128, 19]

import * as ort from "onnxruntime-web";
import { Fbank } from "./fbank.js";
import { CMVN, DEFAULT_CMVN, loadCMVN } from "./cmvn.js";
import {
  StreamVadPostProcessor,
  streamResultsToTimestamps,
} from "./post-process.js";
import type {
  VADResult,
  StreamVADFrameResult,
  StreamVADCreateOptions,
} from "./types.js";

const SAMPLE_RATE = 16000;
const FRAME_LENGTH_SAMPLES = 400; // 25ms @ 16kHz
const FRAME_SHIFT_SAMPLES = 160; // 10ms @ 16kHz
const NUM_BINS = 80;

// Cache dimensions: [8, 1, 128, 19] — 8 DFSMN blocks, batch=1, P=128, lookback=19
const CACHE_SHAPE = [8, 1, 128, 19] as const;

/**
 * Aggressiveness mode presets for streaming VAD.
 *   Mode 0: Very permissive — low threshold, catches most speech
 *   Mode 1: Permissive — balanced (default)
 *   Mode 2: Aggressive — higher threshold, fewer false positives
 *   Mode 3: Very aggressive — highest threshold, strictest
 */
const MODE_PRESETS: Record<
  number,
  { speechThreshold: number; minSpeechFrame: number; minSilenceFrame: number }
> = {
  0: { speechThreshold: 0.3, minSpeechFrame: 8, minSilenceFrame: 20 },
  1: { speechThreshold: 0.5, minSpeechFrame: 10, minSilenceFrame: 15 },
  2: { speechThreshold: 0.7, minSpeechFrame: 15, minSilenceFrame: 10 },
  3: { speechThreshold: 0.9, minSpeechFrame: 20, minSilenceFrame: 5 },
};

export class OmniStreamVAD {
  private session: ort.InferenceSession;
  private fbank: Fbank;
  private cmvn: CMVN;
  private postProcessor: StreamVadPostProcessor;
  private chunkMaxFrames: number;

  /** Model cache state, null until first frame is processed */
  private caches: Float32Array | null;

  /**
   * Audio sample accumulator for streaming processFrame().
   * We accumulate raw PCM samples and track how many fbank frames
   * we have already extracted so far, then extract only new frames.
   */
  private sampleBuffer: Float32Array;
  private sampleBufferLen: number;
  private framesExtracted: number;

  private constructor(
    session: ort.InferenceSession,
    fbank: Fbank,
    cmvn: CMVN,
    postProcessor: StreamVadPostProcessor,
    chunkMaxFrames: number,
  ) {
    this.session = session;
    this.fbank = fbank;
    this.cmvn = cmvn;
    this.postProcessor = postProcessor;
    this.chunkMaxFrames = chunkMaxFrames;
    this.caches = null;

    // Initial capacity: enough for ~10 seconds. Will grow if needed.
    this.sampleBuffer = new Float32Array(SAMPLE_RATE * 10);
    this.sampleBufferLen = 0;
    this.framesExtracted = 0;
  }

  /**
   * Create an OmniStreamVAD instance.
   *
   * @param options - Model path and streaming VAD configuration
   * @returns Promise resolving to a ready-to-use OmniStreamVAD instance
   *
   * @example
   * ```ts
   * const svad = await OmniStreamVAD.create({
   *   modelPath: '/models/fireredvad_stream_vad_with_cache.onnx',
   * });
   * svad.setMode(1); // permissive
   *
   * // Feed 160 samples (10ms) at a time
   * for (const chunk of audioChunks) {
   *   const result = await svad.processFrame(chunk);
   *   if (result.isSpeechStart) console.log('Speech started');
   *   if (result.isSpeechEnd) console.log('Speech ended');
   * }
   * ```
   */
  static async create(
    options: StreamVADCreateOptions,
  ): Promise<OmniStreamVAD> {
    const session = await ort.InferenceSession.create(options.modelPath, {
      executionProviders: ["wasm"],
    });

    const fbank = new Fbank(
      NUM_BINS,
      SAMPLE_RATE,
      FRAME_LENGTH_SAMPLES,
      FRAME_SHIFT_SAMPLES,
    );

    let cmvnData = DEFAULT_CMVN;
    if (options.cmvnData) {
      cmvnData = options.cmvnData;
    } else if (options.cmvnPath) {
      cmvnData = await loadCMVN(options.cmvnPath);
    }
    const cmvn = new CMVN(cmvnData);

    const postProcessor = new StreamVadPostProcessor(
      options.smoothWindowSize ?? 5,
      options.speechThreshold ?? 0.5,
      options.padStartFrames ?? 5,
      options.minSpeechFrames ?? 8,
      options.maxSpeechFrames ?? 2000,
      options.minSilenceFrames ?? 20,
    );

    const chunkMaxFrames = 30000;

    return new OmniStreamVAD(
      session,
      fbank,
      cmvn,
      postProcessor,
      chunkMaxFrames,
    );
  }

  /**
   * Process exactly 160 samples (one 10ms frame at 16kHz).
   *
   * Internally accumulates samples. When enough samples are available
   * (400 for the first frame, then 160 more for each subsequent frame),
   * a new fbank feature frame is computed, CMVN-normalized, and passed
   * through the ONNX model with cache state.
   *
   * Returns null when not enough samples have been accumulated yet
   * (first 2 calls return null since we need 400 samples = 3x160 = 480,
   * with the first frame produced after exactly 400 samples are available,
   * which happens on the 3rd call of 160 samples — the 80 extra are buffered).
   *
   * @param pcm160 - Exactly 160 PCM samples (10ms at 16kHz)
   * @returns StreamVADFrameResult, or null if no new frame is available yet
   */
  async processFrame(
    pcm160: Int16Array | Float32Array,
  ): Promise<StreamVADFrameResult | null> {
    if (pcm160.length !== FRAME_SHIFT_SAMPLES) {
      throw new Error(
        `Expected ${FRAME_SHIFT_SAMPLES} samples, got ${pcm160.length}`,
      );
    }

    // Grow buffer if needed
    if (this.sampleBufferLen + pcm160.length > this.sampleBuffer.length) {
      const newBuf = new Float32Array(this.sampleBuffer.length * 2);
      newBuf.set(this.sampleBuffer.subarray(0, this.sampleBufferLen));
      this.sampleBuffer = newBuf;
    }

    // Append samples
    for (let i = 0; i < pcm160.length; ++i) {
      this.sampleBuffer[this.sampleBufferLen + i] = pcm160[i];
    }
    this.sampleBufferLen += pcm160.length;

    // Check if we have enough samples for a new frame.
    // With snip_edges=true: numFrames = 1 + floor((n - frameLength) / frameShift)
    // A new frame is available when:
    //   framesExtracted < 1 + floor((sampleBufferLen - frameLength) / frameShift)
    const totalFramesPossible =
      this.sampleBufferLen >= FRAME_LENGTH_SAMPLES
        ? 1 +
          Math.floor(
            (this.sampleBufferLen - FRAME_LENGTH_SAMPLES) / FRAME_SHIFT_SAMPLES,
          )
        : 0;

    if (totalFramesPossible <= this.framesExtracted) {
      // No new frame available yet
      return null;
    }

    // Extract exactly one new frame using computeFrame.
    // The frame starts at framesExtracted * frameShift.
    const frameStart = this.framesExtracted * FRAME_SHIFT_SAMPLES;
    const frameData = this.sampleBuffer.subarray(
      frameStart,
      frameStart + FRAME_LENGTH_SAMPLES,
    );
    const features = this.fbank.computeFrame(frameData);
    if (features === null) {
      return null;
    }
    this.framesExtracted += 1;

    // Apply CMVN (1 frame, 80 dims)
    this.cmvn.apply(features, 1);

    // Run ONNX inference with cache
    const prob = await this.runStreamInference(features);

    // Post-process
    return this.postProcessor.processOneFrame(prob);
  }

  /**
   * Run detection on a complete audio buffer using the streaming model.
   * Processes all frames at once with the batch model, then runs the
   * streaming post-processor on the output probabilities.
   *
   * @param audio - Complete PCM audio (Int16Array or Float32Array)
   * @param sampleRate - Sample rate (must be 16000)
   * @returns Frame-level results and aggregated VADResult
   */
  async detectFull(
    audio: Float32Array | Int16Array,
    sampleRate: number = SAMPLE_RATE,
  ): Promise<{ frames: StreamVADFrameResult[]; result: VADResult }> {
    if (sampleRate !== SAMPLE_RATE) {
      throw new Error(
        `Expected ${SAMPLE_RATE}Hz audio, got ${sampleRate}Hz.`,
      );
    }

    this.reset();

    const duration = audio.length / sampleRate;

    // Extract all fbank features at once
    this.fbank.reset();
    const { features, numFrames } = this.fbank.extract(audio);
    if (numFrames === 0) {
      return {
        frames: [],
        result: {
          duration: Math.round(duration * 1000) / 1000,
          timestamps: [],
        },
      };
    }

    // Apply CMVN
    this.cmvn.apply(features, numFrames);

    // Run inference on all frames (chunked if necessary)
    let allProbs: number[];
    if (numFrames <= this.chunkMaxFrames) {
      allProbs = await this.runBatchInference(features, numFrames);
    } else {
      allProbs = [];
      let offset = 0;
      while (offset < numFrames) {
        const chunkFrames = Math.min(
          this.chunkMaxFrames,
          numFrames - offset,
        );
        const chunkFeatures = new Float32Array(chunkFrames * NUM_BINS);
        chunkFeatures.set(
          features.subarray(
            offset * NUM_BINS,
            (offset + chunkFrames) * NUM_BINS,
          ),
        );
        const chunkProbs = await this.runBatchInference(
          chunkFeatures,
          chunkFrames,
        );
        allProbs.push(...chunkProbs);
        offset += chunkFrames;
      }
    }

    // Process through streaming post-processor
    const frames: StreamVADFrameResult[] = [];
    for (const prob of allProbs) {
      const frameResult = this.postProcessor.processOneFrame(prob);
      frames.push(frameResult);
    }

    // Convert to timestamps
    const timestamps = streamResultsToTimestamps(frames);

    this.reset();

    return {
      frames,
      result: {
        duration: Math.round(duration * 1000) / 1000,
        timestamps,
      },
    };
  }

  /**
   * Set the aggressiveness mode.
   *
   * @param mode - 0=very permissive, 1=permissive, 2=aggressive, 3=very aggressive
   */
  setMode(mode: 0 | 1 | 2 | 3): void {
    const preset = MODE_PRESETS[mode];
    if (!preset) {
      throw new Error(`Invalid mode: ${mode}. Must be 0, 1, 2, or 3.`);
    }
    this.postProcessor.speechThreshold = preset.speechThreshold;
    this.postProcessor.minSpeechFrame = preset.minSpeechFrame;
    this.postProcessor.minSilenceFrame = preset.minSilenceFrame;
  }

  /**
   * Reset all internal state (cache, post-processor, fbank, sample buffer).
   * Call this before processing a new audio stream.
   */
  reset(): void {
    this.caches = null;
    this.fbank.reset();
    this.postProcessor.reset();
    this.sampleBufferLen = 0;
    this.framesExtracted = 0;
  }

  /**
   * Release ONNX session resources.
   */
  dispose(): void {
    this.session.release();
  }

  /**
   * Run streaming inference for a single frame with cache management.
   */
  private async runStreamInference(features: Float32Array): Promise<number> {
    const featTensor = new ort.Tensor("float32", features, [1, 1, NUM_BINS]);
    const feeds: Record<string, ort.Tensor> = { feat: featTensor };

    if (this.caches !== null) {
      const cacheTensor = new ort.Tensor(
        "float32",
        this.caches,
        [CACHE_SHAPE[0], CACHE_SHAPE[1], CACHE_SHAPE[2], CACHE_SHAPE[3]],
      );
      feeds["caches_in"] = cacheTensor;
    }

    const results = await this.session.run(feeds);

    // Extract probability
    const probsData = results["probs"].data as Float32Array;
    const prob = probsData[0];

    // Update cache
    if (results["caches_out"]) {
      this.caches = new Float32Array(
        results["caches_out"].data as Float32Array,
      );
    }

    return prob;
  }

  /**
   * Run inference on a batch of frames with cache propagation.
   * Used by detectFull() for efficient batch processing.
   */
  private async runBatchInference(
    features: Float32Array,
    numFrames: number,
  ): Promise<number[]> {
    const featTensor = new ort.Tensor("float32", features, [
      1,
      numFrames,
      NUM_BINS,
    ]);
    const feeds: Record<string, ort.Tensor> = { feat: featTensor };

    if (this.caches !== null) {
      const cacheTensor = new ort.Tensor(
        "float32",
        this.caches,
        [CACHE_SHAPE[0], CACHE_SHAPE[1], CACHE_SHAPE[2], CACHE_SHAPE[3]],
      );
      feeds["caches_in"] = cacheTensor;
    }

    const results = await this.session.run(feeds);

    // Extract probabilities [1, numFrames, 1]
    const probsData = results["probs"].data as Float32Array;
    const probs: number[] = new Array(numFrames);
    for (let i = 0; i < numFrames; ++i) {
      probs[i] = probsData[i];
    }

    // Update cache
    if (results["caches_out"]) {
      this.caches = new Float32Array(
        results["caches_out"].data as Float32Array,
      );
    }

    return probs;
  }
}
