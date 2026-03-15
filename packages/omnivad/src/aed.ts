// OmniAED — Audio Event Detection (3-class)
// Uses ONNX Runtime Web for model inference.
//
// Model: fireredvad_aed.onnx
// Input: feat [batch, time, 80]
// Output: probs [batch, time, 3]
//
// Classes: index 0 = speech, index 1 = singing, index 2 = music

import * as ort from "onnxruntime-web";
import { Fbank } from "./fbank.js";
import { CMVN, DEFAULT_CMVN, loadCMVN } from "./cmvn.js";
import { VadPostProcessor } from "./post-process.js";
import type { AEDResult, AEDCreateOptions } from "./types.js";

const SAMPLE_RATE = 16000;
const FRAME_LENGTH_SAMPLES = 400;
const FRAME_SHIFT_SAMPLES = 160;
const NUM_BINS = 80;
const NUM_CLASSES = 3;

/** Mapping from class index to event name */
const IDX2EVENT: Record<number, string> = {
  0: "speech",
  1: "singing",
  2: "music",
};

export class OmniAED {
  private session: ort.InferenceSession;
  private fbank: Fbank;
  private cmvn: CMVN;
  private eventPostProcessors: Record<string, VadPostProcessor>;
  private eventThresholds: Record<string, number>;
  private chunkMaxFrames: number;

  private constructor(
    session: ort.InferenceSession,
    fbank: Fbank,
    cmvn: CMVN,
    eventPostProcessors: Record<string, VadPostProcessor>,
    eventThresholds: Record<string, number>,
    chunkMaxFrames: number,
  ) {
    this.session = session;
    this.fbank = fbank;
    this.cmvn = cmvn;
    this.eventPostProcessors = eventPostProcessors;
    this.eventThresholds = eventThresholds;
    this.chunkMaxFrames = chunkMaxFrames;
  }

  /**
   * Create an OmniAED instance by loading the ONNX model.
   *
   * @param options - Model path and AED configuration
   * @returns Promise resolving to a ready-to-use OmniAED instance
   *
   * @example
   * ```ts
   * const aed = await OmniAED.create({
   *   modelPath: '/models/fireredvad_aed.onnx',
   * });
   * const result = await aed.detect(audioSamples);
   * console.log(result.events);
   * // { speech: [[0.4, 3.5]], singing: [[1.8, 20.0]], music: [[0.1, 22.0]] }
   * ```
   */
  static async create(options: AEDCreateOptions): Promise<OmniAED> {
    const session = await ort.InferenceSession.create(options.modelPath, {
      executionProviders: ["wasm"],
    });

    const fbank = new Fbank(NUM_BINS, SAMPLE_RATE, FRAME_LENGTH_SAMPLES, FRAME_SHIFT_SAMPLES);

    let cmvnData = DEFAULT_CMVN;
    if (options.cmvnData) {
      cmvnData = options.cmvnData;
    } else if (options.cmvnPath) {
      cmvnData = await loadCMVN(options.cmvnPath);
    }
    const cmvn = new CMVN(cmvnData);

    // Build per-event post-processors with their own thresholds
    const speechThreshold = options.speechThreshold ?? 0.4;
    const singingThreshold = options.singingThreshold ?? 0.5;
    const musicThreshold = options.musicThreshold ?? 0.5;

    const thresholdMap: Record<string, number> = {
      speech: speechThreshold,
      singing: singingThreshold,
      music: musicThreshold,
    };

    const eventPostProcessors: Record<string, VadPostProcessor> = {};
    for (const event of Object.values(IDX2EVENT)) {
      eventPostProcessors[event] = new VadPostProcessor(
        options.smoothWindowSize ?? 5,
        thresholdMap[event],
        options.minSpeechFrames ?? 20,
        options.maxSpeechFrames ?? 2000,
        options.minSilenceFrames ?? 20,
        options.mergeSilenceFrames ?? 0,
        options.extendSpeechFrames ?? 0,
      );
    }

    const chunkMaxFrames = options.chunkMaxFrames ?? 30000;

    return new OmniAED(
      session,
      fbank,
      cmvn,
      eventPostProcessors,
      thresholdMap,
      chunkMaxFrames,
    );
  }

  /**
   * Run Audio Event Detection on an audio buffer.
   *
   * @param audio - PCM audio samples (Int16Array or Float32Array in int16 range)
   * @param sampleRate - Sample rate (must be 16000)
   * @returns AEDResult with per-event timestamps and raw ratios
   */
  async detect(
    audio: Float32Array | Int16Array,
    sampleRate: number = SAMPLE_RATE,
  ): Promise<AEDResult> {
    if (sampleRate !== SAMPLE_RATE) {
      throw new Error(
        `Expected ${SAMPLE_RATE}Hz audio, got ${sampleRate}Hz.`,
      );
    }

    const numSamples = audio.length;
    const duration = numSamples / sampleRate;

    // Extract fbank features
    this.fbank.reset();
    const { features, numFrames } = this.fbank.extract(audio);
    if (numFrames === 0) {
      return {
        duration: Math.round(duration * 1000) / 1000,
        events: { speech: [], singing: [], music: [] },
        ratios: { speech: 0, singing: 0, music: 0 },
      };
    }

    // Apply CMVN
    this.cmvn.apply(features, numFrames);

    // Run inference (with chunking for long audio)
    // allProbs: [numFrames][NUM_CLASSES]
    let allProbs: number[][];

    if (numFrames <= this.chunkMaxFrames) {
      allProbs = await this.runInference(features, numFrames);
    } else {
      allProbs = [];
      let frameOffset = 0;
      while (frameOffset < numFrames) {
        const chunkFrames = Math.min(
          this.chunkMaxFrames,
          numFrames - frameOffset,
        );
        const chunkFeatures = new Float32Array(chunkFrames * NUM_BINS);
        chunkFeatures.set(
          features.subarray(
            frameOffset * NUM_BINS,
            (frameOffset + chunkFrames) * NUM_BINS,
          ),
        );
        const chunkProbs = await this.runInference(chunkFeatures, chunkFrames);
        allProbs.push(...chunkProbs);
        frameOffset += chunkFrames;
      }
    }

    // Post-process each event class independently
    const events: Record<string, [number, number][]> = {};
    const ratios: Record<string, number> = {};

    for (let idx = 0; idx < NUM_CLASSES; ++idx) {
      const event = IDX2EVENT[idx];
      const threshold = this.eventThresholds[event];
      const postProcessor = this.eventPostProcessors[event];

      // Extract per-class probabilities
      const eventProbs: number[] = new Array(numFrames);
      for (let t = 0; t < numFrames; ++t) {
        eventProbs[t] = allProbs[t][idx];
      }

      // Post-process
      const decisions = postProcessor.process(eventProbs);
      const timestamps = postProcessor.decisionToSegment(decisions, duration);
      events[event] = timestamps;

      // Compute raw ratio (fraction of frames above threshold before post-processing)
      let aboveCount = 0;
      for (const p of eventProbs) {
        if (p >= threshold) aboveCount++;
      }
      ratios[event] =
        Math.round((aboveCount / eventProbs.length) * 1000) / 1000;
    }

    return {
      duration: Math.round(duration * 1000) / 1000,
      events,
      ratios,
    };
  }

  /**
   * Release ONNX session resources.
   */
  dispose(): void {
    this.session.release();
  }

  /**
   * Run ONNX inference on a feature chunk.
   * @returns Array of probability vectors, one per frame, each with NUM_CLASSES values
   */
  private async runInference(
    features: Float32Array,
    numFrames: number,
  ): Promise<number[][]> {
    const tensor = new ort.Tensor("float32", features, [1, numFrames, NUM_BINS]);
    const feeds: Record<string, ort.Tensor> = { feat: tensor };
    const results = await this.session.run(feeds);

    // Output: probs [1, numFrames, 3]
    const probsData = results["probs"].data as Float32Array;
    const probs: number[][] = new Array(numFrames);
    for (let t = 0; t < numFrames; ++t) {
      probs[t] = new Array(NUM_CLASSES);
      for (let c = 0; c < NUM_CLASSES; ++c) {
        probs[t][c] = probsData[t * NUM_CLASSES + c];
      }
    }
    return probs;
  }
}
