// OmniVAD — Non-streaming Voice Activity Detection
// Uses ONNX Runtime Web for model inference.
//
// Model: fireredvad_vad.onnx
// Input: feat [batch, time, 80]
// Output: probs [batch, time, 1]

import * as ort from "onnxruntime-web";
import { Fbank } from "./fbank.js";
import { CMVN, DEFAULT_CMVN, loadCMVN } from "./cmvn.js";
import { VadPostProcessor } from "./post-process.js";
import type { VADResult, VADCreateOptions } from "./types.js";

const SAMPLE_RATE = 16000;
const FRAME_LENGTH_SAMPLES = 400; // 25ms @ 16kHz
const FRAME_SHIFT_SAMPLES = 160; // 10ms @ 16kHz
const NUM_BINS = 80;

export class OmniVAD {
  private session: ort.InferenceSession;
  private fbank: Fbank;
  private cmvn: CMVN;
  private postProcessor: VadPostProcessor;
  private chunkMaxFrames: number;

  private constructor(
    session: ort.InferenceSession,
    fbank: Fbank,
    cmvn: CMVN,
    postProcessor: VadPostProcessor,
    chunkMaxFrames: number,
  ) {
    this.session = session;
    this.fbank = fbank;
    this.cmvn = cmvn;
    this.postProcessor = postProcessor;
    this.chunkMaxFrames = chunkMaxFrames;
  }

  /**
   * Create an OmniVAD instance by loading the ONNX model.
   *
   * @param options - Model path and VAD configuration
   * @returns Promise resolving to a ready-to-use OmniVAD instance
   *
   * @example
   * ```ts
   * const vad = await OmniVAD.create({
   *   modelPath: '/models/fireredvad_vad.onnx',
   * });
   * const result = await vad.detect(audioSamples);
   * console.log(result.timestamps); // [[0.44, 1.82], [3.10, 5.60]]
   * ```
   */
  static async create(options: VADCreateOptions): Promise<OmniVAD> {
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

    const postProcessor = new VadPostProcessor(
      options.smoothWindowSize ?? 5,
      options.speechThreshold ?? 0.4,
      options.minSpeechFrames ?? 20,
      options.maxSpeechFrames ?? 2000,
      options.minSilenceFrames ?? 20,
      options.mergeSilenceFrames ?? 0,
      options.extendSpeechFrames ?? 0,
    );

    const chunkMaxFrames = options.chunkMaxFrames ?? 30000;

    return new OmniVAD(session, fbank, cmvn, postProcessor, chunkMaxFrames);
  }

  /**
   * Run VAD detection on an audio buffer.
   *
   * @param audio - PCM audio samples. Int16Array (raw int16) or Float32Array
   *                (values in int16 range, i.e. -32768 to 32767).
   * @param sampleRate - Sample rate of the input audio (default: 16000).
   *                     If not 16000, the audio will NOT be resampled — caller
   *                     must provide 16kHz audio.
   * @returns VADResult with duration and speech timestamps
   */
  async detect(
    audio: Float32Array | Int16Array,
    sampleRate: number = SAMPLE_RATE,
  ): Promise<VADResult> {
    if (sampleRate !== SAMPLE_RATE) {
      throw new Error(
        `Expected ${SAMPLE_RATE}Hz audio, got ${sampleRate}Hz. Please resample before calling detect().`,
      );
    }

    // Compute duration
    const numSamples = audio.length;
    const duration = numSamples / sampleRate;

    // Extract fbank features
    this.fbank.reset();
    const { features, numFrames } = this.fbank.extract(audio);
    if (numFrames === 0) {
      return { duration: Math.round(duration * 1000) / 1000, timestamps: [] };
    }

    // Apply CMVN
    this.cmvn.apply(features, numFrames);

    // Run inference (with chunking for long audio)
    let allProbs: number[];

    if (numFrames <= this.chunkMaxFrames) {
      allProbs = await this.runInference(features, numFrames);
    } else {
      // Split into chunks
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

    // Post-process
    const decisions = this.postProcessor.process(allProbs);
    const timestamps = this.postProcessor.decisionToSegment(
      decisions,
      duration,
    );

    return {
      duration: Math.round(duration * 1000) / 1000,
      timestamps,
    };
  }

  /**
   * Run ONNX inference on a feature chunk.
   * @returns Array of speech probabilities, one per frame
   */
  private async runInference(
    features: Float32Array,
    numFrames: number,
  ): Promise<number[]> {
    // Create tensor: [1, numFrames, 80]
    const tensor = new ort.Tensor("float32", features, [1, numFrames, NUM_BINS]);
    const feeds: Record<string, ort.Tensor> = { feat: tensor };
    const results = await this.session.run(feeds);

    // Output: probs [1, numFrames, 1]
    const probsData = results["probs"].data as Float32Array;
    const probs: number[] = new Array(numFrames);
    for (let i = 0; i < numFrames; ++i) {
      probs[i] = probsData[i];
    }
    return probs;
  }

  /**
   * Release ONNX session resources.
   */
  dispose(): void {
    this.session.release();
  }
}
