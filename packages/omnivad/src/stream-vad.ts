/**
 * Streaming Voice Activity Detection (WASM/ncnn backend).
 * Processes audio frame-by-frame (10ms chunks of 160 samples @ 16kHz).
 *
 * Audio format: Float32Array in [-1, 1] or Int16Array PCM. Wrappers
 * dispatch by dtype; all scaling lives in the C entries.
 */

import type { StreamVADConfig, StreamVADFrameResult, StreamVADFullResult } from "./types.js";
import {
  initWasm,
  getModule,
  dispatchAudio,
  loadModel,
  streamVadCreate,
  streamVadClone,
  streamVadProcess,
  streamVadDetectFull,
  streamVadReset,
  streamVadDestroy,
} from "./wasm-binding.js";

const SAMPLE_RATE = 16000;

export class OmniStreamVAD {
  private handle: number;

  private constructor(handle: number) {
    this.handle = handle;
  }

  /**
   * Create a new OmniStreamVAD instance.
   * Loads model from CDN (browser), local package (Node.js), or custom source.
   */
  static async create(options: StreamVADConfig = {}): Promise<OmniStreamVAD> {
    await initWasm();
    const M = getModule();
    const modelBuffer = await loadModel("stream-vad", options.modelUrl, options.modelData);
    const handle = streamVadCreate(M, modelBuffer, {
      threshold:        options.threshold,
      smoothWindowSize: options.smoothWindowSize,
      padStartFrame:    options.padStartFrame,
      minSpeechFrame:   options.minSpeechFrame,
      maxSpeechFrame:   options.maxSpeechFrame,
      minSilenceFrame:  options.minSilenceFrame,
    });
    return new OmniStreamVAD(handle);
  }

  /**
   * Create a lightweight clone sharing the same underlying model weights.
   * The clone has fresh per-instance state (empty audio buffer, zeroed cache).
   * This is synchronous and extremely fast — ideal for multi-stream scenarios
   * (e.g., handling multiple WebRTC tracks or concurrent audio sessions).
   */
  clone(): OmniStreamVAD {
    if (!this.handle) throw new Error("Cannot clone a disposed instance.");
    const M = getModule();
    const newHandle = streamVadClone(M, this.handle);
    return new OmniStreamVAD(newHandle);
  }

  /**
   * Process one frame of audio (160 samples = 10ms @ 16kHz).
   *
   * Accepts Float32Array in [-1, 1] (Web Audio, soundfile, torch) or
   * Int16Array PCM (WAV, microphone). Dispatches by dtype to the matching
   * C entry — no scaling in JS.
   *
   * Returns null until enough audio is accumulated.
   *
   * Segment-boundary events (isSpeechStart / isSpeechEnd and the matching
   * speech_*_frame indices) come straight from the C-layer state machine
   * (bit-identical to upstream FireRedVAD) — the wrapper is just a marshaller.
   */
  processFrame(audio: Float32Array | Int16Array): StreamVADFrameResult | null {
    const M = getModule();
    const { ptr, length, format } = dispatchAudio(M, audio);

    try {
      const result = streamVadProcess(M, this.handle, ptr, length, format);
      if (!result) return null;

      return {
        confidence:       result.confidence,
        smoothedProb:     result.smoothedProb,
        isSpeech:         result.isSpeech,
        frameIndex:       result.frameIdx,
        isSpeechStart:    result.isSpeechStart,
        isSpeechEnd:      result.isSpeechEnd,
        speechStartFrame: result.speechStartFrame,
        speechEndFrame:   result.speechEndFrame,
      };
    } finally {
      M._free(ptr);
    }
  }

  /**
   * Process entire audio at once and return per-frame probabilities.
   * @param audio - Float32Array in [-1, 1] or Int16Array of 16kHz mono PCM
   */
  detectFull(audio: Float32Array | Int16Array): StreamVADFullResult {
    const M = getModule();
    const { ptr, length, format } = dispatchAudio(M, audio);

    try {
      const { probabilities, numFrames } = streamVadDetectFull(
        M,
        this.handle,
        ptr,
        length,
        format,
      );
      return {
        probabilities,
        numFrames,
        duration: Math.round((length / SAMPLE_RATE) * 1000) / 1000,
      };
    } finally {
      M._free(ptr);
    }
  }

  /** Reset all internal state (model cache, audio buffer, postprocessor). */
  reset(): void {
    streamVadReset(getModule(), this.handle);
  }

  /** Release native resources. */
  dispose(): void {
    if (this.handle) {
      streamVadDestroy(getModule(), this.handle);
      this.handle = 0;
    }
  }
}

