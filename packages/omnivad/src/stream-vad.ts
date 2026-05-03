/**
 * Streaming Voice Activity Detection (WASM/ncnn backend).
 * Processes audio frame-by-frame (10ms chunks of 160 samples @ 16kHz).
 */

import type { StreamVADConfig, StreamVADFrameResult, StreamVADFullResult } from "./types.js";
import {
  initWasm,
  getModule,
  copyAudioToHeap,
  loadModel,
  streamVadCreate,
  streamVadClone,
  streamVadProcess,
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
   * Process one frame of audio (160 int16 samples = 10ms @ 16kHz).
   * Returns null until enough audio is accumulated.
   *
   * Segment-boundary events (isSpeechStart / isSpeechEnd and the matching
   * speech_*_frame indices) come straight from the C-layer state machine
   * (bit-identical to upstream FireRedVAD) — the wrapper is just a marshaller.
   */
  processFrame(pcm160: Int16Array): StreamVADFrameResult | null {
    const M = getModule();
    const ptr = M._malloc(pcm160.length * 2);
    const heap16 = new Int16Array(M.HEAPU8.buffer, ptr, pcm160.length);
    heap16.set(pcm160);

    try {
      const result = streamVadProcess(M, this.handle, ptr, pcm160.length);
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
    const f32 = prepareDetectFullAudio(audio);
    const audioPtr = copyAudioToHeap(M, f32);
    const probsPtrPtr = M._malloc(4);
    const framesPtr = M._malloc(4);

    try {
      const ret = M.ccall(
        "omni_stream_vad_detect_full",
        "number",
        ["number", "number", "number", "number", "number"],
        [this.handle, audioPtr, f32.length, probsPtrPtr, framesPtr],
      );
      if (ret !== 0) throw new Error(`StreamVAD detectFull failed: ${ret}`);

      const numFrames = M.getValue(framesPtr, "i32");
      const probsPtr = M.getValue(probsPtrPtr, "i32");
      const probabilities = probsPtr
        ? new Float32Array(new Float32Array(M.HEAPU8.buffer, probsPtr, numFrames))
        : new Float32Array(0);
      if (probsPtr) M._free(probsPtr);

      return {
        probabilities,
        numFrames,
        duration: Math.round((f32.length / SAMPLE_RATE) * 1000) / 1000,
      };
    } finally {
      M._free(audioPtr);
      M._free(probsPtrPtr);
      M._free(framesPtr);
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

function int16ToFloat32(i16: Int16Array): Float32Array {
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i];
  return f32;
}

function prepareDetectFullAudio(audio: Float32Array | Int16Array): Float32Array {
  if (audio instanceof Int16Array) {
    return int16ToFloat32(audio);
  }
  if (isNormalizedFloat(audio)) {
    const scaled = new Float32Array(audio.length);
    for (let i = 0; i < audio.length; i++) scaled[i] = audio[i] * 32768;
    return scaled;
  }
  return audio;
}

function isNormalizedFloat(audio: Float32Array): boolean {
  const step = Math.max(1, Math.floor(audio.length / 1000));
  let maxAbs = 0;
  for (let i = 0; i < audio.length; i += step) {
    const v = Math.abs(audio[i]);
    if (v > maxAbs) maxAbs = v;
  }
  return maxAbs <= 1.0;
}
