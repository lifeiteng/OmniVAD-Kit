/**
 * Streaming Voice Activity Detection (WASM/ncnn backend).
 * Processes audio frame-by-frame (10ms chunks of 160 samples @ 16kHz).
 */

import type { StreamVADConfig, StreamVADFrameResult } from "./types.js";
import {
  initWasm,
  getModule,
  copyAudioToHeap,
  streamVadCreate,
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
   * Initializes WASM and loads the bundled ncnn model.
   */
  static async create(options: StreamVADConfig = {}): Promise<OmniStreamVAD> {
    await initWasm();
    const M = getModule();
    const threshold = options.speechThreshold ?? 0.5;
    const handle = streamVadCreate(M, threshold);
    return new OmniStreamVAD(handle);
  }

  /**
   * Process one frame of audio (160 int16 samples = 10ms @ 16kHz).
   * Returns null until enough audio is accumulated.
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
        confidence: result.confidence,
        smoothedConfidence: result.confidence,
        isSpeech: result.isSpeech,
        frameIndex: result.frameOffset,
        isSpeechStart: false,
        isSpeechEnd: false,
        speechStartFrame: 0,
        speechEndFrame: 0,
      };
    } finally {
      M._free(ptr);
    }
  }

  /**
   * Process entire audio at once and return per-frame probabilities.
   * @param audio - Float32Array (int16 range) or Int16Array of 16kHz mono PCM
   */
  detectFull(audio: Float32Array | Int16Array): { numFrames: number; duration: number } {
    const M = getModule();
    const f32 = audio instanceof Int16Array ? int16ToFloat32(audio) : audio;
    const audioPtr = copyAudioToHeap(M, f32);
    const probsPtrPtr = M._malloc(4);
    const framesPtr = M._malloc(4);

    try {
      const ret = M.ccall(
        "omni_vad_stream_detect_full",
        "number",
        ["number", "number", "number", "number", "number"],
        [this.handle, audioPtr, f32.length, probsPtrPtr, framesPtr],
      );
      if (ret !== 0) throw new Error(`StreamVAD detectFull failed: ${ret}`);

      const numFrames = M.getValue(framesPtr, "i32");
      const probsPtr = M.getValue(probsPtrPtr, "i32");
      if (probsPtr) M._free(probsPtr);

      return {
        numFrames,
        duration: Math.round((f32.length / SAMPLE_RATE) * 1000) / 1000,
      };
    } finally {
      M._free(audioPtr);
      M._free(probsPtrPtr);
      M._free(framesPtr);
    }
  }

  /** Reset all internal state. */
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
