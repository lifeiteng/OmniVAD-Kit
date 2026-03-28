/**
 * Streaming Voice Activity Detection (WASM/ncnn backend).
 * Processes audio frame-by-frame (10ms chunks of 160 samples @ 16kHz).
 */

import type { StreamVADConfig, StreamVADFrameResult, StreamVADFullResult } from "./types.js";
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
  private inSpeech = false;
  private speechStartFrame = 0;

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
      if (!result || result.frameOffset === 0) return null;

      const frameIndex = result.frameOffset;
      const isSpeechStart = result.isSpeech && !this.inSpeech;
      const isSpeechEnd = !result.isSpeech && this.inSpeech;

      if (isSpeechStart) {
        this.speechStartFrame = frameIndex;
      }

      const activeSpeechStartFrame = isSpeechEnd ? this.speechStartFrame : result.isSpeech ? this.speechStartFrame : 0;
      const speechEndFrame = isSpeechEnd ? Math.max(1, frameIndex - 1) : 0;

      this.inSpeech = result.isSpeech;
      if (isSpeechEnd) {
        this.speechStartFrame = 0;
      }

      return {
        confidence: result.confidence,
        smoothedConfidence: result.confidence,
        isSpeech: result.isSpeech,
        frameIndex,
        isSpeechStart,
        isSpeechEnd,
        speechStartFrame: activeSpeechStartFrame,
        speechEndFrame,
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

  /** Reset all internal state. */
  reset(): void {
    streamVadReset(getModule(), this.handle);
    this.inSpeech = false;
    this.speechStartFrame = 0;
  }

  /** Release native resources. */
  dispose(): void {
    if (this.handle) {
      streamVadDestroy(getModule(), this.handle);
      this.handle = 0;
    }
    this.inSpeech = false;
    this.speechStartFrame = 0;
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
