// OmniVAD — Voice Activity Detection & Audio Event Detection
// NPM package using ncnn via WebAssembly. Zero external dependencies.
// Based on FireRedVAD by Xiaohongshu.

// Main classes
export { OmniVAD } from "./vad.js";
export { OmniStreamVAD } from "./stream-vad.js";
export { OmniAED } from "./aed.js";

// Pure-algorithm utilities (no model load required)
export { mergeChunks, DEFAULT_CHUNK_CONFIG } from "./chunking.js";

// WASM initialization (auto-called by create(), exposed for manual control)
export { initWasm, loadModel, VERSION, DEFAULT_CDN_BASE, MODEL_FILES } from "./wasm-binding.js";

// Backward-compatible aliases
export { OmniVAD as FireRedVAD } from "./vad.js";
export { OmniStreamVAD as FireRedStreamVAD } from "./stream-vad.js";
export { OmniAED as FireRedAED } from "./aed.js";

// Types
export type {
  ModelSource,
  VADResult,
  AEDResult,
  StreamVADFrameResult,
  StreamVADFullResult,
  VADConfig,
  AEDConfig,
  StreamVADConfig,
  ChunkOptions,
  ChunkResult,
} from "./types.js";
