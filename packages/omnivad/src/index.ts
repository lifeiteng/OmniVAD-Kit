// OmniVAD — Voice Activity Detection & Audio Event Detection
// NPM package using ncnn via WebAssembly. Zero external dependencies.
// Based on FireRedVAD by Xiaohongshu.

// Main classes
export { OmniVAD } from "./vad.js";
export { OmniStreamVAD } from "./stream-vad.js";
export { OmniAED } from "./aed.js";
export { OmniAEDOverlapSegmenter } from "./aed-overlap.js";

// Pure-algorithm utilities (no model load required)
export { mergeChunks, DEFAULT_CHUNK_CONFIG } from "./chunking.js";

// AED overlap segmenter kind constants
export {
  AED_KIND_MASK_SPEECH,
  AED_KIND_MASK_SINGING,
  AED_KIND_MASK_MUSIC,
  AED_KIND_MASK_TRANSCRIBABLE,
} from "./wasm-binding.js";

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
  AEDOverlapConfig,
  AEDOverlapEvent,
  AEDOverlapSegment,
  AEDOverlapResult,
  ChunkOptions,
  ChunkResult,
} from "./types.js";
