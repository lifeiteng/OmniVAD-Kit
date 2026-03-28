// OmniVAD — Voice Activity Detection & Audio Event Detection
// NPM package using ncnn via WebAssembly. Zero external dependencies.
// Based on FireRedVAD by Xiaohongshu.

// Main classes
export { OmniVAD } from "./vad.js";
export { OmniStreamVAD } from "./stream-vad.js";
export { OmniAED } from "./aed.js";

// WASM initialization (auto-called by create(), exposed for manual control)
export { initWasm } from "./wasm-binding.js";

// Backward-compatible aliases
export { OmniVAD as FireRedVAD } from "./vad.js";
export { OmniStreamVAD as FireRedStreamVAD } from "./stream-vad.js";
export { OmniAED as FireRedAED } from "./aed.js";

// Types
export type {
  VADResult,
  AEDResult,
  StreamVADFrameResult,
  VADConfig,
  AEDConfig,
  StreamVADConfig,
} from "./types.js";
