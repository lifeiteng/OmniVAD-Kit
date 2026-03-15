// OmniVAD — Voice Activity Detection & Audio Event Detection
// NPM package supporting both browser and Node.js via ONNX Runtime Web.
// Based on FireRedVAD by Xiaohongshu.

// Main classes
export { OmniVAD } from "./vad.js";
export { OmniStreamVAD } from "./stream-vad.js";
export { OmniAED } from "./aed.js";

// Backward-compatible aliases
export { OmniVAD as FireRedVAD } from "./vad.js";
export { OmniStreamVAD as FireRedStreamVAD } from "./stream-vad.js";
export { OmniAED as FireRedAED } from "./aed.js";

// Feature extraction (exposed for advanced usage and testing)
export { Fbank } from "./fbank.js";
export { FFTComputer } from "./fft.js";
export { CMVN, DEFAULT_CMVN, loadCMVN } from "./cmvn.js";

// Post-processing (exposed for advanced usage and testing)
export {
  VadPostProcessor,
  StreamVadPostProcessor,
  streamResultsToTimestamps,
} from "./post-process.js";

// Types
export type {
  VADResult,
  AEDResult,
  StreamVADFrameResult,
  VADConfig,
  AEDConfig,
  StreamVADConfig,
  CMVNData,
  VADCreateOptions,
  StreamVADCreateOptions,
  AEDCreateOptions,
} from "./types.js";
