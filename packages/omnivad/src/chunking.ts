/**
 * Pure-algorithm chunking utility — wraps the C function omni_merge_chunks
 * compiled into the WASM module.
 *
 * WhisperX-style binarize+merge, minus the binarize half because OmniVAD
 * already returns binarized timestamps.
 *
 * Usage:
 *
 *   import { mergeChunks } from "omnivad";
 *
 *   const chunks = await mergeChunks(
 *     [[0.0, 5.0], [6.0, 10.0]],
 *     { chunkSize: 30.0, maxGap: 2.0 }
 *   );
 *   // [{ start: 0, end: 10, segStartIdx: 0, segCount: 2 }]
 */

import type { ChunkOptions, ChunkResult } from "./types.js";
import {
  chunkMerge,
  DEFAULT_CHUNK_CONFIG,
  getModule,
  initWasm,
  type ChunkConfig,
} from "./wasm-binding.js";

/**
 * Merge a sorted array of [start, end] speech segments into duration-bounded
 * chunks.
 *
 * Lazily initializes the WASM module on first call (so the caller doesn't have
 * to await `initWasm()` separately). Subsequent calls reuse the cached module.
 *
 * @param segments  array of [start, end] pairs in seconds, sorted by start
 * @param options   chunking configuration; missing fields fall back to
 *                  {@link DEFAULT_CHUNK_CONFIG}
 */
export async function mergeChunks(
  segments: Array<[number, number]>,
  options: ChunkOptions = {},
): Promise<ChunkResult[]> {
  await initWasm();
  const M = getModule();

  const cfg: ChunkConfig = {
    chunkSize:      options.chunkSize      ?? DEFAULT_CHUNK_CONFIG.chunkSize,
    maxGap:         options.maxGap         ?? DEFAULT_CHUNK_CONFIG.maxGap,
    padOnset:       options.padOnset       ?? DEFAULT_CHUNK_CONFIG.padOnset,
    padOffset:      options.padOffset      ?? DEFAULT_CHUNK_CONFIG.padOffset,
    minDurationOn:  options.minDurationOn  ?? DEFAULT_CHUNK_CONFIG.minDurationOn,
    minDurationOff: options.minDurationOff ?? DEFAULT_CHUNK_CONFIG.minDurationOff,
    mode:           options.mode           ?? DEFAULT_CHUNK_CONFIG.mode,
  };

  const records = chunkMerge(M, segments, cfg);
  return records.map((r) => ({
    start: r.start,
    end: r.end,
    segStartIdx: r.segStartIdx,
    segCount: r.segCount,
  }));
}

export { DEFAULT_CHUNK_CONFIG } from "./wasm-binding.js";
export type { ChunkOptions, ChunkResult } from "./types.js";
