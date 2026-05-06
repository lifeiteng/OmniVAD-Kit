/**
 * Stream VAD throughput bench for the WASM module.
 *
 * Usage:
 *   node bench-stream.cjs <build_dir> [audio_path] [repeat]
 *
 *   build_dir   directory containing omnivad.js + omnivad.wasm
 *   audio_path  16kHz mono WAV (default: tests/data/zh_medium.wav)
 *   repeat      number of timed runs per mode (default: 5)
 *
 * Reports RTF for two paths:
 *   1. per-frame loop  — 160-sample chunks via omni_stream_vad_process
 *   2. detect_full     — entire audio in a single ncnn forward
 */
const fs = require("fs");
const path = require("path");

if (process.argv.length < 3) {
  console.error("usage: node bench-stream.cjs <build_dir> [audio] [repeat]");
  process.exit(1);
}
const BUILD_DIR = path.resolve(process.argv[2]);
const REPO_ROOT = path.resolve(__dirname, "..", "..", "..");
const AUDIO = path.resolve(process.argv[3] || path.join(REPO_ROOT, "tests", "data", "zh_medium.wav"));
const REPEAT = parseInt(process.argv[4] || "5", 10);
const MODELS_DIR = path.join(REPO_ROOT, "models");

function readWav16k(p) {
  const buf = fs.readFileSync(p);
  // assume 44-byte WAV header, 16-bit PCM mono @ 16kHz
  const int16 = new Int16Array(buf.buffer, buf.byteOffset + 44, (buf.length - 44) / 2);
  const f32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768;
  return f32;
}

function loadModelBuf(M, p) {
  const b = fs.readFileSync(p);
  const ptr = M._malloc(b.length);
  M.HEAPU8.set(new Uint8Array(b.buffer, b.byteOffset, b.byteLength), ptr);
  return { ptr, size: b.length };
}

function writeStreamConfig(M, cfgPtr) {
  // OmniStreamVadConfig: f32 + 5 * i32 = 24 bytes
  M.setValue(cfgPtr,      0.5,  "float"); // threshold
  M.setValue(cfgPtr +  4, 5,    "i32");   // smooth_window_size
  M.setValue(cfgPtr +  8, 5,    "i32");   // pad_start_frame
  M.setValue(cfgPtr + 12, 8,    "i32");   // min_speech_frame
  M.setValue(cfgPtr + 16, 2000, "i32");   // max_speech_frame
  M.setValue(cfgPtr + 20, 20,   "i32");   // min_silence_frame
}

function hrSec() { return Number(process.hrtime.bigint()) / 1e9; }

(async () => {
  const audio = readWav16k(AUDIO);
  const dur = audio.length / 16000;
  console.log(`# build_dir : ${BUILD_DIR}`);
  console.log(`# audio     : ${path.basename(AUDIO)}  ${dur.toFixed(2)}s  ${audio.length} samples`);
  console.log(`# repeats   : ${REPEAT}`);

  // Force CommonJS resolution: package.json has "type":"module", so a .js
  // sibling would be loaded as ESM. Mirror to .cjs once and require that.
  const cjsPath = path.join(BUILD_DIR, "omnivad.cjs");
  if (!fs.existsSync(cjsPath)) {
    fs.copyFileSync(path.join(BUILD_DIR, "omnivad.js"), cjsPath);
  }
  const create = require(cjsPath);
  const M = await create();

  // Persistent buffers
  const audioPtr = M._malloc(audio.length * 4);
  new Float32Array(M.HEAPU8.buffer, audioPtr, audio.length).set(audio);

  const model = loadModelBuf(M, path.join(MODELS_DIR, "stream-vad.omnivad"));
  const cfgPtr = M._malloc(24);
  writeStreamConfig(M, cfgPtr);
  const errPtr = M._malloc(4);

  const handle = M.ccall(
    "omni_stream_vad_create_from_buffer", "number",
    ["number", "number", "number", "number"],
    [model.ptr, model.size, cfgPtr, errPtr]
  );
  if (handle === 0) {
    console.error(`stream-vad create failed: err=${M.getValue(errPtr, "i32")}`);
    process.exit(1);
  }
  M._free(model.ptr);

  const resPtr = M._malloc(24); // OmniStreamVadResult: 24 bytes (with padding)

  // ---- Mode 1: per-frame loop (160 samples) ----
  const FRAME = 160;
  const numFrames = Math.floor(audio.length / FRAME);
  const loopTimes = [];
  for (let r = 0; r < REPEAT; r++) {
    M.ccall("omni_stream_vad_reset", null, ["number"], [handle]);
    const t0 = hrSec();
    for (let i = 0; i < numFrames; i++) {
      M.ccall(
        "omni_stream_vad_process", "number",
        ["number", "number", "number", "number"],
        [handle, audioPtr + i * FRAME * 4, FRAME, resPtr]
      );
    }
    loopTimes.push(hrSec() - t0);
  }

  // ---- Mode 2: detect_full (single batch forward) ----
  const probsPP = M._malloc(4);
  const nFrameP = M._malloc(4);
  const fullTimes = [];
  for (let r = 0; r < REPEAT; r++) {
    M.ccall("omni_stream_vad_reset", null, ["number"], [handle]);
    const t0 = hrSec();
    M.ccall(
      "omni_stream_vad_detect_full", "number",
      ["number", "number", "number", "number", "number"],
      [handle, audioPtr, audio.length, probsPP, nFrameP]
    );
    fullTimes.push(hrSec() - t0);
    const outPtr = M.getValue(probsPP, "i32");
    if (outPtr) M.ccall("omni_free", null, ["number"], [outPtr]);
  }

  M.ccall("omni_stream_vad_destroy", null, ["number"], [handle]);
  M._free(audioPtr); M._free(cfgPtr); M._free(errPtr);
  M._free(resPtr); M._free(probsPP); M._free(nFrameP);

  const stats = (xs) => {
    const sorted = [...xs].sort((a, b) => a - b);
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
    return { min: sorted[0], median: sorted[Math.floor(xs.length / 2)], mean };
  };
  const sLoop = stats(loopTimes);
  const sFull = stats(fullTimes);

  console.log("");
  console.log("mode             min(ms)   median(ms)  mean(ms)  RTF(min)  RTF(median)");
  console.log(`per-frame loop   ${(sLoop.min*1000).toFixed(1).padStart(7)}   ${(sLoop.median*1000).toFixed(1).padStart(8)}   ${(sLoop.mean*1000).toFixed(1).padStart(7)}  ${(sLoop.min/dur).toFixed(4)}    ${(sLoop.median/dur).toFixed(4)}`);
  console.log(`detect_full      ${(sFull.min*1000).toFixed(1).padStart(7)}   ${(sFull.median*1000).toFixed(1).padStart(8)}   ${(sFull.mean*1000).toFixed(1).padStart(7)}  ${(sFull.min/dur).toFixed(4)}    ${(sFull.median/dur).toFixed(4)}`);
})();
