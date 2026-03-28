/**
 * Benchmark: omnivad WASM inference speed.
 *
 * Usage: node bench.js [audio.wav]
 * Default: uses bundled test audio (hello_en.wav)
 */

const fs = require("fs");
const path = require("path");

const WASM_DIR = path.join(__dirname, "..", "dist", "wasm");
const TEST_AUDIO = path.join(
  __dirname,
  "..",
  "..",
  "..",
  "tests",
  "data",
  "hello_en.wav",
);
const SAMPLE_RATE = 16000;

function readWav16k(filePath) {
  const buf = fs.readFileSync(filePath);
  // Skip 44-byte WAV header, read as int16, convert to float32 in int16 range
  const int16 = new Int16Array(
    buf.buffer,
    buf.byteOffset + 44,
    (buf.length - 44) / 2,
  );
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i];
  return float32;
}

// Struct sizes must match C definitions
const SIZEOF_POST_CONFIG = 7 * 4; // 7 x int/float
const SIZEOF_AED_POST_CONFIG = 3 * SIZEOF_POST_CONFIG;
const SIZEOF_SEGMENT = 2 * 4; // start, end (float)
const SIZEOF_AED_SEGMENT = 4 * 4; // start, end, cls, confidence

async function main() {
  const audioPath = process.argv[2] || TEST_AUDIO;
  const audio = readWav16k(audioPath);
  const duration = audio.length / SAMPLE_RATE;

  console.log(`Audio: ${audioPath}`);
  console.log(`Duration: ${duration.toFixed(3)}s, Samples: ${audio.length}\n`);

  // Load WASM (.cjs to avoid Node.js ESM detection from parent package.json)
  process.chdir(WASM_DIR);
  const createOmniVAD = require(path.join(WASM_DIR, "omnivad.cjs"));

  let t0 = performance.now();
  const M = await createOmniVAD();
  const initTime = performance.now() - t0;
  console.log(`WASM init: ${initTime.toFixed(1)}ms\n`);

  // Copy audio to WASM heap
  const audioPtr = M._malloc(audio.length * 4);
  const heapF32 = new Float32Array(M.HEAPU8.buffer, audioPtr, audio.length);
  heapF32.set(audio);

  // === VAD ===
  t0 = performance.now();
  const vadHandle = M.ccall(
    "omni_vad_create",
    "number",
    ["string"],
    ["models/vad.omnivad"],
  );
  const vadCreateTime = performance.now() - t0;

  // Set default config via setValue
  const cfgPtr = M._malloc(SIZEOF_POST_CONFIG);
  M.setValue(cfgPtr + 0, 0.4, "float"); // threshold
  M.setValue(cfgPtr + 4, 5, "i32"); // smooth_window_size
  M.setValue(cfgPtr + 8, 20, "i32"); // min_speech_frames
  M.setValue(cfgPtr + 12, 20, "i32"); // min_silence_frames
  M.setValue(cfgPtr + 16, 2000, "i32"); // max_speech_frames
  M.setValue(cfgPtr + 20, 0, "i32"); // merge_silence_frames
  M.setValue(cfgPtr + 24, 0, "i32"); // extend_speech_frames

  const segPtrPtr = M._malloc(4);
  const countPtr = M._malloc(4);

  t0 = performance.now();
  const vadRet = M.ccall(
    "omni_vad_detect",
    "number",
    ["number", "number", "number", "number", "number", "number"],
    [vadHandle, audioPtr, audio.length, cfgPtr, segPtrPtr, countPtr],
  );
  const vadTime = performance.now() - t0;

  const vadCount = M.getValue(countPtr, "i32");
  const vadSegPtr = M.getValue(segPtrPtr, "i32");
  const vadSegments = [];
  for (let i = 0; i < vadCount; i++) {
    const base = vadSegPtr + i * SIZEOF_SEGMENT;
    vadSegments.push({
      start: M.getValue(base, "float").toFixed(3),
      end: M.getValue(base + 4, "float").toFixed(3),
    });
  }
  if (vadSegPtr) M._free(vadSegPtr);

  console.log(`--- VAD ---`);
  console.log(`Create: ${vadCreateTime.toFixed(1)}ms`);
  console.log(
    `Detect: ${vadTime.toFixed(1)}ms, RTF=${(vadTime / 1000 / duration).toFixed(4)}`,
  );
  console.log(`Segments: ${vadCount}`);
  vadSegments.forEach((s, i) => console.log(`  [${i + 1}] ${s.start} - ${s.end}`));

  // === AED ===
  t0 = performance.now();
  const aedHandle = M.ccall(
    "omni_aed_create",
    "number",
    ["string"],
    ["models/aed.omnivad"],
  );
  const aedCreateTime = performance.now() - t0;

  // AED config (3 x PostConfig)
  const aedCfgPtr = M._malloc(SIZEOF_AED_POST_CONFIG);
  for (let c = 0; c < 3; c++) {
    const off = aedCfgPtr + c * SIZEOF_POST_CONFIG;
    M.setValue(off + 0, c === 0 ? 0.4 : 0.5, "float");
    M.setValue(off + 4, 5, "i32");
    M.setValue(off + 8, 20, "i32");
    M.setValue(off + 12, 20, "i32");
    M.setValue(off + 16, 2000, "i32");
    M.setValue(off + 20, 0, "i32");
    M.setValue(off + 24, 0, "i32");
  }

  const aedSegPtrPtr = M._malloc(4);
  const aedCountPtr = M._malloc(4);

  t0 = performance.now();
  const aedRet = M.ccall(
    "omni_aed_detect",
    "number",
    ["number", "number", "number", "number", "number", "number"],
    [aedHandle, audioPtr, audio.length, aedCfgPtr, aedSegPtrPtr, aedCountPtr],
  );
  const aedTime = performance.now() - t0;

  const aedCount = M.getValue(aedCountPtr, "i32");
  const aedSegP = M.getValue(aedSegPtrPtr, "i32");
  const AED_CLASSES = { 0: "speech", 1: "singing", 2: "music" };
  const events = { speech: [], singing: [], music: [] };
  for (let i = 0; i < aedCount; i++) {
    const base = aedSegP + i * SIZEOF_AED_SEGMENT;
    const cls = M.getValue(base + 8, "i32");
    const name = AED_CLASSES[cls] || "unknown";
    if (events[name])
      events[name].push({
        start: M.getValue(base, "float").toFixed(3),
        end: M.getValue(base + 4, "float").toFixed(3),
      });
  }
  if (aedSegP) M._free(aedSegP);

  console.log(`\n--- AED ---`);
  console.log(`Create: ${aedCreateTime.toFixed(1)}ms`);
  console.log(
    `Detect: ${aedTime.toFixed(1)}ms, RTF=${(aedTime / 1000 / duration).toFixed(4)}`,
  );
  for (const [cls, segs] of Object.entries(events)) {
    if (segs.length)
      segs.forEach((s, i) => console.log(`  ${cls}[${i + 1}] ${s.start} - ${s.end}`));
  }

  // === StreamVAD ===
  t0 = performance.now();
  const svadHandle = M.ccall(
    "omni_stream_vad_create",
    "number",
    ["string", "number"],
    ["models/stream-vad.omnivad", 0.5],
  );
  const svadCreateTime = performance.now() - t0;

  const probsPtrPtr = M._malloc(4);
  const framesPtrC = M._malloc(4);

  t0 = performance.now();
  const svadRet = M.ccall(
    "omni_stream_vad_detect_full",
    "number",
    ["number", "number", "number", "number", "number"],
    [svadHandle, audioPtr, audio.length, probsPtrPtr, framesPtrC],
  );
  const svadTime = performance.now() - t0;
  const svadFrames = M.getValue(framesPtrC, "i32");
  const probsP = M.getValue(probsPtrPtr, "i32");
  if (probsP) M._free(probsP);

  console.log(`\n--- StreamVAD ---`);
  console.log(`Create: ${svadCreateTime.toFixed(1)}ms`);
  console.log(
    `DetectFull: ${svadTime.toFixed(1)}ms, RTF=${(svadTime / 1000 / duration).toFixed(4)}, ${svadFrames} frames`,
  );

  // Cleanup
  M.ccall("omni_vad_destroy", null, ["number"], [vadHandle]);
  M.ccall("omni_aed_destroy", null, ["number"], [aedHandle]);
  M.ccall("omni_stream_vad_destroy", null, ["number"], [svadHandle]);
  M._free(audioPtr);
  M._free(cfgPtr);
  M._free(aedCfgPtr);
  M._free(segPtrPtr);
  M._free(countPtr);
  M._free(aedSegPtrPtr);
  M._free(aedCountPtr);
  M._free(probsPtrPtr);
  M._free(framesPtrC);

  console.log(`\n=== Summary ===`);
  console.log(`Audio: ${duration.toFixed(3)}s`);
  console.log(
    `VAD:       ${vadTime.toFixed(1)}ms (RTF=${(vadTime / 1000 / duration).toFixed(4)})`,
  );
  console.log(
    `AED:       ${aedTime.toFixed(1)}ms (RTF=${(aedTime / 1000 / duration).toFixed(4)})`,
  );
  console.log(
    `StreamVAD: ${svadTime.toFixed(1)}ms (RTF=${(svadTime / 1000 / duration).toFixed(4)})`,
  );
}

main().catch(console.error);
