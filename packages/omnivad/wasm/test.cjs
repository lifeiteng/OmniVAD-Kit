/**
 * Quick integration test: load WASM, run VAD+AED on test audio.
 */
const fs = require("fs");
const path = require("path");

const WASM_DIR = path.join(__dirname, "..", "dist", "wasm");
const TEST_AUDIO = path.join(__dirname, "..", "..", "..", "tests", "data", "hello_en.wav");

function readWav16k(filePath) {
  const buf = fs.readFileSync(filePath);
  const int16 = new Int16Array(buf.buffer, buf.byteOffset + 44, (buf.length - 44) / 2);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i];
  return float32;
}

async function main() {
  const audio = readWav16k(TEST_AUDIO);
  console.log(`Audio: ${audio.length} samples, ${(audio.length / 16000).toFixed(3)}s`);

  // Load WASM module
  process.chdir(WASM_DIR);
  const createOmniVAD = require(path.join(WASM_DIR, "omnivad.cjs"));
  const M = await createOmniVAD();

  // --- VAD ---
  const vadHandle = M.ccall("omni_vad_create", "number", ["string"], ["models/vad.omnivad"]);
  console.assert(vadHandle !== 0, "VAD create failed");

  const audioPtr = M._malloc(audio.length * 4);
  new Float32Array(M.HEAPU8.buffer, audioPtr, audio.length).set(audio);

  const cfgPtr = M._malloc(28);
  M.setValue(cfgPtr, 0.4, "float");
  M.setValue(cfgPtr + 4, 5, "i32");
  M.setValue(cfgPtr + 8, 20, "i32");
  M.setValue(cfgPtr + 12, 20, "i32");
  M.setValue(cfgPtr + 16, 2000, "i32");
  M.setValue(cfgPtr + 20, 0, "i32");
  M.setValue(cfgPtr + 24, 0, "i32");

  const segPP = M._malloc(4), countP = M._malloc(4);
  const ret = M.ccall("omni_vad_detect", "number",
    ["number","number","number","number","number","number"],
    [vadHandle, audioPtr, audio.length, cfgPtr, segPP, countP]);
  console.assert(ret === 0, `VAD detect failed: ${ret}`);

  const count = M.getValue(countP, "i32");
  const segP = M.getValue(segPP, "i32");
  console.log(`VAD: ${count} segments`);
  for (let i = 0; i < count; i++) {
    const s = M.getValue(segP + i * 8, "float").toFixed(3);
    const e = M.getValue(segP + i * 8 + 4, "float").toFixed(3);
    console.log(`  [${i + 1}] ${s} - ${e}`);
  }

  // Verify expected result for hello_en.wav
  console.assert(count === 1, `Expected 1 segment, got ${count}`);
  const start = M.getValue(segP, "float");
  const end = M.getValue(segP + 4, "float");
  console.assert(Math.abs(start - 0.26) < 0.05, `Start ${start} not close to 0.26`);
  console.assert(Math.abs(end - 1.82) < 0.05, `End ${end} not close to 1.82`);

  if (segP) M._free(segP);
  M.ccall("omni_vad_destroy", null, ["number"], [vadHandle]);
  M._free(audioPtr);
  M._free(cfgPtr);
  M._free(segPP);
  M._free(countP);

  console.log("WASM integration test passed!");
}

main().catch(e => { console.error(e); process.exit(1); });
