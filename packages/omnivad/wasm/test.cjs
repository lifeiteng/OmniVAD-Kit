/**
 * Quick integration test: load WASM, load model from buffer, run VAD on test
 * audio, and assert against the expected segment timing.
 *
 * Uses node:assert (throws + non-zero exit on failure) so CI fails loudly
 * if the wasm output drifts. console.assert() is a trap here — Node's
 * console.assert just logs and keeps going.
 */
const fs = require("fs");
const path = require("path");
const assert = require("node:assert/strict");

const WASM_DIR = path.join(__dirname, "..", "dist", "wasm");
const MODELS_DIR = path.join(__dirname, "..", "..", "..", "models");
const TEST_AUDIO = path.join(__dirname, "..", "..", "..", "tests", "data", "hello_en.wav");

function readWav16k(filePath) {
  const buf = fs.readFileSync(filePath);
  const int16 = new Int16Array(buf.buffer, buf.byteOffset + 44, (buf.length - 44) / 2);
  const float32 = new Float32Array(int16.length);
  // Normalize int16 → float32 in [-1, 1]. OmniVAD expects normalized
  // float input; without the divide we'd feed it values up to ±32768,
  // which saturates the model and triggers VAD on near-silence.
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
  return float32;
}

function loadModelBuffer(M, modelPath) {
  const buf = fs.readFileSync(modelPath);
  const ptr = M._malloc(buf.length);
  M.HEAPU8.set(new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength), ptr);
  return { ptr, size: buf.length };
}

async function main() {
  const audio = readWav16k(TEST_AUDIO);
  console.log(`Audio: ${audio.length} samples, ${(audio.length / 16000).toFixed(3)}s`);

  // Load WASM module
  const createOmniVAD = require(path.join(WASM_DIR, "omnivad.cjs"));
  const M = await createOmniVAD();

  // --- VAD (buffer-based create) ---
  const vadModel = loadModelBuffer(M, path.join(MODELS_DIR, "vad.omnivad"));
  const errPtr = M._malloc(4);

  const vadHandle = M.ccall(
    "omni_vad_create_from_buffer", "number",
    ["number", "number", "number"],
    [vadModel.ptr, vadModel.size, errPtr]
  );
  assert.notStrictEqual(vadHandle, 0, `VAD create failed: error ${M.getValue(errPtr, "i32")}`);
  M._free(vadModel.ptr);

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
  assert.strictEqual(ret, 0, `VAD detect failed: ${ret}`);

  const count = M.getValue(countP, "i32");
  const segP = M.getValue(segPP, "i32");
  console.log(`VAD: ${count} segments`);
  for (let i = 0; i < count; i++) {
    const s = M.getValue(segP + i * 8, "float").toFixed(3);
    const e = M.getValue(segP + i * 8 + 4, "float").toFixed(3);
    console.log(`  [${i + 1}] ${s} - ${e}`);
  }

  // Verify expected result for hello_en.wav (matches Python reference).
  assert.strictEqual(count, 1, `Expected 1 segment, got ${count}`);
  const start = M.getValue(segP, "float");
  const end = M.getValue(segP + 4, "float");
  assert.ok(Math.abs(start - 0.26) < 0.05, `Start ${start} not close to 0.26`);
  assert.ok(Math.abs(end   - 1.82) < 0.05, `End ${end} not close to 1.82`);

  if (segP) M._free(segP);
  M.ccall("omni_vad_destroy", null, ["number"], [vadHandle]);
  M._free(audioPtr);
  M._free(cfgPtr);
  M._free(segPP);
  M._free(countP);
  M._free(errPtr);

  // --- AED overlap segmenter (pseudo-streaming whole-window AED) ---
  // Hardcoded struct offsets here independently cross-check the offsets used
  // in src/wasm-binding.ts — if either drifts from native/include/omnivad.h
  // this test fails loudly in CI before publish.
  testAedOverlapSegmenter(M, audio);

  console.log("WASM integration test passed!");
}

function testAedOverlapSegmenter(M, audio) {
  const aedModel = loadModelBuffer(M, path.join(MODELS_DIR, "aed.omnivad"));

  // OmniAedOverlapConfig: 10 i32 (ms) + 3 float = 52 bytes. Use native defaults.
  const cfgPtr = M._malloc(52);
  M.setValue(cfgPtr + 0,  2000,  "i32"); // hop_ms
  M.setValue(cfgPtr + 4,  250,   "i32"); // overlap_ms
  M.setValue(cfgPtr + 8,  0,     "i32"); // edge_guard_ms
  M.setValue(cfgPtr + 12, 2000,  "i32"); // hard_split_pause_ms
  M.setValue(cfgPtr + 16, 60000, "i32"); // max_chunk_ms
  M.setValue(cfgPtr + 20, 200,   "i32"); // min_speech_ms
  M.setValue(cfgPtr + 24, 200,   "i32"); // merge_gap_ms
  M.setValue(cfgPtr + 28, 0,     "i32"); // music_gap_tolerance_ms
  M.setValue(cfgPtr + 32, 0,     "i32"); // pad_start_ms
  M.setValue(cfgPtr + 36, 0,     "i32"); // pad_end_ms
  M.setValue(cfgPtr + 40, 0.5,   "float"); // speech_threshold
  M.setValue(cfgPtr + 44, 0.5,   "float"); // singing_threshold
  M.setValue(cfgPtr + 48, 0.5,   "float"); // music_threshold

  const errPtr = M._malloc(4);
  const handle = M.ccall(
    "omni_aed_overlap_segmenter_create_from_buffer", "number",
    ["number", "number", "number", "number"],
    [aedModel.ptr, aedModel.size, cfgPtr, errPtr]
  );
  assert.notStrictEqual(handle, 0, `AED overlap create failed: error ${M.getValue(errPtr, "i32")}`);
  M._free(aedModel.ptr);
  M._free(cfgPtr);
  M._free(errPtr);

  const audioPtr = M._malloc(audio.length * 4);
  new Float32Array(M.HEAPU8.buffer, audioPtr, audio.length).set(audio);

  const KIND_NAMES = { 0: "silence", 1: "speech", 2: "singing", 3: "music", 4: "mixed" };
  const MASK_TRANSCRIBABLE = (1 << 0) | (1 << 1);

  function readResult(segPP, segCP, evPP, evCP) {
    const out = { segments: [], events: [] };
    const evCount = M.getValue(evCP, "i32");
    const evP = M.getValue(evPP, "i32");
    for (let i = 0; i < evCount; i++) {
      const b = evP + i * 32; // OmniAedOnlineEvent = 32 bytes
      const mask = M.getValue(b + 12, "i32") >>> 0;
      const ev = {
        start: M.getValue(b + 0, "float"),
        end: M.getValue(b + 4, "float"),
        primaryKind: KIND_NAMES[M.getValue(b + 8, "i32")],
        kindMask: mask,
        isTranscribable: (mask & MASK_TRANSCRIBABLE) !== 0,
      };
      assert.ok(ev.start <= ev.end + 1e-6, `event start ${ev.start} > end ${ev.end}`);
      assert.ok(ev.primaryKind !== undefined, `unknown event kind at ${i}`);
      out.events.push(ev);
    }
    const segCount = M.getValue(segCP, "i32");
    const segP = M.getValue(segPP, "i32");
    for (let i = 0; i < segCount; i++) {
      const b = segP + i * 16; // OmniAedOnlineSegment = 16 bytes
      const seg = {
        start: M.getValue(b + 0, "float"),
        end: M.getValue(b + 4, "float"),
        eventStartIdx: M.getValue(b + 8, "i32"),
        eventCount: M.getValue(b + 12, "i32"),
      };
      assert.ok(seg.start <= seg.end + 1e-6, `segment start ${seg.start} > end ${seg.end}`);
      out.segments.push(seg);
    }
    if (evP) M._free(evP);
    if (segP) M._free(segP);
    return out;
  }

  const segPP = M._malloc(4), segCP = M._malloc(4), evPP = M._malloc(4), evCP = M._malloc(4);
  M.setValue(segPP, 0, "i32");
  M.setValue(evPP, 0, "i32");

  let ret = M.ccall(
    "omni_aed_overlap_segmenter_ingest", "number",
    ["number", "number", "number", "number", "number", "number", "number"],
    [handle, audioPtr, audio.length, segPP, segCP, evPP, evCP]
  );
  assert.strictEqual(ret, 0, `AED overlap ingest failed: ${ret}`);
  const ingested = readResult(segPP, segCP, evPP, evCP);

  M.setValue(segPP, 0, "i32");
  M.setValue(evPP, 0, "i32");
  ret = M.ccall(
    "omni_aed_overlap_segmenter_flush", "number",
    ["number", "number", "number", "number", "number"],
    [handle, segPP, segCP, evPP, evCP]
  );
  assert.strictEqual(ret, 0, `AED overlap flush failed: ${ret}`);
  const flushed = readResult(segPP, segCP, evPP, evCP);

  const events = [...ingested.events, ...flushed.events];
  const segments = [...ingested.segments, ...flushed.segments];
  console.log(`AED overlap: ${events.length} events, ${segments.length} segments`);

  // hello_en.wav is ~1.8s of real speech → expect at least one committed
  // transcribable event/segment after flush runs the final partial window.
  assert.ok(events.length >= 1, `Expected >= 1 AED overlap event, got ${events.length}`);
  assert.ok(
    events.some((e) => e.isTranscribable),
    "Expected at least one transcribable (speech/singing) event",
  );
  assert.ok(segments.length >= 1, `Expected >= 1 transcribable segment, got ${segments.length}`);

  M.ccall("omni_aed_overlap_segmenter_reset", null, ["number"], [handle]);
  M.ccall("omni_aed_overlap_segmenter_destroy", null, ["number"], [handle]);
  M._free(audioPtr);
  M._free(segPP); M._free(segCP); M._free(evPP); M._free(evCP);
}

main().catch(e => { console.error(e); process.exit(1); });
