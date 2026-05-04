/**
 * End-to-end test of the TS wrapper's dtype dispatch.
 *
 * Verifies that:
 *   1. OmniVAD / OmniAED / OmniStreamVAD all accept BOTH Float32Array
 *      [-1, 1] and Int16Array — the wrappers route by dtype to the
 *      matching C entry.
 *   2. Results from the two paths agree to within int16 quantization
 *      noise (< 5e-3 per-frame probability).
 *
 * This is the regression guard against the f98ecc7-class bug where a
 * wrapper-side conversion silently saturated the model.
 */
const fs = require("fs");
const path = require("path");
const assert = require("node:assert/strict");

const TEST_AUDIO = path.join(
  __dirname, "..", "..", "..", "tests", "data", "hello_en.wav",
);
const PROB_TOL = 5e-3;
const TIME_TOL = 0.02;

function readWavInt16(filePath) {
  const buf = fs.readFileSync(filePath);
  // Skip 44-byte WAV header; samples are little-endian int16.
  return new Int16Array(buf.buffer, buf.byteOffset + 44, (buf.length - 44) / 2);
}

function int16ToFloat32(i16) {
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  return f32;
}

function maxAbsDelta(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function main() {
  const i16 = readWavInt16(TEST_AUDIO);
  // Copy buffers so the two paths can't share the same backing store.
  const i16Copy = new Int16Array(i16);
  const f32 = int16ToFloat32(i16);

  const { OmniVAD, OmniAED, OmniStreamVAD } = require(
    path.join(__dirname, "..", "dist", "index.cjs"),
  );

  // --- Non-stream VAD ---
  {
    const vad = await OmniVAD.create();
    const r_f32 = vad.detect(f32);
    const r_i16 = vad.detect(i16Copy);
    assert.strictEqual(
      r_f32.timestamps.length, r_i16.timestamps.length,
      `VAD: segment count differs (f32=${r_f32.timestamps.length}, i16=${r_i16.timestamps.length})`,
    );
    for (let i = 0; i < r_f32.timestamps.length; i++) {
      const [s1, e1] = r_f32.timestamps[i];
      const [s2, e2] = r_i16.timestamps[i];
      assert.ok(Math.abs(s1 - s2) < TIME_TOL, `VAD seg ${i} start: ${s1} vs ${s2}`);
      assert.ok(Math.abs(e1 - e2) < TIME_TOL, `VAD seg ${i} end:   ${e1} vs ${e2}`);
    }
    // Sanity: hello_en.wav should produce a single ~(0.26, 1.82) segment.
    assert.strictEqual(r_f32.timestamps.length, 1);
    assert.ok(Math.abs(r_f32.timestamps[0][0] - 0.26) < 0.05);
    assert.ok(Math.abs(r_f32.timestamps[0][1] - 1.82) < 0.05);
    vad.dispose();
    console.log(`VAD       OK — ${r_f32.timestamps.length} segments, dispatch verified`);
  }

  // --- AED ---
  {
    const aed = await OmniAED.create();
    const r_f32 = aed.detect(f32);
    const r_i16 = aed.detect(new Int16Array(i16));
    for (const cls of ["speech", "singing", "music"]) {
      assert.strictEqual(
        r_f32.events[cls].length, r_i16.events[cls].length,
        `AED ${cls}: segment count differs`,
      );
    }
    aed.dispose();
    console.log(`AED       OK — ${r_f32.events.speech.length} speech segments, dispatch verified`);
  }

  // --- Streaming VAD: detectFull (batch) ---
  {
    const svad = await OmniStreamVAD.create();
    const r_f32 = svad.detectFull(f32);
    const r_i16 = svad.detectFull(new Int16Array(i16));
    assert.strictEqual(r_f32.numFrames, r_i16.numFrames, "detectFull frame count");
    const delta = maxAbsDelta(r_f32.probabilities, r_i16.probabilities);
    assert.ok(
      delta < PROB_TOL,
      `StreamVAD detectFull: max prob delta ${delta} >= ${PROB_TOL}`,
    );
    svad.dispose();
    console.log(`StreamVAD detectFull OK — max delta ${delta.toExponential(2)}`);
  }

  // --- Streaming VAD: processFrame (per-frame, both dtypes) ---
  {
    const svad = await OmniStreamVAD.create();
    const CHUNK = 160;

    const probs_f32 = [];
    svad.reset();
    for (let off = 0; off + CHUNK <= f32.length; off += CHUNK) {
      const r = svad.processFrame(f32.subarray(off, off + CHUNK));
      if (r) probs_f32.push(r.confidence);
    }

    const probs_i16 = [];
    svad.reset();
    const i16_2 = new Int16Array(i16);
    for (let off = 0; off + CHUNK <= i16_2.length; off += CHUNK) {
      const r = svad.processFrame(i16_2.subarray(off, off + CHUNK));
      if (r) probs_i16.push(r.confidence);
    }

    assert.strictEqual(probs_f32.length, probs_i16.length, "processFrame frame count");
    const delta = maxAbsDelta(probs_f32, probs_i16);
    assert.ok(
      delta < PROB_TOL,
      `StreamVAD processFrame: max prob delta ${delta} >= ${PROB_TOL}`,
    );
    svad.dispose();
    console.log(`StreamVAD processFrame OK — max delta ${delta.toExponential(2)}`);
  }

  // --- Wrong dtype rejection ---
  {
    const svad = await OmniStreamVAD.create();
    let threw = false;
    try {
      // Float64Array is neither Float32Array nor Int16Array
      svad.processFrame(new Float64Array(160));
    } catch (e) {
      threw = true;
      assert.match(e.message, /Float32Array|Int16Array/, "TypeError message");
    }
    assert.ok(threw, "processFrame should reject Float64Array");
    svad.dispose();
    console.log("StreamVAD rejects unsupported dtype OK");
  }

  console.log("\nAll TS wrapper dispatch tests passed!");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
