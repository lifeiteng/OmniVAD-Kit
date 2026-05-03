/**
 * Tests for OmniStreamSegmenter — TS wrapper around the C streaming segmenter.
 *
 * Mirrors native/test/test_stream_segmenter.cpp and tests/test_stream_segmenter.py
 * scenario-by-scenario so the C, Python, and TS views stay bit-identical.
 *
 * Run as plain Node script: requires `pnpm build` first to produce dist/index.cjs.
 */
const path = require("path");
const { OmniStreamSegmenter } = require(path.join(__dirname, "..", "dist", "index.cjs"));

let failed = 0;

function approxEq(a, b, eps = 1e-4) { return Math.abs(a - b) <= eps; }

function check(label, cond, msg) {
  if (cond) console.log(`  PASS [${label}]`);
  else { console.error(`  FAIL [${label}]: ${msg ?? ""}`); failed++; }
}

function makeProbs(runs) {
  const out = [];
  for (const [n, v] of runs) for (let i = 0; i < n; i++) out.push(v);
  return new Float32Array(out);
}

(async () => {
  console.log("=== stream-segmenter.test ===");

  // --- Config + lifecycle -----------------------------------------------
  {
    const seg = await OmniStreamSegmenter.create();
    check("B1: create(default)", !seg.isInSpeech && seg.activeStart === null);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    seg.close();
    seg.close();   // idempotent
    let threw = false;
    try { seg.processFrame(0.5); } catch { threw = true; }
    check("B2: use after close throws", threw);
  }

  // --- Algorithm correctness --------------------------------------------
  {
    const seg = await OmniStreamSegmenter.create();
    const out = seg.processProbs(new Float32Array(100));
    check("T1: all-silence -> 0", out.length === 0);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    const out = seg.processProbs(new Float32Array(100).fill(1));
    check("T2: all-speech -> 0 emit", out.length === 0);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    const probs = makeProbs([[5, 0], [10, 1], [50, 0]]);
    const out = seg.processProbs(probs);
    check("T3: short pulse rejected", out.length === 0);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    const probs = makeProbs([[5, 0], [30, 1], [25, 0]]);
    const out = seg.processProbs(probs);
    check("T4: one clean segment count", out.length === 1);
    if (out.length === 1) {
      check("T4: one clean segment start",
        approxEq(out[0].start, 0.01),
        `start=${out[0].start}`);
      check("T4: one clean segment end",
        approxEq(out[0].end, 0.38),
        `end=${out[0].end}`);
    }
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    const probs = makeProbs([[5, 0], [30, 1], [35, 0], [30, 1], [25, 0]]);
    const out = seg.processProbs(probs);
    check("T5: two segments", out.length === 2);
    seg.close();
  }

  // --- Chunk-size invariance --------------------------------------------
  {
    const probs = makeProbs([
      [5, 0], [30, 1], [35, 0],
      [30, 1], [25, 0], [25, 1], [30, 0],
    ]);

    const segA = await OmniStreamSegmenter.create();
    const a = segA.processProbs(probs);
    segA.close();

    const segB = await OmniStreamSegmenter.create();
    const b = [];
    for (const p of probs) for (const s of segB.processFrame(p)) b.push(s);
    segB.close();

    const segC = await OmniStreamSegmenter.create();
    const c = [];
    const sizes = [1, 7, 3, 13, 1, 50, 100];
    let idx = 0, k = 0;
    while (idx < probs.length) {
      const n = Math.min(sizes[k % sizes.length], probs.length - idx);
      for (const s of segC.processProbs(probs.slice(idx, idx + n))) c.push(s);
      idx += n; k++;
    }
    segC.close();

    let ok = a.length === b.length && b.length === c.length;
    for (let i = 0; ok && i < a.length; i++) {
      ok = ok && approxEq(a[i].start, b[i].start) && approxEq(a[i].end, b[i].end);
      ok = ok && approxEq(a[i].start, c[i].start) && approxEq(a[i].end, c[i].end);
    }
    check("T6: chunk-size invariance (3 paths)", ok,
      `a=${JSON.stringify(a)} b=${JSON.stringify(b)} c=${JSON.stringify(c)}`);
  }

  // --- Force-split -------------------------------------------------------
  {
    const seg = await OmniStreamSegmenter.create({ maxChunkSecs: 0.20 });
    const out = seg.processProbs(new Float32Array(50).fill(1));
    check("T9: force-split count", out.length === 3, `got ${out.length}`);
    if (out.length === 3) {
      check("T9: split[0]", approxEq(out[0].start, 0.00) && approxEq(out[0].end, 0.10));
      check("T9: split[1]", approxEq(out[1].start, 0.11) && approxEq(out[1].end, 0.21));
      check("T9: split[2]", approxEq(out[2].start, 0.22) && approxEq(out[2].end, 0.32));
    }
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create({
      smoothWindowSize: 1, minSpeechSecs: 0.01, maxChunkSecs: 0.20,
    });
    const probs = new Float32Array(25).fill(1);
    probs[15] = 0.5;
    const out = seg.processProbs(probs);
    check("T10: split picks min-prob frame",
      out.length === 1 && approxEq(out[0].end, 0.15),
      `out=${JSON.stringify(out)}`);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create({ maxChunkSecs: 0 });
    const out = seg.processProbs(new Float32Array(1000).fill(1));
    check("T11: max=0 disables split", out.length === 0);
    seg.close();
  }

  // --- Flush -------------------------------------------------------------
  {
    const seg = await OmniStreamSegmenter.create();
    seg.processProbs(new Float32Array(50));
    check("T_flush1: silence-only -> 0", seg.flush(0).length === 0);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    seg.processProbs(new Float32Array(15).fill(1));
    check("T_flush2: unconfirmed candidate -> 0", seg.flush(0).length === 0);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    seg.processProbs(new Float32Array(100).fill(1));
    const out = seg.flush(0);
    check("T_flush3: SPEECH -> trailing tail",
      out.length === 1 && approxEq(out[0].start, 0.000) && approxEq(out[0].end, 1.025),
      `out=${JSON.stringify(out)}`);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    seg.processProbs(new Float32Array(100).fill(1));
    const out = seg.flush(16000);   // wav_dur = 1.0s
    check("T_flush4: clamps to wav_dur",
      out.length === 1 && approxEq(out[0].end, 1.000),
      `out=${JSON.stringify(out)}`);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create({ maxChunkSecs: 0.20 });
    const pre = seg.processProbs(new Float32Array(50).fill(1));
    const tail = seg.flush(0);
    check("T_flush6: tail after force-split",
      pre.length === 3 && tail.length === 1 &&
      approxEq(tail[0].start, 0.33) && approxEq(tail[0].end, 0.525),
      `pre=${pre.length} tail=${JSON.stringify(tail)}`);
    seg.close();
  }
  {
    const seg = await OmniStreamSegmenter.create();
    seg.processProbs(new Float32Array(100).fill(1));
    seg.flush(0);
    check("T_flush7: flush twice idempotent", seg.flush(0).length === 0);
    seg.close();
  }

  // --- State queries -----------------------------------------------------
  {
    const seg = await OmniStreamSegmenter.create();
    check("T8a: not in speech initially", !seg.isInSpeech);
    seg.processProbs(new Float32Array(40).fill(1));
    check("T8b: in speech after long burst", seg.isInSpeech);
    check("T8c: active_start >= 0", typeof seg.activeStart === "number" && seg.activeStart >= 0);
    seg.reset();
    check("T8d: not in speech after reset", !seg.isInSpeech && seg.activeStart === null);
    seg.close();
  }

  if (failed > 0) {
    console.error(`\n=== ${failed} failure(s) ===`);
    process.exit(1);
  }
  console.log("\n=== all tests passed ===");
})().catch((e) => { console.error(e); process.exit(1); });
