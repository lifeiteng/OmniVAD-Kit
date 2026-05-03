/**
 * Tests for omnivad mergeChunks (pure-algorithm chunking via WASM).
 *
 * Mirrors native/test/test_chunking.cpp scenario-by-scenario so the C, Python,
 * and TS views of the same algorithm stay bit-identical. If either side's
 * output drifts, all three test files must be updated.
 *
 * Runs as a plain Node script — no test runner. Intended for `node` invocation
 * post-`pnpm build` (which produces dist/index.cjs).
 */
const path = require("path");
const { mergeChunks, DEFAULT_CHUNK_CONFIG } = require(path.join(__dirname, "..", "dist", "index.cjs"));

let failed = 0;

function approxEq(a, b, eps = 1e-4) {
  return Math.abs(a - b) <= eps;
}

function check(label, actual, expected) {
  const passed = expected.every((exp, i) => {
    const got = actual[i];
    if (!got) return false;
    return (
      approxEq(got.start, exp[0]) &&
      approxEq(got.end, exp[1]) &&
      got.segStartIdx === exp[2] &&
      got.segCount === exp[3]
    );
  }) && actual.length === expected.length;

  if (passed) {
    console.log(`  PASS [${label}]`);
  } else {
    console.error(`  FAIL [${label}]`);
    console.error(`    expected: ${JSON.stringify(expected)}`);
    console.error(`    actual:   ${JSON.stringify(actual)}`);
    failed++;
  }
}

(async () => {
  console.log("=== chunking.test ===");

  // -- Scenario 1 ------------------------------------------------------
  check(
    "1: short audio < max_chunk_secs",
    await mergeChunks([[0.0, 5.0], [6.0, 10.0]], { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 }),
    [[0.0, 10.0, 0, 2]],
  );

  // -- Scenario 2 ------------------------------------------------------
  check(
    "2: long audio multiple splits",
    await mergeChunks(
      [[0.0, 10.0], [11.0, 20.0], [21.0, 30.0], [31.0, 40.0]],
      { maxChunkSecs: 20.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 20.0, 0, 2], [21.0, 40.0, 2, 2]],
  );

  // -- Scenario 3 ------------------------------------------------------
  check(
    "3: gap > max_gap_secs force split",
    await mergeChunks(
      [[0.0, 5.0], [8.0, 10.0], [20.0, 25.0]],
      { maxChunkSecs: 30.0, maxGapSecs: 2.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 5.0, 0, 1], [8.0, 10.0, 1, 1], [20.0, 25.0, 2, 1]],
  );

  // -- Scenario 4 ------------------------------------------------------
  check(
    "4: single segment > max_chunk_secs hard split",
    await mergeChunks([[0.0, 100.0]], { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 }),
    [
      [0.0, 30.0, 0, 1],
      [30.0, 60.0, 0, 1],
      [60.0, 90.0, 0, 1],
      [90.0, 100.0, 0, 1],
    ],
  );

  // -- Scenario 5 ------------------------------------------------------
  {
    const out = await mergeChunks([], { maxChunkSecs: 30.0 });
    if (out.length === 0) console.log("  PASS [5: empty input]");
    else { console.error("  FAIL [5: empty input]:", out); failed++; }
  }

  // -- Scenario 6 ------------------------------------------------------
  check(
    "6: min_speech_secs filter",
    await mergeChunks(
      [[0.0, 0.1], [1.0, 5.0]],
      { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSpeechSecs: 0.5, minSilenceSecs: 0 },
    ),
    [[1.0, 5.0, 0, 1]],
  );

  // -- Scenario 7 ------------------------------------------------------
  check(
    "7: min_silence_secs merge",
    await mergeChunks(
      [[0.0, 5.0], [5.1, 10.0]],
      { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0.5 },
    ),
    [[0.0, 10.0, 0, 1]],
  );

  // -- Scenario 8a -----------------------------------------------------
  check(
    "8a: pad applied",
    await mergeChunks([[5.0, 10.0]], { maxChunkSecs: 30.0, padOnsetSecs: 0.5, padOffsetSecs: 0.5, minSilenceSecs: 0 }),
    [[4.5, 10.5, 0, 1]],
  );

  // -- Scenario 8b -----------------------------------------------------
  check(
    "8b: pad_onset_secs clamped to 0",
    await mergeChunks([[0.1, 5.0]], { maxChunkSecs: 30.0, padOnsetSecs: 0.5, padOffsetSecs: 0, minSilenceSecs: 0 }),
    [[0.0, 5.0, 0, 1]],
  );

  // -- Boundary --------------------------------------------------------
  {
    let threw = false;
    try {
      await mergeChunks([[0.0, 5.0]], { maxChunkSecs: 0 });
    } catch {
      threw = true;
    }
    if (threw) console.log("  PASS [B1: max_chunk_secs=0 throws]");
    else { console.error("  FAIL [B1: max_chunk_secs=0 should throw]"); failed++; }
  }

  // -- Defaults --------------------------------------------------------
  {
    const d = DEFAULT_CHUNK_CONFIG;
    const ok = (
      approxEq(d.maxChunkSecs, 30.0) &&
      d.maxGapSecs === Infinity &&
      approxEq(d.padOnsetSecs, 0.04) &&
      approxEq(d.padOffsetSecs, 0.04) &&
      approxEq(d.minSpeechSecs, 0.0) &&
      approxEq(d.minSilenceSecs, 0.20) &&
      d.mode === "greedy"
    );
    if (ok) console.log("  PASS [B2: defaults (incl. mode='greedy')]");
    else { console.error("  FAIL [B2: defaults]:", d); failed++; }
  }

  // -- B3: default-options path uses DEFAULT_CHUNK_CONFIG --------------
  // Calling mergeChunks([...]) with NO options must apply DEFAULT_CHUNK_CONFIG
  // (pad_onset_secs=0.04 etc), NOT zero defaults. This locks the `?? DEFAULT_*`
  // fallback in src/chunking.ts:46-53.
  //
  // Note: this is precisely where the Python convenience function
  // ``merge_chunks()`` DIVERGES — Python uses zero defaults. See
  // tests/test_chunking.py::test_python_convenience_defaults_differ_from_canonical.
  {
    const out = await mergeChunks([[0.0, 5.0], [5.1, 10.0]]);
    // gap 0.1 < min_silence_secs 0.20 -> pre-merged into 1 seg
    // pad_onset_secs=0.04 -> start 0-0.04 clamped to 0; pad_offset_secs=0.04 -> 10.04
    const ok = out.length === 1 &&
      approxEq(out[0].start, 0.0) &&
      approxEq(out[0].end, 10.04) &&
      out[0].segCount === 1;
    if (ok) console.log("  PASS [B3: default options apply DEFAULT_CHUNK_CONFIG]");
    else { console.error("  FAIL [B3]:", JSON.stringify(out)); failed++; }
  }

  // -- B4: ABI struct sizes (locked against C and Python) --------------
  // SIZEOF_CHUNK_CONFIG=24 and SIZEOF_CHUNK=16 are hard-coded in
  // wasm-binding.ts. Drift would silently corrupt the WASM heap layout.
  // Re-export them from the bundle so tests can self-check.
  // (Constants live in wasm-binding.ts; we sanity-check via behaviour
  // because they're not currently re-exported from index.ts.)
  // — instead probe behaviour: a known-shape input MUST produce a known
  // output; if struct layout drifted, fields would be misaligned and
  // segStartIdx/segCount would come back garbage.
  {
    const out = await mergeChunks([[1.5, 7.25], [10.0, 15.5]], {
      maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0,
    });
    // If layout drifted, segCount/segStartIdx would be non-{0,2}.
    const ok = out.length === 1 &&
      approxEq(out[0].start, 1.5) &&
      approxEq(out[0].end, 15.5) &&
      out[0].segStartIdx === 0 &&
      out[0].segCount === 2;
    if (ok) console.log("  PASS [B4: struct field alignment via end-to-end probe]");
    else { console.error("  FAIL [B4]:", JSON.stringify(out)); failed++; }
  }

  // -- NaN / Inf edge cases --------------------------------------------
  {
    let threw = false;
    try { await mergeChunks([[0.0, 5.0]], { maxChunkSecs: NaN }); } catch { threw = true; }
    if (threw) console.log("  PASS [N1: maxChunkSecs=NaN throws]");
    else { console.error("  FAIL [N1: maxChunkSecs=NaN should throw]"); failed++; }
  }
  {
    let threw = false;
    try { await mergeChunks([[0.0, 5.0]], { maxChunkSecs: -1.0 }); } catch { threw = true; }
    if (threw) console.log("  PASS [N2: maxChunkSecs<0 throws]");
    else { console.error("  FAIL [N2: maxChunkSecs<0 should throw]"); failed++; }
  }
  {
    // maxChunkSecs=Infinity > 0 → accepted; everything fits in one chunk.
    const out = await mergeChunks(
      [[0.0, 5.0], [10.0, 20.0]],
      { maxChunkSecs: Infinity, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    );
    const ok = out.length === 1 && approxEq(out[0].start, 0.0) && approxEq(out[0].end, 20.0);
    if (ok) console.log("  PASS [N3: maxChunkSecs=Infinity accepted]");
    else { console.error("  FAIL [N3]:", JSON.stringify(out)); failed++; }
  }

  // -- Mirror new C scenarios 9-19 (algorithm equivalence) -------------
  check(
    "9: Step1 before Step2 (filter then merge)",
    await mergeChunks(
      [[0.0, 5.0], [5.4, 5.5], [5.6, 10.0]],
      { maxChunkSecs: 30.0, maxGapSecs: 0.55, padOnsetSecs: 0, padOffsetSecs: 0, minSpeechSecs: 0.2, minSilenceSecs: 0.5 },
    ),
    [[0.0, 5.0, 0, 1], [5.6, 10.0, 1, 1]],
  );

  check(
    "10: seg_start_idx after filter+merge",
    await mergeChunks(
      [[0.0, 0.1], [1.0, 5.0], [5.1, 10.0], [20.0, 25.0]],
      { maxChunkSecs: 20.0, padOnsetSecs: 0, padOffsetSecs: 0, minSpeechSecs: 0.5, minSilenceSecs: 0.5 },
    ),
    [[1.0, 10.0, 0, 1], [20.0, 25.0, 1, 1]],
  );

  check(
    "11: min_speech_secs drops all → empty",
    await mergeChunks(
      [[0.0, 0.1], [1.0, 1.05]],
      { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSpeechSecs: 1.0, minSilenceSecs: 0 },
    ),
    [],
  );

  check(
    "12: min_silence_secs cascade max(end)",
    await mergeChunks(
      [[0.0, 10.0], [0.1, 5.0], [0.2, 8.0]],
      { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0.5 },
    ),
    [[0.0, 10.0, 0, 1]],
  );

  check(
    "13: min_silence_secs then size split",
    await mergeChunks(
      [[0.0, 5.0], [5.1, 10.0], [15.0, 20.0]],
      { maxChunkSecs: 12.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0.5 },
    ),
    [[0.0, 10.0, 0, 1], [15.0, 20.0, 1, 1]],
  );

  check(
    "15: max_chunk_secs == segment dur (no Step4)",
    await mergeChunks([[0.0, 30.0]], { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 }),
    [[0.0, 30.0, 0, 1]],
  );

  check(
    "16: max_gap_secs == real gap (no split)",
    await mergeChunks(
      [[0.0, 5.0], [7.0, 10.0]],
      { maxChunkSecs: 30.0, maxGapSecs: 2.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 10.0, 0, 2]],
  );

  check(
    "17: min_silence_secs == real gap (no merge)",
    await mergeChunks(
      [[0.0, 5.0], [5.5, 10.0]],
      { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0.5 },
    ),
    [[0.0, 10.0, 0, 2]],
  );

  check(
    "18: min_speech_secs == segment dur (kept)",
    await mergeChunks(
      [[0.0, 0.5], [1.0, 5.0]],
      { maxChunkSecs: 30.0, padOnsetSecs: 0, padOffsetSecs: 0, minSpeechSecs: 0.5, minSilenceSecs: 0 },
    ),
    [[0.0, 5.0, 0, 2]],
  );

  check(
    "8c: pad allows chunk overlap",
    await mergeChunks(
      [[0.0, 5.0], [6.0, 10.0]],
      { maxChunkSecs: 30.0, maxGapSecs: 0.5, padOnsetSecs: 2.0, padOffsetSecs: 2.0, minSilenceSecs: 0 },
    ),
    [[0.0, 7.0, 0, 1], [4.0, 12.0, 1, 1]],
  );

  // ====================================================================
  //  LONGEST_GAP mode (mode: "longest_gap") — mirrors C scenarios LG1-LG10
  // ====================================================================

  check(
    "LG1: total fits, single chunk",
    await mergeChunks(
      [[0.0, 5.0], [6.0, 10.0]],
      { maxChunkSecs: 30.0, mode: "longest_gap", padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 10.0, 0, 2]],
  );

  check(
    "LG2: simple cut at longest gap",
    await mergeChunks(
      [[0.0, 5.0], [8.0, 10.0], [20.0, 25.0]],
      { maxChunkSecs: 20.0, mode: "longest_gap", padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 10.0, 0, 2], [20.0, 25.0, 2, 1]],
  );

  check(
    "LG3: recursive splits",
    await mergeChunks(
      [[0.0, 5.0], [7.0, 10.0], [20.0, 25.0], [40.0, 50.0]],
      { maxChunkSecs: 15.0, mode: "longest_gap", padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 10.0, 0, 2], [20.0, 25.0, 2, 1], [40.0, 50.0, 3, 1]],
  );

  check(
    "LG4: single seg > max_chunk_secs hard-split fallback",
    await mergeChunks(
      [[0.0, 100.0]],
      { maxChunkSecs: 30.0, mode: "longest_gap", padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [
      [0.0, 30.0, 0, 1],
      [30.0, 60.0, 0, 1],
      [60.0, 90.0, 0, 1],
      [90.0, 100.0, 0, 1],
    ],
  );

  check(
    "LG5: tie-break leftmost gap",
    await mergeChunks(
      [[0.0, 5.0], [10.0, 15.0], [20.0, 25.0]],
      { maxChunkSecs: 10.0, mode: "longest_gap", padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 5.0, 0, 1], [10.0, 15.0, 1, 1], [20.0, 25.0, 2, 1]],
  );

  check(
    "LG6: max_gap_secs honored in LONGEST_GAP",
    await mergeChunks(
      [[0.0, 5.0], [6.0, 10.0]],
      { maxChunkSecs: 30.0, mode: "longest_gap", maxGapSecs: 0.1, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 5.0, 0, 1], [6.0, 10.0, 1, 1]],
  );

  check(
    "LG6b: max_gap_secs forces split inside fitting span",
    await mergeChunks(
      [[0.0, 5.0], [8.0, 10.0], [15.0, 25.0]],
      { maxChunkSecs: 30.0, mode: "longest_gap", maxGapSecs: 4.0, padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 10.0, 0, 2], [15.0, 25.0, 2, 1]],
  );

  check(
    "LG7: filter+merge then longest-gap split",
    await mergeChunks(
      [[0.0, 0.1], [1.0, 5.0], [5.1, 10.0], [20.0, 30.0]],
      {
        maxChunkSecs: 15.0, mode: "longest_gap",
        padOnsetSecs: 0, padOffsetSecs: 0, minSpeechSecs: 0.5, minSilenceSecs: 0.5,
      },
    ),
    [[1.0, 10.0, 0, 1], [20.0, 30.0, 1, 1]],
  );

  check(
    "LG8: pad applied",
    await mergeChunks(
      [[0.0, 5.0], [8.0, 15.0]],
      { maxChunkSecs: 10.0, mode: "longest_gap", padOnsetSecs: 0.5, padOffsetSecs: 0.5, minSilenceSecs: 0 },
    ),
    [[0.0, 5.5, 0, 1], [7.5, 15.5, 1, 1]],
  );

  {
    const out = await mergeChunks([], { maxChunkSecs: 30.0, mode: "longest_gap" });
    if (out.length === 0) console.log("  PASS [LG9: empty input]");
    else { console.error("  FAIL [LG9]:", out); failed++; }
  }

  check(
    "LG10: single seg fits, no hard-split",
    await mergeChunks(
      [[0.0, 30.0]],
      { maxChunkSecs: 30.0, mode: "longest_gap", padOnsetSecs: 0, padOffsetSecs: 0, minSilenceSecs: 0 },
    ),
    [[0.0, 30.0, 0, 1]],
  );

  // -- Mode validation: invalid mode string throws ---------------------
  {
    let threw = false;
    try {
      await mergeChunks([[0.0, 5.0]], { maxChunkSecs: 30.0, mode: "invalid" });
    } catch { threw = true; }
    if (threw) console.log("  PASS [LG-V1: unknown mode throws]");
    else { console.error("  FAIL [LG-V1: unknown mode should throw]"); failed++; }
  }

  if (failed > 0) {
    console.error(`\n=== ${failed} failure(s) ===`);
    process.exit(1);
  }
  console.log("\n=== all tests passed ===");
})().catch((e) => {
  console.error("TEST CRASHED:", e);
  process.exit(1);
});
