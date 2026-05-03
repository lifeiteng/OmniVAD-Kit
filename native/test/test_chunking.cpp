/*
 * Test: omni_merge_chunks (pure-algorithm chunking).
 *
 * Hard-coded scenarios shared with Python tests/test_chunking.py and TS
 * packages/omnivad/tests/chunking.test.cjs — three-language cross-check
 * that all bindings produce identical output.
 *
 * Usage: ./test_chunking            (no arguments — no model files needed)
 */

#include "omnivad.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_failed = 0;

static bool floats_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

#define EXPECT_EQ_INT(actual, expected, msg) \
    do { \
        if ((actual) != (expected)) { \
            std::fprintf(stderr, "  FAIL [%s] %s: expected %d, got %d (line %d)\n", \
                         scenario_name, msg, (int)(expected), (int)(actual), __LINE__); \
            ++g_failed; \
            return; \
        } \
    } while (0)

#define EXPECT_EQ_FLOAT(actual, expected, msg) \
    do { \
        if (!floats_equal((actual), (expected))) { \
            std::fprintf(stderr, "  FAIL [%s] %s: expected %.4f, got %.4f (line %d)\n", \
                         scenario_name, msg, (float)(expected), (float)(actual), __LINE__); \
            ++g_failed; \
            return; \
        } \
    } while (0)

#define EXPECT_RET(actual, expected, msg) \
    do { \
        if ((actual) != (expected)) { \
            std::fprintf(stderr, "  FAIL [%s] %s: expected ret %d, got %d (line %d)\n", \
                         scenario_name, msg, (int)(expected), (int)(actual), __LINE__); \
            ++g_failed; \
            return; \
        } \
    } while (0)

struct ExpectedChunk {
    float start;
    float end;
    int   seg_start_idx;
    int   seg_count;
};

static void run_case(
    const char* scenario_name,
    const std::vector<OmniSegment>& input,
    const OmniChunkConfig& cfg,
    const std::vector<ExpectedChunk>& expected
) {
    OmniChunk* out = nullptr;
    int count = 0;
    int rc = omni_merge_chunks(
        input.empty() ? nullptr : input.data(),
        (int)input.size(),
        &cfg,
        &out, &count
    );
    EXPECT_RET(rc, OMNI_OK, "return code");
    EXPECT_EQ_INT(count, (int)expected.size(), "chunk count");

    for (int i = 0; i < count; ++i) {
        char field_msg[64];
        std::snprintf(field_msg, sizeof(field_msg), "chunk[%d].start", i);
        EXPECT_EQ_FLOAT(out[i].start, expected[i].start, field_msg);
        std::snprintf(field_msg, sizeof(field_msg), "chunk[%d].end", i);
        EXPECT_EQ_FLOAT(out[i].end, expected[i].end, field_msg);
        std::snprintf(field_msg, sizeof(field_msg), "chunk[%d].seg_start_idx", i);
        EXPECT_EQ_INT(out[i].seg_start_idx, expected[i].seg_start_idx, field_msg);
        std::snprintf(field_msg, sizeof(field_msg), "chunk[%d].seg_count", i);
        EXPECT_EQ_INT(out[i].seg_count, expected[i].seg_count, field_msg);
    }

    omni_free(out);
    std::fprintf(stdout, "  PASS [%s]\n", scenario_name);
}

static OmniChunkConfig zero_pad_cfg(float max_chunk_secs, float max_gap_secs = INFINITY) {
    OmniChunkConfig cfg = omni_chunk_config_default();
    cfg.max_chunk_secs   = max_chunk_secs;
    cfg.max_gap_secs     = max_gap_secs;
    cfg.pad_onset_secs   = 0.0f;
    cfg.pad_offset_secs  = 0.0f;
    cfg.min_speech_secs  = 0.0f;
    cfg.min_silence_secs = 0.0f;
    cfg.mode             = OMNI_CHUNK_GREEDY;
    return cfg;
}

static OmniChunkConfig longest_gap_cfg(float max_chunk_secs) {
    OmniChunkConfig cfg = zero_pad_cfg(max_chunk_secs);
    cfg.mode = OMNI_CHUNK_LONGEST_GAP;
    return cfg;
}

int main(int /*argc*/, char** /*argv*/) {
    std::fprintf(stdout, "=== test_chunking ===\n");

    // -- Scenario 1: short audio fits in one chunk -----------------------
    {
        const char* scenario_name = "1: short audio < max_chunk_secs";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {6.0f, 10.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {{0.0f, 10.0f, 0, 2}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 2: total exceeds max_chunk_secs -> multiple chunks --------
    //
    // Trace with max_chunk_secs=20:
    //   accept (0,10)             -> cur=(0,10), seg_count=1
    //   accept (11,20) gap=1 OK   -> cur=(0,20), seg_count=2 (would_exceed: 20-0=20 not >20)
    //   reject (21,30) would 30>20 -> emit (0,20,0,2); cur=(21,30), seg_count=1
    //   accept (31,40) gap=1 OK   -> cur=(21,40), seg_count=2 (40-21=19 not >20)
    //   final emit (21,40,2,2)
    {
        const char* scenario_name = "2: long audio multiple splits";
        std::vector<OmniSegment> input = {
            {0.0f, 10.0f}, {11.0f, 20.0f}, {21.0f, 30.0f}, {31.0f, 40.0f}
        };
        OmniChunkConfig cfg = zero_pad_cfg(20.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 20.0f, 0, 2},
            {21.0f, 40.0f, 2, 2},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 3: gap > max_gap_secs forces split between every pair ------
    {
        const char* scenario_name = "3: gap > max_gap_secs force split";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {8.0f, 10.0f}, {20.0f, 25.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f, /*max_gap_secs=*/2.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 5.0f, 0, 1},
            {8.0f, 10.0f, 1, 1},
            {20.0f, 25.0f, 2, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 4: single segment > max_chunk_secs -> equal hard-split ----
    {
        const char* scenario_name = "4: single segment > max_chunk_secs";
        std::vector<OmniSegment> input = {{0.0f, 100.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 30.0f, 0, 1},
            {30.0f, 60.0f, 0, 1},
            {60.0f, 90.0f, 0, 1},
            {90.0f, 100.0f, 0, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 5: empty input -> zero chunks --------------------------
    {
        const char* scenario_name = "5: empty input";
        std::vector<OmniSegment> input = {};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 6: min_speech_secs filters short segments --------------
    //
    // (0, 0.1) has dur=0.1 < 0.5 -> dropped
    // (1, 5) has dur=4 >= 0.5    -> kept
    // active = [(1,5)] -> 1 chunk; seg_start_idx=0 refers to the *active*
    // (post-filter) array, not the original input.
    {
        const char* scenario_name = "6: min_speech_secs drops short";
        std::vector<OmniSegment> input = {{0.0f, 0.1f}, {1.0f, 5.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.min_speech_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {{1.0f, 5.0f, 0, 1}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 7: min_silence_secs merges adjacent --------------------
    //
    // gap (0,5)->(5.1,10) is 0.1 < 0.5 -> merged into (0,10)
    // active = [(0,10)] (single segment) -> 1 chunk seg_count=1
    {
        const char* scenario_name = "7: min_silence_secs merges close";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {5.1f, 10.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.min_silence_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {{0.0f, 10.0f, 0, 1}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 8: pad_onset_secs / pad_offset_secs applied correctly -----------
    //
    // (5, 10) with pad=0.5 -> chunk (4.5, 10.5)
    // (0.1, 5) with pad_onset_secs=0.5 -> start clamped to 0 (not -0.4)
    {
        const char* scenario_name = "8a: pad applied";
        std::vector<OmniSegment> input = {{5.0f, 10.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.pad_onset_secs = 0.5f;
        cfg.pad_offset_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {{4.5f, 10.5f, 0, 1}};
        run_case(scenario_name, input, cfg, expected);
    }
    {
        const char* scenario_name = "8b: pad_onset_secs clamped to 0";
        std::vector<OmniSegment> input = {{0.1f, 5.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.pad_onset_secs = 0.5f;
        cfg.pad_offset_secs = 0.0f;
        std::vector<ExpectedChunk> expected = {{0.0f, 5.0f, 0, 1}};
        run_case(scenario_name, input, cfg, expected);
    }
    {
        // Adjacent chunks may overlap after pad — algorithm does not de-dupe.
        // Locks current behaviour: gap=1 forces split (max_gap_secs=0.5),
        // pad=2 extends chunk1.end=7 past chunk2.start=4 → overlap.
        const char* scenario_name = "8c: pad allows chunk overlap";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {6.0f, 10.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f, /*max_gap_secs=*/0.5f);
        cfg.pad_onset_secs = 2.0f;
        cfg.pad_offset_secs = 2.0f;
        std::vector<ExpectedChunk> expected = {
            {0.0f, 7.0f, 0, 1},   // chunk1: pad_onset_secs clamps to 0; end = 5+2 = 7
            {4.0f, 12.0f, 1, 1},  // chunk2: 6-2=4, 10+2=12. overlaps with chunk1 [4,7]
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 9: Step1→Step2 ordering --------------------------------
    //
    // P0: filter (Step1) MUST run before merge (Step2). Construction:
    //   inputs (0,5),(5.4,5.5 short),(5.6,10), min_on=0.2, min_off=0.5
    //   max_gap_secs=0.55 (so Step3 splits gap > 0.55).
    //
    // Correct order (Step1 then Step2):
    //   Step1 drops (5.4,5.5) -> active=[(0,5),(5.6,10)] gap=0.6
    //   Step2: 0.6 > 0.5 -> NOT merged. active unchanged.
    //   Step3: max_gap_secs=0.55, gap=0.6 > 0.55 -> SPLIT.
    //   Output: [(0,5,0,1), (5.6,10,1,1)]  ← what we expect.
    //
    // Reversed order (Step2 then Step1):
    //   Step2: gap (0,5)→(5.4,5.5)=0.4<0.5 merge to (0,5.5)
    //          then gap (0,5.5)→(5.6,10)=0.1<0.5 merge to (0,10)
    //   Step1: dur=10>0.2 keep -> active=[(0,10)]
    //   Step3: 1 chunk [(0,10)]  ← would FAIL this test.
    {
        const char* scenario_name = "9: Step1 before Step2 (filter then merge)";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {5.4f, 5.5f}, {5.6f, 10.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f, /*max_gap_secs=*/0.55f);
        cfg.min_speech_secs  = 0.2f;
        cfg.min_silence_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {
            {0.0f, 5.0f, 0, 1},
            {5.6f, 10.0f, 1, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 10: post-filter/post-merge view of seg_start_idx -------
    //
    // P0: confirm seg_start_idx counts on POST-filter+merge view.
    // Inputs (0,0.1),(1,5),(5.1,10),(20,25) max_chunk_secs=20
    //   Step1 min_on=0.5: drop (0,0.1) -> active=[(1,5),(5.1,10),(20,25)]
    //   Step2 min_off=0.5: merge (1,5)+(5.1,10) -> [(1,10),(20,25)]
    //   Step3: max_chunk_secs=20, gap (1,10)→(20,25)=10, both fit ->
    //          (1,10,0,1)+(10..25?). Trace:
    //          accept (1,10) cur=(1,10)
    //          (20,25): would_exceed: 25-1=24 >20 split -> emit (1,10,0,1)
    //          cur=(20,25,1,1) emit -> [(1,10,0,1),(20,25,1,1)]
    {
        const char* scenario_name = "10: seg_start_idx after filter+merge";
        std::vector<OmniSegment> input = {{0.0f, 0.1f}, {1.0f, 5.0f}, {5.1f, 10.0f}, {20.0f, 25.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(20.0f);
        cfg.min_speech_secs  = 0.5f;
        cfg.min_silence_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {
            {1.0f, 10.0f, 0, 1},
            {20.0f, 25.0f, 1, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 11: min_speech_secs drops everything -> empty ---------
    {
        const char* scenario_name = "11: min_speech_secs drops all";
        std::vector<OmniSegment> input = {{0.0f, 0.1f}, {1.0f, 1.05f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.min_speech_secs = 1.0f;
        std::vector<ExpectedChunk> expected = {};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 12: min_silence_secs cascade merge w/ max(end) --------
    //
    // (0,10),(0.1,5),(0.2,8): gap1=-9.9<0.5 merge to (0,10) [max(10,5)=10],
    // gap2=(0.2-10)=-9.8<0.5 merge to (0,10) [max(10,8)=10]. seg_count=1.
    // Locks the max(end) branch on chunking.cpp:86-87.
    {
        const char* scenario_name = "12: min_silence_secs cascade max(end)";
        std::vector<OmniSegment> input = {{0.0f, 10.0f}, {0.1f, 5.0f}, {0.2f, 8.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.min_silence_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {{0.0f, 10.0f, 0, 1}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 13: min_silence_secs then size split ------------------
    //
    // After Step2 merges, total exceeds max_chunk_secs -> Step3 splits.
    // (0,5),(5.1,10),(15,20) min_off=0.5 max_chunk_secs=12:
    //   Step2: merge (0,5)+(5.1,10) -> active=[(0,10),(15,20)]
    //   Step3: cur=(0,10); (15,20) would_exceed 20-0=20>12 split ->
    //          emit (0,10,0,1) cur=(15,20,1,1) emit
    {
        const char* scenario_name = "13: min_silence_secs then size split";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {5.1f, 10.0f}, {15.0f, 20.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(12.0f);
        cfg.min_silence_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {
            {0.0f, 10.0f, 0, 1},
            {15.0f, 20.0f, 1, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 14: Step 4 hard-split with multiple input segments ----
    //
    // After Step2 merges (0,40)+(40.1,80)+(80.1,120) into one (0,120),
    // max_chunk_secs=50 triggers Step 4. Sub-chunks span multiple post-merge
    // segments -> seg_count must reflect overlap count.
    //
    // Wait — Step2 merges into a single OmniSegment in `active`, so
    // active=[(0,120)]. Step4 splits into sub-chunks; each sub overlaps
    // 1 segment. To exercise multi-segment Step4, need: chunk that goes
    // through Step3 with multiple segments AND still > max_chunk_secs.
    //
    // Set max_gap_secs=INF, max_chunk_secs=50, segments tightly packed but each
    // longer than max_chunk_secs won't help (cur_has_content split). Better:
    // pack via min_silence_secs then keep individual segments.
    // Actually min_silence_secs MERGES into single segments, so Step4
    // can never see multi-segment chunks unless Step3 packed them and
    // they exceed max_chunk_secs — but Step3 won't pack past max_chunk_secs.
    //
    // The only way Step4 sees multiple segments is when ONE input
    // segment alone exceeds max_chunk_secs — which is single-segment case.
    // So multi-segment Step4 is unreachable. Document this.
    {
        // Sub-chunk with no overlapping segment: when max_chunk_secs < segment
        // duration AND segment starts mid-chunk, an early sub-chunk could
        // theoretically have no overlap. But Step4 sub-chunks span [s, e)
        // strictly inside [c.start, c.end] which is bounded by segment
        // edges, so every sub-chunk overlaps at least one segment.
        // sub_start<0 guard is defensive; lock its behaviour:
        // Single (0,100) max_chunk_secs=33: subs (0,33),(33,66),(66,99),(99,100)
        // all overlap segment 0.
        const char* scenario_name = "14: Step4 hard-split partial-overlap accounting";
        std::vector<OmniSegment> input = {{0.0f, 100.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(33.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 33.0f, 0, 1},
            {33.0f, 66.0f, 0, 1},
            {66.0f, 99.0f, 0, 1},
            {99.0f, 100.0f, 0, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 15: max_chunk_secs exactly equals segment duration --------
    //
    // (0,30) max_chunk_secs=30: dur==max_chunk_secs, NOT > max_chunk_secs -> Step4 skipped.
    // Locks `<=` boundary on chunking.cpp:145.
    {
        const char* scenario_name = "15: max_chunk_secs == segment duration";
        std::vector<OmniSegment> input = {{0.0f, 30.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {{0.0f, 30.0f, 0, 1}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 16: max_gap_secs exactly equals real gap -------------------
    //
    // Two segments with gap=2, max_gap_secs=2: gap > max_gap_secs is FALSE (2>2 false)
    // -> NOT split by gap. Locks `>` strict boundary on chunking.cpp:119.
    {
        const char* scenario_name = "16: max_gap_secs == real gap (no split)";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {7.0f, 10.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f, /*max_gap_secs=*/2.0f);
        std::vector<ExpectedChunk> expected = {{0.0f, 10.0f, 0, 2}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 17: min_silence_secs exactly equals real gap ----------
    //
    // gap=0.5, min_off=0.5: gap < min_off is FALSE (0.5<0.5 false) -> NOT merged.
    // Locks `<` strict boundary on chunking.cpp:85.
    {
        const char* scenario_name = "17: min_silence_secs == real gap (no merge)";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {5.5f, 10.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.min_silence_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {{0.0f, 10.0f, 0, 2}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 18: min_speech_secs exactly equals segment duration ---
    //
    // dur=0.5, min_on=0.5: dur >= min_on TRUE -> KEPT.
    // Locks `>=` boundary on chunking.cpp:64.
    {
        const char* scenario_name = "18: min_speech_secs == segment dur (kept)";
        std::vector<OmniSegment> input = {{0.0f, 0.5f}, {1.0f, 5.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(30.0f);
        cfg.min_speech_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {{0.0f, 5.0f, 0, 2}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Scenario 19: cur_has_content guard with zero-duration first seg
    //
    // First segment is zero-duration (start==end). cur_has_content is FALSE.
    // Subsequent (would_exceed) condition is suppressed -> no split until
    // cur has real content. Locks chunking.cpp:118.
    // (0,0),(1,40) max_chunk_secs=20: cur=(0,0)→accept (1,40) (no split because
    // cur empty), cur=(0,40), seg_count=2. Step4 splits because dur=40>20.
    {
        const char* scenario_name = "19: zero-duration first seg, cur_has_content guard";
        std::vector<OmniSegment> input = {{0.0f, 0.0f}, {1.0f, 40.0f}};
        OmniChunkConfig cfg = zero_pad_cfg(20.0f);
        // After Step3: cur=(0, 40), seg_count=2. Step4 splits:
        //   sub (0,20): overlaps seg 0 (0,0): start<20 && end>0? end=0 not>0 → no.
        //               overlaps seg 1 (1,40): 1<20 && 40>0 → yes. sub_start=1, count=1
        //   sub (20,40): overlaps seg 1 only. sub_start=1, count=1
        std::vector<ExpectedChunk> expected = {
            {0.0f, 20.0f, 1, 1},
            {20.0f, 40.0f, 1, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // ====================================================================
    //  LONGEST_GAP mode (mode = OMNI_CHUNK_LONGEST_GAP)
    // ====================================================================

    // -- LG1: total span <= max_chunk_secs -> single chunk (no recursion) ---
    {
        const char* scenario_name = "LG1: total fits, single chunk";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {6.0f, 10.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {{0.0f, 10.0f, 0, 2}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG2: simple two-way split at the longest gap -------------------
    //
    // Trace with max_chunk_secs=20:
    //   span=25-0=25 > 20. gaps: (8-5)=3, (20-10)=10. Max at i=1.
    //   Split → [(0,5),(8,10)] | [(20,25)]
    //   Left span=10 ≤ 20 → emit (0,10,0,2)
    //   Right span=5  ≤ 20 → emit (20,25,2,1)
    {
        const char* scenario_name = "LG2: simple cut at longest gap";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {8.0f, 10.0f}, {20.0f, 25.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(20.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 10.0f, 0, 2},
            {20.0f, 25.0f, 2, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG3: recursive splits ------------------------------------------
    //
    // max_chunk_secs=15, input=[(0,5),(7,10),(20,25),(40,50)]
    //   span=50, max gap (40-25)=15 at i=2 → [(0,5),(7,10),(20,25)] | [(40,50)]
    //   Left span=25 > 15. gaps: 2, 10 → max at i=1. Split → [(0,5),(7,10)] | [(20,25)]
    //     ll span=10 ≤ 15 → (0,10,0,2);  lr span=5 ≤ 15 → (20,25,2,1)
    //   Right span=10 ≤ 15 → (40,50,3,1)
    {
        const char* scenario_name = "LG3: recursive splits";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {7.0f, 10.0f}, {20.0f, 25.0f}, {40.0f, 50.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(15.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 10.0f, 0, 2},
            {20.0f, 25.0f, 2, 1},
            {40.0f, 50.0f, 3, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG4: single segment > max_chunk_secs -> hard-split fallback --------
    //
    // n==1 stop condition emits single chunk; Step 4 then equal-splits.
    // Same expected output as GREEDY Scenario 4.
    {
        const char* scenario_name = "LG4: single seg > max_chunk_secs hard-split";
        std::vector<OmniSegment> input = {{0.0f, 100.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 30.0f, 0, 1},
            {30.0f, 60.0f, 0, 1},
            {60.0f, 90.0f, 0, 1},
            {90.0f, 100.0f, 0, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG5: tie-break — leftmost gap wins -----------------------------
    //
    // max_chunk_secs=10, two equal gaps of 5: (10-5) and (20-15).
    //   span=25 > 10. Max gap=5, FIRST occurrence wins → cut at i=0.
    //   → [(0,5)] | [(10,15),(20,25)]
    //   Left span=5 → (0,5,0,1)
    //   Right span=15 > 10. gap (20-15)=5 → cut at i=1 (relative). → [(10,15)] | [(20,25)]
    //   → (10,15,1,1), (20,25,2,1)
    {
        const char* scenario_name = "LG5: tie-break leftmost";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {10.0f, 15.0f}, {20.0f, 25.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(10.0f);
        std::vector<ExpectedChunk> expected = {
            {0.0f, 5.0f, 0, 1},
            {10.0f, 15.0f, 1, 1},
            {20.0f, 25.0f, 2, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG6: max_gap_secs is HONORED in longest_gap mode --------------------
    //
    // Both modes treat max_gap_secs as a hard split boundary. Even though span
    // fits max_chunk_secs, gap=1.0 > max_gap_secs=0.1 forces a split at the gap.
    // longest_gap cuts at the longest gap (here only one) → 2 chunks.
    {
        const char* scenario_name = "LG6: max_gap_secs honored in LONGEST_GAP";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {6.0f, 10.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(30.0f);
        cfg.max_gap_secs = 0.1f;  // gap=1.0 > 0.1 → must split
        std::vector<ExpectedChunk> expected = {
            {0.0f, 5.0f, 0, 1},
            {6.0f, 10.0f, 1, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }
    // -- LG6b: max_gap_secs respected together with max_chunk_secs ---------------
    //
    // 3 segs span 25, max_chunk_secs=30 fits, but max_gap_secs=4 splits at (8,15)
    // gap=7. After splitting at i=1, left span=10 fits → 1 chunk;
    // right has 1 seg → 1 chunk. Result: 2 chunks total.
    {
        const char* scenario_name = "LG6b: max_gap_secs forces split inside fitting span";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {8.0f, 10.0f}, {15.0f, 25.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(30.0f);
        cfg.max_gap_secs = 4.0f;  // gaps: 3 OK, 5 > 4 → split there
        std::vector<ExpectedChunk> expected = {
            {0.0f, 10.0f, 0, 2},
            {15.0f, 25.0f, 2, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG7: filter+merge then longest-gap split -----------------------
    //
    // Step 1 drops (0, 0.1); Step 2 merges (1,5)+(5.1,10)→(1,10). Then
    // active=[(1,10),(20,30)] LONGEST_GAP with max_chunk_secs=15:
    //   span=29 > 15, only one gap=10 → split → (1,10,0,1), (20,30,1,1).
    // seg_start_idx counts on post-filter+merge view.
    {
        const char* scenario_name = "LG7: filter+merge then split";
        std::vector<OmniSegment> input = {{0.0f, 0.1f}, {1.0f, 5.0f}, {5.1f, 10.0f}, {20.0f, 30.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(15.0f);
        cfg.min_speech_secs  = 0.5f;
        cfg.min_silence_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {
            {1.0f, 10.0f, 0, 1},
            {20.0f, 30.0f, 1, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG8: padding applied to longest-gap chunks ---------------------
    //
    // max_chunk_secs=10, input=[(0,5),(8,15)]:
    //   span=15 > 10, gap=3 → split → [(0,5)] | [(8,15)]
    //   With pad=0.5: (0-0.5 clamp 0, 5+0.5=5.5) and (8-0.5=7.5, 15+0.5=15.5).
    {
        const char* scenario_name = "LG8: pad applied";
        std::vector<OmniSegment> input = {{0.0f, 5.0f}, {8.0f, 15.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(10.0f);
        cfg.pad_onset_secs = 0.5f;
        cfg.pad_offset_secs = 0.5f;
        std::vector<ExpectedChunk> expected = {
            {0.0f, 5.5f, 0, 1},
            {7.5f, 15.5f, 1, 1},
        };
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG9: empty input still returns 0 chunks ------------------------
    {
        const char* scenario_name = "LG9: empty input";
        std::vector<OmniSegment> input = {};
        OmniChunkConfig cfg = longest_gap_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- LG10: single segment fitting (no Step 4 fallback) --------------
    {
        const char* scenario_name = "LG10: single seg fits, no hard-split";
        std::vector<OmniSegment> input = {{0.0f, 30.0f}};
        OmniChunkConfig cfg = longest_gap_cfg(30.0f);
        std::vector<ExpectedChunk> expected = {{0.0f, 30.0f, 0, 1}};
        run_case(scenario_name, input, cfg, expected);
    }

    // -- Boundary cases (inline checks; EXPECT_* macros use early `return`
    //    which is illegal inside main, so we inline-check here).
    auto check_int = [](const char* name, const char* field, int actual, int expected) {
        if (actual != expected) {
            std::fprintf(stderr, "  FAIL [%s] %s: expected %d, got %d\n",
                         name, field, expected, actual);
            ++g_failed;
            return false;
        }
        return true;
    };
    auto check_float = [](const char* name, const char* field, float actual, float expected) {
        if (!floats_equal(actual, expected)) {
            std::fprintf(stderr, "  FAIL [%s] %s: expected %.4f, got %.4f\n",
                         name, field, expected, actual);
            ++g_failed;
            return false;
        }
        return true;
    };

    {
        const char* n = "B1: NULL out_chunks";
        OmniChunkConfig cfg = omni_chunk_config_default();
        OmniSegment seg = {0.0f, 5.0f};
        int count = 0;
        int rc = omni_merge_chunks(&seg, 1, &cfg, nullptr, &count);
        if (check_int(n, "rc", rc, OMNI_ERR_NULL_POINTER)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B2: NULL config";
        OmniSegment seg = {0.0f, 5.0f};
        OmniChunk* out = nullptr;
        int count = 0;
        int rc = omni_merge_chunks(&seg, 1, nullptr, &out, &count);
        if (check_int(n, "rc", rc, OMNI_ERR_NULL_POINTER)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B3: max_chunk_secs <= 0";
        OmniSegment seg = {0.0f, 5.0f};
        OmniChunkConfig cfg = omni_chunk_config_default();
        cfg.max_chunk_secs = 0.0f;
        OmniChunk* out = nullptr;
        int count = 0;
        int rc = omni_merge_chunks(&seg, 1, &cfg, &out, &count);
        if (check_int(n, "rc", rc, OMNI_ERR_INVALID_ARG)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B4: num=0 with NULL data OK";
        OmniChunkConfig cfg = omni_chunk_config_default();
        OmniChunk* out = nullptr;
        int count = -1;
        int rc = omni_merge_chunks(nullptr, 0, &cfg, &out, &count);
        bool ok = check_int(n, "rc", rc, OMNI_OK);
        ok = check_int(n, "count", count, 0) && ok;
        if (ok) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B5: defaults";
        OmniChunkConfig cfg = omni_chunk_config_default();
        bool ok = check_float(n, "max_chunk_secs", cfg.max_chunk_secs, 30.0f);
        ok = check_float(n, "pad_onset_secs", cfg.pad_onset_secs, 0.04f) && ok;
        ok = check_float(n, "pad_offset_secs", cfg.pad_offset_secs, 0.04f) && ok;
        ok = check_float(n, "min_silence_secs", cfg.min_silence_secs, 0.20f) && ok;
        if (!std::isinf(cfg.max_gap_secs)) {
            std::fprintf(stderr, "  FAIL [%s] max_gap_secs not INFINITY\n", n);
            ++g_failed;
            ok = false;
        }
        if (ok) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B6: NULL out_count";
        OmniChunkConfig cfg = omni_chunk_config_default();
        OmniSegment seg = {0.0f, 5.0f};
        OmniChunk* out = nullptr;
        int rc = omni_merge_chunks(&seg, 1, &cfg, &out, nullptr);
        if (check_int(n, "rc", rc, OMNI_ERR_NULL_POINTER)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B7: NULL segments with num>0";
        OmniChunkConfig cfg = omni_chunk_config_default();
        OmniChunk* out = nullptr;
        int count = 0;
        int rc = omni_merge_chunks(nullptr, 5, &cfg, &out, &count);
        if (check_int(n, "rc", rc, OMNI_ERR_NULL_POINTER)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B8: num_segments < 0";
        OmniChunkConfig cfg = omni_chunk_config_default();
        OmniSegment seg = {0.0f, 5.0f};
        OmniChunk* out = nullptr;
        int count = 0;
        int rc = omni_merge_chunks(&seg, -1, &cfg, &out, &count);
        if (check_int(n, "rc", rc, OMNI_ERR_INVALID_ARG)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B9: max_chunk_secs = NaN rejected";
        OmniChunkConfig cfg = omni_chunk_config_default();
        cfg.max_chunk_secs = std::nanf("");
        OmniSegment seg = {0.0f, 5.0f};
        OmniChunk* out = nullptr;
        int count = 0;
        int rc = omni_merge_chunks(&seg, 1, &cfg, &out, &count);
        // NaN > 0.0f is false, so guard rejects.
        if (check_int(n, "rc", rc, OMNI_ERR_INVALID_ARG)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B10: max_chunk_secs negative rejected";
        OmniChunkConfig cfg = omni_chunk_config_default();
        cfg.max_chunk_secs = -5.0f;
        OmniSegment seg = {0.0f, 5.0f};
        OmniChunk* out = nullptr;
        int count = 0;
        int rc = omni_merge_chunks(&seg, 1, &cfg, &out, &count);
        if (check_int(n, "rc", rc, OMNI_ERR_INVALID_ARG)) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        // ABI self-check: struct sizes must match what every binding hard-codes.
        // packages/omnivad/src/wasm-binding.ts has SIZEOF_CHUNK_CONFIG=28,
        // SIZEOF_CHUNK=16; omnivad/_binding.py relies on ctypes auto-layout but
        // any layout drift here would silently corrupt all three bindings.
        // 28 = 6 floats + 1 i32 (mode); no tail padding since alignof <= 4.
        const char* n = "B11: ABI struct sizes";
        bool ok = check_int(n, "sizeof(OmniSegment)",     (int)sizeof(OmniSegment),     8);
        ok = check_int(n, "sizeof(OmniChunk)",            (int)sizeof(OmniChunk),       16) && ok;
        ok = check_int(n, "sizeof(OmniChunkConfig)",      (int)sizeof(OmniChunkConfig), 28) && ok;
        if (ok) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        // OmniChunkMode enum values are part of the public API — bindings hard-code
        // them as integer constants. Lock them so the enum can't be reordered.
        const char* n = "B12: OmniChunkMode enum values";
        bool ok = check_int(n, "OMNI_CHUNK_GREEDY",       (int)OMNI_CHUNK_GREEDY,       0);
        ok = check_int(n, "OMNI_CHUNK_LONGEST_GAP",       (int)OMNI_CHUNK_LONGEST_GAP,  1) && ok;
        if (ok) std::fprintf(stdout, "  PASS [%s]\n", n);
    }
    {
        const char* n = "B13: default mode is GREEDY";
        OmniChunkConfig cfg = omni_chunk_config_default();
        if (check_int(n, "mode", cfg.mode, OMNI_CHUNK_GREEDY))
            std::fprintf(stdout, "  PASS [%s]\n", n);
    }

    if (g_failed > 0) {
        std::fprintf(stderr, "\n=== %d failure(s) ===\n", g_failed);
        return 1;
    }
    std::fprintf(stdout, "\n=== all tests passed ===\n");
    return 0;
}
