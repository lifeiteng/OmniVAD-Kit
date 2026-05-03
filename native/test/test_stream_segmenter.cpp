/*
 * Test: omni_stream_segmenter_* (pure-algorithm streaming VAD post-processor).
 *
 * Phase 1 (this commit): create/destroy + config validation only. Algorithm
 * tests are added incrementally in Steps 3-6 of the plan.
 *
 * Usage: ./test_stream_segmenter      (no model files needed)
 */

#include "omnivad.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failed = 0;

#define EXPECT_RET(actual, expected, scenario, msg)                                          \
    do {                                                                                       \
        if ((actual) != (expected)) {                                                          \
            std::fprintf(stderr, "  FAIL [%s] %s: expected %d, got %d (line %d)\n",            \
                         scenario, msg, (int)(expected), (int)(actual), __LINE__);            \
            ++g_failed;                                                                        \
            return;                                                                            \
        }                                                                                      \
    } while (0)

#define EXPECT_NULL(ptr, scenario, msg)                                                        \
    do {                                                                                       \
        if ((ptr) != nullptr) {                                                                \
            std::fprintf(stderr, "  FAIL [%s] %s: expected NULL handle (line %d)\n",           \
                         scenario, msg, __LINE__);                                             \
            ++g_failed;                                                                        \
            return;                                                                            \
        }                                                                                      \
    } while (0)

#define EXPECT_NONNULL(ptr, scenario, msg)                                                     \
    do {                                                                                       \
        if ((ptr) == nullptr) {                                                                \
            std::fprintf(stderr, "  FAIL [%s] %s: expected non-NULL (line %d)\n",              \
                         scenario, msg, __LINE__);                                             \
            ++g_failed;                                                                        \
            return;                                                                            \
        }                                                                                      \
    } while (0)

#define EXPECT_TRUE(cond, scenario, msg)                                                       \
    do {                                                                                       \
        if (!(cond)) {                                                                         \
            std::fprintf(stderr, "  FAIL [%s] %s (line %d)\n", scenario, msg, __LINE__);       \
            ++g_failed;                                                                        \
            return;                                                                            \
        }                                                                                      \
    } while (0)

/* ------------------------------------------------------------------------- */
/*  Test scenarios                                                            */
/* ------------------------------------------------------------------------- */

/* B1: create with default config succeeds (incl. max_speech_frames=3000). */
static void t_create_with_default_config() {
    const char* s = "B1: create(default config)";
    OmniPostConfig cfg = omni_post_config_default();
    int err = -1;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");
    EXPECT_RET(err, OMNI_OK, s, "out_error");
    omni_stream_segmenter_destroy(h);
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B2: NULL config rejected. */
static void t_create_null_config() {
    const char* s = "B2: create(NULL config) rejected";
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(nullptr, &err);
    EXPECT_NULL(h, s, "handle");
    EXPECT_RET(err, OMNI_ERR_NULL_POINTER, s, "out_error");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B3: merge_silence_frames > 0 rejected (Phase 1 limitation). */
static void t_create_rejects_merge_silence() {
    const char* s = "B3: merge_silence_frames>0 rejected";
    OmniPostConfig cfg = omni_post_config_default();
    cfg.merge_silence_frames = 1;
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NULL(h, s, "handle");
    EXPECT_RET(err, OMNI_ERR_INVALID_ARG, s, "out_error");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B4: extend_speech_frames > 0 rejected (Phase 1 limitation). */
static void t_create_rejects_extend_speech() {
    const char* s = "B4: extend_speech_frames>0 rejected";
    OmniPostConfig cfg = omni_post_config_default();
    cfg.extend_speech_frames = 1;
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NULL(h, s, "handle");
    EXPECT_RET(err, OMNI_ERR_INVALID_ARG, s, "out_error");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B5: smooth_window_size < 1 rejected. */
static void t_create_rejects_zero_smooth_window() {
    const char* s = "B5: smooth_window_size=0 rejected";
    OmniPostConfig cfg = omni_post_config_default();
    cfg.smooth_window_size = 0;
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NULL(h, s, "handle");
    EXPECT_RET(err, OMNI_ERR_INVALID_ARG, s, "out_error");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B6: max_speech_frames=0 accepted (disables Step 7). */
static void t_create_accepts_zero_max_speech() {
    const char* s = "B6: max_speech_frames=0 accepted";
    OmniPostConfig cfg = omni_post_config_default();
    cfg.max_speech_frames = 0;
    int err = -1;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");
    EXPECT_RET(err, OMNI_OK, s, "out_error");
    omni_stream_segmenter_destroy(h);
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B7: NULL handle to all functions returns appropriate error / safe defaults. */
static void t_null_handle_safety() {
    const char* s = "B7: NULL handle safety";
    OmniSegment* out = nullptr;
    int count = 0;
    EXPECT_RET(omni_stream_segmenter_process_frame(nullptr, 0.5f, &out, &count),
               OMNI_ERR_NULL_HANDLE, s, "process_frame");
    EXPECT_RET(omni_stream_segmenter_process_probs(nullptr, nullptr, 0, &out, &count),
               OMNI_ERR_NULL_HANDLE, s, "process_probs");
    EXPECT_RET(omni_stream_segmenter_flush(nullptr, 0, &out, &count),
               OMNI_ERR_NULL_HANDLE, s, "flush");
    EXPECT_TRUE(omni_stream_segmenter_is_in_speech(nullptr) == false, s, "is_in_speech");
    EXPECT_TRUE(omni_stream_segmenter_get_active_start(nullptr) < 0.0f, s, "get_active_start");
    omni_stream_segmenter_reset(nullptr);    /* must not crash */
    omni_stream_segmenter_destroy(nullptr);  /* must not crash */
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B8: NULL out pointers in process_frame rejected. */
static void t_null_out_pointers() {
    const char* s = "B8: NULL out pointers rejected";
    OmniPostConfig cfg = omni_post_config_default();
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");

    OmniSegment* out = nullptr;
    int count = 0;
    EXPECT_RET(omni_stream_segmenter_process_frame(h, 0.5f, nullptr, &count),
               OMNI_ERR_NULL_POINTER, s, "process_frame NULL out_segments");
    EXPECT_RET(omni_stream_segmenter_process_frame(h, 0.5f, &out, nullptr),
               OMNI_ERR_NULL_POINTER, s, "process_frame NULL out_count");
    EXPECT_RET(omni_stream_segmenter_flush(h, 0, nullptr, &count),
               OMNI_ERR_NULL_POINTER, s, "flush NULL out_segments");

    omni_stream_segmenter_destroy(h);
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B9: initial state — not in speech, no active start. */
static void t_initial_state() {
    const char* s = "B9: initial state";
    OmniPostConfig cfg = omni_post_config_default();
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");

    EXPECT_TRUE(omni_stream_segmenter_is_in_speech(h) == false, s, "is_in_speech == false");
    EXPECT_TRUE(omni_stream_segmenter_get_active_start(h) < 0.0f, s, "get_active_start < 0");

    omni_stream_segmenter_destroy(h);
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* B10: reset is safe before/after any process call. */
static void t_reset_safe() {
    const char* s = "B10: reset is idempotent";
    OmniPostConfig cfg = omni_post_config_default();
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");

    omni_stream_segmenter_reset(h);
    omni_stream_segmenter_reset(h);
    EXPECT_TRUE(omni_stream_segmenter_is_in_speech(h) == false, s, "still not in speech");

    omni_stream_segmenter_destroy(h);
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* ------------------------------------------------------------------------- */
/*  Algorithm tests (Step 3: causal steps 1-4)                                */
/* ------------------------------------------------------------------------- */

static bool floats_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

/* Make a probability sequence: `n_silence` zeros + `n_speech` ones (etc). */
static std::vector<float> make_probs_pattern(const std::vector<std::pair<int, float>>& runs) {
    std::vector<float> out;
    for (const auto& r : runs) {
        for (int i = 0; i < r.first; ++i) out.push_back(r.second);
    }
    return out;
}

/* Test fixture: feed `probs` with config; return collected segments. */
static std::vector<OmniSegment> run_segmenter(
    const OmniPostConfig& cfg, const std::vector<float>& probs)
{
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    if (!h) return {};

    std::vector<OmniSegment> all;
    OmniSegment* out = nullptr;
    int count = 0;
    omni_stream_segmenter_process_probs(h, probs.data(), (int)probs.size(), &out, &count);
    for (int i = 0; i < count; ++i) all.push_back(out[i]);
    std::free(out);

    omni_stream_segmenter_destroy(h);
    return all;
}

/* T1: all-zero probs -> no segments emitted. */
static void t_all_silence_no_emit() {
    const char* s = "T1: all-silence -> 0 segments";
    OmniPostConfig cfg = omni_post_config_default();
    auto probs = std::vector<float>(100, 0.0f);
    auto segs = run_segmenter(cfg, probs);
    EXPECT_RET((int)segs.size(), 0, s, "no emit");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* T2: all-1.0 probs -> still 0 emitted (no silence end seen yet). */
static void t_all_speech_no_emit_until_flush() {
    const char* s = "T2: all-speech -> 0 emitted (flush handles tail)";
    OmniPostConfig cfg = omni_post_config_default();
    auto probs = std::vector<float>(100, 1.0f);
    auto segs = run_segmenter(cfg, probs);
    EXPECT_RET((int)segs.size(), 0, s, "no emit before flush");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* T3: short pulse below min_speech_frames -> POSSIBLE_SPEECH cancelled, no emit. */
static void t_short_pulse_no_emit() {
    const char* s = "T3: short pulse < min_speech_frames -> 0";
    OmniPostConfig cfg = omni_post_config_default();
    /* 5 silence + 10 speech (< 20 min) + 50 silence */
    auto probs = make_probs_pattern({{5, 0.0f}, {10, 1.0f}, {50, 0.0f}});
    auto segs = run_segmenter(cfg, probs);
    EXPECT_RET((int)segs.size(), 0, s, "short pulse rejected");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* T4: clean segment 5 silence + 30 speech + 25 silence -> 1 segment.
 * Hand-trace with smooth_window_size=5, threshold=0.4, min_speech=20,
 * min_silence=20 (default config):
 *   - frame 5: buf=[0,0,0,0,1] smoothed=0.2 < 0.4 -> SILENCE
 *   - frame 6: buf=[0,0,0,1,1] smoothed=0.4 >= 0.4 -> POSSIBLE_SPEECH @6
 *   - frame 26: 26-6=20 >= 20 -> SPEECH; Step 4: confirmed_start = max(0,6-5)=1
 *   - speech runs until frame 34 (last `1.0`)
 *   - frame 35: buf=[1,1,1,1,0] smoothed=0.8 OK
 *   - frame 36: buf=[1,1,1,0,0] smoothed=0.6 OK
 *   - frame 37: buf=[1,1,0,0,0] smoothed=0.4 OK
 *   - frame 38: buf=[1,0,0,0,0] smoothed=0.2 < 0.4 -> POSSIBLE_SILENCE @38
 *   - frame 58: 58-38=20 -> emit (0.01, 0.38)
 */
static void t_one_clean_segment() {
    const char* s = "T4: one clean segment";
    OmniPostConfig cfg = omni_post_config_default();
    auto probs = make_probs_pattern({{5, 0.0f}, {30, 1.0f}, {25, 0.0f}});
    auto segs = run_segmenter(cfg, probs);
    EXPECT_RET((int)segs.size(), 1, s, "exactly 1 segment");
    EXPECT_TRUE(floats_equal(segs[0].start, 0.01f), s, "start == 0.01");
    EXPECT_TRUE(floats_equal(segs[0].end, 0.38f), s, "end == 0.38");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* T5: two segments separated by long silence -> emit 2. */
static void t_two_segments() {
    const char* s = "T5: two segments";
    OmniPostConfig cfg = omni_post_config_default();
    /* seg1: [5..35) speech, then [35..70) silence, seg2: [70..100) speech, [100..125) silence */
    auto probs = make_probs_pattern({
        {5,  0.0f}, {30, 1.0f},
        {35, 0.0f}, {30, 1.0f},
        {25, 0.0f},
    });
    auto segs = run_segmenter(cfg, probs);
    EXPECT_RET((int)segs.size(), 2, s, "exactly 2 segments");
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* T6: chunk-size invariance — process_frame N times == process_probs(N).
 * Critical correctness invariant: streaming output must not depend on
 * the chunking pattern. */
static void t_chunk_size_invariance() {
    const char* s = "T6: chunk-size invariance";
    OmniPostConfig cfg = omni_post_config_default();
    auto probs = make_probs_pattern({
        {5, 0.0f}, {30, 1.0f}, {35, 0.0f},
        {30, 1.0f}, {25, 0.0f}, {25, 1.0f}, {30, 0.0f},
    });

    /* Path A: process_probs in one shot. */
    auto segs_a = run_segmenter(cfg, probs);

    /* Path B: process_frame one-by-one. */
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");
    std::vector<OmniSegment> segs_b;
    for (float p : probs) {
        OmniSegment* out = nullptr;
        int count = 0;
        omni_stream_segmenter_process_frame(h, p, &out, &count);
        for (int i = 0; i < count; ++i) segs_b.push_back(out[i]);
        std::free(out);
    }
    omni_stream_segmenter_destroy(h);

    /* Path C: process_probs in random small chunks. */
    OmniStreamSegmenterHandle h2 = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h2, s, "handle 2");
    std::vector<OmniSegment> segs_c;
    int chunks[] = {1, 7, 3, 13, 1, 50, 100};   /* arbitrary chunk schedule */
    int idx = 0;
    int chunk_i = 0;
    while (idx < (int)probs.size()) {
        int n = chunks[chunk_i % (int)(sizeof(chunks) / sizeof(int))];
        if (idx + n > (int)probs.size()) n = (int)probs.size() - idx;
        OmniSegment* out = nullptr;
        int count = 0;
        omni_stream_segmenter_process_probs(h2, probs.data() + idx, n, &out, &count);
        for (int i = 0; i < count; ++i) segs_c.push_back(out[i]);
        std::free(out);
        idx += n;
        chunk_i++;
    }
    omni_stream_segmenter_destroy(h2);

    /* All three paths must produce identical output. */
    EXPECT_RET((int)segs_a.size(), (int)segs_b.size(), s, "A == B count");
    EXPECT_RET((int)segs_b.size(), (int)segs_c.size(), s, "B == C count");
    for (size_t i = 0; i < segs_a.size(); ++i) {
        EXPECT_TRUE(floats_equal(segs_a[i].start, segs_b[i].start), s, "A.start == B.start");
        EXPECT_TRUE(floats_equal(segs_a[i].end,   segs_b[i].end),   s, "A.end == B.end");
        EXPECT_TRUE(floats_equal(segs_a[i].start, segs_c[i].start), s, "A.start == C.start");
        EXPECT_TRUE(floats_equal(segs_a[i].end,   segs_c[i].end),   s, "A.end == C.end");
    }
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* T7: reset clears state — feeding pattern after reset behaves like fresh. */
static void t_reset_clears_state() {
    const char* s = "T7: reset clears state";
    OmniPostConfig cfg = omni_post_config_default();
    auto probs = make_probs_pattern({{5, 0.0f}, {30, 1.0f}, {25, 0.0f}});

    auto reference = run_segmenter(cfg, probs);  /* fresh handle */

    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");
    /* Pollute state: push 50 frames of garbage. */
    auto garbage = make_probs_pattern({{50, 0.5f}});
    OmniSegment* out = nullptr; int count = 0;
    omni_stream_segmenter_process_probs(h, garbage.data(), (int)garbage.size(), &out, &count);
    std::free(out);

    omni_stream_segmenter_reset(h);

    std::vector<OmniSegment> after_reset;
    out = nullptr; count = 0;
    omni_stream_segmenter_process_probs(h, probs.data(), (int)probs.size(), &out, &count);
    for (int i = 0; i < count; ++i) after_reset.push_back(out[i]);
    std::free(out);
    omni_stream_segmenter_destroy(h);

    EXPECT_RET((int)after_reset.size(), (int)reference.size(), s, "same count after reset");
    for (size_t i = 0; i < reference.size(); ++i) {
        EXPECT_TRUE(floats_equal(after_reset[i].start, reference[i].start), s, "start matches");
        EXPECT_TRUE(floats_equal(after_reset[i].end,   reference[i].end),   s, "end matches");
    }
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

/* T8: is_in_speech transitions correctly through SPEECH state. */
static void t_is_in_speech_transitions() {
    const char* s = "T8: is_in_speech state transitions";
    OmniPostConfig cfg = omni_post_config_default();
    int err = OMNI_OK;
    OmniStreamSegmenterHandle h = omni_stream_segmenter_create(&cfg, &err);
    EXPECT_NONNULL(h, s, "handle");

    /* Initial: not in speech. */
    EXPECT_TRUE(!omni_stream_segmenter_is_in_speech(h), s, "initially silent");

    /* Push silence: still not in speech. */
    OmniSegment* out = nullptr; int count = 0;
    auto sil = std::vector<float>(5, 0.0f);
    omni_stream_segmenter_process_probs(h, sil.data(), 5, &out, &count); std::free(out);
    EXPECT_TRUE(!omni_stream_segmenter_is_in_speech(h), s, "still silent after sil");

    /* Push enough speech to confirm. */
    auto sp = std::vector<float>(40, 1.0f);
    out = nullptr; count = 0;
    omni_stream_segmenter_process_probs(h, sp.data(), 40, &out, &count); std::free(out);
    EXPECT_TRUE(omni_stream_segmenter_is_in_speech(h), s, "in speech after long burst");
    EXPECT_TRUE(omni_stream_segmenter_get_active_start(h) >= 0.0f, s, "active_start >= 0");

    omni_stream_segmenter_destroy(h);
    std::fprintf(stdout, "  PASS [%s]\n", s);
}

int main(int /*argc*/, char** /*argv*/) {
    std::fprintf(stdout, "=== test_stream_segmenter ===\n");

    t_create_with_default_config();
    t_create_null_config();
    t_create_rejects_merge_silence();
    t_create_rejects_extend_speech();
    t_create_rejects_zero_smooth_window();
    t_create_accepts_zero_max_speech();
    t_null_handle_safety();
    t_null_out_pointers();
    t_initial_state();
    t_reset_safe();

    t_all_silence_no_emit();
    t_all_speech_no_emit_until_flush();
    t_short_pulse_no_emit();
    t_one_clean_segment();
    t_two_segments();
    t_chunk_size_invariance();
    t_reset_clears_state();
    t_is_in_speech_transitions();

    if (g_failed > 0) {
        std::fprintf(stderr, "\n=== %d failure(s) ===\n", g_failed);
        return 1;
    }
    std::fprintf(stdout, "\n=== all tests passed ===\n");
    return 0;
}
