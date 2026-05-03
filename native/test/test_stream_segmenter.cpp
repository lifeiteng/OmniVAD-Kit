/*
 * Test: omni_stream_segmenter_* (pure-algorithm streaming VAD post-processor).
 *
 * Phase 1 (this commit): create/destroy + config validation only. Algorithm
 * tests are added incrementally in Steps 3-6 of the plan.
 *
 * Usage: ./test_stream_segmenter      (no model files needed)
 */

#include "omnivad.h"

#include <cstdio>
#include <cstdlib>

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

    if (g_failed > 0) {
        std::fprintf(stderr, "\n=== %d failure(s) ===\n", g_failed);
        return 1;
    }
    std::fprintf(stdout, "\n=== all tests passed ===\n");
    return 0;
}
