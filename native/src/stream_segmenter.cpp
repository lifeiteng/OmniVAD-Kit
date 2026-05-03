/*
 * OmniVAD Streaming Segmenter — pure-algorithm utility, no model / ncnn dep.
 *
 * Implements the omni_stream_segmenter_* family declared in omnivad.h.
 * Mirrors the causal subset of omni_vad_detect()'s post-processing
 * (steps 1-4 + 7) so that streaming output equals batch output
 * (modulo emission latency).
 *
 * Phase 1 scope: causal steps only.
 *   - merge_silence_frames  > 0  -> rejected at create() time
 *   - extend_speech_frames  > 0  -> rejected at create() time
 *
 * Thread-safety: NO. Each handle is single-threaded.
 */

#include "omnivad.h"

#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>

/* ------------------------------------------------------------------------- */
/*  Constants                                                                 */
/* ------------------------------------------------------------------------- */

namespace {
constexpr float FRAME_SHIFT_SEC  = 0.01f;   // 10ms
constexpr float FRAME_LENGTH_SEC = 0.025f;  // 25ms (matches omni_vad_detect tail rule)
constexpr int   STREAM_VAD_SAMPLE_RATE = 16000;
}

/* ------------------------------------------------------------------------- */
/*  Internal state                                                            */
/* ------------------------------------------------------------------------- */

/* State machine states (must match omnivad.cpp::VadState enum). */
enum SegState {
    SEG_SILENCE          = 0,
    SEG_POSSIBLE_SPEECH  = 1,
    SEG_SPEECH           = 2,
    SEG_POSSIBLE_SILENCE = 3,
};

struct OmniStreamSegmenterCtx {
    /* Config (copied from caller; must outlive create() call). */
    float threshold;
    int   smooth_window_size;
    int   min_speech_frames;
    int   min_silence_frames;
    int   max_speech_frames;

    /* Step 1: causal moving-average state (last smooth_window_size samples). */
    std::deque<float> smooth_buf;

    /* Step 3: state machine. */
    SegState state;
    int      candidate_speech_start;   /* frame index where POSSIBLE_SPEECH began */
    int      candidate_silence_start;  /* frame index where POSSIBLE_SILENCE began */
    int      confirmed_start;          /* with Step 4 applied; -1 if no active segment */

    /* Step 7: ring of raw probs covering current confirmed segment (for force-split). */
    std::vector<float> seg_raw_probs;

    /* Global counter — index of the NEXT frame to be processed (0-based). */
    int total_frames;
};

/* ------------------------------------------------------------------------- */
/*  Public API                                                                */
/* ------------------------------------------------------------------------- */

extern "C" {

OMNIVAD_API OmniStreamSegmenterHandle omni_stream_segmenter_create(
    const OmniPostConfig* config,
    int* out_error)
{
    if (!config) {
        if (out_error) *out_error = OMNI_ERR_NULL_POINTER;
        return nullptr;
    }
    /* Phase 1 reject: non-causal steps. */
    if (config->merge_silence_frames > 0 || config->extend_speech_frames > 0) {
        if (out_error) *out_error = OMNI_ERR_INVALID_ARG;
        return nullptr;
    }
    /* Sanity. */
    if (config->smooth_window_size < 1 || config->min_speech_frames < 0 ||
        config->min_silence_frames < 0 || config->max_speech_frames < 0)
    {
        if (out_error) *out_error = OMNI_ERR_INVALID_ARG;
        return nullptr;
    }

    auto* seg = new (std::nothrow) OmniStreamSegmenterCtx();
    if (!seg) {
        if (out_error) *out_error = OMNI_ERR_OUT_OF_MEMORY;
        return nullptr;
    }

    seg->threshold          = config->threshold;
    seg->smooth_window_size = config->smooth_window_size;
    seg->min_speech_frames  = config->min_speech_frames;
    seg->min_silence_frames = config->min_silence_frames;
    seg->max_speech_frames  = config->max_speech_frames;

    seg->state                  = SEG_SILENCE;
    seg->candidate_speech_start = -1;
    seg->candidate_silence_start = -1;
    seg->confirmed_start        = -1;
    seg->total_frames           = 0;

    if (out_error) *out_error = OMNI_OK;
    return seg;
}

OMNIVAD_API int omni_stream_segmenter_process_frame(
    OmniStreamSegmenterHandle seg,
    float prob,
    OmniSegment** out_segments,
    int* out_count)
{
    if (!seg) return OMNI_ERR_NULL_HANDLE;
    if (!out_segments || !out_count) return OMNI_ERR_NULL_POINTER;
    (void)prob;
    /* Phase 1 stub: no algorithm yet — Step 3 implements it. */
    *out_segments = nullptr;
    *out_count    = 0;
    return OMNI_OK;
}

OMNIVAD_API int omni_stream_segmenter_process_probs(
    OmniStreamSegmenterHandle seg,
    const float* probs,
    int num_frames,
    OmniSegment** out_segments,
    int* out_count)
{
    if (!seg) return OMNI_ERR_NULL_HANDLE;
    if (!out_segments || !out_count) return OMNI_ERR_NULL_POINTER;
    if (num_frames < 0) return OMNI_ERR_INVALID_ARG;
    if (num_frames > 0 && !probs) return OMNI_ERR_NULL_POINTER;
    (void)probs;
    (void)num_frames;
    /* Phase 1 stub. */
    *out_segments = nullptr;
    *out_count    = 0;
    return OMNI_OK;
}

OMNIVAD_API int omni_stream_segmenter_flush(
    OmniStreamSegmenterHandle seg,
    int total_samples_seen,
    OmniSegment** out_segments,
    int* out_count)
{
    if (!seg) return OMNI_ERR_NULL_HANDLE;
    if (!out_segments || !out_count) return OMNI_ERR_NULL_POINTER;
    if (total_samples_seen < 0) return OMNI_ERR_INVALID_ARG;
    (void)total_samples_seen;
    /* Phase 1 stub. */
    *out_segments = nullptr;
    *out_count    = 0;
    return OMNI_OK;
}

OMNIVAD_API bool omni_stream_segmenter_is_in_speech(OmniStreamSegmenterHandle seg) {
    if (!seg) return false;
    return seg->state == SEG_SPEECH || seg->state == SEG_POSSIBLE_SILENCE;
}

OMNIVAD_API float omni_stream_segmenter_get_active_start(OmniStreamSegmenterHandle seg) {
    if (!seg) return -1.0f;
    if (seg->confirmed_start < 0) return -1.0f;
    return (float)seg->confirmed_start * FRAME_SHIFT_SEC;
}

OMNIVAD_API void omni_stream_segmenter_reset(OmniStreamSegmenterHandle seg) {
    if (!seg) return;
    seg->smooth_buf.clear();
    seg->state                   = SEG_SILENCE;
    seg->candidate_speech_start  = -1;
    seg->candidate_silence_start = -1;
    seg->confirmed_start         = -1;
    seg->seg_raw_probs.clear();
    seg->total_frames            = 0;
}

OMNIVAD_API void omni_stream_segmenter_destroy(OmniStreamSegmenterHandle seg) {
    if (!seg) return;
    delete seg;
}

}  /* extern "C" */
