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

    /* Step 1: causal moving-average state (rolling window of last
     * smooth_window_size raw probs + running sum for O(1) update). */
    std::deque<float> smooth_buf;
    double            smooth_running_sum;

    /* Step 3: state machine. */
    SegState state;
    int      candidate_speech_start;   /* frame index where POSSIBLE_SPEECH began */
    int      candidate_silence_start;  /* frame index where POSSIBLE_SILENCE began */
    int      confirmed_start;          /* with Step 4 applied; -1 if no active segment */

    /* Raw probs accumulated during POSSIBLE_SPEECH candidate window — kept
     * separately because smooth_buf only retains the last smooth_window_size
     * frames. On confirm, this is prepended with smooth_window_size * 1.0
     * "neutral" fill to seed seg_raw_probs (the prefix corresponds to the
     * Step 4 backward-extension region; raw probs there are not preserved
     * but they would be silence-zone values which never win the force-split
     * min-search anyway, since the search window starts at max_speech/2 >>
     * smooth_window_size). */
    std::vector<float> candidate_raw_probs;

    /* Step 7: raw probs covering current confirmed segment (for force-split). */
    std::vector<float> seg_raw_probs;

    /* Global frame counter — number of frames pushed so far (1-based count).
     * The frame index of the most recent push is total_frames - 1. */
    int total_frames;
};

/* ------------------------------------------------------------------------- */
/*  Helpers                                                                   */
/* ------------------------------------------------------------------------- */

/* Step 4: extend a candidate start backward by up to smooth_window_size
 * frames. Mirrors omnivad.cpp::fix_smooth_window_start exactly:
 *   `start = (cand - W > 0) ? (cand - W) : 0`
 * Note the strict `> 0` (not `>= 0`) — when cand == W, start = 0. */
static inline int apply_smooth_window_fix(int cand, int W) {
    return (cand - W > 0) ? (cand - W) : 0;
}

/* Append a segment to the emit queue, growing as needed. */
static int push_segment(OmniSegment** out_segments, int* out_count,
                        int* capacity, float start_time, float end_time)
{
    if (*out_count >= *capacity) {
        int new_cap = (*capacity == 0) ? 2 : (*capacity * 2);
        OmniSegment* grown = (OmniSegment*)std::realloc(*out_segments, sizeof(OmniSegment) * new_cap);
        if (!grown) return OMNI_ERR_OUT_OF_MEMORY;
        *out_segments = grown;
        *capacity = new_cap;
    }
    (*out_segments)[*out_count].start = start_time;
    (*out_segments)[*out_count].end   = end_time;
    (*out_count)++;
    return OMNI_OK;
}

/* Step 7: while seg_raw_probs has more than max_speech_frames entries,
 * find the min-probability frame in the second half [N/2, N) of the
 * leading window and split there. Mirrors omnivad.cpp::split_long_segments,
 * but executes incrementally as frames stream in.
 *
 * Each split:
 *   - emits [confirmed_start, confirmed_start + min_idx)
 *   - drops the split frame (treated as silence) and advances the active
 *     segment to start at confirmed_start + min_idx + 1
 *   - keeps seg_raw_probs[min_idx+1 ..] for the next iteration
 *
 * Note: the new segment's start is NOT re-extended by Step 4 (matches
 * batch behaviour: Step 4 runs before Step 7, so freshly-cut starts
 * inside Step 7 do not get backward extension).
 */
static int maybe_force_split(OmniStreamSegmenterCtx* seg,
                             OmniSegment** out_segments,
                             int* out_count,
                             int* capacity)
{
    const int W = seg->max_speech_frames;
    if (W <= 0) return OMNI_OK;

    while ((int)seg->seg_raw_probs.size() > W) {
        const int window_start = W / 2;
        const int window_end   = W;   /* exclusive; never exceeds size since size > W */

        int   min_idx = window_start;
        float min_val = seg->seg_raw_probs[window_start];
        for (int j = window_start + 1; j < window_end; ++j) {
            if (seg->seg_raw_probs[j] < min_val) {
                min_val = seg->seg_raw_probs[j];
                min_idx = j;
            }
        }

        /* Emit the prefix segment [confirmed_start, confirmed_start + min_idx). */
        const float start_sec = (float)seg->confirmed_start * FRAME_SHIFT_SEC;
        const float end_sec   = (float)(seg->confirmed_start + min_idx) * FRAME_SHIFT_SEC;
        int rc = push_segment(out_segments, out_count, capacity, start_sec, end_sec);
        if (rc != OMNI_OK) return rc;

        /* Advance: drop the silence frame at min_idx, keep [min_idx+1 ..]. */
        seg->confirmed_start += (min_idx + 1);
        seg->seg_raw_probs.erase(seg->seg_raw_probs.begin(),
                                  seg->seg_raw_probs.begin() + (min_idx + 1));
    }
    return OMNI_OK;
}

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

    seg->smooth_running_sum      = 0.0;
    seg->state                   = SEG_SILENCE;
    seg->candidate_speech_start  = -1;
    seg->candidate_silence_start = -1;
    seg->confirmed_start         = -1;
    seg->total_frames            = 0;

    if (out_error) *out_error = OMNI_OK;
    return seg;
}

/* ------------------------------------------------------------------------- */
/*  Core algorithm: push one (frame_idx, prob) pair through steps 1-4,       */
/*  appending any emitted segments to *out_segments.                         */
/* ------------------------------------------------------------------------- */

static int push_one_frame_internal(
    OmniStreamSegmenterCtx* seg,
    float prob,
    OmniSegment** out_segments,
    int* out_count,
    int* capacity)
{
    /* This frame's 0-based index, then bump counter. */
    const int frame_idx = seg->total_frames;
    seg->total_frames++;

    /* ----- Step 1: causal moving-average smoothing ----- */
    seg->smooth_buf.push_back(prob);
    seg->smooth_running_sum += prob;
    if ((int)seg->smooth_buf.size() > seg->smooth_window_size) {
        seg->smooth_running_sum -= seg->smooth_buf.front();
        seg->smooth_buf.pop_front();
    }
    /* boundary phase (size < window) -> divides by current size, matches
     * omnivad.cpp's "first window-1 frames cumulative average" branch. */
    const float smoothed = (float)(seg->smooth_running_sum / (double)seg->smooth_buf.size());

    /* ----- Step 2: binary threshold ----- */
    const bool is_speech = smoothed >= seg->threshold;

    /* ----- Step 3: 4-state state machine -----
     * Mirrors omnivad.cpp::state_machine_smooth, but emits segments on
     * the POSSIBLE_SILENCE -> SILENCE transition (with Step 4 applied). */
    switch (seg->state) {
    case SEG_SILENCE:
        if (is_speech) {
            seg->state                  = SEG_POSSIBLE_SPEECH;
            seg->candidate_speech_start = frame_idx;
            if (seg->max_speech_frames > 0) {
                seg->candidate_raw_probs.clear();
                seg->candidate_raw_probs.push_back(prob);
            }
        }
        break;

    case SEG_POSSIBLE_SPEECH:
        if (is_speech) {
            if (seg->max_speech_frames > 0) {
                seg->candidate_raw_probs.push_back(prob);
            }
            if (frame_idx - seg->candidate_speech_start >= seg->min_speech_frames) {
                /* Confirm START. Apply Step 4 backward extension here. */
                seg->state           = SEG_SPEECH;
                seg->confirmed_start = apply_smooth_window_fix(
                    seg->candidate_speech_start, seg->smooth_window_size);
                /* Seed Step 7 buffer:
                 *   [confirmed_start .. candidate_speech_start)  -> neutral 1.0
                 *   [candidate_speech_start .. frame_idx]         -> candidate_raw_probs
                 * The 1.0 prefix is safe because the force-split min-search
                 * window is [max_speech/2, max_speech), which is far past
                 * any reasonable smooth_window_size prefix. */
                if (seg->max_speech_frames > 0) {
                    seg->seg_raw_probs.clear();
                    const int prefix_extend = seg->candidate_speech_start - seg->confirmed_start;
                    for (int i = 0; i < prefix_extend; ++i) {
                        seg->seg_raw_probs.push_back(1.0f);
                    }
                    seg->seg_raw_probs.insert(seg->seg_raw_probs.end(),
                                              seg->candidate_raw_probs.begin(),
                                              seg->candidate_raw_probs.end());
                    seg->candidate_raw_probs.clear();
                    /* Force-split may already need to fire if the candidate
                     * window itself was very long. */
                    int rc = maybe_force_split(seg, out_segments, out_count, capacity);
                    if (rc != OMNI_OK) return rc;
                }
            }
        } else {
            /* Cancel candidate. */
            seg->state                  = SEG_SILENCE;
            seg->candidate_speech_start = -1;
            seg->candidate_raw_probs.clear();
        }
        break;

    case SEG_SPEECH:
        if (!is_speech) {
            seg->state                   = SEG_POSSIBLE_SILENCE;
            seg->candidate_silence_start = frame_idx;
        }
        if (seg->max_speech_frames > 0) {
            seg->seg_raw_probs.push_back(prob);
            int rc = maybe_force_split(seg, out_segments, out_count, capacity);
            if (rc != OMNI_OK) return rc;
        }
        break;

    case SEG_POSSIBLE_SILENCE:
        if (!is_speech) {
            if (frame_idx - seg->candidate_silence_start >= seg->min_silence_frames) {
                /* Confirm END — emit segment.
                 * end_frame is candidate_silence_start (exclusive end), matches
                 * omnivad.cpp::decisions_to_segments which records `t` (the
                 * first 0-frame) as the segment's end. */
                const int end_frame   = seg->candidate_silence_start;
                const float start_sec = (float)seg->confirmed_start * FRAME_SHIFT_SEC;
                const float end_sec   = (float)end_frame * FRAME_SHIFT_SEC;
                int rc = push_segment(out_segments, out_count, capacity, start_sec, end_sec);
                if (rc != OMNI_OK) return rc;

                seg->state                   = SEG_SILENCE;
                seg->confirmed_start         = -1;
                seg->candidate_speech_start  = -1;
                seg->candidate_silence_start = -1;
                seg->seg_raw_probs.clear();
            } else {
                /* Still in POSSIBLE_SILENCE — keep accumulating raw probs but
                 * DO NOT force-split here. Force-split only fires in SPEECH
                 * state to keep its split-point semantics clean (avoids
                 * cutting through silence frames whose end-time would clash
                 * with candidate_silence_start). The buffer can grow up to
                 * max_speech_frames + min_silence_frames before the segment
                 * either confirms END or returns to SPEECH (where the deferred
                 * split fires). */
                if (seg->max_speech_frames > 0) {
                    seg->seg_raw_probs.push_back(prob);
                }
            }
        } else {
            seg->state                   = SEG_SPEECH;
            seg->candidate_silence_start = -1;
            if (seg->max_speech_frames > 0) {
                seg->seg_raw_probs.push_back(prob);
                int rc = maybe_force_split(seg, out_segments, out_count, capacity);
                if (rc != OMNI_OK) return rc;
            }
        }
        break;
    }

    return OMNI_OK;
}

OMNIVAD_API int omni_stream_segmenter_process_frame(
    OmniStreamSegmenterHandle seg,
    float prob,
    OmniSegment** out_segments,
    int* out_count)
{
    if (!seg) return OMNI_ERR_NULL_HANDLE;
    if (!out_segments || !out_count) return OMNI_ERR_NULL_POINTER;

    *out_segments = nullptr;
    *out_count    = 0;
    int capacity  = 0;

    int rc = push_one_frame_internal(seg, prob, out_segments, out_count, &capacity);
    if (rc != OMNI_OK) {
        std::free(*out_segments);
        *out_segments = nullptr;
        *out_count    = 0;
    }
    return rc;
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

    *out_segments = nullptr;
    *out_count    = 0;
    int capacity  = 0;

    for (int i = 0; i < num_frames; ++i) {
        int rc = push_one_frame_internal(seg, probs[i], out_segments, out_count, &capacity);
        if (rc != OMNI_OK) {
            std::free(*out_segments);
            *out_segments = nullptr;
            *out_count    = 0;
            return rc;
        }
    }
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
    seg->smooth_running_sum      = 0.0;
    seg->state                   = SEG_SILENCE;
    seg->candidate_speech_start  = -1;
    seg->candidate_silence_start = -1;
    seg->confirmed_start         = -1;
    seg->candidate_raw_probs.clear();
    seg->seg_raw_probs.clear();
    seg->total_frames            = 0;
}

OMNIVAD_API void omni_stream_segmenter_destroy(OmniStreamSegmenterHandle seg) {
    if (!seg) return;
    delete seg;
}

}  /* extern "C" */
