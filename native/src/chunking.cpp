/*
 * OmniVAD Chunking — pure-algorithm utility, no model / ncnn dependency.
 *
 * Implements omni_merge_chunks() and omni_chunk_config_default(), declared
 * in omnivad.h. WhisperX-style binarize+merge, but we skip the binarize
 * half because OmniVAD already returns binarized timestamps.
 *
 * Thread-safe: stateless. Output is malloc'd; caller frees via omni_free().
 */

#include "omnivad.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

OMNIVAD_API OmniChunkConfig omni_chunk_config_default(void) {
    OmniChunkConfig cfg;
    cfg.max_chunk_secs   = 30.0f;
    cfg.max_gap_secs     = INFINITY;
    cfg.pad_onset_secs   = 0.04f;
    cfg.pad_offset_secs  = 0.04f;
    cfg.min_speech_secs  = 0.0f;
    cfg.min_silence_secs = 0.20f;  // matches VAD min_silence_frames=20 @ 10ms frame shift
    cfg.mode             = OMNI_CHUNK_GREEDY;
    return cfg;
}

OMNIVAD_API int omni_merge_chunks(
    const OmniSegment*     segments,
    int                    num_segments,
    const OmniChunkConfig* config,
    OmniChunk**            out_chunks,
    int*                   out_count
) {
    if (out_chunks == nullptr || out_count == nullptr || config == nullptr) {
        return OMNI_ERR_NULL_POINTER;
    }
    if (segments == nullptr && num_segments != 0) {
        return OMNI_ERR_NULL_POINTER;
    }
    if (num_segments < 0) {
        return OMNI_ERR_INVALID_ARG;
    }
    if (!(config->max_chunk_secs > 0.0f)) {
        return OMNI_ERR_INVALID_ARG;
    }

    *out_chunks = nullptr;
    *out_count  = 0;

    // ----- Step 1: filter input by min_speech_secs -----------------------
    //
    // Drop segments shorter than this duration (whisperX-style Binarize).
    std::vector<OmniSegment> active;
    active.reserve(static_cast<size_t>(num_segments));
    const float min_on = config->min_speech_secs;
    for (int i = 0; i < num_segments; ++i) {
        const float dur = segments[i].end - segments[i].start;
        if (min_on <= 0.0f || dur >= min_on) {
            active.push_back(segments[i]);
        }
    }

    if (active.empty()) {
        return OMNI_OK;
    }

    // ----- Step 2: pre-merge tiny gaps (min_silence_secs) ---------------
    //
    // Fill inactive regions shorter than this many seconds (whisperX-style
    // Binarize.support(collar=min_silence_secs)). We operate on already-
    // sorted segments; if input isn't sorted by start time the output is
    // undefined — that's the caller's contract.
    if (config->min_silence_secs > 0.0f) {
        std::vector<OmniSegment> merged;
        merged.reserve(active.size());
        merged.push_back(active[0]);
        for (size_t i = 1; i < active.size(); ++i) {
            const float gap = active[i].start - merged.back().end;
            if (gap < config->min_silence_secs) {
                if (active[i].end > merged.back().end) {
                    merged.back().end = active[i].end;
                }
            } else {
                merged.push_back(active[i]);
            }
        }
        active = std::move(merged);
    }

    // ----- Step 3: pack into duration-bounded chunks --------------------
    //
    // Two strategies dispatch on config->mode. Both honor max_gap_secs as a
    // hard split boundary and max_chunk_secs as a hard upper bound on chunk
    // duration; they only differ in WHERE they cut when forced to.
    //
    //   GREEDY      — sequential append (whisperX-style merge_chunks main
    //                 loop). Splits at the FIRST point that violates either
    //                 constraint.
    //   LONGEST_GAP — recursive divide-and-conquer. When a constraint is
    //                 violated, splits at the LONGEST internal gap so the
    //                 cut lands on the most natural pause.
    //
    // Both strategies emit chunks in increasing seg_start order, so Step 4
    // can iterate them uniformly.
    struct PendingChunk {
        float start;
        float end;
        int   seg_start;  // index into `active` (post-merge view)
        int   seg_count;
    };
    std::vector<PendingChunk> chunks;
    chunks.reserve(active.size());

    if (config->mode == OMNI_CHUNK_LONGEST_GAP) {
        // Recursive split. Stack-based to bound depth and avoid C-stack
        // overflow on pathological inputs (deep recursion on already-split
        // chunks). Each work item is a contiguous range [begin, end) of
        // `active` indices.
        struct Range {
            int begin;
            int end;  // exclusive
        };
        std::vector<Range> stack;
        stack.reserve(active.size());
        // Process in-order so output chunks are sorted by seg_start. Push
        // ranges in reverse so we pop the leftmost first.
        stack.push_back({0, static_cast<int>(active.size())});

        while (!stack.empty()) {
            Range r = stack.back();
            stack.pop_back();
            const int n = r.end - r.begin;
            if (n <= 0) continue;

            // Single segment: nothing to cut. Step 4 will equal-split if
            // it still exceeds max_chunk_secs.
            if (n == 1) {
                PendingChunk c;
                c.start     = active[r.begin].start;
                c.end       = active[r.end - 1].end;
                c.seg_start = r.begin;
                c.seg_count = n;
                chunks.push_back(c);
                continue;
            }

            // Find the longest internal gap. Tie-break: leftmost (i.e.,
            // first occurrence wins) so output is deterministic.
            int   best_i   = r.begin;
            float best_gap = -1.0f;
            for (int i = r.begin; i < r.end - 1; ++i) {
                const float gap = active[i + 1].start - active[i].end;
                if (gap > best_gap) {
                    best_gap = gap;
                    best_i   = i;
                }
            }

            // Stop conditions: range fits max_chunk_secs AND no internal gap
            // exceeds max_gap_secs. If either is violated, we must cut. The
            // longest gap is necessarily >= every "bad" gap, so cutting
            // there resolves at least one violation per recursion step.
            const float span             = active[r.end - 1].end - active[r.begin].start;
            const bool  fits_size        = span <= config->max_chunk_secs;
            const bool  honors_max_gap   = best_gap <= config->max_gap_secs;
            if (fits_size && honors_max_gap) {
                PendingChunk c;
                c.start     = active[r.begin].start;
                c.end       = active[r.end - 1].end;
                c.seg_start = r.begin;
                c.seg_count = n;
                chunks.push_back(c);
                continue;
            }

            // Split into [begin, best_i+1) and [best_i+1, end). Push the
            // RIGHT half first so the LEFT half is popped/processed next,
            // preserving in-order emission.
            stack.push_back({best_i + 1, r.end});
            stack.push_back({r.begin, best_i + 1});
        }
    } else {
        // GREEDY (default).
        PendingChunk cur;
        cur.start     = active[0].start;
        cur.end       = active[0].end;
        cur.seg_start = 0;
        cur.seg_count = 1;

        for (size_t i = 1; i < active.size(); ++i) {
            const float gap                  = active[i].start - cur.end;
            const bool  cur_has_content      = (cur.end - cur.start) > 0.0f;
            const bool  split_by_gap         = gap > config->max_gap_secs && cur_has_content;
            const bool  would_exceed         = (active[i].end - cur.start) > config->max_chunk_secs;
            const bool  split_by_size        = would_exceed && cur_has_content;

            if (split_by_gap || split_by_size) {
                chunks.push_back(cur);
                cur.start     = active[i].start;
                cur.end       = active[i].end;
                cur.seg_start = static_cast<int>(i);
                cur.seg_count = 1;
            } else {
                cur.end = active[i].end;
                cur.seg_count++;
            }
        }
        chunks.push_back(cur);
    }

    // ----- Step 4: hard-split chunks that still exceed max_chunk_secs -------
    //
    // Triggered when a single segment is longer than max_chunk_secs — Step 3
    // alone cannot enforce the cap because it can't split mid-segment.
    std::vector<PendingChunk> final_chunks;
    final_chunks.reserve(chunks.size());
    for (const PendingChunk& c : chunks) {
        const float dur = c.end - c.start;
        if (dur <= config->max_chunk_secs) {
            final_chunks.push_back(c);
            continue;
        }
        // Equal-size sub-chunks. Sub-chunk boundaries do not respect
        // segment edges (mid-segment splits are inherent to this case).
        // We attribute every input segment that overlaps the sub-chunk
        // window to its seg_start_idx / seg_count, even if partially.
        float s = c.start;
        while (s < c.end) {
            const float e = std::fmin(s + config->max_chunk_secs, c.end);
            int sub_start = -1;
            int sub_count = 0;
            for (int j = c.seg_start; j < c.seg_start + c.seg_count; ++j) {
                if (active[j].end > s && active[j].start < e) {
                    if (sub_start < 0) sub_start = j;
                    sub_count++;
                }
            }
            if (sub_start < 0) {
                sub_start = c.seg_start;
                sub_count = 0;
            }
            PendingChunk sub;
            sub.start     = s;
            sub.end       = e;
            sub.seg_start = sub_start;
            sub.seg_count = sub_count;
            final_chunks.push_back(sub);
            s = e;
        }
    }

    // ----- Step 5: apply pad_onset_secs / pad_offset_secs and emit -----------------
    const int n = static_cast<int>(final_chunks.size());
    OmniChunk* out = static_cast<OmniChunk*>(std::malloc(sizeof(OmniChunk) * n));
    if (out == nullptr) {
        return OMNI_ERR_OUT_OF_MEMORY;
    }
    for (int i = 0; i < n; ++i) {
        float padded_start = final_chunks[i].start - config->pad_onset_secs;
        if (padded_start < 0.0f) padded_start = 0.0f;
        out[i].start         = padded_start;
        out[i].end           = final_chunks[i].end + config->pad_offset_secs;
        out[i].seg_start_idx = final_chunks[i].seg_start;
        out[i].seg_count     = final_chunks[i].seg_count;
    }

    *out_chunks = out;
    *out_count  = n;
    return OMNI_OK;
}

}  // extern "C"
