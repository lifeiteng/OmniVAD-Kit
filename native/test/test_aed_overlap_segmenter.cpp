/*
 * Test: AED overlap segmenter C API.
 *
 * Usage: test_aed_overlap_segmenter <aed.omnivad> <wav_file>
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<int16_t> load_pcm16(const char* wav_file) {
    vad::WavReader reader;
    if (!reader.Open(wav_file)) {
        fprintf(stderr, "Failed to open WAV: %s\n", wav_file);
        std::exit(1);
    }
    std::vector<float> mono = reader.GetMonoData();
    std::vector<int16_t> pcm(mono.size());
    for (size_t i = 0; i < mono.size(); ++i) {
        float v = mono[i];
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }
    return pcm;
}

static bool run_with_step(
    OmniAedOverlapSegmenterHandle segmenter,
    const std::vector<int16_t>& pcm,
    int step_samples,
    std::vector<OmniAedOnlineSegment>& out_segments)
{
    for (size_t start = 0; start < pcm.size(); start += (size_t)step_samples) {
        size_t end = start + (size_t)step_samples;
        if (end > pcm.size()) end = pcm.size();

        OmniAedOnlineSegment* segments = NULL;
        OmniAedOnlineEvent* events = NULL;
        int segment_count = 0;
        int event_count = 0;
        int ret = omni_aed_overlap_segmenter_ingest_int16(
            segmenter,
            pcm.data() + start,
            (int)(end - start),
            &segments,
            &segment_count,
            &events,
            &event_count);
        if (ret != OMNI_OK) {
            fprintf(stderr, "ingest failed: %s\n", omni_error_string(ret));
            return false;
        }
        for (int i = 0; i < segment_count; ++i) {
            if (segments[i].event_start_idx < 0 ||
                segments[i].event_start_idx + segments[i].event_count > event_count ||
                segments[i].event_count <= 0) {
                fprintf(stderr, "invalid event index range from ingest\n");
                omni_free(segments);
                omni_free(events);
                return false;
            }
            for (int j = 0; j < segments[i].event_count; ++j) {
                const OmniAedOnlineEvent& ev = events[segments[i].event_start_idx + j];
                if (ev.start < segments[i].start || ev.end > segments[i].end || ev.end <= ev.start) {
                    fprintf(stderr, "event outside segment bounds from ingest\n");
                    omni_free(segments);
                    omni_free(events);
                    return false;
                }
            }
        }
        for (int i = 0; i < segment_count; ++i) {
            out_segments.push_back(segments[i]);
        }
        omni_free(segments);
        omni_free(events);
    }

    OmniAedOnlineSegment* segments = NULL;
    OmniAedOnlineEvent* events = NULL;
    int segment_count = 0;
    int event_count = 0;
    int ret = omni_aed_overlap_segmenter_flush(segmenter, &segments, &segment_count, &events, &event_count);
    if (ret != OMNI_OK) {
        fprintf(stderr, "flush failed: %s\n", omni_error_string(ret));
        return false;
    }
    for (int i = 0; i < segment_count; ++i) {
        if (segments[i].event_start_idx < 0 ||
            segments[i].event_start_idx + segments[i].event_count > event_count ||
            segments[i].event_count <= 0) {
            fprintf(stderr, "invalid event index range from flush\n");
            omni_free(segments);
            omni_free(events);
            return false;
        }
        for (int j = 0; j < segments[i].event_count; ++j) {
            const OmniAedOnlineEvent& ev = events[segments[i].event_start_idx + j];
            if (ev.start < segments[i].start || ev.end > segments[i].end || ev.end <= ev.start) {
                fprintf(stderr, "event outside segment bounds from flush\n");
                omni_free(segments);
                omni_free(events);
                return false;
            }
        }
    }
    for (int i = 0; i < segment_count; ++i) {
        out_segments.push_back(segments[i]);
    }
    omni_free(segments);
    omni_free(events);
    return true;
}

static bool run_tiny_silence_case(OmniAedOverlapSegmenterHandle segmenter, int num_samples) {
    std::vector<int16_t> pcm((size_t)num_samples, 0);
    OmniAedOnlineSegment* segments = NULL;
    OmniAedOnlineEvent* events = NULL;
    int segment_count = 0;
    int event_count = 0;

    int split = num_samples > 120 ? 120 : num_samples;
    int ret = omni_aed_overlap_segmenter_ingest_int16(
        segmenter, pcm.data(), split, &segments, &segment_count, &events, &event_count);
    if (ret != OMNI_OK || segment_count != 0 || event_count != 0) {
        fprintf(stderr, "tiny ingest failed for %d samples: %s\n", num_samples, omni_error_string(ret));
        omni_free(segments);
        omni_free(events);
        return false;
    }
    omni_free(segments);
    omni_free(events);

    if (split < num_samples) {
        ret = omni_aed_overlap_segmenter_ingest_int16(
            segmenter,
            pcm.data() + split,
            num_samples - split,
            &segments,
            &segment_count,
            &events,
            &event_count);
        if (ret != OMNI_OK || segment_count != 0 || event_count != 0) {
            fprintf(stderr, "tiny second ingest failed for %d samples: %s\n", num_samples, omni_error_string(ret));
            omni_free(segments);
            omni_free(events);
            return false;
        }
        omni_free(segments);
        omni_free(events);
    }

    ret = omni_aed_overlap_segmenter_flush(segmenter, &segments, &segment_count, &events, &event_count);
    if (ret != OMNI_OK || segment_count != 0 || event_count != 0) {
        fprintf(stderr, "tiny flush failed for %d samples: %s\n", num_samples, omni_error_string(ret));
        omni_free(segments);
        omni_free(events);
        return false;
    }
    omni_free(segments);
    omni_free(events);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <aed.omnivad> <wav_file>\n", argv[0]);
        return 1;
    }

    OmniAedOverlapConfig cfg = omni_aed_overlap_config_default();
    cfg.hop_ms = 500;
    cfg.overlap_ms = 100;
    cfg.hard_split_pause_ms = 200;
    cfg.max_chunk_ms = 2000;

    OmniAedOverlapConfig bad = cfg;
    bad.hop_ms = 505;
    int err = OMNI_OK;
    OmniAedOverlapSegmenterHandle bad_handle = omni_aed_overlap_segmenter_create(argv[1], &bad, &err);
    if (bad_handle || err != OMNI_ERR_INVALID_ARG) {
        fprintf(stderr, "invalid config was not rejected\n");
        if (bad_handle) omni_aed_overlap_segmenter_destroy(bad_handle);
        return 1;
    }
    bad = cfg;
    bad.edge_guard_ms = cfg.hop_ms;
    bad_handle = omni_aed_overlap_segmenter_create(argv[1], &bad, &err);
    if (bad_handle || err != OMNI_ERR_INVALID_ARG) {
        fprintf(stderr, "invalid edge guard was not rejected\n");
        if (bad_handle) omni_aed_overlap_segmenter_destroy(bad_handle);
        return 1;
    }
    bad = cfg;
    bad.hard_split_lookahead_ms = -1;
    bad_handle = omni_aed_overlap_segmenter_create(argv[1], &bad, &err);
    if (bad_handle || err != OMNI_ERR_INVALID_ARG) {
        fprintf(stderr, "invalid hard split lookahead was not rejected\n");
        if (bad_handle) omni_aed_overlap_segmenter_destroy(bad_handle);
        return 1;
    }

    std::vector<int16_t> pcm = load_pcm16(argv[2]);

    int create_err = OMNI_OK;
    OmniAedOverlapSegmenterHandle segmenter = omni_aed_overlap_segmenter_create(argv[1], &cfg, &create_err);
    if (!segmenter) {
        fprintf(stderr, "create failed: %s\n", omni_error_string(create_err));
        return 1;
    }

    const int tiny_cases[] = {0, 1, 120, 399, 400, 500, 559, 560, 600, 16000, 24000, 32000};
    for (size_t i = 0; i < sizeof(tiny_cases) / sizeof(tiny_cases[0]); ++i) {
        omni_aed_overlap_segmenter_reset(segmenter);
        if (!run_tiny_silence_case(segmenter, tiny_cases[i])) {
            omni_aed_overlap_segmenter_destroy(segmenter);
            return 1;
        }
    }
    omni_aed_overlap_segmenter_reset(segmenter);

    std::vector<OmniAedOnlineSegment> segments_a;
    if (!run_with_step(segmenter, pcm, 800, segments_a)) {
        omni_aed_overlap_segmenter_destroy(segmenter);
        return 1;
    }

    omni_aed_overlap_segmenter_reset(segmenter);
    std::vector<OmniAedOnlineSegment> segments_b;
    if (!run_with_step(segmenter, pcm, 5000, segments_b)) {
        omni_aed_overlap_segmenter_destroy(segmenter);
        return 1;
    }

    if (segments_a.size() != segments_b.size()) {
        fprintf(stderr, "segment count drift: %zu vs %zu\n", segments_a.size(), segments_b.size());
        omni_aed_overlap_segmenter_destroy(segmenter);
        return 1;
    }
    for (size_t i = 0; i < segments_a.size(); ++i) {
        if (segments_a[i].start != segments_b[i].start || segments_a[i].end != segments_b[i].end) {
            fprintf(stderr, "segment boundary drift at %zu\n", i);
            omni_aed_overlap_segmenter_destroy(segmenter);
            return 1;
        }
        if (i > 0 && segments_a[i - 1].end > segments_a[i].start) {
            fprintf(stderr, "segments are not monotonic\n");
            omni_aed_overlap_segmenter_destroy(segmenter);
            return 1;
        }
    }

    printf("AED overlap segmenter: %zu segments\n", segments_a.size());
    omni_aed_overlap_segmenter_destroy(segmenter);
    return 0;
}
