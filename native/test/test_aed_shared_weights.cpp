/*
 * Test: AED overlap segmenter clones share model weights.
 *
 * Covers:
 *   1) weight sharing   — cloning must not reload the model (RSS delta of many
 *                          clones is far below the delta of one create)
 *   2) concurrency      — N clones ingest different audio on N threads at the
 *                          same time; each output is byte-identical to a serial run
 *   3) config isolation — clone_with_config uses the passed config, not the
 *                          template's; two clones with different hard_split_pause_ms
 *                          each match a fresh create with their own config
 *   4) lifetime         — the template can be destroyed before its clones; the
 *                          clones keep working and release cleanly
 *   5) argument checks  — clone_with_config rejects NULL handle / invalid config
 *
 * Usage: test_aed_shared_weights <aed.omnivad> <test_data_dir> [threads] [rounds]
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

static const int kStepSamples = 16000; /* 1 s per ingest, like the desktop host */

/* Returns resident set size in bytes, or 0 when the platform is unsupported. */
static size_t current_rss_bytes(void) {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (size_t)info.resident_size;
#elif defined(__linux__)
    FILE* fp = fopen("/proc/self/statm", "r");
    if (!fp) return 0;
    long pages_total = 0, pages_resident = 0;
    int n = fscanf(fp, "%ld %ld", &pages_total, &pages_resident);
    fclose(fp);
    if (n != 2) return 0;
    return (size_t)pages_resident * (size_t)sysconf(_SC_PAGESIZE);
#else
    return 0;
#endif
}

static bool load_pcm16(const std::string& path, std::vector<int16_t>& pcm) {
    vad::WavReader reader;
    if (!reader.Open(path)) {
        fprintf(stderr, "Failed to open WAV: %s\n", path.c_str());
        return false;
    }
    std::vector<float> mono = reader.GetMonoData();
    pcm.resize(mono.size());
    for (size_t i = 0; i < mono.size(); ++i) {
        float v = mono[i];
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }
    return true;
}

template <typename T>
static void append_bytes(std::vector<unsigned char>& out, const T& value) {
    const unsigned char* p = (const unsigned char*)&value;
    out.insert(out.end(), p, p + sizeof(T));
}

static void append_outputs(std::vector<unsigned char>& out,
                           const OmniAedOnlineSegment* segments, int segment_count,
                           const OmniAedOnlineEvent* events, int event_count) {
    /* Field-wise so struct padding never leaks into the comparison. */
    append_bytes(out, segment_count);
    for (int i = 0; i < segment_count; ++i) {
        append_bytes(out, segments[i].start);
        append_bytes(out, segments[i].end);
        append_bytes(out, segments[i].event_start_idx);
        append_bytes(out, segments[i].event_count);
    }
    append_bytes(out, event_count);
    for (int i = 0; i < event_count; ++i) {
        append_bytes(out, events[i].start);
        append_bytes(out, events[i].end);
        append_bytes(out, (int)events[i].primary_kind);
        append_bytes(out, events[i].kind_mask);
        append_bytes(out, events[i].speech_confidence);
        append_bytes(out, events[i].singing_confidence);
        append_bytes(out, events[i].music_confidence);
        append_bytes(out, events[i].confidence);
    }
}

/* Ingest the whole clip in 1 s steps, flush, and return a byte transcript of
 * every segment/event emitted along the way. Empty vector on API failure. */
static std::vector<unsigned char> run_session(OmniAedOverlapSegmenterHandle h,
                                              const std::vector<int16_t>& pcm,
                                              int* out_segment_total) {
    std::vector<unsigned char> out;
    int segment_total = 0;
    for (size_t start = 0; start < pcm.size(); start += (size_t)kStepSamples) {
        size_t end = start + (size_t)kStepSamples;
        if (end > pcm.size()) end = pcm.size();
        OmniAedOnlineSegment* segments = NULL;
        OmniAedOnlineEvent* events = NULL;
        int segment_count = 0, event_count = 0;
        int ret = omni_aed_overlap_segmenter_ingest_int16(
            h, pcm.data() + start, (int)(end - start), &segments, &segment_count, &events, &event_count);
        if (ret != OMNI_OK) {
            fprintf(stderr, "ingest failed: %s\n", omni_error_string(ret));
            omni_free(segments);
            omni_free(events);
            return std::vector<unsigned char>();
        }
        append_outputs(out, segments, segment_count, events, event_count);
        segment_total += segment_count;
        omni_free(segments);
        omni_free(events);
    }
    OmniAedOnlineSegment* segments = NULL;
    OmniAedOnlineEvent* events = NULL;
    int segment_count = 0, event_count = 0;
    int ret = omni_aed_overlap_segmenter_flush(h, &segments, &segment_count, &events, &event_count);
    if (ret != OMNI_OK) {
        fprintf(stderr, "flush failed: %s\n", omni_error_string(ret));
        omni_free(segments);
        omni_free(events);
        return std::vector<unsigned char>();
    }
    append_outputs(out, segments, segment_count, events, event_count);
    segment_total += segment_count;
    omni_free(segments);
    omni_free(events);
    out.push_back(0xA5); /* terminator: distinguishes "ran, no output" from failure */
    if (out_segment_total) *out_segment_total = segment_total;
    return out;
}

static std::vector<unsigned char> run_fresh(const char* bundle, const OmniAedOverlapConfig& cfg,
                                            const std::vector<int16_t>& pcm, int* out_segment_total) {
    int err = OMNI_OK;
    OmniAedOverlapSegmenterHandle h = omni_aed_overlap_segmenter_create(bundle, &cfg, &err);
    if (!h) {
        fprintf(stderr, "create failed: %s\n", omni_error_string(err));
        return std::vector<unsigned char>();
    }
    std::vector<unsigned char> out = run_session(h, pcm, out_segment_total);
    omni_aed_overlap_segmenter_destroy(h);
    return out;
}

static OmniAedOverlapConfig base_config(void) {
    OmniAedOverlapConfig cfg = omni_aed_overlap_config_default();
    cfg.max_chunk_ms = 60000;
    cfg.hard_split_lookahead_ms = 0;
    return cfg;
}

/* ---- 1) weight sharing ---- */

static bool test_weight_sharing(const char* bundle) {
    const int kClones = 16;
    OmniAedOverlapConfig cfg = base_config();

    int err = OMNI_OK;
    /* Warm-up create: absorbs one-time process costs (ncnn layer registry,
     * allocator arenas) so they are not charged to the measured create. */
    OmniAedOverlapSegmenterHandle warm = omni_aed_overlap_segmenter_create(bundle, &cfg, &err);
    if (!warm) {
        fprintf(stderr, "warm-up create failed: %s\n", omni_error_string(err));
        return false;
    }

    size_t rss0 = current_rss_bytes();
    OmniAedOverlapSegmenterHandle tmpl = omni_aed_overlap_segmenter_create(bundle, &cfg, &err);
    size_t rss1 = current_rss_bytes();
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        omni_aed_overlap_segmenter_destroy(warm);
        return false;
    }

    std::vector<OmniAedOverlapSegmenterHandle> clones;
    for (int i = 0; i < kClones; ++i) {
        OmniAedOverlapSegmenterHandle c = (i % 2 == 0)
            ? omni_aed_overlap_segmenter_clone(tmpl, &err)
            : omni_aed_overlap_segmenter_clone_with_config(tmpl, &cfg, &err);
        if (!c) {
            fprintf(stderr, "clone %d failed: %s\n", i, omni_error_string(err));
            break;
        }
        clones.push_back(c);
    }
    size_t rss2 = current_rss_bytes();

    bool ok = (int)clones.size() == kClones;
    if (ok && rss0 == 0) {
        printf("  weight sharing: RSS unavailable on this platform, skipped\n");
    } else if (ok) {
        double create_mb = (double)(rss1 > rss0 ? rss1 - rss0 : 0) / (1024.0 * 1024.0);
        double clones_mb = (double)(rss2 > rss1 ? rss2 - rss1 : 0) / (1024.0 * 1024.0);
        printf("  weight sharing: one create +%.2f MB, %d clones +%.2f MB (%.3f MB/clone)\n",
               create_mb, kClones, clones_mb, clones_mb / kClones);
        /* A clone that reloads the model costs about one create each, so 16 of
         * them land at ~16x the create delta. Shared weights keep the whole
         * batch below half of a single create. */
        if (create_mb < 0.5) {
            fprintf(stderr, "  create delta %.2f MB too small to judge sharing\n", create_mb);
            ok = false;
        } else if (clones_mb >= create_mb * 0.5) {
            fprintf(stderr, "  %d clones cost %.2f MB >= half of one create (%.2f MB): weights are not shared\n",
                    kClones, clones_mb, create_mb);
            ok = false;
        }
    }

    for (size_t i = 0; i < clones.size(); ++i) omni_aed_overlap_segmenter_destroy(clones[i]);
    omni_aed_overlap_segmenter_destroy(tmpl);
    omni_aed_overlap_segmenter_destroy(warm);
    return ok;
}

/* ---- 2) concurrency ---- */

static bool test_concurrent_clones(const char* bundle, const std::vector<std::vector<int16_t> >& clips,
                                   int threads, int rounds) {
    OmniAedOverlapConfig cfg = base_config();
    int err = OMNI_OK;
    OmniAedOverlapSegmenterHandle tmpl = omni_aed_overlap_segmenter_create(bundle, &cfg, &err);
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        return false;
    }

    /* Serial reference: each clip on its own fresh clone, one at a time. */
    std::vector<std::vector<unsigned char> > expected((size_t)threads);
    for (int i = 0; i < threads; ++i) {
        OmniAedOverlapSegmenterHandle c = omni_aed_overlap_segmenter_clone(tmpl, &err);
        if (!c) {
            fprintf(stderr, "reference clone failed: %s\n", omni_error_string(err));
            omni_aed_overlap_segmenter_destroy(tmpl);
            return false;
        }
        expected[(size_t)i] = run_session(c, clips[(size_t)i % clips.size()], NULL);
        omni_aed_overlap_segmenter_destroy(c);
        if (expected[(size_t)i].empty()) {
            omni_aed_overlap_segmenter_destroy(tmpl);
            return false;
        }
    }

    bool ok = true;
    for (int round = 0; round < rounds && ok; ++round) {
        std::vector<OmniAedOverlapSegmenterHandle> handles((size_t)threads, NULL);
        for (int i = 0; i < threads; ++i) {
            handles[(size_t)i] = omni_aed_overlap_segmenter_clone_with_config(tmpl, &cfg, &err);
            if (!handles[(size_t)i]) {
                fprintf(stderr, "clone failed: %s\n", omni_error_string(err));
                ok = false;
            }
        }
        std::vector<std::vector<unsigned char> > actual((size_t)threads);
        if (ok) {
            std::vector<std::thread> workers;
            for (int i = 0; i < threads; ++i) {
                workers.push_back(std::thread([&, i]() {
                    actual[(size_t)i] = run_session(handles[(size_t)i], clips[(size_t)i % clips.size()], NULL);
                }));
            }
            for (size_t i = 0; i < workers.size(); ++i) workers[i].join();
            for (int i = 0; i < threads; ++i) {
                if (actual[(size_t)i] != expected[(size_t)i]) {
                    fprintf(stderr, "  round %d handle %d: concurrent output differs from serial (%zu vs %zu bytes)\n",
                            round, i, actual[(size_t)i].size(), expected[(size_t)i].size());
                    ok = false;
                }
            }
        }
        for (size_t i = 0; i < handles.size(); ++i) omni_aed_overlap_segmenter_destroy(handles[i]);
    }
    omni_aed_overlap_segmenter_destroy(tmpl);
    if (ok) printf("  concurrency: %d threads x %d rounds byte-identical to serial\n", threads, rounds);
    return ok;
}

/* ---- 3) config isolation + legacy clone semantics ---- */

static bool test_config_isolation(const char* bundle, const std::vector<int16_t>& pcm) {
    OmniAedOverlapConfig cfg_a = base_config();
    cfg_a.hard_split_pause_ms = 200;
    OmniAedOverlapConfig cfg_b = base_config();
    cfg_b.hard_split_pause_ms = 800;
    OmniAedOverlapConfig cfg_t = base_config(); /* template: default 2000 ms pause */

    int seg_a = 0, seg_b = 0, seg_t = 0;
    std::vector<unsigned char> ref_a = run_fresh(bundle, cfg_a, pcm, &seg_a);
    std::vector<unsigned char> ref_b = run_fresh(bundle, cfg_b, pcm, &seg_b);
    std::vector<unsigned char> ref_t = run_fresh(bundle, cfg_t, pcm, &seg_t);
    if (ref_a.empty() || ref_b.empty() || ref_t.empty()) return false;
    printf("  config isolation: fresh create segments pause=200ms:%d 800ms:%d 2000ms:%d\n", seg_a, seg_b, seg_t);
    if (ref_a == ref_b || ref_b == ref_t) {
        fprintf(stderr, "  test audio does not separate 200/800/2000 ms pauses; test is not discriminating\n");
        return false;
    }

    int err = OMNI_OK;
    OmniAedOverlapSegmenterHandle tmpl = omni_aed_overlap_segmenter_create(bundle, &cfg_t, &err);
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        return false;
    }
    OmniAedOverlapSegmenterHandle a = omni_aed_overlap_segmenter_clone_with_config(tmpl, &cfg_a, &err);
    OmniAedOverlapSegmenterHandle b = omni_aed_overlap_segmenter_clone_with_config(tmpl, &cfg_b, &err);
    OmniAedOverlapSegmenterHandle legacy = omni_aed_overlap_segmenter_clone(tmpl, &err);
    bool ok = a && b && legacy;
    if (!ok) fprintf(stderr, "  clone failed: %s\n", omni_error_string(err));

    /* Interleave A and B step by step on one thread so any state shared
     * between clones would bleed across immediately. */
    std::vector<unsigned char> out_a, out_b;
    if (ok) {
        std::thread ta([&]() { out_a = run_session(a, pcm, NULL); });
        std::thread tb([&]() { out_b = run_session(b, pcm, NULL); });
        ta.join();
        tb.join();
        if (out_a != ref_a) {
            fprintf(stderr, "  clone A (200 ms) does not match a fresh create with 200 ms\n");
            ok = false;
        }
        if (out_b != ref_b) {
            fprintf(stderr, "  clone B (800 ms) does not match a fresh create with 800 ms\n");
            ok = false;
        }
        std::vector<unsigned char> out_legacy = run_session(legacy, pcm, NULL);
        if (out_legacy != ref_t) {
            fprintf(stderr, "  legacy clone does not reproduce the template's config\n");
            ok = false;
        }
        /* Template's own config must be untouched by clone_with_config. */
        std::vector<unsigned char> out_tmpl = run_session(tmpl, pcm, NULL);
        if (out_tmpl != ref_t) {
            fprintf(stderr, "  template output changed after clone_with_config\n");
            ok = false;
        }
    }

    /* Legacy clone of a clone copies that clone's (B) config. */
    if (ok) {
        OmniAedOverlapSegmenterHandle bb = omni_aed_overlap_segmenter_clone(b, &err);
        if (!bb) {
            fprintf(stderr, "  clone of clone failed: %s\n", omni_error_string(err));
            ok = false;
        } else {
            if (run_session(bb, pcm, NULL) != ref_b) {
                fprintf(stderr, "  legacy clone of B does not reproduce B's config\n");
                ok = false;
            }
            omni_aed_overlap_segmenter_destroy(bb);
        }
    }

    omni_aed_overlap_segmenter_destroy(legacy);
    omni_aed_overlap_segmenter_destroy(b);
    omni_aed_overlap_segmenter_destroy(a);
    omni_aed_overlap_segmenter_destroy(tmpl);
    return ok;
}

/* ---- 4) template destroyed first ---- */

static bool test_template_outlived(const char* bundle, const std::vector<int16_t>& pcm) {
    OmniAedOverlapConfig cfg_t = base_config();
    OmniAedOverlapConfig cfg_a = base_config();
    cfg_a.hard_split_pause_ms = 200;
    std::vector<unsigned char> ref_t = run_fresh(bundle, cfg_t, pcm, NULL);
    std::vector<unsigned char> ref_a = run_fresh(bundle, cfg_a, pcm, NULL);
    if (ref_t.empty() || ref_a.empty()) return false;

    int err = OMNI_OK;
    OmniAedOverlapSegmenterHandle tmpl = omni_aed_overlap_segmenter_create(bundle, &cfg_t, &err);
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        return false;
    }
    OmniAedOverlapSegmenterHandle a = omni_aed_overlap_segmenter_clone_with_config(tmpl, &cfg_a, &err);
    OmniAedOverlapSegmenterHandle legacy = omni_aed_overlap_segmenter_clone(tmpl, &err);
    omni_aed_overlap_segmenter_destroy(tmpl);
    tmpl = NULL;

    bool ok = a && legacy;
    if (!ok) fprintf(stderr, "  clone failed: %s\n", omni_error_string(err));
    if (ok && run_session(a, pcm, NULL) != ref_a) {
        fprintf(stderr, "  clone_with_config output changed after template destroy\n");
        ok = false;
    }
    if (ok && run_session(legacy, pcm, NULL) != ref_t) {
        fprintf(stderr, "  legacy clone output changed after template destroy\n");
        ok = false;
    }
    /* A clone must itself be a usable template once the original is gone. */
    if (ok) {
        OmniAedOverlapSegmenterHandle again = omni_aed_overlap_segmenter_clone_with_config(legacy, &cfg_a, &err);
        if (!again || run_session(again, pcm, NULL) != ref_a) {
            fprintf(stderr, "  re-clone after template destroy failed\n");
            ok = false;
        }
        omni_aed_overlap_segmenter_destroy(again);
    }
    omni_aed_overlap_segmenter_destroy(a);
    omni_aed_overlap_segmenter_destroy(legacy);
    return ok;
}

/* ---- 5) argument checks ---- */

static bool test_argument_checks(const char* bundle) {
    int err = OMNI_OK;
    OmniAedOverlapConfig cfg = base_config();
    if (omni_aed_overlap_segmenter_clone_with_config(NULL, &cfg, &err) || err != OMNI_ERR_NULL_HANDLE) {
        fprintf(stderr, "  NULL handle not rejected\n");
        return false;
    }
    OmniAedOverlapSegmenterHandle tmpl = omni_aed_overlap_segmenter_create(bundle, &cfg, &err);
    if (!tmpl) return false;
    bool ok = true;
    OmniAedOverlapConfig bad = cfg;
    bad.hop_ms = 505;
    OmniAedOverlapSegmenterHandle h = omni_aed_overlap_segmenter_clone_with_config(tmpl, &bad, &err);
    if (h || err != OMNI_ERR_INVALID_ARG) {
        fprintf(stderr, "  invalid config not rejected\n");
        omni_aed_overlap_segmenter_destroy(h);
        ok = false;
    }
    /* NULL config means library defaults, same as create. */
    h = omni_aed_overlap_segmenter_clone_with_config(tmpl, NULL, &err);
    if (!h || err != OMNI_OK) {
        fprintf(stderr, "  NULL config (defaults) rejected: %s\n", omni_error_string(err));
        ok = false;
    }
    omni_aed_overlap_segmenter_destroy(h);
    omni_aed_overlap_segmenter_destroy(tmpl);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <aed.omnivad> <test_data_dir> [threads] [rounds]\n", argv[0]);
        return 1;
    }
    const char* bundle = argv[1];
    std::string data_dir = argv[2];
    int threads = argc > 3 ? atoi(argv[3]) : 8;
    int rounds = argc > 4 ? atoi(argv[4]) : 3;
    if (threads <= 0) threads = 8;
    if (rounds <= 0) rounds = 3;

    const char* names[] = {"en_medium.wav", "zh_medium.wav", "event.wav", "hello_en.wav", "hello_zh.wav"};
    std::vector<std::vector<int16_t> > clips;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        std::vector<int16_t> pcm;
        if (!load_pcm16(data_dir + "/" + names[i], pcm)) return 1;
        clips.push_back(pcm);
    }
    /* More distinct inputs than files: rotated copies of the longer clips. */
    const size_t base_count = clips.size();
    for (size_t i = 0; clips.size() < (size_t)threads; ++i) {
        std::vector<int16_t> rotated = clips[i % base_count];
        size_t shift = (rotated.size() / 3 + 7919 * (i + 1)) % rotated.size();
        std::rotate(rotated.begin(), rotated.begin() + (long)shift, rotated.end());
        clips.push_back(rotated);
    }

    struct Case { const char* name; bool ok; };
    std::vector<Case> results;
    results.push_back(Case{"weight_sharing", test_weight_sharing(bundle)});
    results.push_back(Case{"concurrent_clones", test_concurrent_clones(bundle, clips, threads, rounds)});
    results.push_back(Case{"config_isolation", test_config_isolation(bundle, clips[0])});
    results.push_back(Case{"template_outlived", test_template_outlived(bundle, clips[0])});
    results.push_back(Case{"argument_checks", test_argument_checks(bundle)});

    bool all = true;
    for (size_t i = 0; i < results.size(); ++i) {
        printf("%s %s\n", results[i].ok ? "PASS" : "FAIL", results[i].name);
        all = all && results[i].ok;
    }
    return all ? 0 : 1;
}
