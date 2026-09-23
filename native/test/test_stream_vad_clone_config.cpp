/*
 * Test: omni_stream_vad_clone_with_config.
 *
 * Covers:
 *   1) weight sharing   — clone_with_config must not reload the model (RSS
 *                          delta of many clones is far below that of as many creates)
 *   2) config isolation — clones with different threshold / min_silence_frame
 *                          each match a fresh create with their own config; the
 *                          template and legacy omni_stream_vad_clone keep theirs
 *   3) config rejection — every config create rejects, clone_with_config rejects
 *                          with the same error; NULL config means defaults
 *   4) lifetime         — the template can be destroyed before its clones
 *   5) concurrency      — N clones stream different audio on N threads at the
 *                          same time; each output is byte-identical to a serial run
 *
 * Usage: test_stream_vad_clone_config <stream-vad.omnivad> <test_data_dir> [threads] [rounds]
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <algorithm>
#include <cmath>
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

static const int kChunkSamples = 160; /* 10 ms per process call */

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

struct SessionStats {
    int starts;
    int ends;
};

/* Stream the clip in 10 ms chunks and return a byte transcript of every
 * per-frame result (field-wise, so struct padding never leaks in). Empty
 * vector on API failure. */
static std::vector<unsigned char> run_session(OmniStreamVadHandle h, const std::vector<int16_t>& pcm,
                                              SessionStats* stats) {
    std::vector<unsigned char> out;
    SessionStats s = {0, 0};
    for (size_t start = 0; start + (size_t)kChunkSamples <= pcm.size(); start += (size_t)kChunkSamples) {
        OmniStreamVadResult r;
        int ret = omni_stream_vad_process_int16(h, pcm.data() + start, kChunkSamples, &r);
        if (ret == OMNI_ERR_NO_FRAMES) {
            append_bytes(out, ret);
            continue;
        }
        if (ret != OMNI_OK) {
            fprintf(stderr, "process failed: %s\n", omni_error_string(ret));
            return std::vector<unsigned char>();
        }
        append_bytes(out, r.confidence);
        append_bytes(out, r.smoothed_prob);
        append_bytes(out, (int)r.is_speech);
        append_bytes(out, (int)r.is_speech_start);
        append_bytes(out, (int)r.is_speech_end);
        append_bytes(out, r.frame_idx);
        append_bytes(out, r.speech_start_frame);
        append_bytes(out, r.speech_end_frame);
        if (r.is_speech_start) s.starts++;
        if (r.is_speech_end) s.ends++;
    }
    out.push_back(0xA5); /* terminator: distinguishes "ran, no output" from failure */
    if (stats) *stats = s;
    return out;
}

static std::vector<unsigned char> run_fresh(const char* bundle, const OmniStreamVadConfig* cfg,
                                            const std::vector<int16_t>& pcm, SessionStats* stats) {
    int err = OMNI_OK;
    OmniStreamVadHandle h = omni_stream_vad_create(bundle, cfg, &err);
    if (!h) {
        fprintf(stderr, "create failed: %s\n", omni_error_string(err));
        return std::vector<unsigned char>();
    }
    std::vector<unsigned char> out = run_session(h, pcm, stats);
    omni_stream_vad_destroy(h);
    return out;
}

static OmniStreamVadConfig config_with(float threshold, int min_silence_frame) {
    OmniStreamVadConfig c = omni_stream_vad_config_default();
    c.threshold = threshold;
    c.min_silence_frame = min_silence_frame;
    return c;
}

/* ---- 1) weight sharing ---- */

static bool test_weight_sharing(const char* bundle) {
    const int kHandles = 16;
    OmniStreamVadConfig tmpl_cfg = omni_stream_vad_config_default();
    OmniStreamVadConfig clone_cfg = config_with(0.3f, 10);

    int err = OMNI_OK;
    /* Warm-up create: absorbs one-time process costs (ncnn layer registry,
     * allocator arenas) so they are not charged to the measurement. */
    OmniStreamVadHandle warm = omni_stream_vad_create(bundle, &tmpl_cfg, &err);
    if (!warm) {
        fprintf(stderr, "warm-up create failed: %s\n", omni_error_string(err));
        return false;
    }
    OmniStreamVadHandle tmpl = omni_stream_vad_create(bundle, &tmpl_cfg, &err);
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        omni_stream_vad_destroy(warm);
        return false;
    }

    size_t rss0 = current_rss_bytes();
    std::vector<OmniStreamVadHandle> clones;
    for (int i = 0; i < kHandles; ++i) {
        OmniStreamVadHandle c = omni_stream_vad_clone_with_config(tmpl, &clone_cfg, &err);
        if (!c) {
            fprintf(stderr, "clone %d failed: %s\n", i, omni_error_string(err));
            break;
        }
        clones.push_back(c);
    }
    size_t rss1 = current_rss_bytes();
    std::vector<OmniStreamVadHandle> creates;
    for (int i = 0; i < kHandles; ++i) {
        OmniStreamVadHandle c = omni_stream_vad_create(bundle, &clone_cfg, &err);
        if (!c) {
            fprintf(stderr, "create %d failed: %s\n", i, omni_error_string(err));
            break;
        }
        creates.push_back(c);
    }
    size_t rss2 = current_rss_bytes();

    bool ok = (int)clones.size() == kHandles && (int)creates.size() == kHandles;
    if (ok && rss0 == 0) {
        printf("  weight sharing: RSS unavailable on this platform, skipped\n");
    } else if (ok) {
        double clones_mb = (double)(rss1 > rss0 ? rss1 - rss0 : 0) / (1024.0 * 1024.0);
        double creates_mb = (double)(rss2 > rss1 ? rss2 - rss1 : 0) / (1024.0 * 1024.0);
        printf("  weight sharing: %d creates +%.2f MB (%.3f MB each), %d clone_with_config +%.2f MB (%.3f MB each)\n",
               kHandles, creates_mb, creates_mb / kHandles, kHandles, clones_mb, clones_mb / kHandles);
        /* A clone keeps only its runtime buffers (fbank, DFSMN cache); a create
         * also loads the net + CMVN. Shared weights keep the clones well under
         * a quarter of the creates. */
        if (creates_mb < 1.0) {
            fprintf(stderr, "  create delta %.2f MB too small to judge sharing\n", creates_mb);
            ok = false;
        } else if (clones_mb >= creates_mb * 0.25) {
            fprintf(stderr, "  %d clones cost %.2f MB >= a quarter of %d creates (%.2f MB): weights are not shared\n",
                    kHandles, clones_mb, kHandles, creates_mb);
            ok = false;
        }
    }

    for (size_t i = 0; i < creates.size(); ++i) omni_stream_vad_destroy(creates[i]);
    for (size_t i = 0; i < clones.size(); ++i) omni_stream_vad_destroy(clones[i]);
    omni_stream_vad_destroy(tmpl);
    omni_stream_vad_destroy(warm);
    return ok;
}

/* ---- 2) config isolation + legacy clone semantics ---- */

static bool test_config_isolation(const char* bundle, const std::vector<int16_t>& pcm) {
    OmniStreamVadConfig cfg_t = omni_stream_vad_config_default();  /* template */
    OmniStreamVadConfig cfg_thr = config_with(0.8f, cfg_t.min_silence_frame);  /* threshold only */
    OmniStreamVadConfig cfg_sil = config_with(cfg_t.threshold, 80);            /* min_silence only */

    SessionStats st_t, st_thr, st_sil;
    std::vector<unsigned char> ref_t = run_fresh(bundle, &cfg_t, pcm, &st_t);
    std::vector<unsigned char> ref_thr = run_fresh(bundle, &cfg_thr, pcm, &st_thr);
    std::vector<unsigned char> ref_sil = run_fresh(bundle, &cfg_sil, pcm, &st_sil);
    if (ref_t.empty() || ref_thr.empty() || ref_sil.empty()) return false;
    printf("  config isolation: fresh create segment ends default:%d threshold=0.8:%d min_silence=80:%d\n",
           st_t.ends, st_thr.ends, st_sil.ends);
    if (ref_t == ref_thr || ref_t == ref_sil || ref_thr == ref_sil || st_t.ends == st_sil.ends) {
        fprintf(stderr, "  test audio does not separate the configs; test is not discriminating\n");
        return false;
    }

    int err = OMNI_OK;
    OmniStreamVadHandle tmpl = omni_stream_vad_create(bundle, &cfg_t, &err);
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        return false;
    }
    OmniStreamVadHandle a = omni_stream_vad_clone_with_config(tmpl, &cfg_thr, &err);
    OmniStreamVadHandle b = omni_stream_vad_clone_with_config(tmpl, &cfg_sil, &err);
    OmniStreamVadHandle legacy = omni_stream_vad_clone(tmpl, &err);
    bool ok = a && b && legacy;
    if (!ok) fprintf(stderr, "  clone failed: %s\n", omni_error_string(err));

    if (ok) {
        std::vector<unsigned char> out_a, out_b;
        std::thread ta([&]() { out_a = run_session(a, pcm, NULL); });
        std::thread tb([&]() { out_b = run_session(b, pcm, NULL); });
        ta.join();
        tb.join();
        if (out_a != ref_thr) {
            fprintf(stderr, "  clone A (threshold=0.8) does not match a fresh create with threshold=0.8\n");
            ok = false;
        }
        if (out_b != ref_sil) {
            fprintf(stderr, "  clone B (min_silence=80) does not match a fresh create with min_silence=80\n");
            ok = false;
        }
        if (run_session(legacy, pcm, NULL) != ref_t) {
            fprintf(stderr, "  legacy clone does not reproduce the template's config\n");
            ok = false;
        }
        /* The template's own config must be untouched by clone_with_config. */
        if (run_session(tmpl, pcm, NULL) != ref_t) {
            fprintf(stderr, "  template output changed after clone_with_config\n");
            ok = false;
        }
    }

    /* Legacy clone of a clone copies that clone's (B) config. */
    if (ok) {
        OmniStreamVadHandle bb = omni_stream_vad_clone(b, &err);
        if (!bb || run_session(bb, pcm, NULL) != ref_sil) {
            fprintf(stderr, "  legacy clone of B does not reproduce B's config\n");
            ok = false;
        }
        omni_stream_vad_destroy(bb);
    }

    omni_stream_vad_destroy(legacy);
    omni_stream_vad_destroy(b);
    omni_stream_vad_destroy(a);
    omni_stream_vad_destroy(tmpl);
    return ok;
}

/* ---- 3) config rejection matches create ---- */

static bool test_config_rejection(const char* bundle, const std::vector<int16_t>& pcm) {
    int err = OMNI_OK;
    OmniStreamVadConfig def = omni_stream_vad_config_default();
    if (omni_stream_vad_clone_with_config(NULL, &def, &err) || err != OMNI_ERR_NULL_HANDLE) {
        fprintf(stderr, "  NULL handle not rejected with OMNI_ERR_NULL_HANDLE\n");
        return false;
    }
    OmniStreamVadHandle tmpl = omni_stream_vad_create(bundle, &def, &err);
    if (!tmpl) return false;

    struct Variant { const char* name; OmniStreamVadConfig cfg; };
    std::vector<Variant> variants;
    OmniStreamVadConfig c;
    /* Invalid values */
    c = def; c.threshold = -0.01f;        variants.push_back(Variant{"threshold=-0.01", c});
    c = def; c.threshold = 1.01f;         variants.push_back(Variant{"threshold=1.01", c});
    c = def; c.threshold = NAN;           variants.push_back(Variant{"threshold=NaN", c});
    c = def; c.smooth_window_size = 0;    variants.push_back(Variant{"smooth_window_size=0", c});
    c = def; c.pad_start_frame = -1;      variants.push_back(Variant{"pad_start_frame=-1", c});
    c = def; c.min_speech_frame = 0;      variants.push_back(Variant{"min_speech_frame=0", c});
    c = def; c.max_speech_frame = 0;      variants.push_back(Variant{"max_speech_frame=0", c});
    c = def; c.min_silence_frame = 0;     variants.push_back(Variant{"min_silence_frame=0", c});
    c = def; c.min_silence_frame = -5;    variants.push_back(Variant{"min_silence_frame=-5", c});
    /* Boundary values that create accepts */
    c = def; c.threshold = 0.0f;          variants.push_back(Variant{"threshold=0", c});
    c = def; c.threshold = 1.0f;          variants.push_back(Variant{"threshold=1", c});
    c = def; c.smooth_window_size = 1;    variants.push_back(Variant{"smooth_window_size=1", c});
    c = def; c.pad_start_frame = 0;       variants.push_back(Variant{"pad_start_frame=0", c});
    c = def; c.min_speech_frame = 1; c.max_speech_frame = 1; c.min_silence_frame = 1;
    variants.push_back(Variant{"all frame limits=1", c});

    bool ok = true;
    int rejected = 0;
    for (size_t i = 0; i < variants.size(); ++i) {
        int err_create = OMNI_OK, err_clone = OMNI_OK;
        OmniStreamVadHandle hc = omni_stream_vad_create(bundle, &variants[i].cfg, &err_create);
        OmniStreamVadHandle hk = omni_stream_vad_clone_with_config(tmpl, &variants[i].cfg, &err_clone);
        if ((hc == NULL) != (hk == NULL) || err_create != err_clone) {
            fprintf(stderr, "  %s: create -> %s (%s), clone_with_config -> %s (%s)\n", variants[i].name,
                    hc ? "handle" : "NULL", omni_error_string(err_create),
                    hk ? "handle" : "NULL", omni_error_string(err_clone));
            ok = false;
        }
        if (!hc) rejected++;
        omni_stream_vad_destroy(hc);
        omni_stream_vad_destroy(hk);
    }
    if (rejected != 9) {
        fprintf(stderr, "  expected create to reject 9 variants, got %d\n", rejected);
        ok = false;
    }

    /* NULL config means library defaults, same as create. */
    OmniStreamVadHandle h = omni_stream_vad_clone_with_config(tmpl, NULL, &err);
    if (!h || err != OMNI_OK) {
        fprintf(stderr, "  NULL config (defaults) rejected: %s\n", omni_error_string(err));
        ok = false;
    } else if (run_session(h, pcm, NULL) != run_fresh(bundle, NULL, pcm, NULL)) {
        fprintf(stderr, "  NULL config clone does not match create with NULL config\n");
        ok = false;
    }
    omni_stream_vad_destroy(h);
    omni_stream_vad_destroy(tmpl);
    if (ok) printf("  config rejection: %zu variants agree with create (%d rejected)\n", variants.size(), rejected);
    return ok;
}

/* ---- 4) template destroyed first ---- */

static bool test_template_outlived(const char* bundle, const std::vector<int16_t>& pcm) {
    OmniStreamVadConfig cfg_t = omni_stream_vad_config_default();
    OmniStreamVadConfig cfg_a = config_with(0.8f, 80);
    std::vector<unsigned char> ref_t = run_fresh(bundle, &cfg_t, pcm, NULL);
    std::vector<unsigned char> ref_a = run_fresh(bundle, &cfg_a, pcm, NULL);
    if (ref_t.empty() || ref_a.empty()) return false;

    int err = OMNI_OK;
    OmniStreamVadHandle tmpl = omni_stream_vad_create(bundle, &cfg_t, &err);
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        return false;
    }
    OmniStreamVadHandle a = omni_stream_vad_clone_with_config(tmpl, &cfg_a, &err);
    OmniStreamVadHandle legacy = omni_stream_vad_clone(tmpl, &err);
    omni_stream_vad_destroy(tmpl);
    tmpl = NULL;

    bool ok = a && legacy;
    if (!ok) fprintf(stderr, "  clone failed: %s\n", omni_error_string(err));
    if (ok && run_session(a, pcm, NULL) != ref_a) {
        fprintf(stderr, "  clone A output changed after template destroy\n");
        ok = false;
    }
    if (ok && run_session(legacy, pcm, NULL) != ref_t) {
        fprintf(stderr, "  legacy clone output changed after template destroy\n");
        ok = false;
    }
    /* A clone must itself be a usable template once the original is gone. */
    if (ok) {
        OmniStreamVadHandle again = omni_stream_vad_clone_with_config(legacy, &cfg_a, &err);
        if (!again || run_session(again, pcm, NULL) != ref_a) {
            fprintf(stderr, "  re-clone after template destroy failed\n");
            ok = false;
        }
        omni_stream_vad_destroy(again);
    }
    omni_stream_vad_destroy(a);
    omni_stream_vad_destroy(legacy);
    return ok;
}

/* ---- 5) concurrency ---- */

static bool test_concurrent_clones(const char* bundle, const std::vector<std::vector<int16_t> >& clips,
                                   int threads, int rounds) {
    /* Each thread gets its own config so config and state isolation are both exercised. */
    std::vector<OmniStreamVadConfig> cfgs((size_t)threads);
    for (int i = 0; i < threads; ++i) {
        cfgs[(size_t)i] = config_with(0.3f + 0.1f * (float)(i % 5), 10 + 10 * (i % 4));
    }

    int err = OMNI_OK;
    OmniStreamVadHandle tmpl = omni_stream_vad_create(bundle, NULL, &err);
    if (!tmpl) {
        fprintf(stderr, "template create failed: %s\n", omni_error_string(err));
        return false;
    }

    /* Serial reference: each clip on its own fresh clone, one at a time. */
    std::vector<std::vector<unsigned char> > expected((size_t)threads);
    for (int i = 0; i < threads; ++i) {
        OmniStreamVadHandle c = omni_stream_vad_clone_with_config(tmpl, &cfgs[(size_t)i], &err);
        if (!c) {
            fprintf(stderr, "reference clone failed: %s\n", omni_error_string(err));
            omni_stream_vad_destroy(tmpl);
            return false;
        }
        expected[(size_t)i] = run_session(c, clips[(size_t)i % clips.size()], NULL);
        omni_stream_vad_destroy(c);
        if (expected[(size_t)i].empty()) {
            omni_stream_vad_destroy(tmpl);
            return false;
        }
    }

    bool ok = true;
    for (int round = 0; round < rounds && ok; ++round) {
        std::vector<OmniStreamVadHandle> handles((size_t)threads, NULL);
        for (int i = 0; i < threads; ++i) {
            handles[(size_t)i] = omni_stream_vad_clone_with_config(tmpl, &cfgs[(size_t)i], &err);
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
        for (size_t i = 0; i < handles.size(); ++i) omni_stream_vad_destroy(handles[i]);
    }
    omni_stream_vad_destroy(tmpl);
    if (ok) printf("  concurrency: %d threads x %d rounds byte-identical to serial\n", threads, rounds);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <stream-vad.omnivad> <test_data_dir> [threads] [rounds]\n", argv[0]);
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
    results.push_back(Case{"config_isolation", test_config_isolation(bundle, clips[0])});
    results.push_back(Case{"config_rejection", test_config_rejection(bundle, clips[0])});
    results.push_back(Case{"template_outlived", test_template_outlived(bundle, clips[0])});
    results.push_back(Case{"concurrent_clones", test_concurrent_clones(bundle, clips, threads, rounds)});

    bool all = true;
    for (size_t i = 0; i < results.size(); ++i) {
        printf("%s %s\n", results[i].ok ? "PASS" : "FAIL", results[i].name);
        all = all && results[i].ok;
    }
    return all ? 0 : 1;
}
