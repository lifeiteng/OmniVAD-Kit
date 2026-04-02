/*
 * Test: Thread-safety regression for OmniVAD / OmniAED / OmniStreamVAD
 *
 * Covers the "plan 2" matrix:
 *   1) shared VAD handle, concurrent detect_probs
 *   2) isolated VAD handles, concurrent detect_probs
 *   3) shared AED handle, concurrent detect_probs
 *   4) shared StreamVAD handle, concurrent process (negative contract test)
 *   5) isolated StreamVAD handles, concurrent process (positive control)
 *
 * Usage: test_thread_safety <models_dir> <wav_file> [threads] [repeats]
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static const int kDefaultThreads = 4;
static const int kDefaultRepeats = 100;
static const int kChunkSize = 160; /* 10 ms @ 16 kHz */

static const char* pass_fail(bool ok) { return ok ? "PASS" : "FAIL"; }

/* ---- Helpers ---- */

static bool vectors_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    return memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static std::vector<float> run_vad_probs(OmniVadHandle h, const int16_t* pcm, int n) {
    float* raw = NULL;
    int frames = 0;
    int ret = omni_vad_detect_probs_int16(h, pcm, n, &raw, &frames);
    if (ret != OMNI_OK || !raw || frames <= 0) return {};
    std::vector<float> out(raw, raw + frames);
    omni_free(raw);
    return out;
}

static std::vector<float> run_aed_probs(OmniAedHandle h, const int16_t* pcm, int n) {
    float* raw = NULL;
    int frames = 0;
    int ret = omni_aed_detect_probs_int16(h, pcm, n, &raw, &frames);
    if (ret != OMNI_OK || !raw || frames <= 0) return {};
    std::vector<float> out(raw, raw + frames * 3);
    omni_free(raw);
    return out;
}

struct StreamFrame {
    int frame_offset;
    float confidence;
};

static std::vector<StreamFrame> run_stream_sequence(OmniStreamVadHandle h, const int16_t* pcm, int n) {
    std::vector<StreamFrame> seq;
    omni_stream_vad_reset(h);
    for (int off = 0; off + kChunkSize <= n; off += kChunkSize) {
        OmniStreamVadResult res;
        int ret = omni_stream_vad_process(h, pcm + off, kChunkSize, &res);
        if (ret == OMNI_ERR_NO_FRAMES) continue;
        if (ret != OMNI_OK) return {};
        seq.push_back({res.frame_offset, res.confidence});
    }
    return seq;
}

static bool stream_seqs_equal(const std::vector<StreamFrame>& a, const std::vector<StreamFrame>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].frame_offset != b[i].frame_offset) return false;
        if (a[i].confidence != b[i].confidence) return false;
    }
    return true;
}

/* ---- Test 1: Shared VAD handle concurrent detect_probs ---- */
static bool test_shared_vad(const std::string& bundle, const int16_t* pcm, int n,
                            int threads, int repeats) {
    int err = 0;
    OmniVadHandle h = omni_vad_create(bundle.c_str(), &err);
    if (!h) { fprintf(stderr, "  Failed to create VAD: %d\n", err); return false; }

    /* Serial baseline */
    std::vector<float> baseline = run_vad_probs(h, pcm, n);
    if (baseline.empty()) { omni_vad_destroy(h); return false; }

    std::atomic<int> mismatches(0);

    for (int r = 0; r < repeats; ++r) {
        std::vector<std::thread> workers;
        std::vector<std::vector<float>> results(threads);

        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t]() {
                results[t] = run_vad_probs(h, pcm, n);
            });
        }
        for (auto& w : workers) w.join();

        for (int t = 0; t < threads; ++t) {
            if (!vectors_equal(results[t], baseline)) {
                mismatches.fetch_add(1);
            }
        }
    }

    omni_vad_destroy(h);
    if (mismatches.load() > 0) {
        printf("  %d/%d thread-runs mismatched baseline\n", mismatches.load(), repeats * threads);
    }
    return mismatches.load() == 0;
}

/* ---- Test 2: Isolated VAD handles concurrent detect_probs ---- */
static bool test_isolated_vad(const std::string& bundle, const int16_t* pcm, int n,
                              int threads, int repeats) {
    /* Compute baseline with one handle */
    int err = 0;
    OmniVadHandle bh = omni_vad_create(bundle.c_str(), &err);
    if (!bh) return false;
    std::vector<float> baseline = run_vad_probs(bh, pcm, n);
    omni_vad_destroy(bh);
    if (baseline.empty()) return false;

    std::atomic<int> mismatches(0);

    for (int r = 0; r < repeats; ++r) {
        std::vector<std::thread> workers;
        std::vector<std::vector<float>> results(threads);

        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t]() {
                int e = 0;
                OmniVadHandle local = omni_vad_create(bundle.c_str(), &e);
                if (!local) { mismatches.fetch_add(1); return; }
                results[t] = run_vad_probs(local, pcm, n);
                omni_vad_destroy(local);
            });
        }
        for (auto& w : workers) w.join();

        for (int t = 0; t < threads; ++t) {
            if (!vectors_equal(results[t], baseline)) {
                mismatches.fetch_add(1);
            }
        }
    }

    if (mismatches.load() > 0) {
        printf("  %d/%d thread-runs mismatched baseline\n", mismatches.load(), repeats * threads);
    }
    return mismatches.load() == 0;
}

/* ---- Test 3: Shared AED handle concurrent detect_probs ---- */
static bool test_shared_aed(const std::string& bundle, const int16_t* pcm, int n,
                            int threads, int repeats) {
    int err = 0;
    OmniAedHandle h = omni_aed_create(bundle.c_str(), &err);
    if (!h) { fprintf(stderr, "  Failed to create AED: %d\n", err); return false; }

    std::vector<float> baseline = run_aed_probs(h, pcm, n);
    if (baseline.empty()) { omni_aed_destroy(h); return false; }

    std::atomic<int> mismatches(0);

    for (int r = 0; r < repeats; ++r) {
        std::vector<std::thread> workers;
        std::vector<std::vector<float>> results(threads);

        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t]() {
                results[t] = run_aed_probs(h, pcm, n);
            });
        }
        for (auto& w : workers) w.join();

        for (int t = 0; t < threads; ++t) {
            if (!vectors_equal(results[t], baseline)) {
                mismatches.fetch_add(1);
            }
        }
    }

    omni_aed_destroy(h);
    if (mismatches.load() > 0) {
        printf("  %d/%d thread-runs mismatched baseline\n", mismatches.load(), repeats * threads);
    }
    return mismatches.load() == 0;
}

/* ---- Test 4: Shared StreamVAD handle concurrent process (negative) ---- */
static bool test_shared_stream_negative(const std::string& bundle, const int16_t* pcm, int n,
                                        int threads, int repeats) {
    /*
     * This test intentionally exercises undefined behavior (concurrent mutation
     * of std::deque, non-atomic frame_offset, etc.) to prove that sharing a
     * StreamVAD handle across threads is unsafe.
     *
     * To avoid crashing (memory corruption from concurrent std::deque ops),
     * we cap at 20 repeats and stop early once we observe an invariant break.
     */
    int safe_repeats = std::min(repeats, 20);

    /* Build serial reference with a separate handle */
    int err = 0;
    OmniStreamVadHandle ref_h = omni_stream_vad_create(bundle.c_str(), 0.5f, &err);
    if (!ref_h) return false;
    std::vector<StreamFrame> serial = run_stream_sequence(ref_h, pcm, n);
    omni_stream_vad_destroy(ref_h);
    if (serial.empty()) return false;

    /* Shared handle for concurrent abuse */
    OmniStreamVadHandle shared = omni_stream_vad_create(bundle.c_str(), 0.5f, &err);
    if (!shared) return false;

    int inconsistent_runs = 0;

    for (int r = 0; r < safe_repeats; ++r) {
        omni_stream_vad_reset(shared);

        /* Each thread processes interleaved chunks against the shared handle */
        std::vector<std::thread> workers;
        std::vector<std::vector<StreamFrame>> thread_results(threads);
        std::atomic<int> errors(0);

        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t]() {
                std::vector<StreamFrame> local_seq;
                for (int off = t * kChunkSize; off + kChunkSize <= n; off += threads * kChunkSize) {
                    OmniStreamVadResult res;
                    int ret = omni_stream_vad_process(shared, pcm + off, kChunkSize, &res);
                    if (ret == OMNI_ERR_NO_FRAMES) continue;
                    if (ret != OMNI_OK) { errors.fetch_add(1); return; }
                    local_seq.push_back({res.frame_offset, res.confidence});
                }
                thread_results[t] = std::move(local_seq);
            });
        }
        for (auto& w : workers) w.join();

        /* Any inference error is itself proof of unsafety */
        if (errors.load() > 0) {
            inconsistent_runs++;
            break; /* Stop early to avoid prolonged UB */
        }

        /* Merge and sort by frame_offset */
        std::vector<StreamFrame> merged;
        for (auto& tr : thread_results)
            merged.insert(merged.end(), tr.begin(), tr.end());
        std::sort(merged.begin(), merged.end(),
                  [](const StreamFrame& a, const StreamFrame& b) { return a.frame_offset < b.frame_offset; });

        /* Check invariant breaks */
        if (merged.size() != serial.size() || !stream_seqs_equal(merged, serial)) {
            inconsistent_runs++;
        } else {
            int fo = omni_stream_vad_get_frame_offset(shared);
            if (fo != (int)serial.size()) {
                inconsistent_runs++;
            }
        }

        /* Stop early once proven unsafe */
        if (inconsistent_runs > 0) break;
    }

    omni_stream_vad_destroy(shared);

    printf("  inconsistent_runs=%d/%d (expect >= 1 for negative test)\n", inconsistent_runs, safe_repeats);
    /* Negative test passes if we observed at least one invariant break */
    return inconsistent_runs >= 1;
}

/* ---- Test 5: Isolated StreamVAD handles concurrent process (positive) ---- */
static bool test_isolated_stream(const std::string& bundle, const int16_t* pcm, int n,
                                 int threads, int repeats) {
    int err = 0;
    OmniStreamVadHandle ref_h = omni_stream_vad_create(bundle.c_str(), 0.5f, &err);
    if (!ref_h) return false;
    std::vector<StreamFrame> serial = run_stream_sequence(ref_h, pcm, n);
    omni_stream_vad_destroy(ref_h);
    if (serial.empty()) return false;

    std::atomic<int> mismatches(0);

    for (int r = 0; r < repeats; ++r) {
        std::vector<std::thread> workers;
        std::vector<std::vector<StreamFrame>> results(threads);

        for (int t = 0; t < threads; ++t) {
            workers.emplace_back([&, t]() {
                int e = 0;
                OmniStreamVadHandle local = omni_stream_vad_create(bundle.c_str(), 0.5f, &e);
                if (!local) { mismatches.fetch_add(1); return; }
                results[t] = run_stream_sequence(local, pcm, n);
                omni_stream_vad_destroy(local);
            });
        }
        for (auto& w : workers) w.join();

        for (int t = 0; t < threads; ++t) {
            if (!stream_seqs_equal(results[t], serial)) {
                mismatches.fetch_add(1);
            }
        }
    }

    if (mismatches.load() > 0) {
        printf("  %d/%d thread-runs mismatched baseline\n", mismatches.load(), repeats * threads);
    }
    return mismatches.load() == 0;
}

/* ---- Main ---- */

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <models_dir> <wav_file> [threads] [repeats]\n"
            "  models_dir: directory with stream-vad.omnivad, vad.omnivad, aed.omnivad\n"
            "  threads:    default %d\n"
            "  repeats:    default %d\n",
            argv[0], kDefaultThreads, kDefaultRepeats);
        return 1;
    }

    std::string models_dir = argv[1];
    if (!models_dir.empty() && models_dir.back() != '/') models_dir += '/';
    const char* wav_file = argv[2];
    int threads = argc > 3 ? std::max(1, std::atoi(argv[3])) : kDefaultThreads;
    int repeats = argc > 4 ? std::max(1, std::atoi(argv[4])) : kDefaultRepeats;

    std::string vad_bundle    = models_dir + "vad.omnivad";
    std::string aed_bundle    = models_dir + "aed.omnivad";
    std::string stream_bundle = models_dir + "stream-vad.omnivad";

    /* Load WAV */
    vad::WavReader reader;
    if (!reader.Open(wav_file)) {
        fprintf(stderr, "Failed to open WAV: %s\n", wav_file);
        return 1;
    }
    std::vector<float> mono = reader.GetMonoData();
    int num_samples = (int)mono.size();

    /* Convert to int16 for the APIs that need it */
    std::vector<int16_t> pcm(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float v = mono[i];
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }

    printf("============================================================\n");
    printf("  OmniVAD Thread-Safety Regression\n");
    printf("============================================================\n");
    printf("threads : %d\n", threads);
    printf("repeats : %d\n", repeats);
    printf("samples : %d\n\n", num_samples);

    int pass = 0, fail = 0;

    auto run_test = [&](const char* name, bool result) {
        printf("[%s] %s\n", pass_fail(result), name);
        if (result) pass++; else fail++;
    };

    run_test("shared VAD handle concurrent detect_probs",
             test_shared_vad(vad_bundle, pcm.data(), num_samples, threads, repeats));

    run_test("isolated VAD handles concurrent detect_probs",
             test_isolated_vad(vad_bundle, pcm.data(), num_samples, threads, repeats));

    run_test("shared AED handle concurrent detect_probs",
             test_shared_aed(aed_bundle, pcm.data(), num_samples, threads, repeats));

    run_test("shared StreamVAD handle concurrent process (negative)",
             test_shared_stream_negative(stream_bundle, pcm.data(), num_samples, threads, repeats));

    run_test("isolated StreamVAD handles concurrent process",
             test_isolated_stream(stream_bundle, pcm.data(), num_samples, threads, repeats));

    printf("\nSummary: %d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
