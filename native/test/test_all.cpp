/*
 * Test: All 3 OmniVAD APIs
 *
 * Exercises stream VAD, non-stream VAD, and non-stream AED on the same WAV file.
 * This test validates that all APIs can be created, used, and destroyed
 * without errors.
 *
 * Usage: test_all <models_dir> <wav_file>
 *   models_dir should contain: stream-vad.omnivad, vad.omnivad, aed.omnivad
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char* pass_fail(bool ok) { return ok ? "PASS" : "FAIL"; }

static const char* aed_class_name(OmniAedClass cls) {
    switch (cls) {
        case OMNI_AED_SPEECH:  return "speech ";
        case OMNI_AED_SINGING: return "singing";
        case OMNI_AED_MUSIC:   return "music  ";
        default:                  return "unknown";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <models_dir> <wav_file>\n"
            "  models_dir should contain: stream-vad.omnivad, vad.omnivad, aed.omnivad\n",
            argv[0]);
        return 1;
    }

    std::string models_dir = argv[1];
    if (models_dir.back() != '/') models_dir += '/';
    std::string stream_bundle   = models_dir + "stream-vad.omnivad";
    std::string vad_bundle      = models_dir + "vad.omnivad";
    std::string aed_bundle      = models_dir + "aed.omnivad";
    const char* wav_file        = argv[2];

    int pass_count = 0;
    int fail_count = 0;
    int test_count = 0;

    /* Load WAV */
    vad::WavReader reader;
    if (!reader.Open(wav_file)) {
        fprintf(stderr, "Failed to open WAV: %s\n", wav_file);
        return 1;
    }

    std::vector<float> mono = reader.GetMonoData();
    int num_samples = (int)mono.size();

    /* Convert to 16-bit PCM for stream API */
    const float* fdata = reader.data();
    int total_samples = reader.num_samples();
    std::vector<int16_t> pcm(total_samples);
    for (int i = 0; i < total_samples; ++i) {
        float v = fdata[i];
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }

    printf("============================================================\n");
    printf("  OmniVAD Unified API Test Suite\n");
    printf("  Audio: %s (%d samples, %.2f seconds)\n",
           wav_file, num_samples, (float)num_samples / 16000.0f);
    printf("============================================================\n\n");

    /* ------------------------------------------------------------------ */
    /*  Test 1: Stream VAD                                                 */
    /* ------------------------------------------------------------------ */
    printf("--- Test 1: Stream VAD ---\n");

    test_count++;
    int create_err = OMNI_OK;
    OmniStreamVadHandle stream_vad = omni_stream_vad_create(stream_bundle.c_str(), NULL, &create_err);

    bool t1_create = (stream_vad != NULL);
    printf("  Create:  %s", pass_fail(t1_create));
    if (!t1_create) printf(" (%s)", omni_error_string(create_err));
    printf("\n");
    if (t1_create) pass_count++; else fail_count++;

    if (stream_vad) {
        /* Process first 100 frames (1 second) */
        int chunk_size = 160;
        int frames_to_process = 100;
        int speech_frames = 0;
        int processed = 0;
        bool t1_process_ok = true;

        for (int offset = 0;
             offset + chunk_size <= total_samples && processed < frames_to_process;
             offset += chunk_size, processed++)
        {
            OmniStreamVadResult result;
            int ret = omni_stream_vad_process_int16(stream_vad, pcm.data() + offset, chunk_size, &result);
            if (ret == OMNI_ERR_NO_FRAMES) {
                continue;
            }
            if (ret != OMNI_OK) {
                t1_process_ok = false;
                break;
            }
            if (result.is_speech) speech_frames++;
        }

        test_count++;
        printf("  Process: %s (%d frames, %d speech)\n",
               pass_fail(t1_process_ok), processed, speech_frames);
        if (t1_process_ok) pass_count++; else fail_count++;

        /* Test reset */
        test_count++;
        omni_stream_vad_reset(stream_vad);
        bool t1_reset = (omni_stream_vad_get_frame_offset(stream_vad) == 0);
        printf("  Reset:   %s\n", pass_fail(t1_reset));
        if (t1_reset) pass_count++; else fail_count++;

        omni_stream_vad_destroy(stream_vad);
    }

    /* ------------------------------------------------------------------ */
    /*  Test 2: Non-stream VAD                                             */
    /* ------------------------------------------------------------------ */
    printf("\n--- Test 2: Non-stream VAD ---\n");

    test_count++;
    create_err = OMNI_OK;
    OmniVadHandle nonstream_vad = omni_vad_create(vad_bundle.c_str(), &create_err);

    bool t2_create = (nonstream_vad != NULL);
    printf("  Create:  %s", pass_fail(t2_create));
    if (!t2_create) printf(" (%s)", omni_error_string(create_err));
    printf("\n");
    if (t2_create) pass_count++; else fail_count++;

    if (nonstream_vad) {
        /* Process with default config */
        OmniPostConfig cfg = omni_post_config_default();
        OmniSegment* segments = NULL;
        int seg_count = 0;

        test_count++;
        int ret = omni_vad_detect_int16(
            nonstream_vad, pcm.data(), num_samples, &cfg, &segments, &seg_count);
        bool t2_process = (ret == OMNI_OK);
        printf("  Process: %s (%d segments)\n", pass_fail(t2_process), seg_count);
        if (t2_process) pass_count++; else fail_count++;

        if (segments) {
            for (int i = 0; i < seg_count && i < 5; ++i) {
                printf("    [%d] %.3f - %.3f\n", i + 1, segments[i].start, segments[i].end);
            }
            if (seg_count > 5) printf("    ... (%d more)\n", seg_count - 5);
            omni_free(segments);
        }

        /* Test raw probabilities */
        test_count++;
        float* raw_probs = NULL;
        int raw_frames = 0;
        ret = omni_vad_detect_probs_int16(
            nonstream_vad, pcm.data(), num_samples, &raw_probs, &raw_frames);
        bool t2_raw = (ret == OMNI_OK && raw_probs != NULL && raw_frames > 0);
        printf("  Raw:     %s (%d frames)\n", pass_fail(t2_raw), raw_frames);
        if (t2_raw) pass_count++; else fail_count++;
        omni_free(raw_probs);

        omni_vad_destroy(nonstream_vad);
    }

    /* ------------------------------------------------------------------ */
    /*  Test 3: Non-stream AED                                             */
    /* ------------------------------------------------------------------ */
    printf("\n--- Test 3: Non-stream AED ---\n");

    test_count++;
    create_err = OMNI_OK;
    OmniAedHandle aed = omni_aed_create(aed_bundle.c_str(), &create_err);

    bool t3_create = (aed != NULL);
    printf("  Create:  %s", pass_fail(t3_create));
    if (!t3_create) printf(" (%s)", omni_error_string(create_err));
    printf("\n");
    if (t3_create) pass_count++; else fail_count++;

    if (aed) {
        OmniAedPostConfig aed_cfg = omni_aed_post_config_default();
        OmniAedSegment* aed_segments = NULL;
        int aed_count = 0;

        test_count++;
        int ret = omni_aed_detect_int16(
            aed, pcm.data(), num_samples, &aed_cfg, &aed_segments, &aed_count);
        bool t3_process = (ret == OMNI_OK);
        printf("  Process: %s (%d segments)\n", pass_fail(t3_process), aed_count);
        if (t3_process) pass_count++; else fail_count++;

        if (aed_segments) {
            for (int i = 0; i < aed_count && i < 10; ++i) {
                printf("    [%d] %s  %.3f - %.3f  (conf=%.3f)\n",
                       i + 1,
                       aed_class_name(aed_segments[i].cls),
                       aed_segments[i].start, aed_segments[i].end,
                       aed_segments[i].confidence);
            }
            if (aed_count > 10) printf("    ... (%d more)\n", aed_count - 10);
            omni_free(aed_segments);
        }

        /* Test raw probabilities */
        test_count++;
        float* aed_raw = NULL;
        int aed_frames = 0;
        ret = omni_aed_detect_probs_int16(
            aed, pcm.data(), num_samples, &aed_raw, &aed_frames);
        bool t3_raw = (ret == OMNI_OK && aed_raw != NULL && aed_frames > 0);
        printf("  Raw:     %s (%d frames x 3 classes)\n", pass_fail(t3_raw), aed_frames);
        if (t3_raw) pass_count++; else fail_count++;
        omni_free(aed_raw);

        omni_aed_destroy(aed);
    }

    /* ------------------------------------------------------------------ */
    /*  Test 4: Error handling                                             */
    /* ------------------------------------------------------------------ */
    printf("\n--- Test 4: Error handling ---\n");

    test_count++;
    int ret = omni_stream_vad_process_int16(NULL, pcm.data(), 160, NULL);
    bool t4_null_handle = (ret == OMNI_ERR_NULL_HANDLE);
    printf("  Null handle:     %s (ret=%d)\n", pass_fail(t4_null_handle), ret);
    if (t4_null_handle) pass_count++; else fail_count++;

    test_count++;
    const char* err_str = omni_error_string(OMNI_ERR_LOAD_BUNDLE);
    bool t4_err_string = (err_str != NULL && strlen(err_str) > 0);
    printf("  Error string:    %s (\"%s\")\n", pass_fail(t4_err_string), err_str);
    if (t4_err_string) pass_count++; else fail_count++;

    test_count++;
    omni_free(NULL);  /* should not crash */
    printf("  Free NULL:       PASS\n");
    pass_count++;

    /* ------------------------------------------------------------------ */
    /*  Summary                                                            */
    /* ------------------------------------------------------------------ */
    printf("\n============================================================\n");
    printf("  Results: %d/%d passed", pass_count, test_count);
    if (fail_count > 0) {
        printf(" (%d FAILED)", fail_count);
    }
    printf("\n============================================================\n");

    return fail_count > 0 ? 1 : 0;
}
