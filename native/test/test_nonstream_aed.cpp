/*
 * Test: Non-stream AED (Audio Event Detection)
 *
 * Reads a WAV file, runs 3-class AED inference (speech/singing/music),
 * applies per-class post-processing, and prints detected event segments.
 *
 * Usage: test_nonstream_aed <model.param> <model.bin> <cmvn_means.bin> <cmvn_istd.bin> <wav_file>
 *        [threshold]
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static const char* aed_class_name(OmniAedClass cls) {
    switch (cls) {
        case OMNI_AED_SPEECH:  return "speech ";
        case OMNI_AED_SINGING: return "singing";
        case OMNI_AED_MUSIC:   return "music  ";
        default:                  return "unknown";
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <aed.param> <aed.bin> <cmvn_means.bin> <cmvn_istd.bin> <wav_file>"
            " [threshold]\n",
            argv[0]);
        return 1;
    }

    const char* model_param = argv[1];
    const char* model_bin   = argv[2];
    const char* cmvn_means  = argv[3];
    const char* cmvn_istd   = argv[4];
    const char* wav_file    = argv[5];

    /* Parse optional threshold (applies to all classes) */
    OmniAedPostConfig cfg = omni_aed_post_config_default();
    if (argc > 6) {
        float t = (float)atof(argv[6]);
        cfg.speech.threshold  = t;
        cfg.singing.threshold = t;
        cfg.music.threshold   = t;
    }

    /* Load WAV */
    vad::WavReader reader;
    if (!reader.Open(wav_file)) {
        fprintf(stderr, "Failed to open WAV: %s\n", wav_file);
        return 1;
    }

    std::vector<float> mono = reader.GetMonoData();
    int num_samples = (int)mono.size();

    printf("Non-stream AED: processing %d samples (%.2f seconds)\n",
           num_samples, (float)num_samples / 16000.0f);
    printf("Thresholds: speech=%.2f, singing=%.2f, music=%.2f\n",
           cfg.speech.threshold, cfg.singing.threshold, cfg.music.threshold);

    /* Create AED */
    OmniAedNonStreamHandle aed = omni_aed_nonstream_create(
        model_param, model_bin, cmvn_means, cmvn_istd);
    if (!aed) {
        fprintf(stderr, "Failed to create AED\n");
        return 1;
    }

    /* Process */
    OmniAedSegment* segments = NULL;
    int count = 0;
    int ret = omni_aed_nonstream_process(aed, mono.data(), num_samples, &cfg, &segments, &count);
    if (ret != OMNI_OK) {
        fprintf(stderr, "Process error: %s\n", omni_error_string(ret));
        omni_aed_nonstream_destroy(aed);
        return 1;
    }

    /* Print segments */
    printf("\nDetected %d audio event segments:\n", count);
    for (int i = 0; i < count; ++i) {
        float dur = segments[i].end - segments[i].start;
        printf("  [%2d] %s  %.3f - %.3f  (%.3f s, conf=%.3f)\n",
               i + 1,
               aed_class_name(segments[i].cls),
               segments[i].start, segments[i].end,
               dur, segments[i].confidence);
    }

    /* Also demonstrate raw probabilities API */
    printf("\n--- Raw probabilities (first 20 frames) ---\n");
    float* raw_probs = NULL;
    int num_frames = 0;
    ret = omni_aed_nonstream_process_raw(aed, mono.data(), num_samples, &raw_probs, &num_frames);
    if (ret == OMNI_OK && raw_probs) {
        int show = num_frames < 20 ? num_frames : 20;
        for (int t = 0; t < show; ++t) {
            printf("  Frame %4d: time=%.3fs, speech=%.4f, singing=%.4f, music=%.4f\n",
                   t, t * 0.01f,
                   raw_probs[t * 3 + 0],
                   raw_probs[t * 3 + 1],
                   raw_probs[t * 3 + 2]);
        }
        omni_free(raw_probs);
    }

    /* Cleanup */
    omni_free(segments);
    omni_aed_nonstream_destroy(aed);
    return 0;
}
