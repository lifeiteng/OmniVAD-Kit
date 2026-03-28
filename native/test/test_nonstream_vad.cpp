/*
 * Test: Non-stream VAD
 *
 * Reads a WAV file, runs non-stream VAD inference on the whole audio,
 * applies post-processing, and prints detected speech segments.
 *
 * Usage: test_nonstream_vad <model.param> <model.bin> <cmvn_means.bin> <cmvn_istd.bin> <wav_file>
 *        [threshold] [smooth_window] [min_speech_ms] [min_silence_ms]
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <model.param> <model.bin> <cmvn_means.bin> <cmvn_istd.bin> <wav_file>"
            " [threshold] [smooth_window] [min_speech_ms] [min_silence_ms]\n",
            argv[0]);
        return 1;
    }

    const char* model_param = argv[1];
    const char* model_bin   = argv[2];
    const char* cmvn_means  = argv[3];
    const char* cmvn_istd   = argv[4];
    const char* wav_file    = argv[5];

    /* Parse optional post-processing parameters */
    OmniPostConfig cfg = omni_post_config_default();
    if (argc > 6) cfg.threshold          = (float)atof(argv[6]);
    if (argc > 7) cfg.smooth_window_size = atoi(argv[7]);
    if (argc > 8) cfg.min_speech_frames  = atoi(argv[8]) / 10;  /* ms -> frames */
    if (argc > 9) cfg.min_silence_frames = atoi(argv[9]) / 10;  /* ms -> frames */

    /* Load WAV */
    vad::WavReader reader;
    if (!reader.Open(wav_file)) {
        fprintf(stderr, "Failed to open WAV: %s\n", wav_file);
        return 1;
    }

    std::vector<float> mono = reader.GetMonoData();
    int num_samples = (int)mono.size();

    /* Convert to int16 PCM for the public API */
    std::vector<int16_t> pcm(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float v = mono[i];
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }

    printf("Non-stream VAD: processing %d samples (%.2f seconds)\n",
           num_samples, (float)num_samples / 16000.0f);
    printf("Config: threshold=%.2f, smooth_window=%d, min_speech=%d frames, min_silence=%d frames\n",
           cfg.threshold, cfg.smooth_window_size, cfg.min_speech_frames, cfg.min_silence_frames);

    /* Create non-stream VAD */
    OmniVadNonStreamHandle vad = omni_vad_nonstream_create(
        model_param, model_bin, cmvn_means, cmvn_istd);
    if (!vad) {
        fprintf(stderr, "Failed to create non-stream VAD\n");
        return 1;
    }

    /* Process */
    OmniSegment* segments = NULL;
    int count = 0;
    int ret = omni_vad_nonstream_process_int16(vad, pcm.data(), num_samples, &cfg, &segments, &count);
    if (ret != OMNI_OK) {
        fprintf(stderr, "Process error: %s\n", omni_error_string(ret));
        omni_vad_nonstream_destroy(vad);
        return 1;
    }

    /* Print segments */
    printf("\nDetected %d speech segments:\n", count);
    float total_speech = 0.0f;
    for (int i = 0; i < count; ++i) {
        float dur = segments[i].end - segments[i].start;
        total_speech += dur;
        printf("  [%2d] %.3f - %.3f  (%.3f s)\n",
               i + 1, segments[i].start, segments[i].end, dur);
    }
    printf("\nTotal speech: %.3f s / %.3f s (%.1f%%)\n",
           total_speech, (float)num_samples / 16000.0f,
           100.0f * total_speech / ((float)num_samples / 16000.0f));

    /* Cleanup */
    omni_free(segments);
    omni_vad_nonstream_destroy(vad);
    return 0;
}
