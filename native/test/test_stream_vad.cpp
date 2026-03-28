/*
 * Test: Stream VAD
 *
 * Reads a WAV file and processes it frame-by-frame (10ms chunks) through
 * the stream VAD API, printing per-frame speech probability.
 *
 * Usage: test_stream_vad <bundle.omnivad> <wav_file>
 */

#include "omnivad.h"
#include "frontend/wav.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <stream-vad.omnivad> <wav_file>\n",
            argv[0]);
        return 1;
    }

    const char* bundle_path = argv[1];
    const char* wav_file    = argv[2];

    /* Load WAV */
    vad::WavReader reader;
    if (!reader.Open(wav_file)) {
        fprintf(stderr, "Failed to open WAV: %s\n", wav_file);
        return 1;
    }

    int sample_rate = reader.sample_rate();
    if (sample_rate != 16000) {
        fprintf(stderr, "Warning: sample rate %d, expected 16000\n", sample_rate);
    }

    /* Convert to 16-bit PCM */
    int num_samples = reader.num_samples();
    const float* fdata = reader.data();
    std::vector<int16_t> pcm(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float v = fdata[i];
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm[i] = (int16_t)v;
    }

    /* Create stream VAD */
    float threshold = 0.5f;
    int create_err = OMNI_OK;
    OmniStreamVadHandle vad = omni_stream_vad_create(bundle_path, threshold, &create_err);
    if (!vad) {
        fprintf(stderr, "Failed to create stream VAD: %s\n", omni_error_string(create_err));
        return 1;
    }

    /* Also run detect_full for batch comparison */
    {
        std::vector<float> fdata_vec(fdata, fdata + num_samples);
        float* full_probs = NULL;
        int full_frames = 0;
        int ret = omni_stream_vad_detect_full(vad, fdata_vec.data(), num_samples,
                                                  &full_probs, &full_frames);
        if (ret == OMNI_OK && full_probs) {
            printf("detect_full: %d frames\n", full_frames);
            for (int i = 0; i < full_frames; ++i) {
                printf("FullFrame %4d: time=%.3fs, confidence=%.4f, %s\n",
                       i, i * 0.01f, full_probs[i],
                       full_probs[i] > threshold ? "SPEECH" : "silence");
            }
            omni_free(full_probs);
        }
    }

    /* Reset for streaming test */
    omni_stream_vad_reset(vad);

    printf("\nStream VAD: processing %d samples (%.2f seconds)\n",
           num_samples, (float)num_samples / 16000.0f);

    /* Process 10ms chunks (160 samples) */
    int chunk_size = 160;
    int speech_frames = 0;
    int total_frames = 0;

    for (int offset = 0; offset + chunk_size <= num_samples; offset += chunk_size) {
        OmniStreamVadResult result;
        int ret = omni_stream_vad_process(vad, pcm.data() + offset, chunk_size, &result);
        if (ret == OMNI_ERR_NO_FRAMES) {
            continue;
        }
        if (ret != OMNI_OK) {
            fprintf(stderr, "Process error at offset %d: %s\n", offset, omni_error_string(ret));
            continue;
        }

        if (result.frame_offset > 0) {
            total_frames++;
            if (result.is_speech) speech_frames++;
            printf("Frame %4d: time=%.3fs, confidence=%.4f, %s\n",
                   result.frame_offset,
                   result.frame_offset * 0.01f,
                   result.confidence,
                   result.is_speech ? "SPEECH" : "silence");
        }
    }

    printf("\nSummary: %d/%d frames are speech (%.1f%%)\n",
           speech_frames, total_frames,
           total_frames > 0 ? 100.0f * speech_frames / total_frames : 0.0f);

    omni_stream_vad_destroy(vad);
    return 0;
}
