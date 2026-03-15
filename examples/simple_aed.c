/*
 * Simple AED example in C — detect speech, singing, music using omnivad library.
 *
 * Build:
 *   gcc -o simple_aed simple_aed.c -I../native/include -L../native/build -lomnivad
 *
 * Usage:
 *   ./simple_aed model.param model.bin cmvn_means.bin cmvn_istd.bin audio.wav
 */

#include "omnivad.h"
#include <stdio.h>
#include <stdlib.h>

/* Minimal WAV loader (same as simple_vad.c) */
static int load_wav_mono(const char* path, float** out_data, int* out_samples) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

    unsigned char header[44];
    if (fread(header, 1, 44, fp) != 44) { fclose(fp); return -1; }

    unsigned int data_size = *(unsigned int*)(header + 40);
    int num_samples = data_size / 2;

    float* data = (float*)malloc(sizeof(float) * num_samples);
    if (!data) { fclose(fp); return -1; }

    for (int i = 0; i < num_samples; i++) {
        short sample;
        if (fread(&sample, 2, 1, fp) != 1) break;
        data[i] = (float)sample;
    }

    fclose(fp);
    *out_data = data;
    *out_samples = num_samples;
    return 0;
}

static const char* class_name(OmniAedClass cls) {
    switch (cls) {
        case OMNI_AED_SPEECH:  return "speech";
        case OMNI_AED_SINGING: return "singing";
        case OMNI_AED_MUSIC:   return "music";
        default:                  return "unknown";
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <model.param> <model.bin> <cmvn_means.bin> <cmvn_istd.bin> <audio.wav>\n", argv[0]);
        return 1;
    }

    float* audio = NULL;
    int num_samples = 0;
    if (load_wav_mono(argv[5], &audio, &num_samples) != 0) {
        fprintf(stderr, "Failed to load: %s\n", argv[5]);
        return 1;
    }
    printf("Audio: %d samples (%.2fs)\n", num_samples, num_samples / 16000.0f);

    /* Create AED */
    OmniAedNonStreamHandle aed = omni_aed_nonstream_create(
        argv[1], argv[2], argv[3], argv[4]);
    if (!aed) {
        fprintf(stderr, "Failed to create AED\n");
        free(audio);
        return 1;
    }

    /* Detect events */
    OmniAedPostConfig config = omni_aed_post_config_default();
    OmniAedSegment* segments = NULL;
    int count = 0;

    int ret = omni_aed_nonstream_process(aed, audio, num_samples, &config, &segments, &count);

    if (ret != OMNI_OK) {
        fprintf(stderr, "Error: %s\n", omni_error_string(ret));
    } else {
        printf("Audio events: %d\n\n", count);
        for (int i = 0; i < count; i++) {
            printf("  [%d] %-8s %.3fs - %.3fs  (%.3fs, conf=%.3f)\n",
                   i + 1,
                   class_name(segments[i].cls),
                   segments[i].start, segments[i].end,
                   segments[i].end - segments[i].start,
                   segments[i].confidence);
        }
    }

    omni_free(segments);
    omni_aed_nonstream_destroy(aed);
    free(audio);
    return 0;
}
