/*
 * Simple VAD example in C — detect speech segments using omnivad library.
 *
 * Build:
 *   gcc -o simple_vad simple_vad.c -I../native/include -L../native/build -lomnivad
 *
 * Usage:
 *   ./simple_vad model.param model.bin cmvn_means.bin cmvn_istd.bin audio.wav
 */

#include "omnivad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal WAV loader — reads 16-bit mono PCM into float array */
static int load_wav_mono(const char* path, float** out_data, int* out_samples) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

    /* Skip to data chunk (assume standard 44-byte header) */
    unsigned char header[44];
    if (fread(header, 1, 44, fp) != 44) { fclose(fp); return -1; }

    /* Find "data" marker */
    unsigned int data_size = *(unsigned int*)(header + 40);
    int num_samples = data_size / 2; /* 16-bit = 2 bytes per sample */

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

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <model.param> <model.bin> <cmvn_means.bin> <cmvn_istd.bin> <audio.wav>\n", argv[0]);
        return 1;
    }

    /* Load audio */
    float* audio = NULL;
    int num_samples = 0;
    if (load_wav_mono(argv[5], &audio, &num_samples) != 0) {
        fprintf(stderr, "Failed to load: %s\n", argv[5]);
        return 1;
    }
    printf("Audio: %d samples (%.2fs)\n", num_samples, num_samples / 16000.0f);

    /* Create VAD */
    OmniVadNonStreamHandle vad = omni_vad_nonstream_create(
        argv[1], argv[2], argv[3], argv[4]);
    if (!vad) {
        fprintf(stderr, "Failed to create VAD\n");
        free(audio);
        return 1;
    }

    /* Configure post-processing */
    OmniPostConfig config = omni_post_config_default();

    /* Detect */
    OmniSegment* segments = NULL;
    int count = 0;
    int ret = omni_vad_nonstream_process(vad, audio, num_samples, &config, &segments, &count);

    if (ret != OMNI_OK) {
        fprintf(stderr, "Error: %s\n", omni_error_string(ret));
    } else {
        printf("Speech segments: %d\n", count);
        for (int i = 0; i < count; i++) {
            printf("  [%d] %.3fs - %.3fs  (%.3fs)\n",
                   i + 1, segments[i].start, segments[i].end,
                   segments[i].end - segments[i].start);
        }
    }

    /* Cleanup */
    omni_free(segments);
    omni_vad_nonstream_destroy(vad);
    free(audio);
    return 0;
}
