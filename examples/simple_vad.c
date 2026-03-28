/*
 * Simple VAD example in C — detect speech segments using omnivad library.
 *
 * Build:
 *   gcc -o simple_vad simple_vad.c -I../native/include -L../native/build -lomnivad
 *
 * Usage:
 *   ./simple_vad vad.omnivad audio.wav
 */

#include "omnivad.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal WAV loader — reads 16-bit mono PCM into int16 array */
static int load_wav_int16(const char* path, short** out_data, int* out_samples) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

    /* Skip to data chunk (assume standard 44-byte header) */
    unsigned char header[44];
    if (fread(header, 1, 44, fp) != 44) { fclose(fp); return -1; }

    unsigned int data_size = *(unsigned int*)(header + 40);
    int num_samples = data_size / 2;

    short* data = (short*)malloc(sizeof(short) * num_samples);
    if (!data) { fclose(fp); return -1; }

    size_t read = fread(data, sizeof(short), num_samples, fp);
    fclose(fp);
    *out_data = data;
    *out_samples = (int)read;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <vad.omnivad> <audio.wav>\n", argv[0]);
        return 1;
    }

    /* Load audio */
    short* audio = NULL;
    int num_samples = 0;
    if (load_wav_int16(argv[2], &audio, &num_samples) != 0) {
        fprintf(stderr, "Failed to load: %s\n", argv[2]);
        return 1;
    }
    printf("Audio: %d samples (%.2fs)\n", num_samples, num_samples / 16000.0f);

    /* Create VAD */
    int err = OMNI_OK;
    OmniVadHandle vad = omni_vad_create(argv[1], &err);
    if (!vad) {
        fprintf(stderr, "Failed to create VAD: %s\n", omni_error_string(err));
        free(audio);
        return 1;
    }

    /* Configure post-processing */
    OmniPostConfig config = omni_post_config_default();

    /* Detect */
    OmniSegment* segments = NULL;
    int count = 0;
    int ret = omni_vad_detect_int16(vad, audio, num_samples, &config, &segments, &count);

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
    omni_vad_destroy(vad);
    free(audio);
    return 0;
}
