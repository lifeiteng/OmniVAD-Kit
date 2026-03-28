/*
 * Simple AED example in C — detect speech, singing, music using omnivad library.
 *
 * Build:
 *   gcc -o simple_aed simple_aed.c -I../native/include -L../native/build -lomnivad
 *
 * Usage:
 *   ./simple_aed aed.omnivad audio.wav
 */

#include "omnivad.h"
#include <stdio.h>
#include <stdlib.h>

/* Minimal WAV loader — reads 16-bit mono PCM into int16 array */
static int load_wav_int16(const char* path, short** out_data, int* out_samples) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

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

static const char* class_name(OmniAedClass cls) {
    switch (cls) {
        case OMNI_AED_SPEECH:  return "speech";
        case OMNI_AED_SINGING: return "singing";
        case OMNI_AED_MUSIC:   return "music";
        default:                  return "unknown";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <aed.omnivad> <audio.wav>\n", argv[0]);
        return 1;
    }

    short* audio = NULL;
    int num_samples = 0;
    if (load_wav_int16(argv[2], &audio, &num_samples) != 0) {
        fprintf(stderr, "Failed to load: %s\n", argv[2]);
        return 1;
    }
    printf("Audio: %d samples (%.2fs)\n", num_samples, num_samples / 16000.0f);

    /* Create AED */
    int err = OMNI_OK;
    OmniAedHandle aed = omni_aed_create(argv[1], &err);
    if (!aed) {
        fprintf(stderr, "Failed to create AED: %s\n", omni_error_string(err));
        free(audio);
        return 1;
    }

    /* Detect events */
    OmniAedPostConfig config = omni_aed_post_config_default();
    OmniAedSegment* segments = NULL;
    int count = 0;

    int ret = omni_aed_detect_int16(aed, audio, num_samples, &config, &segments, &count);

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
    omni_aed_destroy(aed);
    free(audio);
    return 0;
}
