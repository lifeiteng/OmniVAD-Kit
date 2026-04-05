/*
 * OmniVAD Unified C API
 *
 * Three model types through a single header:
 *   1. VAD        — whole-audio speech segmentation with post-processing
 *   2. Stream VAD — frame-by-frame speech detection with packed cache
 *   3. AED        — whole-audio audio event detection (speech/singing/music)
 *
 * All models use 80-dim log-mel fbank features (25ms window, 10ms shift,
 * Povey window, pre-emphasis 0.97) and ncnn for inference.
 *
 * Audio input: two formats only.
 *   _detect()       — float* in [-1.0, 1.0] (Web Audio, soundfile, torch)
 *   _detect_int16() — int16_t* PCM (WAV files, microphones)
 *
 * Usage:
 *   OmniVadHandle vad = omni_vad_create("vad.omnivad", NULL);
 *   omni_vad_detect_int16(vad, pcm, n, &config, &segments, &count);
 *   omni_vad_destroy(vad);
 */

#ifndef OMNIVAD_H
#define OMNIVAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* DLL export macro for Windows */
#if defined(_WIN32) && defined(OMNIVAD_SHARED)
  #define OMNIVAD_API __declspec(dllexport)
#elif defined(_WIN32) && defined(OMNIVAD_IMPORT)
  #define OMNIVAD_API __declspec(dllimport)
#else
  #define OMNIVAD_API
#endif

/* -------------------------------------------------------------------------- */
/*  Version                                                                   */
/* -------------------------------------------------------------------------- */

#define OMNIVAD_VERSION_MAJOR 1
#define OMNIVAD_VERSION_MINOR 0
#define OMNIVAD_VERSION_PATCH 0

/* -------------------------------------------------------------------------- */
/*  Shared types                                                              */
/* -------------------------------------------------------------------------- */

/** Status and error codes returned by public API functions. */
typedef enum {
    OMNI_OK                  =  0,
    OMNI_ERR_NULL_HANDLE     = -1,
    OMNI_ERR_NULL_POINTER    = -2,
    OMNI_ERR_LOAD_BUNDLE     = -3,
    OMNI_ERR_LOAD_PARAM      = -4,
    OMNI_ERR_LOAD_MODEL      = -5,
    OMNI_ERR_LOAD_CMVN       = -6,
    OMNI_ERR_NO_FRAMES       = -7,
    OMNI_ERR_INFERENCE       = -8,
    OMNI_ERR_OUT_OF_MEMORY   = -9,
    OMNI_ERR_INVALID_ARG     = -10,
} OmniErrorCode;

/** A time segment with start/end in seconds. */
typedef struct {
    float start;    /* segment start time in seconds */
    float end;      /* segment end time in seconds */
} OmniSegment;

/** AED event class identifiers. */
typedef enum {
    OMNI_AED_SPEECH  = 0,
    OMNI_AED_SINGING = 1,
    OMNI_AED_MUSIC   = 2,
} OmniAedClass;

/** A labeled time segment for AED. */
typedef struct {
    float start;            /* segment start time in seconds */
    float end;              /* segment end time in seconds */
    OmniAedClass cls;       /* event class */
    float confidence;       /* average confidence over the segment */
} OmniAedSegment;

/* -------------------------------------------------------------------------- */
/*  Post-processing configuration                                             */
/* -------------------------------------------------------------------------- */

/**
 * Post-processing parameters applied to raw frame-level probabilities
 * to produce clean segments.
 *
 * Algorithm (matches Python VadPostprocessor exactly):
 *   1. Causal moving-average smoothing of probabilities.
 *   2. Binary threshold: frame is active if smoothed_prob >= threshold.
 *   3. 4-state machine: requires min_speech continuous speech to confirm,
 *      and min_silence continuous silence to end.
 *   4. Fix smooth window start: extend speech start backward.
 *   5. Merge short silence gaps < merge_silence_frames.
 *   6. Extend speech segments by extend_speech_frames in both directions.
 *   7. Force-split segments longer than max_speech_frames at lowest-prob point.
 *   8. Convert frame indices to timestamps: frame_idx * 0.01 seconds.
 */
typedef struct {
    float threshold;            /* activation threshold (default: 0.4)             */
    int   smooth_window_size;   /* causal moving-average window (default: 5)       */
    int   min_speech_frames;    /* min frames to confirm speech (default: 20)       */
    int   min_silence_frames;   /* min frames to confirm silence end (default: 20)  */
    int   max_speech_frames;    /* force-split longer segments (default: 2000)      */
    int   merge_silence_frames; /* merge silence gaps shorter than this (default: 0)*/
    int   extend_speech_frames; /* extend speech N frames each side (default: 0)    */
} OmniPostConfig;

/** Return a default post-processing config. */
OMNIVAD_API OmniPostConfig omni_post_config_default(void);

/* -------------------------------------------------------------------------- */
/*  1. VAD (whole audio -> speech segments)                                   */
/* -------------------------------------------------------------------------- */

/** Opaque handle for VAD. */
typedef struct OmniVadCtx* OmniVadHandle;

/**
 * Create a VAD instance from a .omnivad bundle file.
 *
 * @param bundle_path path to .omnivad bundle file
 * @param out_error   receives OMNI_OK on success or a detailed error code on failure
 * @return handle, or NULL on failure
 */
OMNIVAD_API OmniVadHandle omni_vad_create(const char* bundle_path, int* out_error);

/**
 * Create a VAD instance from an in-memory .omnivad bundle.
 *
 * @param data       pointer to bundle bytes
 * @param size       bundle size in bytes
 * @param out_error  receives OMNI_OK on success or error code on failure
 * @return handle, or NULL on failure
 */
OMNIVAD_API OmniVadHandle omni_vad_create_from_buffer(const void* data, int size, int* out_error);

/** Detect speech segments from float audio [-1.0, 1.0]. */
OMNIVAD_API int omni_vad_detect(
    OmniVadHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments,
    int* out_count
);

/** Detect speech segments from int16 PCM audio. */
OMNIVAD_API int omni_vad_detect_int16(
    OmniVadHandle handle,
    const int16_t* audio_data,
    int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments,
    int* out_count
);

/** Get per-frame speech probabilities from float audio [-1.0, 1.0]. */
OMNIVAD_API int omni_vad_detect_probs(
    OmniVadHandle handle,
    const float* audio_data, int num_samples,
    float** out_probs, int* out_frames
);

/** Get per-frame speech probabilities from int16 PCM audio. */
OMNIVAD_API int omni_vad_detect_probs_int16(
    OmniVadHandle handle,
    const int16_t* audio_data, int num_samples,
    float** out_probs, int* out_frames
);

/** Destroy VAD and free all resources. */
OMNIVAD_API void omni_vad_destroy(OmniVadHandle handle);

/* -------------------------------------------------------------------------- */
/*  2. Stream VAD (frame-by-frame with packed cache [1,1024,19])              */
/* -------------------------------------------------------------------------- */

/** Opaque handle for stream VAD. */
typedef struct OmniStreamVadCtx* OmniStreamVadHandle;

/** Per-frame result from stream VAD. */
typedef struct {
    float confidence;       /* speech probability [0, 1] */
    bool  is_speech;        /* true if confidence > threshold */
    int   frame_offset;     /* frame index (0-based, each frame = 10ms) */
} OmniStreamVadResult;

/**
 * Create a stream VAD instance from a .omnivad bundle file.
 *
 * @param bundle_path  path to .omnivad bundle file
 * @param threshold    speech threshold (typical: 0.5)
 * @param out_error    receives OMNI_OK on success or a detailed error code on failure
 * @return handle, or NULL on failure
 */
OMNIVAD_API OmniStreamVadHandle omni_stream_vad_create(
    const char* bundle_path,
    float threshold,
    int* out_error
);

/**
 * Create a stream VAD instance from an in-memory .omnivad bundle.
 *
 * @param data       pointer to bundle bytes
 * @param size       bundle size in bytes
 * @param threshold  speech threshold (typical: 0.5)
 * @param out_error  receives OMNI_OK on success or error code on failure
 * @return handle, or NULL on failure
 */
OMNIVAD_API OmniStreamVadHandle omni_stream_vad_create_from_buffer(
    const void* data,
    int size,
    float threshold,
    int* out_error
);

/**
 * Create a lightweight clone sharing model weights with fresh per-instance state.
 *
 * The clone shares the ncnn::Net and CMVN data with the source handle (via
 * reference counting), but has its own audio buffer, cache, frame offset, and
 * fbank. Ideal for multi-stream scenarios where many concurrent sessions share
 * the same model.
 *
 * @param handle     source stream VAD handle
 * @param out_error  receives OMNI_OK on success or error code on failure
 * @return new handle, or NULL on failure
 */
OMNIVAD_API OmniStreamVadHandle omni_stream_vad_clone(
    OmniStreamVadHandle handle,
    int* out_error
);

/**
 * Process one chunk of 16-bit PCM audio (typically 160 samples = 10ms @ 16kHz).
 *
 * @param handle      stream VAD handle
 * @param audio_data  16-bit PCM samples
 * @param num_samples number of samples (recommended: 160 for 10ms)
 * @param result      output per-frame result
 * @return OMNI_OK on success, OMNI_ERR_NO_FRAMES if buffering
 */
OMNIVAD_API int omni_stream_vad_process(
    OmniStreamVadHandle handle,
    const int16_t* audio_data,
    int num_samples,
    OmniStreamVadResult* result
);

/** Batch mode: process entire audio as float [-1.0, 1.0], return per-frame probs. */
OMNIVAD_API int omni_stream_vad_detect_full(
    OmniStreamVadHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames
);

/** Batch mode: process entire audio as int16 PCM, return per-frame probs. */
OMNIVAD_API int omni_stream_vad_detect_full_int16(
    OmniStreamVadHandle handle,
    const int16_t* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames
);

/** Reset all internal state (cache, audio buffer, frame offset). */
OMNIVAD_API void omni_stream_vad_reset(OmniStreamVadHandle handle);

/** Get current frame offset. */
OMNIVAD_API int omni_stream_vad_get_frame_offset(OmniStreamVadHandle handle);

/** Destroy stream VAD and free all resources. */
OMNIVAD_API void omni_stream_vad_destroy(OmniStreamVadHandle handle);

/* -------------------------------------------------------------------------- */
/*  3. AED (whole audio -> speech/singing/music segments)                     */
/* -------------------------------------------------------------------------- */

/** Opaque handle for AED. */
typedef struct OmniAedCtx* OmniAedHandle;

/** Per-class post-processing configuration for AED. */
typedef struct {
    OmniPostConfig speech;
    OmniPostConfig singing;
    OmniPostConfig music;
} OmniAedPostConfig;

/** Return default AED post-processing config. */
OMNIVAD_API OmniAedPostConfig omni_aed_post_config_default(void);

/**
 * Create an AED instance from a .omnivad bundle file.
 *
 * @param bundle_path path to .omnivad bundle file
 * @param out_error   receives OMNI_OK on success or a detailed error code on failure
 * @return handle, or NULL on failure
 */
OMNIVAD_API OmniAedHandle omni_aed_create(const char* bundle_path, int* out_error);

/**
 * Create an AED instance from an in-memory .omnivad bundle.
 *
 * @param data       pointer to bundle bytes
 * @param size       bundle size in bytes
 * @param out_error  receives OMNI_OK on success or error code on failure
 * @return handle, or NULL on failure
 */
OMNIVAD_API OmniAedHandle omni_aed_create_from_buffer(const void* data, int size, int* out_error);

/** Detect audio events from float audio [-1.0, 1.0]. */
OMNIVAD_API int omni_aed_detect(
    OmniAedHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments,
    int* out_count
);

/** Detect audio events from int16 PCM audio. */
OMNIVAD_API int omni_aed_detect_int16(
    OmniAedHandle handle,
    const int16_t* audio_data,
    int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments,
    int* out_count
);

/**
 * Get per-frame probabilities (3 classes) from float audio [-1.0, 1.0].
 * Output: out_probs[frame * 3 + class], class 0=speech, 1=singing, 2=music.
 */
OMNIVAD_API int omni_aed_detect_probs(
    OmniAedHandle handle,
    const float* audio_data, int num_samples,
    float** out_probs, int* out_frames
);

/** Get per-frame probabilities (3 classes) from int16 PCM audio. */
OMNIVAD_API int omni_aed_detect_probs_int16(
    OmniAedHandle handle,
    const int16_t* audio_data, int num_samples,
    float** out_probs, int* out_frames
);

/** Destroy AED handle and free all resources. */
OMNIVAD_API void omni_aed_destroy(OmniAedHandle handle);

/* -------------------------------------------------------------------------- */
/*  Memory management                                                         */
/* -------------------------------------------------------------------------- */

/** Free memory allocated by any detect function. Safe to call with NULL. */
OMNIVAD_API void omni_free(void* ptr);

/** Return a human-readable string for an error code. */
OMNIVAD_API const char* omni_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* OMNIVAD_H */
