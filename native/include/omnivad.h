/*
 * OmniVAD Unified C API
 *
 * Provides three model types through a single header:
 *   1. Stream VAD    - frame-by-frame speech detection with packed cache
 *   2. Non-stream VAD - whole-audio speech segmentation with post-processing
 *   3. Non-stream AED - whole-audio audio event detection (speech/singing/music)
 *
 * All models use 80-dim log-mel fbank features (25ms window, 10ms shift,
 * Povey window, pre-emphasis 0.97) and ncnn for inference.
 *
 * Usage:
 *   // Create a model handle
 *   OmniVadHandle vad = omni_vad_stream_create(param, bin, means, istd);
 *
 *   // Process audio
 *   omni_vad_stream_process(vad, pcm, 160, &result);
 *
 *   // Destroy
 *   omni_vad_stream_destroy(vad);
 */

#ifndef OMNIVAD_H
#define OMNIVAD_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* -------------------------------------------------------------------------- */
/*  Version                                                                   */
/* -------------------------------------------------------------------------- */

#define OMNIVAD_VERSION_MAJOR 1
#define OMNIVAD_VERSION_MINOR 0
#define OMNIVAD_VERSION_PATCH 0

/* -------------------------------------------------------------------------- */
/*  Shared types                                                              */
/* -------------------------------------------------------------------------- */

/** Error codes returned by all API functions. */
typedef enum {
    OMNI_OK                  =  0,
    OMNI_ERR_NULL_HANDLE     = -1,
    OMNI_ERR_NULL_INPUT      = -2,
    OMNI_ERR_LOAD_PARAM      = -3,
    OMNI_ERR_LOAD_MODEL      = -4,
    OMNI_ERR_LOAD_CMVN       = -5,
    OMNI_ERR_NO_FRAMES       = -6,
    OMNI_ERR_INFERENCE       = -7,
    OMNI_ERR_OUT_OF_MEMORY   = -8,
    OMNI_ERR_INVALID_ARG     = -9,
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
    OmniAedClass cls;    /* event class */
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
OmniPostConfig omni_post_config_default(void);

/* -------------------------------------------------------------------------- */
/*  1. Stream VAD (frame-by-frame with packed cache [1,1024,19])              */
/* -------------------------------------------------------------------------- */

/** Opaque handle for stream VAD. */
typedef struct OmniVadStreamCtx* OmniVadHandle;

/** Per-frame result from stream VAD. */
typedef struct {
    float confidence;       /* speech probability [0, 1] */
    bool  is_speech;        /* true if confidence > threshold */
    int   frame_offset;     /* frame index (0-based, each frame = 10ms) */
} OmniVadStreamResult;

/**
 * Create a stream VAD instance.
 *
 * @param model_param  ncnn .param file path
 * @param model_bin    ncnn .bin file path
 * @param cmvn_means   binary file of float[80] means (NULL to skip CMVN)
 * @param cmvn_istd    binary file of float[80] inverse-std (NULL to skip CMVN)
 * @param threshold    speech threshold (typical: 0.5)
 * @return handle, or NULL on failure
 */
OmniVadHandle omni_vad_stream_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd,
    float threshold
);

/**
 * Process one chunk of 16-bit PCM audio (typically 160 samples = 10ms @ 16kHz).
 *
 * Internally accumulates a sliding window of 25ms and extracts one fbank frame,
 * then runs ncnn with packed cache [1,1024,19].
 *
 * @param handle      stream VAD handle
 * @param audio_data  16-bit PCM samples
 * @param num_samples number of samples (recommended: 160 for 10ms)
 * @param result      output per-frame result
 * @return OMNI_OK on success, error code otherwise
 */
int omni_vad_stream_process(
    OmniVadHandle handle,
    const int16_t* audio_data,
    int num_samples,
    OmniVadStreamResult* result
);

/**
 * Process a complete audio file (batch mode, matches Python detect_full).
 *
 * Uses whole-file fbank extraction (not per-frame sliding window) then
 * runs the stream model frame-by-frame with cache. This produces results
 * identical to Python's OmniStreamVad.detect_full().
 *
 * @param handle       stream VAD handle
 * @param audio_data   mono float samples (16kHz)
 * @param num_samples  total number of samples
 * @param out_probs    pointer to receive allocated float array of per-frame probs
 * @param out_frames   pointer to receive number of frames
 * @return OMNI_OK on success
 */
int omni_vad_stream_detect_full(
    OmniVadHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames
);

/**
 * Create a stream VAD instance from a .omnivad bundle file.
 * The bundle contains param, bin, and CMVN data in a single file.
 */
OmniVadHandle omni_vad_stream_create_from_bundle(
    const char* bundle_path,
    float threshold
);

/** Reset all internal state (cache, audio buffer, frame offset). */
void omni_vad_stream_reset(OmniVadHandle handle);

/** Get current frame offset. */
int omni_vad_stream_get_frame_offset(OmniVadHandle handle);

/** Destroy stream VAD and free all resources. */
void omni_vad_stream_destroy(OmniVadHandle handle);

/* -------------------------------------------------------------------------- */
/*  2. Non-stream VAD (whole audio -> speech segments)                        */
/* -------------------------------------------------------------------------- */

/** Opaque handle for non-stream VAD. */
typedef struct OmniVadNonStreamCtx* OmniVadNonStreamHandle;

/**
 * Create a non-stream VAD instance.
 *
 * @param model_param  ncnn .param file path (non-stream VAD model)
 * @param model_bin    ncnn .bin file path
 * @param cmvn_means   binary file of float[80] means
 * @param cmvn_istd    binary file of float[80] inverse-std
 * @return handle, or NULL on failure
 */
OmniVadNonStreamHandle omni_vad_nonstream_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd
);

/** Create from a .omnivad bundle file. */
OmniVadNonStreamHandle omni_vad_nonstream_create_from_bundle(const char* bundle_path);

/*
 * Audio format conventions:
 *
 *   _process()     — float* in int16 range [-32768, 32767]. This is the native
 *                    format used internally (ncnn fbank expects unscaled PCM floats).
 *   _process_i16() — int16_t* PCM. Converted to float internally (zero-copy cast).
 *   _process_f32() — float* in normalized [-1.0, 1.0] range (Web Audio API format).
 *                    Scaled by 32768 internally before processing.
 */

/**
 * Process float audio (int16 range) and detect speech segments.
 *
 * @param audio_data    mono float samples in int16 range [-32768, 32767] (16kHz)
 */
int omni_vad_nonstream_process(
    OmniVadNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments,
    int* out_count
);

/** Process int16 PCM audio and detect speech segments. */
int omni_vad_nonstream_process_i16(
    OmniVadNonStreamHandle handle,
    const int16_t* audio_data,
    int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments,
    int* out_count
);

/** Process normalized float audio [-1.0, 1.0] and detect speech segments. */
int omni_vad_nonstream_process_f32(
    OmniVadNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments,
    int* out_count
);

/**
 * Process float audio (int16 range) and return raw frame-level probabilities.
 */
int omni_vad_nonstream_process_raw(
    OmniVadNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames
);

/** Destroy non-stream VAD and free all resources. */
void omni_vad_nonstream_destroy(OmniVadNonStreamHandle handle);

/* -------------------------------------------------------------------------- */
/*  3. Non-stream AED (whole audio -> speech/singing/music segments)          */
/* -------------------------------------------------------------------------- */

/** Opaque handle for non-stream AED. */
typedef struct OmniAedNonStreamCtx* OmniAedNonStreamHandle;

/** Per-class post-processing configuration for AED. */
typedef struct {
    OmniPostConfig speech;
    OmniPostConfig singing;
    OmniPostConfig music;
} OmniAedPostConfig;

/** Return default AED post-processing config. */
OmniAedPostConfig omni_aed_post_config_default(void);

/**
 * Create a non-stream AED instance.
 *
 * @param model_param  ncnn .param file path (AED model, 3-class output)
 * @param model_bin    ncnn .bin file path
 * @param cmvn_means   binary file of float[80] means
 * @param cmvn_istd    binary file of float[80] inverse-std
 * @return handle, or NULL on failure
 */
OmniAedNonStreamHandle omni_aed_nonstream_create(
    const char* model_param,
    const char* model_bin,
    const char* cmvn_means,
    const char* cmvn_istd
);

/** Create from a .omnivad bundle file. */
OmniAedNonStreamHandle omni_aed_nonstream_create_from_bundle(const char* bundle_path);

/** Process float audio (int16 range) and detect audio events. */
int omni_aed_nonstream_process(
    OmniAedNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments,
    int* out_count
);

/** Process int16 PCM audio and detect audio events. */
int omni_aed_nonstream_process_i16(
    OmniAedNonStreamHandle handle,
    const int16_t* audio_data,
    int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments,
    int* out_count
);

/** Process normalized float audio [-1.0, 1.0] and detect audio events. */
int omni_aed_nonstream_process_f32(
    OmniAedNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments,
    int* out_count
);

/**
 * Process float audio (int16 range) and return raw frame-level probabilities.
 * Output layout: out_probs[frame * 3 + class], where class 0=speech, 1=singing, 2=music.
 */
int omni_aed_nonstream_process_raw(
    OmniAedNonStreamHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames
);

/** Destroy AED handle and free all resources. */
void omni_aed_nonstream_destroy(OmniAedNonStreamHandle handle);

/* -------------------------------------------------------------------------- */
/*  Memory management                                                         */
/* -------------------------------------------------------------------------- */

/**
 * Free memory allocated by any omni_*_process function.
 * Safe to call with NULL.
 */
void omni_free(void* ptr);

/** Return a human-readable string for an error code. */
const char* omni_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* OMNIVAD_H */
