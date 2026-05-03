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
    int   max_speech_frames;    /* force-split longer segments (default: 3000 = 30s)*/
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
    int   frame_offset;     /* 1-based count of processed frames (0 = no frame produced yet, each frame = 10ms) */
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

/** Get the 1-based count of processed frames so far (multiply by 0.01s for elapsed audio). */
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
/*  Chunking (segment list -> duration-bounded chunks)                         */
/* -------------------------------------------------------------------------- */

/**
 * Output of omni_merge_chunks(). One chunk bundles a contiguous slice of the
 * input segment array, capped by max_chunk_secs, with optional padding applied.
 */
typedef struct {
    float start;            /* chunk start in seconds (with pad_onset_secs, clamped >= 0) */
    float end;              /* chunk end in seconds (with pad_offset_secs)                */
    int   seg_start_idx;    /* index of first input segment included in this chunk   */
    int   seg_count;        /* number of input segments included                     */
} OmniChunk;

/**
 * Chunking strategy. Selects how omni_merge_chunks() packs segments into
 * duration-bounded chunks after the shared filter+merge preprocessing.
 *
 *   GREEDY        (default, backward-compatible)
 *     Sequentially append the next segment until it would exceed max_chunk_secs
 *     or its gap exceeds max_gap_secs. Predictable, fast, but split points may
 *     fall in the middle of a long pause.
 *     RECOMMENDED FOR fixed-length-input ASR (Whisper / whisperX, which
 *     pad to 30s anyway) — packs each chunk close to max_chunk_secs to
 *     minimize wasted padding.
 *
 *   LONGEST_GAP   (recursive, gap-aware)
 *     Recursively split at the LONGEST internal gap until BOTH constraints
 *     hold for every chunk: span <= max_chunk_secs AND no internal gap exceeds
 *     max_gap_secs. When no gap can be cut (single segment longer than
 *     max_chunk_secs), fall back to equal hard-split.
 *     RECOMMENDED FOR variable-length-input models (forced alignment,
 *     TTS, encoder-style ASR) — splits at natural pauses; no fixed-length
 *     padding required, so chunks of unequal length are fine.
 */
typedef enum {
    OMNI_CHUNK_GREEDY      = 0,
    OMNI_CHUNK_LONGEST_GAP = 1,
} OmniChunkMode;

/**
 * Configuration for omni_merge_chunks().
 *
 * Pipeline (both modes share Steps 1, 2, 5):
 *   1. Drop input segments whose duration < min_speech_secs.
 *   2. Pre-merge consecutive segments whose gap < min_silence_secs.
 *   3. Pack into chunks per `mode`. Both modes honor max_chunk_secs and
 *      max_gap_secs as hard constraints; they only differ in WHERE to cut.
 *      - GREEDY:      sequential append; split at the FIRST point that
 *                     would violate either constraint.
 *      - LONGEST_GAP: recursively split at the LONGEST internal gap until
 *                     every chunk satisfies both constraints.
 *   4. Hard-split any chunk still longer than max_chunk_secs into equal
 *      sub-chunks (handles single-segment-too-long for both modes).
 *   5. Apply pad_onset_secs / pad_offset_secs to each chunk's boundaries.
 */
typedef struct {
    float max_chunk_secs;    /* hard upper bound on chunk duration (seconds), > 0; pairs with VAD max_speech_frames */
    float max_gap_secs;      /* split if gap > this. INFINITY disables. Honored by both modes */
    float pad_onset_secs;    /* extend chunk start backward (seconds, clamped >= 0) */
    float pad_offset_secs;   /* extend chunk end forward (seconds) */
    float min_speech_secs;   /* drop segments shorter than this (seconds); pairs with VAD min_speech_frames */
    float min_silence_secs;  /* merge gaps shorter than this (seconds); pairs with VAD min_silence_frames */
    int   mode;              /* OmniChunkMode (default GREEDY for ABI compatibility) */
} OmniChunkConfig;

/**
 * Return the default chunking config.
 *
 * Defaults:
 *   max_chunk_secs   = 30.0      (matches Whisper 30s input window)
 *   max_gap_secs     = INFINITY  (disabled)
 *   pad_onset_secs   = 0.04
 *   pad_offset_secs  = 0.04
 *   min_speech_secs  = 0.0
 *   min_silence_secs = 0.20      (matches VAD min_silence_frames=20 @ 10ms shift)
 *   mode             = OMNI_CHUNK_GREEDY
 */
OMNIVAD_API OmniChunkConfig omni_chunk_config_default(void);

/**
 * Merge a sorted array of speech segments into duration-bounded chunks.
 *
 * Pure algorithm — does NOT depend on ncnn or any model. Safe to call from
 * any thread; no global state.
 *
 * Selects between two packing strategies via `config->mode`. Both modes
 * honor max_chunk_secs and max_gap_secs as hard constraints; they only differ in
 * WHERE to cut when forced to:
 *   OMNI_CHUNK_GREEDY      — sequential append; cuts at the FIRST point
 *                            that violates a constraint. Deterministic,
 *                            historical default.
 *   OMNI_CHUNK_LONGEST_GAP — recursive divide-and-conquer; cuts at the
 *                            LONGEST internal gap so the split lands on
 *                            the most natural pause. Falls back to equal
 *                            hard-split when a single segment alone
 *                            exceeds max_chunk_secs.
 *
 * @param segments      input array sorted by `start` (overlapping or
 *                      out-of-order is undefined behaviour)
 * @param num_segments  number of input segments (>= 0)
 * @param config        algorithm configuration; must have max_chunk_secs > 0
 * @param out_chunks    receives a malloc'd array (caller frees via omni_free).
 *                      On success and num_segments==0, *out_chunks==NULL and
 *                      *out_count==0.
 * @param out_count     receives chunk count
 * @return OMNI_OK, OMNI_ERR_NULL_POINTER, OMNI_ERR_INVALID_ARG,
 *         or OMNI_ERR_OUT_OF_MEMORY
 */
OMNIVAD_API int omni_merge_chunks(
    const OmniSegment*     segments,
    int                    num_segments,
    const OmniChunkConfig* config,
    OmniChunk**            out_chunks,
    int*                   out_count
);

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
