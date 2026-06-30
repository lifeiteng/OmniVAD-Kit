/*
 * OmniVAD Unified C API - Implementation (ncnn backend)
 *
 * Contains:
 *   - Stream VAD with packed cache [1,1024,19]
 *   - VAD (whole-audio inference + post-processing)
 *   - AED (3-class: speech/singing/music + per-class post-processing)
 *
 * All models share the same 80-dim log-mel fbank frontend:
 *   sample_rate=16000, frame_length=400 (25ms), frame_shift=160 (10ms),
 *   Povey window, pre-emphasis=0.97, remove_dc_offset=true.
 */

#include "omnivad.h"
#include "frontend/fbank.h"
#include "datareader.h"
#include "net.h"

#include <algorithm>
#include <memory>
#ifndef __EMSCRIPTEN__
#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <fcntl.h>
#endif
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <new>
#include <string>
#include <vector>

/* -------------------------------------------------------------------------- */
/*  Exception guards                                                           */
/*                                                                            */
/*  The native and Python builds compile with exceptions enabled, so the      */
/*  std::bad_alloc guards below stay active. The WebAssembly build compiles    */
/*  with -fno-exceptions (size-optimized, ncnn built NCNN_DISABLE_EXCEPTION),  */
/*  where `try`/`catch` is a hard error. There the guards degrade to plain     */
/*  blocks: an allocation failure aborts via emmalloc instead of returning     */
/*  OMNI_ERR_OUT_OF_MEMORY, which is the accepted behavior for that target.    */
/* -------------------------------------------------------------------------- */
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#define OMNI_TRY try
#define OMNI_CATCH_BAD_ALLOC catch (const std::bad_alloc&)
#else
#define OMNI_TRY
#define OMNI_CATCH_BAD_ALLOC if (false)
#endif

/* -------------------------------------------------------------------------- */
/*  Constants                                                                 */
/* -------------------------------------------------------------------------- */

static const int SAMPLE_RATE       = 16000;
static const int FEAT_DIM          = 80;
static const int FRAME_LENGTH      = 400;   /* 25ms @ 16kHz */
static const int FRAME_SHIFT       = 160;   /* 10ms @ 16kHz */
static const float FRAME_SHIFT_SEC = 0.01f; /* 10ms per frame */

/* Stream VAD packed cache dimensions */
static const int CACHE_SIZE = 1024;  /* 8 * 128 */
static const int CACHE_LEN  = 19;

/* -------------------------------------------------------------------------- */
/*  Helpers: .omnivad bundle loading                                          */
/* -------------------------------------------------------------------------- */

/*
 * Bundle format: OVAD(4) + version(4) + param_size(4) + bin_size(4)
 *                + cmvn_means_size(4) + cmvn_istd_size(4)
 *                + param_data + bin_data + cmvn_means + cmvn_istd
 */
struct OmniBundle {
    std::vector<char> param_data;
    std::vector<char> bin_data;
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool has_cmvn;
};

static void set_out_error(int* out_error, int code) {
    if (out_error) *out_error = code;
}

static int validate_num_samples(int num_samples) {
    return (num_samples < 0) ? OMNI_ERR_INVALID_ARG : OMNI_OK;
}

static int validate_threshold(float threshold) {
    if (std::isnan(threshold) || threshold < 0.0f || threshold > 1.0f) {
        return OMNI_ERR_INVALID_ARG;
    }
    return OMNI_OK;
}

static int validate_post_config(const OmniPostConfig* config) {
    if (!config) return OMNI_OK;

    if (std::isnan(config->threshold) || config->threshold < 0.0f || config->threshold > 1.0f) {
        return OMNI_ERR_INVALID_ARG;
    }
    if (config->smooth_window_size < 0 ||
        config->min_speech_frames < 0 ||
        config->min_silence_frames < 0 ||
        config->max_speech_frames < 0 ||
        config->merge_silence_frames < 0 ||
        config->extend_speech_frames < 0) {
        return OMNI_ERR_INVALID_ARG;
    }
    return OMNI_OK;
}

static int validate_aed_post_config(const OmniAedPostConfig* config) {
    if (!config) return OMNI_OK;

    int ret = validate_post_config(&config->speech);
    if (ret != OMNI_OK) return ret;
    ret = validate_post_config(&config->singing);
    if (ret != OMNI_OK) return ret;
    return validate_post_config(&config->music);
}

static int load_bundle(const char* path, OmniBundle& bundle) {
    if (!path) {
        fprintf(stderr, "[omnivad] bundle path is null\n");
        return OMNI_ERR_NULL_POINTER;
    }
    if (path[0] == '\0') {
        fprintf(stderr, "[omnivad] bundle path is empty\n");
        return OMNI_ERR_INVALID_ARG;
    }

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[omnivad] failed to open bundle: %s\n", path);
        return OMNI_ERR_LOAD_BUNDLE;
    }

    /* Read header (24 bytes) */
    char magic[4];
    uint32_t version, param_size, bin_size, means_size, istd_size;

    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "OVAD", 4) != 0) {
        fprintf(stderr, "[omnivad] invalid bundle magic in: %s\n", path);
        fclose(fp);
        return OMNI_ERR_LOAD_BUNDLE;
    }
    if (fread(&version, 4, 1, fp) != 1 || version != 1) {
        fprintf(stderr, "[omnivad] unsupported bundle version %u in: %s\n", version, path);
        fclose(fp);
        return OMNI_ERR_LOAD_BUNDLE;
    }
    if (fread(&param_size, 4, 1, fp) != 1 ||
        fread(&bin_size, 4, 1, fp) != 1 ||
        fread(&means_size, 4, 1, fp) != 1 ||
        fread(&istd_size, 4, 1, fp) != 1) {
        fprintf(stderr, "[omnivad] truncated bundle header in: %s\n", path);
        fclose(fp);
        return OMNI_ERR_LOAD_BUNDLE;
    }

    /* Read data sections */
    bundle.param_data.resize(param_size);
    bundle.bin_data.resize(bin_size);

    if (fread(bundle.param_data.data(), 1, param_size, fp) != param_size ||
        fread(bundle.bin_data.data(), 1, bin_size, fp) != bin_size) {
        fprintf(stderr, "[omnivad] truncated bundle data in: %s\n", path);
        fclose(fp);
        return OMNI_ERR_LOAD_BUNDLE;
    }

    /* CMVN (optional but expected) */
    bundle.has_cmvn = (means_size > 0 && istd_size > 0);
    if (bundle.has_cmvn) {
        if ((means_size % sizeof(float)) != 0 || (istd_size % sizeof(float)) != 0) {
            fprintf(stderr, "[omnivad] invalid CMVN payload size in: %s\n", path);
            fclose(fp);
            return OMNI_ERR_LOAD_CMVN;
        }
        int means_dim = means_size / sizeof(float);
        int istd_dim = istd_size / sizeof(float);
        bundle.cmvn_means.resize(means_dim);
        bundle.cmvn_istd.resize(istd_dim);
        if (fread(bundle.cmvn_means.data(), 1, means_size, fp) != means_size ||
            fread(bundle.cmvn_istd.data(), 1, istd_size, fp) != istd_size) {
            fprintf(stderr, "[omnivad] truncated CMVN data in: %s\n", path);
            fclose(fp);
            return OMNI_ERR_LOAD_CMVN;
        }
    }

    fclose(fp);
    return OMNI_OK;
}

static int load_bundle_from_buffer(const void* data, int size, OmniBundle& bundle) {
    if (!data) {
        fprintf(stderr, "[omnivad] bundle data is null\n");
        return OMNI_ERR_NULL_POINTER;
    }
    if (size < 24) {
        fprintf(stderr, "[omnivad] bundle data too small (%d bytes)\n", size);
        return OMNI_ERR_LOAD_BUNDLE;
    }

    const unsigned char* p = (const unsigned char*)data;
    const unsigned char* end = p + size;

    if (memcmp(p, "OVAD", 4) != 0) {
        fprintf(stderr, "[omnivad] invalid bundle magic in buffer\n");
        return OMNI_ERR_LOAD_BUNDLE;
    }
    p += 4;

    uint32_t version, param_size, bin_size, means_size, istd_size;
    memcpy(&version, p, 4); p += 4;
    if (version != 1) {
        fprintf(stderr, "[omnivad] unsupported bundle version %u in buffer\n", version);
        return OMNI_ERR_LOAD_BUNDLE;
    }
    memcpy(&param_size, p, 4); p += 4;
    memcpy(&bin_size, p, 4); p += 4;
    memcpy(&means_size, p, 4); p += 4;
    memcpy(&istd_size, p, 4); p += 4;

    if (p + param_size + bin_size > end) {
        fprintf(stderr, "[omnivad] truncated bundle data in buffer\n");
        return OMNI_ERR_LOAD_BUNDLE;
    }

    bundle.param_data.assign(p, p + param_size); p += param_size;
    bundle.bin_data.assign(p, p + bin_size); p += bin_size;

    bundle.has_cmvn = (means_size > 0 && istd_size > 0);
    if (bundle.has_cmvn) {
        if ((means_size % sizeof(float)) != 0 || (istd_size % sizeof(float)) != 0) {
            fprintf(stderr, "[omnivad] invalid CMVN payload size in buffer\n");
            return OMNI_ERR_LOAD_CMVN;
        }
        if (p + means_size + istd_size > end) {
            fprintf(stderr, "[omnivad] truncated CMVN data in buffer\n");
            return OMNI_ERR_LOAD_CMVN;
        }
        int means_dim = means_size / sizeof(float);
        int istd_dim = istd_size / sizeof(float);
        bundle.cmvn_means.resize(means_dim);
        bundle.cmvn_istd.resize(istd_dim);
        memcpy(bundle.cmvn_means.data(), p, means_size); p += means_size;
        memcpy(bundle.cmvn_istd.data(), p, istd_size);
    }

    return OMNI_OK;
}

/* Load ncnn model from in-memory param/bin data */
static ncnn::Net* load_ncnn_from_memory(
    const std::vector<char>& param_data,
    const std::vector<char>& bin_data,
    int* out_error)
{
    ncnn::Net* net = new (std::nothrow) ncnn::Net();
    if (!net) {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }

    /* load_param_mem requires a null-terminated C string (it uses sscanf
     * internally). The bundle stores param text without a trailing '\0',
     * so we must copy into a std::string to guarantee null termination.
     * Without this, sscanf reads past the buffer end — harmless in
     * single-threaded runs (heap padding is usually zero) but causes
     * sporadic parse failures under high concurrency due to randomized
     * heap layout. */
    std::string param_str(param_data.begin(), param_data.end());
    if (net->load_param_mem(param_str.c_str()) != 0) {
        fprintf(stderr, "[omnivad] failed to load param from bundle\n");
        delete net;
        set_out_error(out_error, OMNI_ERR_LOAD_PARAM);
        return NULL;
    }

    /* Load model binary directly from memory via DataReaderFromMemory.
     * No temp files needed — works everywhere including WASM. */
    const unsigned char* bin_ptr = (const unsigned char*)bin_data.data();
    ncnn::DataReaderFromMemory dr(bin_ptr);
    if (net->load_model(dr) != 0) {
        fprintf(stderr, "[omnivad] failed to load model bin from memory\n");
        delete net;
        set_out_error(out_error, OMNI_ERR_LOAD_MODEL);
        return NULL;
    }
    set_out_error(out_error, OMNI_OK);
    return net;
}

static void apply_cmvn(const std::vector<float>& means,
                       const std::vector<float>& istd,
                       float* features,
                       int num_frames, int feat_dim) {
    if ((int)means.size() < feat_dim || (int)istd.size() < feat_dim) return;
    for (int t = 0; t < num_frames; ++t) {
        float* f = features + t * feat_dim;
        for (int d = 0; d < feat_dim; ++d) {
            f[d] = (f[d] - means[d]) * istd[d];
        }
    }
}

/* -------------------------------------------------------------------------- */
/*  Helpers: Post-processing                                                  */
/*  Exact port of Python VadPostprocessor from vad_postprocessor.py           */
/* -------------------------------------------------------------------------- */

struct FrameSegment {
    int start;
    int end;
};

/* VAD state machine states (matches Python VadState enum) */
enum VadState {
    VAD_SILENCE          = 0,
    VAD_POSSIBLE_SPEECH  = 1,
    VAD_SPEECH           = 2,
    VAD_POSSIBLE_SILENCE = 3,
};

/*
 * Step 1: Causal moving-average smoothing.
 * Matches Python: np.convolve(probs, kernel, mode='full')[:len(probs)]
 * with boundary fix: first window_size-1 frames use cumulative average.
 */
static void smooth_probs_causal(float* probs, int num_frames, int window_size) {
    if (window_size <= 1 || num_frames <= 0) return;

    std::vector<float> buf(num_frames);

    /* Boundary: first window_size-1 frames use cumulative average */
    int boundary = (window_size - 1 < num_frames) ? window_size - 1 : num_frames;
    for (int i = 0; i < boundary; ++i) {
        float sum = 0.0f;
        for (int j = 0; j <= i; ++j) sum += probs[j];
        buf[i] = sum / (float)(i + 1);
    }

    /* Main: causal moving average with window_size */
    float window_sum = 0.0f;
    for (int j = 0; j < window_size && j < num_frames; ++j) window_sum += probs[j];

    for (int i = boundary; i < num_frames; ++i) {
        /* np.convolve(probs, ones/N, 'full')[:len] is a causal filter:
         * smoothed[i] = mean(probs[i-window_size+1 .. i]) */
        if (i >= window_size) {
            window_sum = 0.0f;
            for (int j = i - window_size + 1; j <= i; ++j) window_sum += probs[j];
        }
        buf[i] = window_sum / (float)window_size;
        /* Slide window forward */
        if (i + 1 < num_frames) {
            window_sum += probs[i + 1];
            window_sum -= probs[i - window_size + 1];
        }
    }

    memcpy(probs, buf.data(), sizeof(float) * num_frames);
}

/*
 * Step 2: Apply threshold. Matches Python: (probs >= threshold).
 */
static void apply_threshold(const float* probs, int* binary, int num_frames, float threshold) {
    for (int i = 0; i < num_frames; ++i) {
        binary[i] = (probs[i] >= threshold) ? 1 : 0;
    }
}

/*
 * Step 3: 4-state machine smoothing.
 * Matches Python _smooth_preds_with_state_machine exactly.
 */
static void state_machine_smooth(const int* binary, int* decisions, int num_frames,
                                  int min_speech_frames, int min_silence_frames) {
    if (min_speech_frames <= 0 && min_silence_frames <= 0) {
        memcpy(decisions, binary, sizeof(int) * num_frames);
        return;
    }

    VadState state = VAD_SILENCE;
    int speech_start = -1;
    int silence_start = -1;

    for (int t = 0; t < num_frames; ++t) {
        int is_speech = binary[t];

        if (state == VAD_SILENCE) {
            if (is_speech) {
                state = VAD_POSSIBLE_SPEECH;
                speech_start = t;
            }
        } else if (state == VAD_POSSIBLE_SPEECH) {
            if (is_speech) {
                if (t - speech_start >= min_speech_frames) {
                    state = VAD_SPEECH;
                    /* Backfill: mark speech_start..t-1 as speech */
                    for (int j = speech_start; j < t; ++j) decisions[j] = 1;
                }
            } else {
                state = VAD_SILENCE;
                speech_start = -1;
            }
        } else if (state == VAD_SPEECH) {
            if (!is_speech) {
                state = VAD_POSSIBLE_SILENCE;
                silence_start = t;
            }
        } else if (state == VAD_POSSIBLE_SILENCE) {
            if (!is_speech) {
                if (t - silence_start >= min_silence_frames) {
                    state = VAD_SILENCE;
                    speech_start = -1;
                }
            } else {
                state = VAD_SPEECH;
                silence_start = -1;
            }
        }

        /* Current frame's decision */
        if (state == VAD_SPEECH || state == VAD_POSSIBLE_SILENCE) {
            decisions[t] = 1;
        } else {
            decisions[t] = 0;
        }
    }
}

/*
 * Step 4: Fix smooth window start.
 * Extend speech starts backward by smooth_window_size frames.
 * Matches Python _fix_smooth_window_start.
 */
static void fix_smooth_window_start(int* decisions, int num_frames, int smooth_window_size) {
    if (smooth_window_size <= 0) return;
    std::vector<int> orig(decisions, decisions + num_frames);
    for (int t = 1; t < num_frames; ++t) {
        if (orig[t - 1] == 0 && orig[t] == 1) {
            int start = (t - smooth_window_size > 0) ? (t - smooth_window_size) : 0;
            for (int j = start; j < t; ++j) decisions[j] = 1;
        }
    }
}

/*
 * Step 5: Merge short silence segments.
 * Matches Python _merge_short_silence_segments.
 */
static void merge_short_silence(int* decisions, int num_frames, int merge_silence_frames) {
    if (merge_silence_frames <= 0) return;
    std::vector<int> orig(decisions, decisions + num_frames);
    int silence_start = -1;
    for (int t = 0; t < num_frames; ++t) {
        if (t > 0 && orig[t - 1] == 1 && orig[t] == 0 && silence_start < 0) {
            silence_start = t;
        } else if (t > 0 && orig[t - 1] == 0 && orig[t] == 1 && silence_start >= 0) {
            int gap = t - silence_start;
            if (gap < merge_silence_frames) {
                for (int j = silence_start; j < t; ++j) decisions[j] = 1;
            }
            silence_start = -1;
        }
    }
}

/*
 * Step 6: Extend speech segments N frames before and after.
 * Matches Python _extend_speech_segments (convolution-based).
 */
static void extend_speech(int* decisions, int num_frames, int extend_frames) {
    if (extend_frames <= 0) return;
    std::vector<int> orig(decisions, decisions + num_frames);
    for (int t = 0; t < num_frames; ++t) {
        if (orig[t] == 1) {
            int lo = (t - extend_frames > 0) ? (t - extend_frames) : 0;
            int hi = (t + extend_frames < num_frames) ? (t + extend_frames + 1) : num_frames;
            for (int j = lo; j < hi; ++j) decisions[j] = 1;
        }
    }
}

/*
 * Step 7: Split long speech segments at lowest-probability point.
 * Matches Python _split_long_speech_segments + _find_split_points.
 */
static void split_long_segments(int* decisions, int num_frames,
                                 int max_speech_frames, const float* raw_probs) {
    if (max_speech_frames <= 0) return;

    /* Find speech segments from decisions */
    int seg_start = -1;
    std::vector<std::pair<int,int>> segments;
    for (int t = 0; t < num_frames; ++t) {
        if (decisions[t] == 1 && seg_start < 0) {
            seg_start = t;
        } else if (decisions[t] == 0 && seg_start >= 0) {
            segments.push_back({seg_start, t});
            seg_start = -1;
        }
    }
    if (seg_start >= 0) segments.push_back({seg_start, num_frames});

    /* For each segment, find split points */
    for (auto& seg : segments) {
        int s = seg.first;
        int e = seg.second;
        int length = e - s;
        int local_start = 0;

        while (local_start < length) {
            if ((length - local_start) <= max_speech_frames) break;

            int window_start = local_start + max_speech_frames / 2;
            int window_end = local_start + max_speech_frames;
            if (window_end > length) window_end = length;

            /* Find min prob in window */
            int min_idx = window_start;
            float min_val = raw_probs[s + window_start];
            for (int j = window_start + 1; j < window_end; ++j) {
                if (raw_probs[s + j] < min_val) {
                    min_val = raw_probs[s + j];
                    min_idx = j;
                }
            }

            /* Insert silence at split point */
            decisions[s + min_idx] = 0;
            local_start = min_idx + 1;
        }
    }
}

/*
 * Convert frame-level decisions to time segments.
 * Matches Python decision_to_segment: uses FRAME_SHIFT_S for start,
 * and for trailing segments adds FRAME_LENGTH_S to end time.
 */
static const float FRAME_LENGTH_SEC = 0.025f;

static std::vector<FrameSegment> decisions_to_segments(
    const int* decisions, int num_frames, int min_speech_frames)
{
    std::vector<FrameSegment> segments;
    int seg_start = -1;
    for (int t = 0; t < num_frames; ++t) {
        if (decisions[t] == 1 && seg_start < 0) {
            seg_start = t;
        } else if (decisions[t] == 0 && seg_start >= 0) {
            segments.push_back({seg_start, t});
            seg_start = -1;
        }
    }
    if (seg_start >= 0) {
        segments.push_back({seg_start, num_frames});
    }
    return segments;
}

/*
 * Full post-processing pipeline matching Python VadPostprocessor.process().
 * raw_probs is the unmodified model output (needed for split_long_segments).
 */
static std::vector<FrameSegment> postprocess_pipeline(
    const float* raw_probs, int num_frames,
    float threshold, int smooth_window_size,
    int min_speech_frames, int max_speech_frames,
    int min_silence_frames, int merge_silence_frames,
    int extend_speech_frames)
{
    /* Work on a copy of probs for smoothing */
    std::vector<float> probs(raw_probs, raw_probs + num_frames);

    /* Step 1: Causal smoothing */
    smooth_probs_causal(probs.data(), num_frames, smooth_window_size);

    /* Step 2: Binary threshold */
    std::vector<int> binary(num_frames);
    apply_threshold(probs.data(), binary.data(), num_frames, threshold);

    /* Step 3: State machine smoothing */
    std::vector<int> decisions(num_frames, 0);
    state_machine_smooth(binary.data(), decisions.data(), num_frames,
                          min_speech_frames, min_silence_frames);

    /* Step 4: Fix smooth window start */
    fix_smooth_window_start(decisions.data(), num_frames, smooth_window_size);

    /* Step 5: Merge short silence */
    merge_short_silence(decisions.data(), num_frames, merge_silence_frames);

    /* Step 6: Extend speech */
    extend_speech(decisions.data(), num_frames, extend_speech_frames);

    /* Step 7: Split long segments (uses raw_probs, not smoothed) */
    split_long_segments(decisions.data(), num_frames, max_speech_frames, raw_probs);

    /* Convert to segments */
    return decisions_to_segments(decisions.data(), num_frames, min_speech_frames);
}

/* -------------------------------------------------------------------------- */
/*  Default configs                                                           */
/* -------------------------------------------------------------------------- */

extern "C" {

OmniPostConfig omni_post_config_default(void) {
    OmniPostConfig cfg;
    cfg.threshold            = 0.4f;
    cfg.smooth_window_size   = 5;
    cfg.min_speech_frames    = 20;    /* 200ms */
    cfg.min_silence_frames   = 20;    /* 200ms */
    cfg.max_speech_frames    = 3000;  /* 30s; matches Whisper input window */
    cfg.merge_silence_frames = 0;     /* disabled */
    cfg.extend_speech_frames = 0;     /* disabled */
    return cfg;
}

OmniAedPostConfig omni_aed_post_config_default(void) {
    OmniAedPostConfig cfg;
    cfg.speech  = omni_post_config_default();
    cfg.singing = omni_post_config_default();
    cfg.music   = omni_post_config_default();
    /* AED classes may benefit from different defaults */
    cfg.speech.threshold  = 0.5f;
    cfg.singing.threshold = 0.5f;
    cfg.music.threshold   = 0.5f;
    return cfg;
}

const char* omni_error_string(int error_code) {
    switch (error_code) {
        case OMNI_OK:                return "success";
        case OMNI_ERR_NULL_HANDLE:   return "null handle";
        case OMNI_ERR_NULL_POINTER:  return "null pointer";
        case OMNI_ERR_LOAD_BUNDLE:   return "failed to load or parse .omnivad bundle";
        case OMNI_ERR_LOAD_MODEL:    return "failed to load ncnn model from bundle";
        case OMNI_ERR_LOAD_CMVN:     return "failed to load CMVN data from bundle";
        case OMNI_ERR_NO_FRAMES:     return "audio does not contain enough samples for one frame";
        case OMNI_ERR_INFERENCE:     return "ncnn inference failed";
        case OMNI_ERR_OUT_OF_MEMORY: return "out of memory";
        case OMNI_ERR_INVALID_ARG:   return "invalid argument";
        default:                         return "unknown error";
    }
}

void omni_free(void* ptr) {
    free(ptr);
}

/* -- int16 and normalized float32 conversion helpers ----------------------- */
/* These internal helpers return std::vector and must keep C++ linkage even   */
/* though they live inside the extern "C" API block: MSVC rejects a           */
/* C-linkage function returning a C++ class (error C2526), while GCC/Clang     */
/* only warn (C4190). extern "C++" restores C++ linkage for them.             */

extern "C++" {

static std::vector<float> i16_to_float(const int16_t* data, int n) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) out[i] = (float)data[i];
    return out;
}

static std::vector<float> f32_normalize_to_int16_range(const float* data, int n) {
    std::vector<float> out(n);
    for (int i = 0; i < n; ++i) out[i] = data[i] * 32768.0f;
    return out;
}

} /* extern "C++" */

/* ========================================================================== */
/*  1. Stream VAD Implementation                                              */
/* ========================================================================== */

/* Shared assets: model weights + CMVN (read-only, ref-counted via shared_ptr) */
struct OmniStreamVadSharedAssets {
    std::shared_ptr<ncnn::Net> net;
    std::shared_ptr<const std::vector<float>> cmvn_means;
    std::shared_ptr<const std::vector<float>> cmvn_istd;
    bool use_cmvn;
};

/* Stream VAD post-processing state machine, bit-identical to upstream
 * fireredvad/core/stream_vad_postprocessor.py::StreamVadPostprocessor. */
enum StreamVadState {
    STREAM_VAD_SILENCE          = 0,
    STREAM_VAD_POSSIBLE_SPEECH  = 1,
    STREAM_VAD_SPEECH           = 2,
    STREAM_VAD_POSSIBLE_SILENCE = 3,
};

struct OmniStreamVadCtx {
    /* Shared model assets (shared across clones) */
    std::shared_ptr<OmniStreamVadSharedAssets> shared;

    /* Per-instance mutable state */
    std::deque<float> audio_buffer;
    ncnn::Mat cache_packed;
    int frame_offset;          /* 1-based count of processed frames (== frame_idx of last result) */
    vad::Fbank* fbank;

    /* Post-processing config (copied at create) */
    OmniStreamVadConfig cfg;
    int effective_pad_start;   /* max(smooth_window_size, pad_start_frame) */

    /* Smoothing window state */
    std::deque<float> smooth_window;
    double smooth_window_sum;

    /* State machine */
    StreamVadState state;
    int speech_cnt;
    int silence_cnt;
    bool hit_max_speech;
    int last_speech_start_frame;
    int last_speech_end_frame;
};

/* Default config — bit-identical to upstream FireRedStreamVadConfig. */
static OmniStreamVadConfig stream_vad_config_default(void) {
    OmniStreamVadConfig c;
    c.threshold          = 0.5f;
    c.smooth_window_size = 5;
    c.pad_start_frame    = 5;
    c.min_speech_frame   = 8;
    c.max_speech_frame   = 2000;
    c.min_silence_frame  = 20;
    return c;
}

OMNIVAD_API OmniStreamVadConfig omni_stream_vad_config_default(void) {
    return stream_vad_config_default();
}

static int validate_stream_vad_config(const OmniStreamVadConfig* c) {
    if (validate_threshold(c->threshold) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    if (c->smooth_window_size < 1) return OMNI_ERR_INVALID_ARG;
    if (c->pad_start_frame    < 0) return OMNI_ERR_INVALID_ARG;
    if (c->min_speech_frame  <= 0) return OMNI_ERR_INVALID_ARG;
    if (c->max_speech_frame  <= 0) return OMNI_ERR_INVALID_ARG;
    if (c->min_silence_frame <= 0) return OMNI_ERR_INVALID_ARG;
    return OMNI_OK;
}

static void stream_vad_postprocessor_reset(OmniStreamVadCtx* ctx) {
    ctx->smooth_window.clear();
    ctx->smooth_window_sum = 0.0;
    ctx->state = STREAM_VAD_SILENCE;
    ctx->speech_cnt = 0;
    ctx->silence_cnt = 0;
    ctx->hit_max_speech = false;
    ctx->last_speech_start_frame = -1;
    ctx->last_speech_end_frame = -1;
}

/* Causal moving-average smoothing — matches upstream smooth_prob(). */
static float stream_vad_smooth(OmniStreamVadCtx* ctx, float prob) {
    if (ctx->cfg.smooth_window_size <= 1) return prob;
    ctx->smooth_window.push_back(prob);
    ctx->smooth_window_sum += prob;
    if ((int)ctx->smooth_window.size() > ctx->cfg.smooth_window_size) {
        ctx->smooth_window_sum -= ctx->smooth_window.front();
        ctx->smooth_window.pop_front();
    }
    return (float)(ctx->smooth_window_sum / (double)ctx->smooth_window.size());
}

/* State transition — bit-identical to upstream state_transition().
 * Updates result fields in-place. */
static void stream_vad_state_transition(
    OmniStreamVadCtx* ctx, bool is_speech, OmniStreamVadResult* result)
{
    /* hit_max_speech carry-over: open new segment immediately on next frame */
    if (ctx->hit_max_speech) {
        result->is_speech_start = true;
        result->speech_start_frame = ctx->frame_offset;
        ctx->last_speech_start_frame = result->speech_start_frame;
        ctx->hit_max_speech = false;
    }

    if (ctx->state == STREAM_VAD_SILENCE) {
        if (is_speech) {
            ctx->state = STREAM_VAD_POSSIBLE_SPEECH;
            ctx->speech_cnt += 1;
        } else {
            ctx->silence_cnt += 1;
            ctx->speech_cnt = 0;
        }
    } else if (ctx->state == STREAM_VAD_POSSIBLE_SPEECH) {
        if (is_speech) {
            ctx->speech_cnt += 1;
            if (ctx->speech_cnt >= ctx->cfg.min_speech_frame) {
                ctx->state = STREAM_VAD_SPEECH;
                int candidate = ctx->frame_offset - ctx->speech_cnt + 1 - ctx->effective_pad_start;
                if (candidate < 1) candidate = 1;
                if (candidate < ctx->last_speech_end_frame + 1) candidate = ctx->last_speech_end_frame + 1;
                result->is_speech_start = true;
                result->speech_start_frame = candidate;
                ctx->last_speech_start_frame = candidate;
                ctx->silence_cnt = 0;
            }
        } else {
            ctx->state = STREAM_VAD_SILENCE;
            ctx->silence_cnt = 1;
            ctx->speech_cnt = 0;
        }
    } else if (ctx->state == STREAM_VAD_SPEECH) {
        ctx->speech_cnt += 1;
        if (is_speech) {
            ctx->silence_cnt = 0;
            if (ctx->speech_cnt >= ctx->cfg.max_speech_frame) {
                ctx->hit_max_speech = true;
                ctx->speech_cnt = 0;
                result->is_speech_end = true;
                result->speech_end_frame = ctx->frame_offset;
                result->speech_start_frame = ctx->last_speech_start_frame;
                ctx->last_speech_start_frame = -1;
                ctx->last_speech_end_frame = result->speech_end_frame;
            }
        } else {
            ctx->state = STREAM_VAD_POSSIBLE_SILENCE;
            ctx->silence_cnt += 1;
        }
    } else { /* STREAM_VAD_POSSIBLE_SILENCE */
        ctx->speech_cnt += 1;
        if (is_speech) {
            ctx->state = STREAM_VAD_SPEECH;
            ctx->silence_cnt = 0;
            if (ctx->speech_cnt >= ctx->cfg.max_speech_frame) {
                ctx->hit_max_speech = true;
                ctx->speech_cnt = 0;
                result->is_speech_end = true;
                result->speech_end_frame = ctx->frame_offset;
                result->speech_start_frame = ctx->last_speech_start_frame;
                ctx->last_speech_start_frame = -1;
                ctx->last_speech_end_frame = result->speech_end_frame;
            }
        } else {
            ctx->silence_cnt += 1;
            if (ctx->silence_cnt >= ctx->cfg.min_silence_frame) {
                ctx->state = STREAM_VAD_SILENCE;
                result->is_speech_end = true;
                result->speech_end_frame = ctx->frame_offset;
                result->speech_start_frame = ctx->last_speech_start_frame;
                ctx->last_speech_end_frame = result->speech_end_frame;
                ctx->last_speech_start_frame = -1;
                ctx->speech_cnt = 0;
            }
        }
    }
}

/* Initialize result with neutral defaults before state_transition mutates it. */
static void stream_vad_init_result(OmniStreamVadResult* r, int frame_idx) {
    r->confidence         = 0.0f;
    r->smoothed_prob      = 0.0f;
    r->is_speech          = false;
    r->is_speech_start    = false;
    r->is_speech_end      = false;
    r->frame_idx          = frame_idx;
    r->speech_start_frame = -1;
    r->speech_end_frame   = -1;
}

/* Shared init helper used by create / create_from_buffer / clone. */
static OmniStreamVadCtx* stream_vad_init_ctx(
    std::shared_ptr<OmniStreamVadSharedAssets> shared,
    const OmniStreamVadConfig& cfg,
    int* out_error)
{
    OmniStreamVadCtx* ctx = new (std::nothrow) OmniStreamVadCtx();
    if (!ctx) {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }

    ctx->shared = shared;
    ctx->fbank = NULL;
    ctx->frame_offset = 0;
    ctx->cache_packed = ncnn::Mat(CACHE_LEN, CACHE_SIZE);
    ctx->cache_packed.fill(0.0f);

    ctx->fbank = new (std::nothrow) vad::Fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    if (!ctx->fbank) {
        delete ctx;
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }

    ctx->cfg = cfg;
    /* Upstream rule: pad_start_frame is at least smooth_window_size. */
    ctx->effective_pad_start = cfg.pad_start_frame > cfg.smooth_window_size
        ? cfg.pad_start_frame : cfg.smooth_window_size;
    stream_vad_postprocessor_reset(ctx);
    return ctx;
}

OmniStreamVadHandle omni_stream_vad_create(
    const char* bundle_path,
    const OmniStreamVadConfig* config,
    int* out_error)
{
    set_out_error(out_error, OMNI_OK);

    OmniStreamVadConfig cfg = config ? *config : stream_vad_config_default();
    int ret = validate_stream_vad_config(&cfg);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    OmniBundle bundle;
    ret = load_bundle(bundle_path, bundle);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    auto shared = std::make_shared<OmniStreamVadSharedAssets>();
    shared->use_cmvn = false;

    ncnn::Net* raw_net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data, &ret);
    if (!raw_net) {
        set_out_error(out_error, ret);
        return NULL;
    }
    shared->net.reset(raw_net);

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        shared->cmvn_means = std::make_shared<const std::vector<float>>(std::move(bundle.cmvn_means));
        shared->cmvn_istd = std::make_shared<const std::vector<float>>(std::move(bundle.cmvn_istd));
        shared->use_cmvn = true;
    } else if (bundle.has_cmvn) {
        set_out_error(out_error, OMNI_ERR_LOAD_CMVN);
        return NULL;
    }

    return stream_vad_init_ctx(shared, cfg, out_error);
}

OmniStreamVadHandle omni_stream_vad_create_from_buffer(
    const void* data,
    int size,
    const OmniStreamVadConfig* config,
    int* out_error)
{
    set_out_error(out_error, OMNI_OK);

    OmniStreamVadConfig cfg = config ? *config : stream_vad_config_default();
    int ret = validate_stream_vad_config(&cfg);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    OmniBundle bundle;
    ret = load_bundle_from_buffer(data, size, bundle);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    auto shared = std::make_shared<OmniStreamVadSharedAssets>();
    shared->use_cmvn = false;

    ncnn::Net* raw_net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data, &ret);
    if (!raw_net) {
        set_out_error(out_error, ret);
        return NULL;
    }
    shared->net.reset(raw_net);

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        shared->cmvn_means = std::make_shared<const std::vector<float>>(std::move(bundle.cmvn_means));
        shared->cmvn_istd = std::make_shared<const std::vector<float>>(std::move(bundle.cmvn_istd));
        shared->use_cmvn = true;
    } else if (bundle.has_cmvn) {
        set_out_error(out_error, OMNI_ERR_LOAD_CMVN);
        return NULL;
    }

    return stream_vad_init_ctx(shared, cfg, out_error);
}

/* -------------------------------------------------------------------------- */
/*  Stream VAD: clone (shares model weights, fresh per-instance state)        */
/* -------------------------------------------------------------------------- */

OmniStreamVadHandle omni_stream_vad_clone(
    OmniStreamVadHandle handle,
    int* out_error)
{
    set_out_error(out_error, OMNI_OK);

    if (!handle) {
        set_out_error(out_error, OMNI_ERR_NULL_HANDLE);
        return NULL;
    }

    return stream_vad_init_ctx(handle->shared, handle->cfg, out_error);
}

/* Internal: run fbank + CMVN + ncnn + state machine on the current
 * audio_buffer contents. Both public entries push samples (with the
 * appropriate scaling) and then call this. Single source of truth for
 * inference; entries differ only in how samples reach audio_buffer. */
static int stream_vad_process_buffered(
    OmniStreamVadCtx* ctx,
    OmniStreamVadResult* result)
{
    /* Keep buffer size at most frame_length (400 samples = 25ms) */
    while ((int)ctx->audio_buffer.size() > FRAME_LENGTH) {
        ctx->audio_buffer.pop_front();
    }

    /* Not enough data for a full frame yet */
    if ((int)ctx->audio_buffer.size() < FRAME_LENGTH) {
        stream_vad_init_result(result, ctx->frame_offset);
        return OMNI_ERR_NO_FRAMES;
    }

    /* Extract fbank features for current window */
    std::vector<float> frame_audio(ctx->audio_buffer.begin(), ctx->audio_buffer.end());
    std::vector<float> features;
    int num_frames = ctx->fbank->Compute(frame_audio, &features);

    if (num_frames == 0 || features.empty()) {
        stream_vad_init_result(result, ctx->frame_offset);
        return OMNI_ERR_NO_FRAMES;
    }

    /* Apply CMVN to the last frame */
    if (ctx->shared->use_cmvn) {
        apply_cmvn(*ctx->shared->cmvn_means, *ctx->shared->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Take the last frame's features */
    std::vector<float> current_feat(FEAT_DIM);
    memcpy(current_feat.data(),
           features.data() + (num_frames - 1) * FEAT_DIM,
           FEAT_DIM * sizeof(float));

    ctx->frame_offset++;
    stream_vad_init_result(result, ctx->frame_offset);

    /* ncnn inference: single frame with packed cache
     *   in0 = feat [w=FEAT_DIM, h=1]
     *   in1 = cache_packed [w=CACHE_LEN, h=CACHE_SIZE]
     *   out0 = probs [1]
     *   out1 = new_cache_packed [w=CACHE_LEN, h=CACHE_SIZE]
     */
    ncnn::Mat in_feat(FEAT_DIM, 1);
    memcpy(in_feat.data, current_feat.data(), FEAT_DIM * sizeof(float));

    ncnn::Mat in_feat_clone = in_feat.clone();
    ncnn::Mat cache_clone = ctx->cache_packed.clone();

    ncnn::Extractor ex = ctx->shared->net->create_extractor();
    ex.input("in0", in_feat_clone);
    ex.input("in1", cache_clone);

    ncnn::Mat out_probs;
    int ret = ex.extract("out0", out_probs);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] stream: ncnn extract out0 failed: %d\n", ret);
        return OMNI_ERR_INFERENCE;
    }

    ncnn::Mat new_cache;
    ret = ex.extract("out1", new_cache);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] stream: ncnn extract out1 failed: %d\n", ret);
        return OMNI_ERR_INFERENCE;
    }

    /* Update cache */
    ctx->cache_packed = new_cache;

    /* Per-frame inference + post-processing (matches upstream
     * StreamVadPostprocessor.process_one_frame). */
    float confidence    = ((float*)out_probs.data)[0];
    float smoothed_prob = stream_vad_smooth(ctx, confidence);
    bool  is_speech     = smoothed_prob >= ctx->cfg.threshold;

    result->confidence    = confidence;
    result->smoothed_prob = smoothed_prob;
    result->is_speech     = is_speech;

    stream_vad_state_transition(ctx, is_speech, result);
    return OMNI_OK;
}

/* Public entry: FP32 input in [-1.0, 1.0]. Scales to int16 magnitude
 * (the format fbank expects) while pushing to audio_buffer. */
int omni_stream_vad_process(
    OmniStreamVadHandle handle,
    const float* audio_data,
    int num_samples,
    OmniStreamVadResult* result)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !result) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

    OmniStreamVadCtx* ctx = handle;
    for (int i = 0; i < num_samples; ++i) {
        ctx->audio_buffer.push_back(audio_data[i] * 32768.0f);
    }
    return stream_vad_process_buffered(ctx, result);
}

/* Public entry: int16 PCM. Cast to float (already in int16 magnitude). */
int omni_stream_vad_process_int16(
    OmniStreamVadHandle handle,
    const int16_t* audio_data,
    int num_samples,
    OmniStreamVadResult* result)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !result) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

    OmniStreamVadCtx* ctx = handle;
    for (int i = 0; i < num_samples; ++i) {
        ctx->audio_buffer.push_back((float)audio_data[i]);
    }
    return stream_vad_process_buffered(ctx, result);
}

/* Internal: detect_full from float audio in int16 range [-32768, 32767]. */
static int stream_vad_detect_full_int16range(
    OmniStreamVadHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !out_probs || !out_frames) return OMNI_ERR_NULL_POINTER;
    if (num_samples < FRAME_LENGTH) return OMNI_ERR_NO_FRAMES;

    OmniStreamVadCtx* ctx = handle;

    /* Step 1: Whole-file fbank (matches Python AudioFeat.extract) */
    std::vector<float> wave(audio_data, audio_data + num_samples);
    vad::Fbank fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    std::vector<float> features;
    int num_frames = fbank.Compute(wave, &features);
    if (num_frames <= 0) return OMNI_ERR_NO_FRAMES;

    /* Apply CMVN */
    if (ctx->shared->use_cmvn) {
        apply_cmvn(*ctx->shared->cmvn_means, *ctx->shared->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Step 2: Reset cache, then run model frame-by-frame with cache */
    ncnn::Mat cache_packed(CACHE_LEN, CACHE_SIZE);
    cache_packed.fill(0.0f);

    float* probs = (float*)malloc(sizeof(float) * num_frames);
    if (!probs) return OMNI_ERR_OUT_OF_MEMORY;

    for (int t = 0; t < num_frames; ++t) {
        /* Prepare single-frame input [w=FEAT_DIM, h=1] */
        ncnn::Mat in_feat(FEAT_DIM, 1);
        memcpy(in_feat.data, features.data() + t * FEAT_DIM, FEAT_DIM * sizeof(float));

        ncnn::Mat in_feat_clone = in_feat.clone();
        ncnn::Mat cache_clone = cache_packed.clone();

        ncnn::Extractor ex = ctx->shared->net->create_extractor();
        ex.input("in0", in_feat_clone);
        ex.input("in1", cache_clone);

        ncnn::Mat out_prob_mat;
        int ret = ex.extract("out0", out_prob_mat);
        if (ret != 0) {
            free(probs);
            return OMNI_ERR_INFERENCE;
        }

        ncnn::Mat new_cache;
        ret = ex.extract("out1", new_cache);
        if (ret != 0) {
            free(probs);
            return OMNI_ERR_INFERENCE;
        }

        cache_packed = new_cache;
        probs[t] = ((float*)out_prob_mat.data)[0];
    }

    *out_probs = probs;
    *out_frames = num_frames;
    return OMNI_OK;
}

/* Public API: float [-1.0, 1.0] → scale × 32768 → internal */
int omni_stream_vad_detect_full(
    OmniStreamVadHandle handle,
    const float* audio_data, int num_samples,
    float** out_probs, int* out_frames)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = f32_normalize_to_int16_range(audio_data, num_samples);
    return stream_vad_detect_full_int16range(handle, buf.data(), num_samples,
                                             out_probs, out_frames);
}

/* Public API: int16 PCM → cast to float → internal */
int omni_stream_vad_detect_full_int16(
    OmniStreamVadHandle handle,
    const int16_t* audio_data, int num_samples,
    float** out_probs, int* out_frames)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = i16_to_float(audio_data, num_samples);
    return stream_vad_detect_full_int16range(handle, buf.data(), num_samples,
                                             out_probs, out_frames);
}

void omni_stream_vad_reset(OmniStreamVadHandle handle) {
    if (!handle) return;
    handle->audio_buffer.clear();
    handle->frame_offset = 0;
    handle->cache_packed.fill(0.0f);
    if (handle->fbank) {
        handle->fbank->reset();
    }
    stream_vad_postprocessor_reset(handle);
}

int omni_stream_vad_get_frame_offset(OmniStreamVadHandle handle) {
    if (!handle) return 0;
    return handle->frame_offset;
}

void omni_stream_vad_destroy(OmniStreamVadHandle handle) {
    if (!handle) return;
    delete handle->fbank;
    handle->shared.reset();  /* shared_ptr decrement; last one frees ncnn::Net + CMVN */
    delete handle;
}

/* ========================================================================== */
/*  2. VAD Implementation                                          */
/* ========================================================================== */

struct OmniVadCtx {
    ncnn::Net* net;

    /* CMVN vectors */
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool use_cmvn;
};

OmniVadHandle omni_vad_create(const char* bundle_path, int* out_error) {
    set_out_error(out_error, OMNI_OK);

    OmniBundle bundle;
    int ret = load_bundle(bundle_path, bundle);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    OmniVadCtx* ctx = new (std::nothrow) OmniVadCtx();
    if (!ctx) {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
    ctx->use_cmvn = false;

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data, &ret);
    if (!ctx->net) {
        delete ctx;
        set_out_error(out_error, ret);
        return NULL;
    }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    } else if (bundle.has_cmvn) {
        delete ctx->net;
        delete ctx;
        set_out_error(out_error, OMNI_ERR_LOAD_CMVN);
        return NULL;
    }
    set_out_error(out_error, OMNI_OK);
    return ctx;
}

OmniVadHandle omni_vad_create_from_buffer(const void* data, int size, int* out_error) {
    set_out_error(out_error, OMNI_OK);

    OmniBundle bundle;
    int ret = load_bundle_from_buffer(data, size, bundle);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    OmniVadCtx* ctx = new (std::nothrow) OmniVadCtx();
    if (!ctx) {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
    ctx->use_cmvn = false;

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data, &ret);
    if (!ctx->net) {
        delete ctx;
        set_out_error(out_error, ret);
        return NULL;
    }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    } else if (bundle.has_cmvn) {
        delete ctx->net;
        delete ctx;
        set_out_error(out_error, OMNI_ERR_LOAD_CMVN);
        return NULL;
    }
    set_out_error(out_error, OMNI_OK);
    return ctx;
}

/*
 * Internal: compute fbank features for the whole audio, apply CMVN,
 * run ncnn inference, return raw per-frame probabilities.
 *
 * For non-stream VAD, the model takes input in0 [w=feat_dim, h=num_frames]
 * and outputs out0 with one probability per frame.
 */
static int vad_infer(
    OmniVadCtx* ctx,
    const float* audio_data,
    int num_samples,
    std::vector<float>& out_probs,
    int& out_num_frames)
{
    if (num_samples < FRAME_LENGTH) return OMNI_ERR_NO_FRAMES;

    /* Wrap audio data into a vector for the fbank API */
    std::vector<float> wave(audio_data, audio_data + num_samples);

    /* Compute fbank features */
    vad::Fbank fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    std::vector<float> features;
    int num_frames = fbank.Compute(wave, &features);
    if (num_frames <= 0) return OMNI_ERR_NO_FRAMES;

    /* Apply CMVN */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Prepare ncnn input: [w=FEAT_DIM, h=num_frames] */
    ncnn::Mat in_feat(FEAT_DIM, num_frames);
    memcpy(in_feat.data, features.data(), sizeof(float) * features.size());
    ncnn::Mat in_feat_clone = in_feat.clone();

    ncnn::Extractor ex = ctx->net->create_extractor();
    ex.input("in0", in_feat_clone);

    ncnn::Mat ncnn_out;
    int ret = ex.extract("out0", ncnn_out);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] vad: ncnn extract failed: %d\n", ret);
        return OMNI_ERR_INFERENCE;
    }

    /* Extract per-frame probabilities.
     * Expected output: 1D array of length num_frames. */
    int T = num_frames;
    out_probs.resize(T);

    /* ncnn may output as dims=1 w=T, or dims=2 w=1 h=T, etc. Handle both. */
    if (ncnn_out.dims == 1) {
        T = ncnn_out.w;
        out_probs.resize(T);
        const float* p = (const float*)ncnn_out.data;
        for (int i = 0; i < T; ++i) {
            out_probs[i] = p[i];
        }
    } else if (ncnn_out.dims == 2) {
        /* Could be [w=1, h=T] or [w=T, h=1] */
        if (ncnn_out.w == 1) {
            T = ncnn_out.h;
            out_probs.resize(T);
            for (int i = 0; i < T; ++i) {
                out_probs[i] = ncnn_out.row(i)[0];
            }
        } else {
            T = ncnn_out.w;
            out_probs.resize(T);
            const float* p = ncnn_out.row(0);
            for (int i = 0; i < T; ++i) {
                out_probs[i] = p[i];
            }
        }
    } else {
        /* Fallback: treat entire data as flat array */
        T = ncnn_out.total();
        out_probs.resize(T);
        const float* p = (const float*)ncnn_out.data;
        for (int i = 0; i < T; ++i) {
            out_probs[i] = p[i];
        }
    }

    out_num_frames = T;
    return OMNI_OK;
}

/* Internal: process float audio in int16 range [-32768, 32767]. */
static int vad_detect_int16range(
    OmniVadHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments,
    int* out_count)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!out_segments || !out_count) return OMNI_ERR_NULL_POINTER;

    OmniPostConfig cfg = config ? *config : omni_post_config_default();

    /* Run inference */
    std::vector<float> probs;
    int num_frames = 0;
    int ret = vad_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_segments = NULL;
        *out_count = 0;
        return ret;
    }

    /* Run full post-processing pipeline (matches Python exactly) */
    std::vector<FrameSegment> segs = postprocess_pipeline(
        probs.data(), num_frames,
        cfg.threshold, cfg.smooth_window_size,
        cfg.min_speech_frames, cfg.max_speech_frames,
        cfg.min_silence_frames, cfg.merge_silence_frames,
        cfg.extend_speech_frames);

    /* Convert to output format with proper timestamp calculation */
    int count = (int)segs.size();
    if (count == 0) {
        *out_segments = NULL;
        *out_count = 0;
        return OMNI_OK;
    }

    float wav_dur = (float)num_samples / (float)SAMPLE_RATE;

    OmniSegment* result = (OmniSegment*)malloc(sizeof(OmniSegment) * count);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;

    for (int i = 0; i < count; ++i) {
        result[i].start = segs[i].start * FRAME_SHIFT_SEC;
        /* For the last segment, match Python: end = num_frames * shift + frame_length */
        if (i == count - 1 && segs[i].end == num_frames) {
            float end_time = (float)num_frames * FRAME_SHIFT_SEC + FRAME_LENGTH_SEC;
            if (end_time > wav_dur) end_time = wav_dur;
            result[i].end = end_time;
        } else {
            result[i].end = segs[i].end * FRAME_SHIFT_SEC;
        }
    }

    *out_segments = result;
    *out_count = count;
    return OMNI_OK;
}

/* Internal: get probs from float audio in int16 range. */
static int vad_detect_probs_int16range(
    OmniVadHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!out_probs || !out_frames) return OMNI_ERR_NULL_POINTER;

    std::vector<float> probs;
    int num_frames = 0;
    int ret = vad_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_probs = NULL;
        *out_frames = 0;
        return ret;
    }

    float* result = (float*)malloc(sizeof(float) * num_frames);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;
    memcpy(result, probs.data(), sizeof(float) * num_frames);

    *out_probs = result;
    *out_frames = num_frames;
    return OMNI_OK;
}

void omni_vad_destroy(OmniVadHandle handle) {
    if (!handle) return;
    delete handle->net;
    delete handle;
}

/* Public API: float [-1.0, 1.0] → scale × 32768 → internal */
int omni_vad_detect(
    OmniVadHandle handle,
    const float* audio_data, int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments, int* out_count)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = f32_normalize_to_int16_range(audio_data, num_samples);
    return vad_detect_int16range(handle, buf.data(), num_samples,
                                             config, out_segments, out_count);
}

/* Public API: int16 PCM → cast to float → internal */
int omni_vad_detect_int16(
    OmniVadHandle handle,
    const int16_t* audio_data, int num_samples,
    const OmniPostConfig* config,
    OmniSegment** out_segments, int* out_count)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = i16_to_float(audio_data, num_samples);
    return vad_detect_int16range(handle, buf.data(), num_samples,
                                             config, out_segments, out_count);
}

int omni_vad_detect_probs(
    OmniVadHandle handle,
    const float* audio_data, int num_samples,
    float** out_probs, int* out_frames)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = f32_normalize_to_int16_range(audio_data, num_samples);
    return vad_detect_probs_int16range(handle, buf.data(), num_samples,
                                          out_probs, out_frames);
}

int omni_vad_detect_probs_int16(
    OmniVadHandle handle,
    const int16_t* audio_data, int num_samples,
    float** out_probs, int* out_frames)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = i16_to_float(audio_data, num_samples);
    return vad_detect_probs_int16range(handle, buf.data(), num_samples,
                                          out_probs, out_frames);
}

/* ========================================================================== */
/*  3. AED Implementation                                          */
/* ========================================================================== */

struct OmniAedCtx {
    ncnn::Net* net;

    /* CMVN vectors */
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool use_cmvn;
};

OmniAedHandle omni_aed_create(const char* bundle_path, int* out_error) {
    set_out_error(out_error, OMNI_OK);

    OmniBundle bundle;
    int ret = load_bundle(bundle_path, bundle);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    OmniAedCtx* ctx = new (std::nothrow) OmniAedCtx();
    if (!ctx) {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
    ctx->use_cmvn = false;

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data, &ret);
    if (!ctx->net) {
        delete ctx;
        set_out_error(out_error, ret);
        return NULL;
    }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    } else if (bundle.has_cmvn) {
        delete ctx->net;
        delete ctx;
        set_out_error(out_error, OMNI_ERR_LOAD_CMVN);
        return NULL;
    }
    set_out_error(out_error, OMNI_OK);
    return ctx;
}

OmniAedHandle omni_aed_create_from_buffer(const void* data, int size, int* out_error) {
    set_out_error(out_error, OMNI_OK);

    OmniBundle bundle;
    int ret = load_bundle_from_buffer(data, size, bundle);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    OmniAedCtx* ctx = new (std::nothrow) OmniAedCtx();
    if (!ctx) {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
    ctx->use_cmvn = false;

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data, &ret);
    if (!ctx->net) {
        delete ctx;
        set_out_error(out_error, ret);
        return NULL;
    }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    } else if (bundle.has_cmvn) {
        delete ctx->net;
        delete ctx;
        set_out_error(out_error, OMNI_ERR_LOAD_CMVN);
        return NULL;
    }
    set_out_error(out_error, OMNI_OK);
    return ctx;
}

/*
 * Internal: run AED inference, return per-frame 3-class probabilities.
 * Output layout: probs[frame * 3 + class], where class 0=speech, 1=singing, 2=music.
 */
static int aed_infer(
    OmniAedCtx* ctx,
    const float* audio_data,
    int num_samples,
    std::vector<float>& out_probs,
    int& out_num_frames)
{
    if (num_samples < FRAME_LENGTH) return OMNI_ERR_NO_FRAMES;

    std::vector<float> wave(audio_data, audio_data + num_samples);

    /* Compute fbank features */
    vad::Fbank fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    std::vector<float> features;
    int num_frames = fbank.Compute(wave, &features);
    if (num_frames <= 0) return OMNI_ERR_NO_FRAMES;

    /* Apply CMVN */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Prepare ncnn input */
    ncnn::Mat in_feat(FEAT_DIM, num_frames);
    memcpy(in_feat.data, features.data(), sizeof(float) * features.size());
    ncnn::Mat in_feat_clone = in_feat.clone();

    ncnn::Extractor ex = ctx->net->create_extractor();
    ex.input("in0", in_feat_clone);

    ncnn::Mat ncnn_out;
    int ret = ex.extract("out0", ncnn_out);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] aed: ncnn extract failed: %d\n", ret);
        return OMNI_ERR_INFERENCE;
    }

    /*
     * AED model outputs 3-class probabilities per frame.
     * Possible ncnn layouts:
     *   a) dims=2, w=3, h=T  -> row(t) gives [speech, singing, music]
     *   b) dims=2, h=3, w=T  -> transposed: row(cls)[t]
     *   c) dims=1, w=T*3     -> flat array
     */
    int T = 0;
    static const int NUM_CLASSES = 3;

    if (ncnn_out.dims == 2 && ncnn_out.w == NUM_CLASSES) {
        /* Layout (a): [h=T, w=3] - most common */
        T = ncnn_out.h;
        out_probs.resize(T * NUM_CLASSES);
        for (int t = 0; t < T; ++t) {
            const float* row = ncnn_out.row(t);
            out_probs[t * NUM_CLASSES + 0] = row[0];  /* speech */
            out_probs[t * NUM_CLASSES + 1] = row[1];  /* singing */
            out_probs[t * NUM_CLASSES + 2] = row[2];  /* music */
        }
    } else if (ncnn_out.dims == 2 && ncnn_out.h == NUM_CLASSES) {
        /* Layout (b): transposed [h=3, w=T] */
        T = ncnn_out.w;
        out_probs.resize(T * NUM_CLASSES);
        const float* row0 = ncnn_out.row(0);
        const float* row1 = ncnn_out.row(1);
        const float* row2 = ncnn_out.row(2);
        for (int t = 0; t < T; ++t) {
            out_probs[t * NUM_CLASSES + 0] = row0[t];
            out_probs[t * NUM_CLASSES + 1] = row1[t];
            out_probs[t * NUM_CLASSES + 2] = row2[t];
        }
    } else if (ncnn_out.dims == 1 && ncnn_out.w % NUM_CLASSES == 0) {
        /* Layout (c): flat [T*3] */
        T = ncnn_out.w / NUM_CLASSES;
        out_probs.resize(T * NUM_CLASSES);
        const float* p = (const float*)ncnn_out.data;
        memcpy(out_probs.data(), p, sizeof(float) * T * NUM_CLASSES);
    } else {
        fprintf(stderr, "[omnivad] aed: unexpected output dims=%d w=%d h=%d c=%d\n",
                ncnn_out.dims, ncnn_out.w, ncnn_out.h, ncnn_out.c);
        return OMNI_ERR_INFERENCE;
    }

    out_num_frames = T;
    return OMNI_OK;
}

/* Internal: AED process from float audio in int16 range. */
static int aed_detect_int16range(
    OmniAedHandle handle,
    const float* audio_data,
    int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments,
    int* out_count)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!out_segments || !out_count) return OMNI_ERR_NULL_POINTER;

    OmniAedPostConfig cfg = config ? *config : omni_aed_post_config_default();

    /* Run inference */
    std::vector<float> probs;
    int num_frames = 0;
    int ret = aed_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_segments = NULL;
        *out_count = 0;
        return ret;
    }

    static const int NUM_CLASSES = 3;
    const OmniPostConfig* class_configs[NUM_CLASSES] = {
        &cfg.speech, &cfg.singing, &cfg.music
    };
    const OmniAedClass class_ids[NUM_CLASSES] = {
        OMNI_AED_SPEECH, OMNI_AED_SINGING, OMNI_AED_MUSIC
    };

    float wav_dur = (float)num_samples / (float)SAMPLE_RATE;

    /* Process each class independently */
    std::vector<OmniAedSegment> all_segments;

    for (int c = 0; c < NUM_CLASSES; ++c) {
        /* Extract per-frame probabilities for this class */
        std::vector<float> class_probs(num_frames);
        for (int t = 0; t < num_frames; ++t) {
            class_probs[t] = probs[t * NUM_CLASSES + c];
        }

        /* Run full post-processing pipeline */
        const OmniPostConfig* cc = class_configs[c];
        std::vector<FrameSegment> segs = postprocess_pipeline(
            class_probs.data(), num_frames,
            cc->threshold, cc->smooth_window_size,
            cc->min_speech_frames, cc->max_speech_frames,
            cc->min_silence_frames, cc->merge_silence_frames,
            cc->extend_speech_frames);

        /* Convert to output format with class label and average confidence */
        for (size_t i = 0; i < segs.size(); ++i) {
            OmniAedSegment seg;
            seg.start = segs[i].start * FRAME_SHIFT_SEC;
            /* Match Python: trailing segment end = num_frames * shift + frame_length */
            if (segs[i].end == num_frames) {
                float end_time = (float)num_frames * FRAME_SHIFT_SEC + FRAME_LENGTH_SEC;
                if (end_time > wav_dur) end_time = wav_dur;
                seg.end = end_time;
            } else {
                seg.end = segs[i].end * FRAME_SHIFT_SEC;
            }
            seg.cls   = class_ids[c];

            /* Compute average confidence over the segment */
            float sum = 0.0f;
            int count = 0;
            for (int t = segs[i].start; t < segs[i].end && t < num_frames; ++t) {
                sum += class_probs[t];
                count++;
            }
            seg.confidence = (count > 0) ? (sum / count) : 0.0f;

            all_segments.push_back(seg);
        }
    }

    /* Sort all segments by start time */
    std::sort(all_segments.begin(), all_segments.end(),
              [](const OmniAedSegment& a, const OmniAedSegment& b) {
                  if (a.start != b.start) return a.start < b.start;
                  return a.cls < b.cls;
              });

    /* Allocate and return */
    int count = (int)all_segments.size();
    if (count == 0) {
        *out_segments = NULL;
        *out_count = 0;
        return OMNI_OK;
    }

    OmniAedSegment* result = (OmniAedSegment*)malloc(sizeof(OmniAedSegment) * count);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;

    memcpy(result, all_segments.data(), sizeof(OmniAedSegment) * count);

    *out_segments = result;
    *out_count = count;
    return OMNI_OK;
}

static int aed_detect_probs_int16range(
    OmniAedHandle handle,
    const float* audio_data,
    int num_samples,
    float** out_probs,
    int* out_frames)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!out_probs || !out_frames) return OMNI_ERR_NULL_POINTER;

    std::vector<float> probs;
    int num_frames = 0;
    int ret = aed_infer(handle, audio_data, num_samples, probs, num_frames);
    if (ret != OMNI_OK) {
        *out_probs = NULL;
        *out_frames = 0;
        return ret;
    }

    static const int NUM_CLASSES = 3;
    int total = num_frames * NUM_CLASSES;
    float* result = (float*)malloc(sizeof(float) * total);
    if (!result) return OMNI_ERR_OUT_OF_MEMORY;
    memcpy(result, probs.data(), sizeof(float) * total);

    *out_probs = result;
    *out_frames = num_frames;
    return OMNI_OK;
}

void omni_aed_destroy(OmniAedHandle handle) {
    if (!handle) return;
    delete handle->net;
    delete handle;
}

/* Public API: float [-1.0, 1.0] → scale × 32768 → internal */
int omni_aed_detect(
    OmniAedHandle handle,
    const float* audio_data, int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments, int* out_count)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = f32_normalize_to_int16_range(audio_data, num_samples);
    return aed_detect_int16range(handle, buf.data(), num_samples,
                                             config, out_segments, out_count);
}

/* Public API: int16 PCM → cast to float → internal */
int omni_aed_detect_int16(
    OmniAedHandle handle,
    const int16_t* audio_data, int num_samples,
    const OmniAedPostConfig* config,
    OmniAedSegment** out_segments, int* out_count)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = i16_to_float(audio_data, num_samples);
    return aed_detect_int16range(handle, buf.data(), num_samples,
                                             config, out_segments, out_count);
}

int omni_aed_detect_probs(
    OmniAedHandle handle,
    const float* audio_data, int num_samples,
    float** out_probs, int* out_frames)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = f32_normalize_to_int16_range(audio_data, num_samples);
    return aed_detect_probs_int16range(handle, buf.data(), num_samples,
                                          out_probs, out_frames);
}

int omni_aed_detect_probs_int16(
    OmniAedHandle handle,
    const int16_t* audio_data, int num_samples,
    float** out_probs, int* out_frames)
{
    if (!audio_data) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    std::vector<float> buf = i16_to_float(audio_data, num_samples);
    return aed_detect_probs_int16range(handle, buf.data(), num_samples,
                                          out_probs, out_frames);
}

/* ========================================================================== */
/*  4. AED Overlap Segmenter                                                  */
/* ========================================================================== */

static const uint32_t AED_MASK_SPEECH = 1u << 0;
static const uint32_t AED_MASK_SINGING = 1u << 1;
static const uint32_t AED_MASK_MUSIC = 1u << 2;

struct AedMergedFrame {
    double speech_sum;
    double singing_sum;
    double music_sum;
    double weight_sum;
    int coverage;
};

struct AedCommittedFrame {
    float speech;
    float singing;
    float music;
    int coverage;
};

struct AedExtractedEvent {
    int start_ms;
    int end_ms;
    uint32_t mask;
    OmniAedEventKind primary_kind;
    float speech_confidence;
    float singing_confidence;
    float music_confidence;
    float confidence;
};

struct AedExtractedSegment {
    int start_ms;
    int end_ms;
    int event_start_idx;
    int event_count;
};

struct AedSplitCandidate {
    bool valid;
    int run_start_idx;
    int run_end_idx;
    int start_ms;
    int end_ms;
};

struct OmniAedOverlapSegmenterCtx {
    OmniAedHandle aed;
    OmniAedOverlapConfig config;
    std::vector<unsigned char> bundle_bytes;

    std::vector<float> audio_buffer;
    int64_t audio_base_sample;
    int64_t total_samples;
    int64_t next_window_start_sample;

    int64_t timeline_base_frame;
    std::vector<AedMergedFrame> timeline;
    int64_t available_until_frame;
    int64_t latest_window_start_frame;
    int64_t committed_until_frame;
    int64_t committed_base_frame;
    bool committed_base_is_transcribable_continuation;
    std::vector<AedCommittedFrame> committed_frames;

    int emitted_until_ms;
    int emitted_padded_until_ms;
    bool flushed;
};

static bool is_grid_10ms(int value) {
    return value >= 0 && (value % 10) == 0;
}

static int validate_aed_overlap_config(const OmniAedOverlapConfig* c) {
    if (!c) return OMNI_ERR_NULL_POINTER;
    if (!is_grid_10ms(c->hop_ms) || c->hop_ms <= 0) return OMNI_ERR_INVALID_ARG;
    if (!is_grid_10ms(c->overlap_ms) || c->overlap_ms >= c->hop_ms) return OMNI_ERR_INVALID_ARG;
    if (!is_grid_10ms(c->edge_guard_ms) || c->edge_guard_ms >= c->hop_ms) return OMNI_ERR_INVALID_ARG;
    if (c->hard_split_pause_ms < 0 ||
        c->max_chunk_ms <= 0 ||
        c->min_speech_ms < 0 ||
        c->merge_gap_ms < 0 ||
        c->music_gap_tolerance_ms < 0 ||
        c->pad_start_ms < 0 ||
        c->pad_end_ms < 0) {
        return OMNI_ERR_INVALID_ARG;
    }
    if (validate_threshold(c->speech_threshold) != OMNI_OK ||
        validate_threshold(c->singing_threshold) != OMNI_OK ||
        validate_threshold(c->music_threshold) != OMNI_OK) {
        return OMNI_ERR_INVALID_ARG;
    }
    return OMNI_OK;
}

OmniAedOverlapConfig omni_aed_overlap_config_default(void) {
    OmniAedOverlapConfig cfg;
    cfg.hop_ms = 2000;
    cfg.overlap_ms = 250;
    cfg.edge_guard_ms = 0;
    cfg.hard_split_pause_ms = 2000;
    cfg.max_chunk_ms = 60000;
    cfg.min_speech_ms = 200;
    cfg.merge_gap_ms = 200;
    cfg.music_gap_tolerance_ms = 0;
    cfg.pad_start_ms = 0;
    cfg.pad_end_ms = 0;
    cfg.speech_threshold = 0.5f;
    cfg.singing_threshold = 0.5f;
    cfg.music_threshold = 0.5f;
    return cfg;
}

static int read_file_bytes(const char* path, std::vector<unsigned char>& out) {
    if (!path) return OMNI_ERR_NULL_POINTER;
    if (path[0] == '\0') return OMNI_ERR_INVALID_ARG;

    FILE* fp = fopen(path, "rb");
    if (!fp) return OMNI_ERR_LOAD_BUNDLE;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return OMNI_ERR_LOAD_BUNDLE;
    }
    long size = ftell(fp);
    if (size <= 0) {
        fclose(fp);
        return OMNI_ERR_LOAD_BUNDLE;
    }
    rewind(fp);
    OMNI_TRY {
        out.resize((size_t)size);
    } OMNI_CATCH_BAD_ALLOC {
        fclose(fp);
        return OMNI_ERR_OUT_OF_MEMORY;
    }
    if (fread(out.data(), 1, out.size(), fp) != out.size()) {
        fclose(fp);
        out.clear();
        return OMNI_ERR_LOAD_BUNDLE;
    }
    fclose(fp);
    return OMNI_OK;
}

static OmniAedOverlapSegmenterHandle create_aed_overlap_from_bytes(
    const void* data,
    int size,
    const OmniAedOverlapConfig* config,
    int* out_error)
{
    set_out_error(out_error, OMNI_OK);
    if (!data) {
        set_out_error(out_error, OMNI_ERR_NULL_POINTER);
        return NULL;
    }
    if (size <= 0) {
        set_out_error(out_error, OMNI_ERR_INVALID_ARG);
        return NULL;
    }

    OmniAedOverlapConfig cfg = config ? *config : omni_aed_overlap_config_default();
    int ret = validate_aed_overlap_config(&cfg);
    if (ret != OMNI_OK) {
        set_out_error(out_error, ret);
        return NULL;
    }

    int aed_err = OMNI_OK;
    OmniAedHandle aed = omni_aed_create_from_buffer(data, size, &aed_err);
    if (!aed) {
        set_out_error(out_error, aed_err);
        return NULL;
    }

    OmniAedOverlapSegmenterCtx* ctx = new (std::nothrow) OmniAedOverlapSegmenterCtx();
    if (!ctx) {
        omni_aed_destroy(aed);
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }

    ctx->aed = aed;
    ctx->config = cfg;
    const unsigned char* p = (const unsigned char*)data;
    OMNI_TRY {
        ctx->bundle_bytes.assign(p, p + size);
    } OMNI_CATCH_BAD_ALLOC {
        omni_aed_destroy(aed);
        delete ctx;
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
    ctx->audio_base_sample = 0;
    ctx->total_samples = 0;
    ctx->next_window_start_sample = 0;
    ctx->timeline_base_frame = 0;
    ctx->available_until_frame = 0;
    ctx->latest_window_start_frame = 0;
    ctx->committed_until_frame = 0;
    ctx->committed_base_frame = 0;
    ctx->committed_base_is_transcribable_continuation = false;
    ctx->emitted_until_ms = 0;
    ctx->emitted_padded_until_ms = 0;
    ctx->flushed = false;
    set_out_error(out_error, OMNI_OK);
    return ctx;
}

OmniAedOverlapSegmenterHandle omni_aed_overlap_segmenter_create(
    const char* bundle_path,
    const OmniAedOverlapConfig* config,
    int* out_error)
{
    OMNI_TRY {
        std::vector<unsigned char> bytes;
        int ret = read_file_bytes(bundle_path, bytes);
        if (ret != OMNI_OK) {
            set_out_error(out_error, ret);
            return NULL;
        }
        return create_aed_overlap_from_bytes(bytes.data(), (int)bytes.size(), config, out_error);
    } OMNI_CATCH_BAD_ALLOC {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
}

OmniAedOverlapSegmenterHandle omni_aed_overlap_segmenter_create_from_buffer(
    const void* data,
    int size,
    const OmniAedOverlapConfig* config,
    int* out_error)
{
    OMNI_TRY {
        return create_aed_overlap_from_bytes(data, size, config, out_error);
    } OMNI_CATCH_BAD_ALLOC {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
}

OmniAedOverlapSegmenterHandle omni_aed_overlap_segmenter_clone(
    OmniAedOverlapSegmenterHandle handle,
    int* out_error)
{
    OMNI_TRY {
        if (!handle) {
            set_out_error(out_error, OMNI_ERR_NULL_HANDLE);
            return NULL;
        }
        return create_aed_overlap_from_bytes(
            handle->bundle_bytes.data(),
            (int)handle->bundle_bytes.size(),
            &handle->config,
            out_error);
    } OMNI_CATCH_BAD_ALLOC {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }
}

static void reset_aed_overlap_state(OmniAedOverlapSegmenterCtx* ctx) {
    ctx->audio_buffer.clear();
    ctx->audio_base_sample = 0;
    ctx->total_samples = 0;
    ctx->next_window_start_sample = 0;
    ctx->timeline_base_frame = 0;
    ctx->timeline.clear();
    ctx->available_until_frame = 0;
    ctx->latest_window_start_frame = 0;
    ctx->committed_until_frame = 0;
    ctx->committed_base_frame = 0;
    ctx->committed_base_is_transcribable_continuation = false;
    ctx->committed_frames.clear();
    ctx->emitted_until_ms = 0;
    ctx->emitted_padded_until_ms = 0;
    ctx->flushed = false;
}

void omni_aed_overlap_segmenter_reset(OmniAedOverlapSegmenterHandle handle) {
    if (!handle) return;
    reset_aed_overlap_state(handle);
}

static int ms_to_samples(int ms) {
    return (int)(((int64_t)ms * SAMPLE_RATE) / 1000);
}

static int ms_to_frames(int ms) {
    return ms / 10;
}

static int min_aed_inference_samples(void) {
    return FRAME_LENGTH + FRAME_SHIFT;
}

static float overlap_frame_weight(const OmniAedOverlapConfig& cfg, int local_frame, int num_frames) {
    if (cfg.overlap_ms <= 0) return 1.0f;
    int local_ms = local_frame * 10;
    int window_ms = cfg.hop_ms + cfg.overlap_ms;
    if (local_ms < cfg.overlap_ms || local_ms >= window_ms - cfg.overlap_ms) {
        return 0.35f;
    }
    if (local_frame < 0 || local_frame >= num_frames) return 0.35f;
    return 1.0f;
}

static int ensure_timeline_frame(OmniAedOverlapSegmenterCtx* ctx, int64_t frame_idx) {
    if (frame_idx < ctx->timeline_base_frame) return OMNI_OK;
    int64_t offset = frame_idx - ctx->timeline_base_frame;
    if (offset >= (int64_t)ctx->timeline.size()) {
        size_t old_size = ctx->timeline.size();
        ctx->timeline.resize((size_t)offset + 1);
        for (size_t i = old_size; i < ctx->timeline.size(); ++i) {
            ctx->timeline[i].speech_sum = 0.0;
            ctx->timeline[i].singing_sum = 0.0;
            ctx->timeline[i].music_sum = 0.0;
            ctx->timeline[i].weight_sum = 0.0;
            ctx->timeline[i].coverage = 0;
        }
    }
    return OMNI_OK;
}

static int run_aed_overlap_window(OmniAedOverlapSegmenterCtx* ctx, int64_t start_sample, int64_t end_sample) {
    if (end_sample - start_sample < min_aed_inference_samples()) return OMNI_OK;
    int64_t offset = start_sample - ctx->audio_base_sample;
    if (offset < 0 || offset + (end_sample - start_sample) > (int64_t)ctx->audio_buffer.size()) {
        return OMNI_ERR_INVALID_ARG;
    }

    std::vector<float> probs;
    int num_frames = 0;
    int ret = aed_infer(ctx->aed, ctx->audio_buffer.data() + offset, (int)(end_sample - start_sample), probs, num_frames);
    if (ret != OMNI_OK) return ret;

    int64_t window_start_frame = start_sample / FRAME_SHIFT;
    for (int t = 0; t < num_frames; ++t) {
        int64_t frame_idx = window_start_frame + t;
        if (frame_idx < ctx->committed_until_frame) continue;
        ret = ensure_timeline_frame(ctx, frame_idx);
        if (ret != OMNI_OK) return ret;
        AedMergedFrame& f = ctx->timeline[(size_t)(frame_idx - ctx->timeline_base_frame)];
        float w = overlap_frame_weight(ctx->config, t, num_frames);
        f.speech_sum += (double)probs[t * 3 + 0] * w;
        f.singing_sum += (double)probs[t * 3 + 1] * w;
        f.music_sum += (double)probs[t * 3 + 2] * w;
        f.weight_sum += w;
        f.coverage += 1;
    }

    int64_t until = window_start_frame + num_frames;
    if (until > ctx->available_until_frame) ctx->available_until_frame = until;
    ctx->latest_window_start_frame = window_start_frame;
    return OMNI_OK;
}

static int run_ready_aed_overlap_windows(OmniAedOverlapSegmenterCtx* ctx) {
    int window_samples = ms_to_samples(ctx->config.hop_ms + ctx->config.overlap_ms);
    int hop_samples = ms_to_samples(ctx->config.hop_ms);
    while (ctx->total_samples - ctx->next_window_start_sample >= window_samples) {
        int64_t start = ctx->next_window_start_sample;
        int64_t end = start + window_samples;
        int ret = run_aed_overlap_window(ctx, start, end);
        if (ret != OMNI_OK) return ret;
        ctx->next_window_start_sample += hop_samples;

        int64_t discard = ctx->next_window_start_sample - ctx->audio_base_sample;
        if (discard > 0 && discard <= (int64_t)ctx->audio_buffer.size()) {
            ctx->audio_buffer.erase(ctx->audio_buffer.begin(), ctx->audio_buffer.begin() + discard);
            ctx->audio_base_sample += discard;
        }
    }
    return OMNI_OK;
}

static int commit_aed_overlap_frames(OmniAedOverlapSegmenterCtx* ctx, int64_t commit_to_frame) {
    if (commit_to_frame > ctx->available_until_frame) commit_to_frame = ctx->available_until_frame;
    if (commit_to_frame <= ctx->committed_until_frame) return OMNI_OK;

    for (int64_t fidx = ctx->committed_until_frame; fidx < commit_to_frame; ++fidx) {
        AedCommittedFrame out;
        out.speech = 0.0f;
        out.singing = 0.0f;
        out.music = 0.0f;
        out.coverage = 0;
        if (fidx >= ctx->timeline_base_frame &&
            fidx < ctx->timeline_base_frame + (int64_t)ctx->timeline.size()) {
            const AedMergedFrame& f = ctx->timeline[(size_t)(fidx - ctx->timeline_base_frame)];
            if (f.weight_sum > 0.0) {
                out.speech = (float)(f.speech_sum / f.weight_sum);
                out.singing = (float)(f.singing_sum / f.weight_sum);
                out.music = (float)(f.music_sum / f.weight_sum);
                out.coverage = f.coverage;
            }
        }
        ctx->committed_frames.push_back(out);
    }
    ctx->committed_until_frame = commit_to_frame;

    int64_t discard_frames = ctx->committed_until_frame - ctx->timeline_base_frame;
    if (discard_frames > 0 && discard_frames <= (int64_t)ctx->timeline.size()) {
        ctx->timeline.erase(ctx->timeline.begin(), ctx->timeline.begin() + discard_frames);
        ctx->timeline_base_frame += discard_frames;
    }
    return OMNI_OK;
}

static uint32_t mask_for_frame(const AedCommittedFrame& f, const OmniAedOverlapConfig& cfg) {
    uint32_t mask = 0;
    if (f.speech >= cfg.speech_threshold) mask |= AED_MASK_SPEECH;
    if (f.singing >= cfg.singing_threshold) mask |= AED_MASK_SINGING;
    if (f.music >= cfg.music_threshold) mask |= AED_MASK_MUSIC;
    return mask;
}

static int count_mask_bits(uint32_t mask) {
    int n = 0;
    while (mask) {
        n += (int)(mask & 1u);
        mask >>= 1;
    }
    return n;
}

static OmniAedEventKind primary_kind_for_mask(uint32_t mask) {
    if (mask == 0) return OMNI_AED_EVENT_SILENCE;
    if (count_mask_bits(mask) > 1) return OMNI_AED_EVENT_MIXED;
    if (mask & AED_MASK_SPEECH) return OMNI_AED_EVENT_SPEECH;
    if (mask & AED_MASK_SINGING) return OMNI_AED_EVENT_SINGING;
    return OMNI_AED_EVENT_MUSIC;
}

static bool is_transcribable_mask(uint32_t mask) {
    return (mask & (AED_MASK_SPEECH | AED_MASK_SINGING)) != 0;
}

static AedSplitCandidate find_longest_aed_internal_gap(
    const std::vector<AedExtractedEvent>& events,
    int event_start_idx,
    int event_end_idx,
    int current_start_ms,
    int max_end_ms)
{
    AedSplitCandidate best;
    best.valid = false;
    best.run_start_idx = -1;
    best.run_end_idx = -1;
    best.start_ms = 0;
    best.end_ms = 0;

    if (event_start_idx < 0 || event_end_idx <= event_start_idx) return best;

    /* Prefer cuts near the max boundary; front-half gaps create short chunks
     * with too little language context for downstream transcription. */
    int min_gap_start_ms = current_start_ms + (max_end_ms - current_start_ms) / 2;

    int transcribable_remaining = 0;
    for (int j = event_start_idx; j <= event_end_idx; ++j) {
        if (is_transcribable_mask(events[(size_t)j].mask)) {
            ++transcribable_remaining;
        }
    }

    bool has_transcribable_before = false;
    int i = event_start_idx;
    while (i <= event_end_idx) {
        if (is_transcribable_mask(events[(size_t)i].mask)) {
            has_transcribable_before = true;
            --transcribable_remaining;
            ++i;
            continue;
        }

        int run_start = i;
        int run_end = i;
        while (run_end + 1 <= event_end_idx &&
               !is_transcribable_mask(events[(size_t)(run_end + 1)].mask)) {
            ++run_end;
        }

        bool has_transcribable_after = transcribable_remaining > 0;

        const AedExtractedEvent& first = events[(size_t)run_start];
        const AedExtractedEvent& last = events[(size_t)run_end];
        int run_start_ms = first.start_ms;
        int run_end_ms = last.end_ms;
        int duration_ms = run_end_ms - run_start_ms;
        int best_duration_ms = best.valid ? best.end_ms - best.start_ms : -1;

        if (has_transcribable_before &&
            has_transcribable_after &&
            run_start_ms >= min_gap_start_ms &&
            run_start_ms <= max_end_ms &&
            duration_ms > best_duration_ms) {
            best.valid = true;
            best.run_start_idx = run_start;
            best.run_end_idx = run_end;
            best.start_ms = run_start_ms;
            best.end_ms = run_end_ms;
        }

        i = run_end + 1;
    }

    return best;
}

static AedExtractedEvent make_event_from_frames(
    const std::vector<AedCommittedFrame>& frames,
    int64_t base_frame,
    int start_frame,
    int end_frame,
    uint32_t mask,
    int real_duration_ms,
    bool flushed)
{
    AedExtractedEvent ev;
    int64_t absolute_start_frame = base_frame + start_frame;
    int64_t absolute_end_frame = base_frame + end_frame;
    ev.start_ms = (int)(absolute_start_frame * 10);
    ev.end_ms = (int)(absolute_end_frame * 10);
    if (flushed && end_frame == (int)frames.size()) {
        int tail_end = (int)(absolute_end_frame * 10 + 25);
        ev.end_ms = tail_end > real_duration_ms ? real_duration_ms : tail_end;
    }
    ev.mask = mask;
    ev.primary_kind = primary_kind_for_mask(mask);
    double speech = 0.0;
    double singing = 0.0;
    double music = 0.0;
    int count = end_frame - start_frame;
    for (int i = start_frame; i < end_frame; ++i) {
        speech += frames[(size_t)i].speech;
        singing += frames[(size_t)i].singing;
        music += frames[(size_t)i].music;
    }
    if (count > 0) {
        ev.speech_confidence = (float)(speech / count);
        ev.singing_confidence = (float)(singing / count);
        ev.music_confidence = (float)(music / count);
    } else {
        ev.speech_confidence = 0.0f;
        ev.singing_confidence = 0.0f;
        ev.music_confidence = 0.0f;
    }
    ev.confidence = ev.speech_confidence;
    if (ev.singing_confidence > ev.confidence) ev.confidence = ev.singing_confidence;
    if (ev.music_confidence > ev.confidence) ev.confidence = ev.music_confidence;
    return ev;
}

static void update_event_confidence(AedExtractedEvent& ev) {
    ev.confidence = ev.speech_confidence;
    if (ev.singing_confidence > ev.confidence) ev.confidence = ev.singing_confidence;
    if (ev.music_confidence > ev.confidence) ev.confidence = ev.music_confidence;
}

static void merge_event_confidence_weighted(AedExtractedEvent& dst, const AedExtractedEvent& src) {
    int dst_ms = dst.end_ms - dst.start_ms;
    int src_ms = src.end_ms - src.start_ms;
    int total_ms = dst_ms + src_ms;
    if (total_ms <= 0) return;
    dst.speech_confidence =
        (dst.speech_confidence * (float)dst_ms + src.speech_confidence * (float)src_ms) / (float)total_ms;
    dst.singing_confidence =
        (dst.singing_confidence * (float)dst_ms + src.singing_confidence * (float)src_ms) / (float)total_ms;
    dst.music_confidence =
        (dst.music_confidence * (float)dst_ms + src.music_confidence * (float)src_ms) / (float)total_ms;
    update_event_confidence(dst);
}

/* extract_aed_overlap_events / build_aed_overlap_segments return std::vector  */
/* and must keep C++ linkage inside this extern "C" block (MSVC error C2526    */
/* otherwise; GCC/Clang only warn C4190). */
extern "C++" {

static std::vector<AedExtractedEvent> extract_aed_overlap_events(
    const std::vector<AedCommittedFrame>& frames,
    int64_t base_frame,
    const OmniAedOverlapConfig& cfg,
    int real_duration_ms,
    bool flushed,
    bool first_event_is_transcribable_continuation)
{
    std::vector<AedExtractedEvent> events;
    if (frames.empty()) return events;

    int start = 0;
    uint32_t current = mask_for_frame(frames[0], cfg);
    for (int i = 1; i < (int)frames.size(); ++i) {
        uint32_t mask = mask_for_frame(frames[(size_t)i], cfg);
        if (mask != current) {
            events.push_back(make_event_from_frames(frames, base_frame, start, i, current, real_duration_ms, flushed));
            start = i;
            current = mask;
        }
    }
    events.push_back(make_event_from_frames(frames, base_frame, start, (int)frames.size(), current, real_duration_ms, flushed));

    for (size_t i = 0; i < events.size(); ++i) {
        if (is_transcribable_mask(events[i].mask)) {
            /* A pruned first event can be the tail of a longer event already
             * emitted before a max-chunk split. Do not reclassify that tail as
             * silence only because the remaining slice is short. */
            if (i == 0 && first_event_is_transcribable_continuation) {
                continue;
            }
            if (events[i].end_ms - events[i].start_ms < cfg.min_speech_ms) {
                events[i].mask = 0;
                events[i].primary_kind = OMNI_AED_EVENT_SILENCE;
            }
        }
    }

    std::vector<AedExtractedEvent> compacted;
    for (size_t i = 0; i < events.size(); ++i) {
        if (!compacted.empty() && compacted.back().mask == events[i].mask) {
            merge_event_confidence_weighted(compacted.back(), events[i]);
            compacted.back().end_ms = events[i].end_ms;
        } else {
            compacted.push_back(events[i]);
        }
    }
    events.swap(compacted);

    if (cfg.merge_gap_ms <= 0 || events.size() < 3) return events;

    std::vector<AedExtractedEvent> merged;
    for (size_t i = 0; i < events.size(); ++i) {
        int gap_ms = events[i].end_ms - events[i].start_ms;
        bool mergeable_silence_gap = events[i].mask == 0 && gap_ms <= cfg.merge_gap_ms;
        bool mergeable_music_gap =
            events[i].mask == AED_MASK_MUSIC && gap_ms <= cfg.music_gap_tolerance_ms;
        if (!merged.empty() &&
            (mergeable_silence_gap || mergeable_music_gap) &&
            i + 1 < events.size() &&
            is_transcribable_mask(merged.back().mask) &&
            is_transcribable_mask(events[i + 1].mask)) {
            AedExtractedEvent next = events[i + 1];
            merge_event_confidence_weighted(merged.back(), next);
            merged.back().end_ms = next.end_ms;
            merged.back().mask |= next.mask;
            merged.back().primary_kind = primary_kind_for_mask(merged.back().mask);
            ++i;
        } else {
            merged.push_back(events[i]);
        }
    }
    return merged;
}

static std::vector<AedExtractedSegment> build_aed_overlap_segments(
    const std::vector<AedExtractedEvent>& events,
    const OmniAedOverlapConfig& cfg,
    bool flushed)
{
    std::vector<AedExtractedSegment> segments;
    int current_start = -1;
    int current_event_start = -1;
    int current_event_count = 0;
    int last_transcribable_end = -1;
    bool is_continuation = false;

    for (int i = 0; i < (int)events.size(); ++i) {
        const AedExtractedEvent& ev = events[(size_t)i];
        bool transcribable = is_transcribable_mask(ev.mask);
        if (transcribable) {
            if (current_start < 0) {
                current_start = ev.start_ms;
                current_event_start = i;
                current_event_count = 0;
                is_continuation = false;
            }
            current_event_count++;
            last_transcribable_end = ev.end_ms;

            while (last_transcribable_end - current_start >= cfg.max_chunk_ms) {
                AedExtractedSegment seg;
                seg.start_ms = current_start;
                AedSplitCandidate gap = find_longest_aed_internal_gap(
                    events,
                    current_event_start,
                    i,
                    current_start,
                    current_start + cfg.max_chunk_ms);
                seg.end_ms = gap.valid ? gap.start_ms : current_start + cfg.max_chunk_ms;
                seg.event_start_idx = current_event_start;
                seg.event_count = gap.valid
                    ? gap.run_start_idx - current_event_start
                    : current_event_count;
                segments.push_back(seg);
                /* The following segment may be the tail of the same long
                 * transcribable event, so preserve it even when the remaining
                 * tail is shorter than min_speech_ms. */
                is_continuation = true;
                if (gap.valid) {
                    current_start = gap.end_ms;
                    current_event_start = gap.run_end_idx + 1;
                    current_event_count = i - current_event_start + 1;
                } else {
                    current_start = seg.end_ms;
                    int next_event_start = current_event_start;
                    while (next_event_start <= i && events[(size_t)next_event_start].end_ms <= current_start) {
                        next_event_start++;
                    }
                    current_event_start = next_event_start;
                    current_event_count = i - current_event_start + 1;
                }
            }
        } else if (current_start >= 0) {
            current_event_count++;
            if (ev.end_ms - last_transcribable_end >= cfg.hard_split_pause_ms) {
                AedExtractedSegment seg;
                seg.start_ms = current_start;
                seg.end_ms = last_transcribable_end;
                seg.event_start_idx = current_event_start;
                seg.event_count = current_event_count;
                if (seg.end_ms > seg.start_ms &&
                    (is_continuation || seg.end_ms - seg.start_ms >= cfg.min_speech_ms)) {
                    segments.push_back(seg);
                }
                current_start = -1;
                current_event_start = -1;
                current_event_count = 0;
                last_transcribable_end = -1;
                is_continuation = false;
            }
        }
    }

    if (flushed && current_start >= 0 && last_transcribable_end > current_start) {
        AedExtractedSegment seg;
        seg.start_ms = current_start;
        seg.end_ms = last_transcribable_end;
        seg.event_start_idx = current_event_start;
        seg.event_count = current_event_count;
        if (is_continuation || seg.end_ms - seg.start_ms >= cfg.min_speech_ms) {
            segments.push_back(seg);
        }
    }

    return segments;
}

} /* extern "C++" */

static void prune_committed_frames_before(OmniAedOverlapSegmenterCtx* ctx, int64_t cutoff_frame) {
    if (cutoff_frame <= ctx->committed_base_frame) return;
    if (cutoff_frame > ctx->committed_until_frame) cutoff_frame = ctx->committed_until_frame;

    int64_t remove_count = cutoff_frame - ctx->committed_base_frame;
    if (remove_count <= 0) return;
    if (remove_count >= (int64_t)ctx->committed_frames.size()) {
        ctx->committed_frames.clear();
        ctx->committed_base_frame = cutoff_frame;
        return;
    }

    ctx->committed_frames.erase(
        ctx->committed_frames.begin(),
        ctx->committed_frames.begin() + remove_count);
    ctx->committed_base_frame = cutoff_frame;
}

static void prune_aed_overlap_history(
    OmniAedOverlapSegmenterCtx* ctx,
    const std::vector<AedExtractedEvent>& events)
{
    bool has_pending_transcribable = false;
    for (size_t i = 0; i < events.size(); ++i) {
        if (is_transcribable_mask(events[i].mask) && events[i].end_ms > ctx->emitted_until_ms) {
            has_pending_transcribable = true;
            break;
        }
    }

    if (!has_pending_transcribable) {
        prune_committed_frames_before(ctx, ctx->committed_until_frame);
        ctx->committed_base_is_transcribable_continuation = false;
        return;
    }

    int64_t cutoff_frame = ctx->emitted_until_ms / 10;
    int cutoff_ms = (int)(cutoff_frame * 10);
    bool cutoff_splits_transcribable = false;
    for (size_t i = 0; i < events.size(); ++i) {
        const AedExtractedEvent& ev = events[i];
        if (is_transcribable_mask(ev.mask) &&
            ev.start_ms < cutoff_ms &&
            ev.end_ms > cutoff_ms) {
            cutoff_splits_transcribable = true;
            break;
        }
    }
    int64_t old_base_frame = ctx->committed_base_frame;
    prune_committed_frames_before(ctx, cutoff_frame);
    if (ctx->committed_base_frame > old_base_frame) {
        ctx->committed_base_is_transcribable_continuation = cutoff_splits_transcribable;
    }
}

static int choose_aed_overlap_padding_boundary(
    int left_end_ms,
    int right_start_ms,
    int left_pad_end_ms,
    int right_pad_start_ms)
{
    if (right_start_ms <= left_end_ms) return left_end_ms;

    int gap_ms = right_start_ms - left_end_ms;
    int requested_ms = left_pad_end_ms + right_pad_start_ms;
    if (requested_ms <= 0) return left_end_ms;
    if (gap_ms >= requested_ms) return left_end_ms + left_pad_end_ms;

    int64_t weighted = (int64_t)gap_ms * (int64_t)left_pad_end_ms;
    int left_extra_ms = (int)((weighted + requested_ms / 2) / requested_ms);
    if (left_extra_ms < 0) left_extra_ms = 0;
    if (left_extra_ms > left_pad_end_ms) left_extra_ms = left_pad_end_ms;
    if (left_extra_ms > gap_ms) left_extra_ms = gap_ms;
    return left_end_ms + left_extra_ms;
}

static int next_transcribable_event_start_ms(
    const std::vector<AedExtractedEvent>& events,
    int after_ms)
{
    /* extract_aed_overlap_events() returns events in chronological order. */
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].start_ms >= after_ms && is_transcribable_mask(events[i].mask)) {
            return events[i].start_ms;
        }
    }
    return -1;
}

static int copy_aed_overlap_outputs(
    OmniAedOverlapSegmenterCtx* ctx,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count)
{
    if (!out_segments || !out_segment_count || !out_events || !out_event_count) {
        return OMNI_ERR_NULL_POINTER;
    }
    *out_segments = NULL;
    *out_segment_count = 0;
    *out_events = NULL;
    *out_event_count = 0;

    int real_duration_ms = (int)((ctx->total_samples * 1000) / SAMPLE_RATE);
    std::vector<AedExtractedEvent> all_events = extract_aed_overlap_events(
        ctx->committed_frames,
        ctx->committed_base_frame,
        ctx->config,
        real_duration_ms,
        ctx->flushed,
        ctx->committed_base_is_transcribable_continuation);
    std::vector<AedExtractedSegment> all_segments = build_aed_overlap_segments(
        all_events, ctx->config, ctx->flushed);

    std::vector<OmniAedOnlineSegment> segments;
    std::vector<OmniAedOnlineEvent> events;

    for (size_t si = 0; si < all_segments.size(); ++si) {
        AedExtractedSegment seg = all_segments[si];
        if (seg.end_ms <= ctx->emitted_until_ms) continue;

        int padded_start = seg.start_ms - ctx->config.pad_start_ms;
        if (padded_start < 0) padded_start = 0;
        if (padded_start < ctx->emitted_padded_until_ms) {
            padded_start = ctx->emitted_padded_until_ms;
        }

        int padded_end = seg.end_ms + ctx->config.pad_end_ms;
        int next_start_ms = -1;
        if (si + 1 < all_segments.size()) {
            next_start_ms = all_segments[si + 1].start_ms;
        } else {
            next_start_ms = next_transcribable_event_start_ms(all_events, seg.end_ms);
        }

        if (next_start_ms >= 0) {
            padded_end = choose_aed_overlap_padding_boundary(
                seg.end_ms,
                next_start_ms,
                ctx->config.pad_end_ms,
                ctx->config.pad_start_ms);
        } else if (!ctx->flushed) {
            int confirmed_until_ms = (int)(ctx->committed_until_frame * 10);
            int confirmed_gap_ms = confirmed_until_ms - seg.end_ms;
            /* A new transcribable run can be hidden until it reaches
             * min_speech_ms, so keep that much extra confirmed silence before
             * granting full right padding without a visible next boundary. */
            int required_gap_ms =
                ctx->config.pad_end_ms + ctx->config.pad_start_ms + ctx->config.min_speech_ms;
            if (confirmed_gap_ms < required_gap_ms) {
                break;
            }
        }
        if (ctx->flushed && padded_end > real_duration_ms) padded_end = real_duration_ms;
        if (padded_end <= padded_start) continue;

        int local_event_start = (int)events.size();
        for (int ei = 0; ei < seg.event_count; ++ei) {
            int src_idx = seg.event_start_idx + ei;
            if (src_idx < 0 || src_idx >= (int)all_events.size()) continue;
            const AedExtractedEvent& src = all_events[(size_t)src_idx];
            int event_start_ms = src.start_ms > seg.start_ms ? src.start_ms : seg.start_ms;
            int event_end_ms = src.end_ms < seg.end_ms ? src.end_ms : seg.end_ms;
            if (event_end_ms <= event_start_ms) continue;
            OmniAedOnlineEvent ev;
            ev.start = (float)event_start_ms / 1000.0f;
            ev.end = (float)event_end_ms / 1000.0f;
            ev.primary_kind = src.primary_kind;
            ev.kind_mask = src.mask;
            ev.speech_confidence = src.speech_confidence;
            ev.singing_confidence = src.singing_confidence;
            ev.music_confidence = src.music_confidence;
            ev.confidence = src.confidence;
            events.push_back(ev);
        }
        if ((int)events.size() == local_event_start) continue;

        OmniAedOnlineSegment out;
        out.start = (float)padded_start / 1000.0f;
        out.end = (float)padded_end / 1000.0f;
        out.event_start_idx = local_event_start;
        out.event_count = (int)events.size() - local_event_start;
        segments.push_back(out);
        if (seg.end_ms > ctx->emitted_until_ms) ctx->emitted_until_ms = seg.end_ms;
        if (padded_end > ctx->emitted_padded_until_ms) ctx->emitted_padded_until_ms = padded_end;
    }

    prune_aed_overlap_history(ctx, all_events);

    if (!segments.empty()) {
        OmniAedOnlineSegment* p = (OmniAedOnlineSegment*)malloc(sizeof(OmniAedOnlineSegment) * segments.size());
        if (!p) return OMNI_ERR_OUT_OF_MEMORY;
        memcpy(p, segments.data(), sizeof(OmniAedOnlineSegment) * segments.size());
        *out_segments = p;
        *out_segment_count = (int)segments.size();
    }

    if (!events.empty()) {
        OmniAedOnlineEvent* p = (OmniAedOnlineEvent*)malloc(sizeof(OmniAedOnlineEvent) * events.size());
        if (!p) {
            if (*out_segments) {
                free(*out_segments);
                *out_segments = NULL;
                *out_segment_count = 0;
            }
            return OMNI_ERR_OUT_OF_MEMORY;
        }
        memcpy(p, events.data(), sizeof(OmniAedOnlineEvent) * events.size());
        *out_events = p;
        *out_event_count = (int)events.size();
    }

    return OMNI_OK;
}

static int aed_overlap_ingest_int16range(
    OmniAedOverlapSegmenterHandle handle,
    const float* audio_data,
    int num_samples,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data && num_samples > 0) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    if (!out_segments || !out_segment_count || !out_events || !out_event_count) return OMNI_ERR_NULL_POINTER;
    *out_segments = NULL;
    *out_segment_count = 0;
    *out_events = NULL;
    *out_event_count = 0;
    if (handle->flushed) return OMNI_ERR_INVALID_ARG;

    if (num_samples > 0) {
        handle->audio_buffer.insert(handle->audio_buffer.end(), audio_data, audio_data + num_samples);
        handle->total_samples += num_samples;
    }

    int ret = run_ready_aed_overlap_windows(handle);
    if (ret != OMNI_OK) return ret;

    int64_t guard_frames = ms_to_frames(handle->config.edge_guard_ms);
    int64_t commit_to = handle->latest_window_start_frame + ms_to_frames(handle->config.hop_ms) - guard_frames;
    if (commit_to < 0) commit_to = 0;
    ret = commit_aed_overlap_frames(handle, commit_to);
    if (ret != OMNI_OK) return ret;

    return copy_aed_overlap_outputs(handle, out_segments, out_segment_count, out_events, out_event_count);
}

int omni_aed_overlap_segmenter_ingest(
    OmniAedOverlapSegmenterHandle handle,
    const float* audio_data,
    int num_samples,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count)
{
    OMNI_TRY {
        if (!audio_data && num_samples > 0) return OMNI_ERR_NULL_POINTER;
        if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
        std::vector<float> buf = f32_normalize_to_int16_range(audio_data, num_samples);
        return aed_overlap_ingest_int16range(
            handle, buf.data(), num_samples,
            out_segments, out_segment_count, out_events, out_event_count);
    } OMNI_CATCH_BAD_ALLOC {
        return OMNI_ERR_OUT_OF_MEMORY;
    }
}

int omni_aed_overlap_segmenter_ingest_int16(
    OmniAedOverlapSegmenterHandle handle,
    const int16_t* audio_data,
    int num_samples,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count)
{
    OMNI_TRY {
        if (!audio_data && num_samples > 0) return OMNI_ERR_NULL_POINTER;
        if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
        std::vector<float> buf = i16_to_float(audio_data, num_samples);
        return aed_overlap_ingest_int16range(
            handle, buf.data(), num_samples,
            out_segments, out_segment_count, out_events, out_event_count);
    } OMNI_CATCH_BAD_ALLOC {
        return OMNI_ERR_OUT_OF_MEMORY;
    }
}

int omni_aed_overlap_segmenter_flush(
    OmniAedOverlapSegmenterHandle handle,
    OmniAedOnlineSegment** out_segments,
    int* out_segment_count,
    OmniAedOnlineEvent** out_events,
    int* out_event_count)
{
    OMNI_TRY {
        if (!handle) return OMNI_ERR_NULL_HANDLE;
        if (!out_segments || !out_segment_count || !out_events || !out_event_count) return OMNI_ERR_NULL_POINTER;
        *out_segments = NULL;
        *out_segment_count = 0;
        *out_events = NULL;
        *out_event_count = 0;
        if (handle->flushed) return OMNI_OK;

        int hop_samples = ms_to_samples(handle->config.hop_ms);
        while (handle->next_window_start_sample < handle->total_samples) {
            int64_t remaining = handle->total_samples - handle->next_window_start_sample;
            if (remaining < min_aed_inference_samples()) break;
            int64_t start = handle->next_window_start_sample;
            int64_t end = handle->total_samples;
            int ret = run_aed_overlap_window(handle, start, end);
            if (ret != OMNI_OK) return ret;
            handle->next_window_start_sample += hop_samples;
        }

        handle->flushed = true;
        int ret = commit_aed_overlap_frames(handle, handle->available_until_frame);
        if (ret != OMNI_OK) return ret;
        return copy_aed_overlap_outputs(handle, out_segments, out_segment_count, out_events, out_event_count);
    } OMNI_CATCH_BAD_ALLOC {
        return OMNI_ERR_OUT_OF_MEMORY;
    }
}

void omni_aed_overlap_segmenter_destroy(OmniAedOverlapSegmenterHandle handle) {
    if (!handle) return;
    omni_aed_destroy(handle->aed);
    delete handle;
}

} /* extern "C" */
