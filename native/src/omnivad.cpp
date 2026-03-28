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
#include <vector>

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

    if (net->load_param_mem(param_data.data()) != 0) {
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
    cfg.max_speech_frames    = 2000;  /* 20s */
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

/* ========================================================================== */
/*  1. Stream VAD Implementation                                              */
/* ========================================================================== */

struct OmniStreamVadCtx {
    ncnn::Net* net;

    /* Audio buffer (sliding window, keeps last 25ms) */
    std::deque<float> audio_buffer;

    /* Packed cache [w=CACHE_LEN, h=CACHE_SIZE] -> [1, 1024, 19] */
    ncnn::Mat cache_packed;

    /* Frame counter */
    int frame_offset;

    /* Speech threshold */
    float threshold;

    /* CMVN vectors */
    std::vector<float> cmvn_means;
    std::vector<float> cmvn_istd;
    bool use_cmvn;

    /* Fbank computer */
    vad::Fbank* fbank;
};

OmniStreamVadHandle omni_stream_vad_create(
    const char* bundle_path,
    float threshold,
    int* out_error)
{
    set_out_error(out_error, OMNI_OK);

    int ret = validate_threshold(threshold);
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

    OmniStreamVadCtx* ctx = new (std::nothrow) OmniStreamVadCtx();
    if (!ctx) {
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }

    ctx->fbank = NULL;
    ctx->frame_offset = 0;
    ctx->threshold = threshold;
    ctx->use_cmvn = false;
    ctx->cache_packed = ncnn::Mat(CACHE_LEN, CACHE_SIZE);
    ctx->cache_packed.fill(0.0f);

    ctx->net = load_ncnn_from_memory(bundle.param_data, bundle.bin_data, &ret);
    if (!ctx->net) {
        delete ctx;
        set_out_error(out_error, ret);
        return NULL;
    }

    ctx->fbank = new (std::nothrow) vad::Fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    if (!ctx->fbank) {
        delete ctx->net;
        delete ctx;
        set_out_error(out_error, OMNI_ERR_OUT_OF_MEMORY);
        return NULL;
    }

    if (bundle.has_cmvn && (int)bundle.cmvn_means.size() >= FEAT_DIM) {
        ctx->cmvn_means = std::move(bundle.cmvn_means);
        ctx->cmvn_istd = std::move(bundle.cmvn_istd);
        ctx->use_cmvn = true;
    } else if (bundle.has_cmvn) {
        delete ctx->fbank;
        delete ctx->net;
        delete ctx;
        set_out_error(out_error, OMNI_ERR_LOAD_CMVN);
        return NULL;
    }

    set_out_error(out_error, OMNI_OK);
    return ctx;
}

int omni_stream_vad_process(
    OmniStreamVadHandle handle,
    const int16_t* audio_data,
    int num_samples,
    OmniStreamVadResult* result)
{
    if (!handle) return OMNI_ERR_NULL_HANDLE;
    if (!audio_data || !result) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

    OmniStreamVadCtx* ctx = handle;

    /* Convert 16-bit PCM to float and push into buffer */
    for (int i = 0; i < num_samples; ++i) {
        ctx->audio_buffer.push_back((float)audio_data[i]);
    }

    /* Keep buffer size at most frame_length (400 samples = 25ms) */
    while ((int)ctx->audio_buffer.size() > FRAME_LENGTH) {
        ctx->audio_buffer.pop_front();
    }

    /* Not enough data for a full frame yet */
    if ((int)ctx->audio_buffer.size() < FRAME_LENGTH) {
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_ERR_NO_FRAMES;
    }

    /* Extract fbank features for current window */
    std::vector<float> frame_audio(ctx->audio_buffer.begin(), ctx->audio_buffer.end());
    std::vector<float> features;
    int num_frames = ctx->fbank->Compute(frame_audio, &features);

    if (num_frames == 0 || features.empty()) {
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_ERR_NO_FRAMES;
    }

    /* Apply CMVN to the last frame */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
                   features.data(), num_frames, FEAT_DIM);
    }

    /* Take the last frame's features */
    std::vector<float> current_feat(FEAT_DIM);
    memcpy(current_feat.data(),
           features.data() + (num_frames - 1) * FEAT_DIM,
           FEAT_DIM * sizeof(float));

    ctx->frame_offset++;

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

    ncnn::Extractor ex = ctx->net->create_extractor();
    ex.input("in0", in_feat_clone);
    ex.input("in1", cache_clone);

    ncnn::Mat out_probs;
    int ret = ex.extract("out0", out_probs);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] stream: ncnn extract out0 failed: %d\n", ret);
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_ERR_INFERENCE;
    }

    ncnn::Mat new_cache;
    ret = ex.extract("out1", new_cache);
    if (ret != 0) {
        fprintf(stderr, "[omnivad] stream: ncnn extract out1 failed: %d\n", ret);
        result->confidence = 0.0f;
        result->is_speech = false;
        result->frame_offset = ctx->frame_offset;
        return OMNI_ERR_INFERENCE;
    }

    /* Update cache */
    ctx->cache_packed = new_cache;

    float confidence = ((float*)out_probs.data)[0];
    result->confidence = confidence;
    result->is_speech = confidence > ctx->threshold;
    result->frame_offset = ctx->frame_offset;
    return OMNI_OK;
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
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;
    if (num_samples < FRAME_LENGTH) return OMNI_ERR_NO_FRAMES;

    OmniStreamVadCtx* ctx = handle;

    /* Step 1: Whole-file fbank (matches Python AudioFeat.extract) */
    std::vector<float> wave(audio_data, audio_data + num_samples);
    vad::Fbank fbank(FEAT_DIM, SAMPLE_RATE, FRAME_LENGTH, FRAME_SHIFT);
    std::vector<float> features;
    int num_frames = fbank.Compute(wave, &features);
    if (num_frames <= 0) return OMNI_ERR_NO_FRAMES;

    /* Apply CMVN */
    if (ctx->use_cmvn) {
        apply_cmvn(ctx->cmvn_means, ctx->cmvn_istd,
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

        ncnn::Extractor ex = ctx->net->create_extractor();
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
}

int omni_stream_vad_get_frame_offset(OmniStreamVadHandle handle) {
    if (!handle) return 0;
    return handle->frame_offset;
}

void omni_stream_vad_destroy(OmniStreamVadHandle handle) {
    if (!handle) return;
    delete handle->net;
    delete handle->fbank;
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
    if (!audio_data || !out_segments || !out_count) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

    OmniPostConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = omni_post_config_default();
    }
    if (validate_post_config(&cfg) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

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
    if (!audio_data || !out_probs || !out_frames) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

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
    if (!audio_data || !out_segments || !out_count) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

    OmniAedPostConfig cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = omni_aed_post_config_default();
    }
    if (validate_aed_post_config(&cfg) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

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
    if (!audio_data || !out_probs || !out_frames) return OMNI_ERR_NULL_POINTER;
    if (validate_num_samples(num_samples) != OMNI_OK) return OMNI_ERR_INVALID_ARG;

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

} /* extern "C" */
